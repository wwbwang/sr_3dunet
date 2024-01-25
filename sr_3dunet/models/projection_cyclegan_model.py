import cv2
import math
import numpy as np
import random
import torch
from collections import OrderedDict
import itertools
from os import path as osp
import tqdm
import copy

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.losses.gan_loss import g_path_regularize, r1_penalty
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel


@MODEL_REGISTRY.register()
class Projection_CycleGAN_Model(BaseModel):
    """Projection_CycleGAN model."""

    def __init__(self, opt):
        super(Projection_CycleGAN_Model, self).__init__(opt)

        # define network net_g
        self.net_g_A = build_network(opt['network_g_A'])
        self.net_g_B = build_network(opt['network_g_B'])
        self.net_g_A = self.model_to_device(self.net_g_A)
        self.net_g_B = self.model_to_device(self.net_g_B)
        # self.net_g_A = copy.copy(self.net_g_B)
        
        # define network net_d
        self.net_d_A = build_network(self.opt['network_d_A'])
        self.net_d_B = build_network(self.opt['network_d_B'])
        self.net_d_A = self.model_to_device(self.net_d_A)
        self.net_d_B = self.model_to_device(self.net_d_B)
        
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d_A', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d_A', 'params')
            self.load_network(self.net_d_A, load_path, self.opt['path'].get('strict_load_d', True), param_key)
            
        load_path = self.opt['path'].get('pretrain_network_d_B', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d_B', 'params')
            self.load_network(self.net_d_B, load_path, self.opt['path'].get('strict_load_d', True), param_key)
        
        # no ema    ### TODO
        self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.net_g_A.train()
        self.net_d_A.train()
        self.net_g_B.train()
        self.net_d_B.train()

        # define losses
        if train_opt.get('cycle_opt'):
            self.cri_cycle = build_loss(train_opt['cycle_opt']).to(self.device)
        else:
            self.cri_cycle = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('projection_ssim_opt'):
            self.cri_projection_ssim = build_loss(train_opt['projection_ssim_opt']).to(self.device)
        else:
            self.cri_projection_ssim = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g_A'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, itertools.chain(self.net_g_A.parameters(),self.net_g_B.parameters()), **train_opt['optim_g_A'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d_A'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_A.parameters(),self.net_d_B.parameters()), **train_opt['optim_d_A'])
        self.optimizers.append(self.optimizer_d)
        
    def feed_data(self, data):
        self.img_aniso = data['img_aniso'].to(self.device)
        self.img_iso = data['img_iso'].to(self.device)

    def optimize_parameters(self, current_iter):
        
        real_A = self.img_aniso
        real_B = self.img_iso
        self.net_d_A.requires_grad = False
        self.net_d_B.requires_grad = False
        
        # optimize net_g
        self.optimizer_g.zero_grad()
        fake_B = self.net_g_A(real_A)
        fake_A = self.net_g_B(real_B)
        
        rec_A = self.net_g_B(fake_B)
        rec_B = self.net_g_A(fake_A)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):              
            # generator loss
            # GAN loss D_A(G_B(B))
            fake_A_pred = self.net_d_A(fake_A)
            l_g_gan_A = self.cri_gan(fake_A_pred, True, is_disc=False)
            # GAN loss D_B(G_A(A))
            fake_B_pred = self.net_d_B(fake_B)
            l_g_gan_B = self.cri_gan(fake_B_pred, True, is_disc=False)
            
            l_g_total += l_g_gan_A + l_g_gan_B
            loss_dict['l_g_gan_A'] = l_g_gan_A
            loss_dict['l_g_gan_B'] = l_g_gan_B
            
            # cycle loss
            if self.cri_cycle:
                l_g_cycle_A = self.cri_cycle(rec_A, real_A)
                l_g_cycle_B = self.cri_cycle(rec_B, real_B)
                l_g_total += l_g_cycle_A + l_g_cycle_B
                loss_dict['l_g_cycle_A'] = l_g_cycle_A
                loss_dict['l_g_cycle_B'] = l_g_cycle_B
            # (cycle) projection ssim loss
            if self.cri_projection_ssim:
                l_g_projection_ssim_A = self.cri_projection_ssim(rec_A, real_A)
                l_g_projection_ssim_B = self.cri_projection_ssim(rec_B, real_B)
                l_g_total += l_g_projection_ssim_A + l_g_projection_ssim_B
                loss_dict['l_g_projection_ssim_A'] = l_g_projection_ssim_A
                loss_dict['l_g_projection_ssim_B'] = l_g_projection_ssim_B
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep_A, l_g_style_A = self.cri_perceptual(rec_A, real_A)
                l_g_percep_B, l_g_style_B = self.cri_perceptual(rec_B, real_B)
                if l_g_percep_A is not None:
                    l_g_cycle_A += l_g_percep_A
                    l_g_cycle_B += l_g_percep_B
                if l_g_style_A is not None:
                    l_g_cycle_A += l_g_style_A
                    l_g_cycle_B += l_g_style_B

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        self.net_d_A.requires_grad = True
        self.net_d_B.requires_grad = True

        self.optimizer_d.zero_grad()
        # discriminator loss
        def backward_D_basic(net_d, real, fake):
            # Real
            real_pred = net_d(real)
            loss_D_real = self.cri_gan(real_pred, True)
            loss_D_real.backward()
            # Fake  
            fake_pred = net_d(fake.detach())
            loss_D_fake = self.cri_gan(fake_pred, False)
            loss_D_fake.backward()
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            # loss_D.backward()
            return loss_D
        l_d_gan_A = backward_D_basic(self.net_d_A, real_A, fake_A)
        loss_dict['l_d_gan_A'] = l_d_gan_A
        l_d_gan_B = backward_D_basic(self.net_d_B, real_B, fake_B)
        loss_dict['l_d_gan_B'] = l_d_gan_B
        self.optimizer_d.step()
        loss_dict['fake_B_mean'] = fake_B.mean()
        loss_dict['real_B_mean'] = real_B.mean()
        loss_dict['rec_B_mean'] = rec_B.mean()

        self.log_dict = self.reduce_loss_dict(loss_dict)    
        # print("real_A: {}, real_B: {}, fake_A: {}, fake_B: {}, rec_A: {}, rec_B: {}"\
        #     .format(real_A.mean(), real_B.mean(), fake_A.mean(), fake_B.mean(), rec_A.mean(), rec_B.mean()))

    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_A, 'net_d_A', current_iter)
        self.save_network(self.net_d_B, 'net_d_B', current_iter)
        self.save_training_state(epoch, current_iter)