import torch
from collections import OrderedDict
from os import path as osp
import os
from tqdm import tqdm
import numpy as np
import random
import itertools

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srgan_model import SRGANModel
from basicsr.models.base_model import BaseModel

from ..utils.data_utils import get_projection

def affine_img(img, iso_dimension=-1, aniso_dimension=-2):

    # list_dimensions = [-1, -2, -3]
    # list_dimensions.remove(iso_dimension)
    # aniso_dimension = random.choice(list_dimensions)

    img = img.transpose(iso_dimension, aniso_dimension)
    return img

@MODEL_REGISTRY.register()
class StepNet_Model(BaseModel):

    def __init__(self, opt):
        super(StepNet_Model, self).__init__(opt)

        # define network
        self.net_g_A = build_network(opt['network_g_A'])
        self.net_g_B = build_network(opt['network_g_B'])
        self.net_g_A = self.model_to_device(self.net_g_A)
        self.net_g_B = self.model_to_device(self.net_g_B)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g_A', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g_A', 'params')
            self.load_network(self.net_g_A, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        load_path = self.opt['path'].get('pretrain_network_g_B', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g_B', 'params')
            self.load_network(self.net_g_B, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d_proj = build_network(self.opt['network_d_proj'])
        self.net_d_proj = self.model_to_device(self.net_d_proj)
        self.net_d_iso = build_network(self.opt['network_d_iso'])
        self.net_d_iso = self.model_to_device(self.net_d_iso)

        # load pretrained models
        load_path = self.opt['path'].get('network_d_proj', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d_proj, load_path, self.opt['path'].get('strict_load_d', True), param_key)
        load_path = self.opt['path'].get('pretrain_network_d_iso', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d_iso, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g_A.train()
        self.net_g_B.train()
        self.net_d_proj.train()
        self.net_d_iso.train()

        # define losses
        if train_opt.get('projection_opt'):
            self.cri_projection = build_loss(train_opt['projection_opt']).to(self.device)
        else:
            self.cri_projection = None
        
        if train_opt.get('projection_ssim_opt'):
            self.cri_projection_ssim = build_loss(train_opt['projection_ssim_opt']).to(self.device)
        else:
            self.cri_projection_ssim = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None
        
        if train_opt.get('cycle_opt'):
            self.cri_cycle = build_loss(train_opt['cycle_opt']).to(self.device)
        else:
            self.cri_cycle = None

        if train_opt.get('cycle_ssim_opt'):
            self.cri_cycle_ssim = build_loss(train_opt['cycle_ssim_opt']).to(self.device)
        else:
            self.cri_cycle_ssim = None
            
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, itertools.chain(self.net_g_A.parameters(),self.net_g_B.parameters()), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_proj.parameters(),self.net_d_iso.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data.to(self.device)

    def optimize_parameters(self, current_iter):
        
        iso_dimension = self.opt['datasets']['train'].get('iso_dimension', None)
        
        # optimize net_g
        for p in self.net_d_proj.parameters():
            p.requires_grad = False
        for p in self.net_d_iso.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.realA = self.lq
        self.fakeB = self.net_g_A(self.realA)
        self.affine_fakeB = affine_img(self.fakeB)
        
        self.recA1 = self.net_g_B(self.fakeB)
        # self.fakeC = self.net_g_B(self.affine_fakeB)
        # self.recA2 = self.net_g_B(affine_img(self.net_g_A(self.fakeC)))
        
        # get iso and aniso projection arrays
        input_iso_proj, input_aiso_proj0, input_aiso_proj1 = get_projection(self.realA, iso_dimension)
        output_iso_proj, output_aiso_proj0, output_aiso_proj1 = get_projection(self.fakeB, iso_dimension)
        aiso_proj_index = random.choice(['0', '1'])
        match = lambda x: {
            '0': (input_aiso_proj0, output_aiso_proj0),
            '1': (input_aiso_proj1, output_aiso_proj1)
        }.get(x, ('error0', 'error1'))
        input_aiso_proj, output_aiso_proj = match(aiso_proj_index) # random.choice([output_aiso_proj0, output_aiso_proj1])

        l_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):     
            # cycle loss
            if self.cri_cycle:
                l_cycle1 = self.cri_cycle(self.realA, self.recA1) + self.cri_cycle_ssim(self.realA, self.recA1)
                # l_cycle2 = self.cri_cycle(self.realA, self.recA2) + self.cri_cycle_ssim(self.realA, self.recA2)
                l_total += l_cycle1 # + l_cycle2
                loss_dict['l_cycle1'] = l_cycle1
                # loss_dict['l_cycle2'] = l_cycle2
            # projection loss
            if self.cri_projection:
                l_iso_proj = self.cri_projection(output_iso_proj, input_iso_proj) + self.cri_projection_ssim(output_iso_proj, input_iso_proj)
                l_total += l_iso_proj
                loss_dict['l_iso_proj'] = l_iso_proj

            # generator loss
            # fakeB_g_anisoproj_pred = self.net_d_proj(output_aiso_proj)
            # l_g_A = self.cri_gan(fakeB_g_anisoproj_pred, True, is_disc=False)
            # l_total += l_g_A
            # loss_dict['l_g_A'] = l_g_A
            
            recA1_g_pred = self.net_d_iso(self.recA1)
            l_g_recA1 = self.cri_gan(recA1_g_pred, True, is_disc=False)
            l_total += l_g_recA1
            loss_dict['l_g_recA1'] = l_g_recA1
            
            l_total.backward()

            # recA2_g_pred = self.net_d_iso(self.recA2)
            # l_g_recA2 = self.cri_gan(recA2_g_pred, True, is_disc=False)
            # l_total += l_g_recA2
            # l_g_recA2.backward(retain_graph=True)
            # loss_dict['l_g_recA2'] = l_g_recA2
            
            # fakeC_g_pred = self.net_d_iso(self.fakeC)
            # l_g_A2C = self.cri_gan(fakeC_g_pred, True, is_disc=False)
            # l_total += l_g_A2C
            # l_g_A2C.backward()
            # loss_dict['l_g_A2C'] = l_g_A2C
            
            loss_dict['l_total'] = l_total
            
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d_proj.parameters():
            p.requires_grad = True
        for p in self.net_d_iso.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # discriminator loss
        # real
        # realB_g_anisoproj_pred = self.net_d_proj(input_iso_proj)
        # l_d_real_B = self.cri_gan(realB_g_anisoproj_pred, True, is_disc=True)
        # loss_dict['l_d_real_B'] = l_d_real_B
        # l_d_real_B.backward()
        realC_g_pred = self.net_d_iso(self.realA)   # same as A
        l_d_real_A2C = self.cri_gan(realC_g_pred, True, is_disc=True)
        loss_dict['l_d_real_A2C'] = l_d_real_A2C
        l_d_real_A2C.backward()
        
        # fake
        # fakeB_d_anisoproj_pred = self.net_d_proj(output_aiso_proj.detach())
        # l_d_fake_B = self.cri_gan(fakeB_d_anisoproj_pred, False, is_disc=False)
        # loss_dict['l_d_fake_B'] = l_d_fake_B
        # l_d_fake_B.backward()
        # fakeC_d_pred = self.net_d_iso(self.fakeC.detach())
        # l_d_fake_A2C = self.cri_gan(fakeC_d_pred, False, is_disc=False)
        # loss_dict['l_d_fake_A2C'] = l_d_fake_A2C
        # l_d_fake_A2C.backward()
        
        recA1_d_pred = self.net_d_iso(self.recA1.detach())
        l_d_recA1 = self.cri_gan(recA1_d_pred, False, is_disc=False)
        loss_dict['l_d_recA1'] = l_d_recA1
        l_d_recA1.backward()
        # recA2_d_pred = self.net_d_iso(self.recA2.detach())
        # l_d_recA2 = self.cri_gan(recA2_d_pred, False, is_disc=False)
        # loss_dict['l_d_recA2'] = l_d_recA2
        # l_d_recA2.backward()
        
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_proj, 'net_d_proj', current_iter)
        self.save_network(self.net_d_iso, 'net_d_iso', current_iter)
        self.save_training_state(epoch, current_iter)
