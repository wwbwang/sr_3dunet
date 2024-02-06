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

@MODEL_REGISTRY.register()
class CycleGAN_2d(BaseModel):

    def __init__(self, opt):
        super(CycleGAN_2d, self).__init__(opt)

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
        self.net_d_A = build_network(self.opt['network_d_A'])
        self.net_d_A = self.model_to_device(self.net_d_A)
        self.net_d_B = build_network(self.opt['network_d_B'])
        self.net_d_B = self.model_to_device(self.net_d_B)
        # self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d_A', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d_A, load_path, self.opt['path'].get('strict_load_d', True), param_key)
        load_path = self.opt['path'].get('pretrain_network_d_B', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d_B, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g_A.train()
        self.net_g_B.train()
        self.net_d_A.train()
        self.net_d_B.train()

        # define losses
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None
        
        if train_opt.get('classifier_opt'):
            self.cri_classifier = build_loss(train_opt['classifier_opt']).to(self.device)
        else:
            self.cri_classifier = None

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
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_A.parameters(),self.net_d_B.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data[0].to(self.device)
        self.hq = data[1].to(self.device)

    def optimize_parameters(self, current_iter):
        
        # optimize net_g
        for p in self.net_d_A.parameters():
            p.requires_grad = False
        for p in self.net_d_B.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.realA = self.lq
        self.realB = self.hq
        
        self.fakeA = self.net_g_B(self.realB)
        self.fakeB = self.net_g_A(self.realA)
        
        self.recB = self.net_g_A(self.fakeA)
        self.recA = self.net_g_B(self.fakeB)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):     
            # cycle loss
            if self.cri_cycle:
                l_g_cycle_A = self.cri_cycle(self.realA, self.recA)
                l_g_cycle_B = self.cri_cycle(self.realB, self.recB)
                
                loss_dict['l_g_cycle_A'] = l_g_cycle_A
                loss_dict['l_g_cycle_B'] = l_g_cycle_B
                l_g_total += l_g_cycle_A + l_g_cycle_B
            
            # cycle_ssim loss
            if self.cri_cycle_ssim:
                l_g_cycle_ssim_A = self.cri_cycle_ssim(self.realA, self.recA)
                l_g_cycle_ssim_B = self.cri_cycle_ssim(self.realB, self.recB)
                
                loss_dict['l_g_cycle_ssim_A'] = l_g_cycle_ssim_A 
                loss_dict['l_g_cycle_ssim_B'] = l_g_cycle_ssim_B 
                l_g_total += l_g_cycle_ssim_A +l_g_cycle_ssim_B
            
            # generator loss
            recA_g_pred, recA_g_label = self.net_d_A(self.recA)
            l_g_B_gan = self.cri_gan(recA_g_pred, True, is_disc=False)
            l_g_B_gan_class = self.cri_classifier(recA_g_label, True, is_disc=False)
            recB_g_pred, recB_g_label = self.net_d_B(self.recB)
            l_g_A_gan = self.cri_gan(recB_g_pred, True, is_disc=False)
            l_g_A_gan_class = self.cri_classifier(recB_g_label, True, is_disc=False)
            
            loss_dict['l_g_A_gan'] = l_g_A_gan
            loss_dict['l_g_B_gan'] = l_g_B_gan
            loss_dict['l_g_A_gan_class'] = l_g_A_gan_class
            loss_dict['l_g_B_gan_class'] = l_g_B_gan_class
            l_g_total += l_g_A_gan + l_g_B_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d_A.parameters():
            p.requires_grad = True
        for p in self.net_d_B.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # discriminator loss
        # real
        realA_d_pred, realA_d_label = self.net_d_A(self.realA)
        l_d_realA = self.cri_gan(realA_d_pred, True, is_disc=True)
        l_d_realA_class = self.cri_gan(realA_d_label, True, is_disc=True)
        loss_dict['l_d_realA'] = l_d_realA
        loss_dict['l_d_realA_class'] = l_d_realA_class
        (l_d_realA+l_d_realA_class).backward()
        realB_d_pred, realB_d_label = self.net_d_B(self.realB)
        l_d_realB = self.cri_gan(realB_d_pred, True, is_disc=True)
        l_d_realB_class = self.cri_gan(realB_d_label, True, is_disc=True)
        loss_dict['l_d_realB'] = l_d_realB
        loss_dict['l_d_realB_class'] = l_d_realB_class
        (l_d_realB+l_d_realB_class).backward()
        
        # fake
        fakeA_d_pred, fakeA_d_label = self.net_d_A(self.recA.detach())
        l_d_fakeA = self.cri_gan(fakeA_d_pred, False, is_disc=True)
        l_d_fakeA_class = self.cri_gan(fakeA_d_label, False, is_disc=True)
        loss_dict['l_d_fakeA'] = l_d_fakeA
        loss_dict['l_d_fakeA_class'] = l_d_fakeA_class
        (l_d_fakeA+l_d_fakeA_class).backward()
        fakeB_d_pred, fakeB_d_label = self.net_d_B(self.recB.detach())
        l_d_fakeB = self.cri_gan(fakeB_d_pred, False, is_disc=True)
        l_d_fakeB_class = self.cri_gan(fakeB_d_label, False, is_disc=True)
        loss_dict['l_d_fakeB'] = l_d_fakeB
        loss_dict['l_d_fakeB_class'] = l_d_fakeB_class
        (l_d_fakeB+l_d_fakeB_class).backward()
        
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_A, 'net_d_A', current_iter)
        self.save_network(self.net_d_B, 'net_d_B', current_iter)
        self.save_training_state(epoch, current_iter)
