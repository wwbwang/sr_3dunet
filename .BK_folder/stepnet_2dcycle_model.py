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

@MODEL_REGISTRY.register()
class Cycle_MIP_Model(BaseModel):

    def __init__(self, opt):
        super(Cycle_MIP_Model, self).__init__(opt)

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

        def define_load_network(opt_name, pretrain_name):
            # define network net_d
            net_d = build_network(opt_name)
            net_d = self.model_to_device(net_d)
        
            # load pretrained models
            load_path = self.opt['path'].get(pretrain_name, None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)
            net_d.train()
            return net_d

        self.net_d_A = define_load_network(self.opt['network_d_A'], 'pretrain_network_d_A')
        self.net_d_B = define_load_network(self.opt['network_d_B'], 'pretrain_network_d_B')

        # define losses
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
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_A.parameters(),self.net_d_B.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.img_aniso = data['img_aniso'].to(self.device)
        self.img_iso = data['img_iso'].to(self.device)

    def optimize_parameters(self, current_iter):
        
        self.realA = self.img_aniso
        self.realB = self.img_iso
        self.net_d_A.requires_grad = False
        self.net_d_B.requires_grad = False
        
        # optimize net_g
        self.optimizer_g.zero_grad()
        self.fakeB = self.net_g_A(self.realA)
        self.fakeA = self.net_g_B(self.realB)
        
        self.recA = self.net_g_B(self.fakeB)
        self.recB = self.net_g_A(self.fakeA)

        l_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):              
            # cycle loss
            if self.cri_cycle:
                l_cycle_A = self.cri_cycle(self.realA, self.recA) + self.cri_cycle_ssim(self.realA, self.recA)
                l_cycle_B = self.cri_cycle(self.realB, self.recB) + self.cri_cycle_ssim(self.realB, self.recB)

                l_total += l_cycle_A + l_cycle_B
                loss_dict['l_cycle_A'] = l_cycle_A
                loss_dict['l_cycle_B'] = l_cycle_B
            
            # generator loss
            fake_A_g_pred = self.net_d_A(self.fakeA)
            l_g_A = self.cri_gan(fake_A_g_pred, True, is_disc=False)
            fake_B_g_pred = self.net_d_B(self.fakeB)
            l_g_B = self.cri_gan(fake_B_g_pred, True, is_disc=False)
            
            l_total += l_g_A + l_g_B
            loss_dict['l_g_A'] = l_g_A
            loss_dict['l_g_B'] = l_g_B

            l_total.backward()
            loss_dict['l_total'] = l_total
            self.optimizer_g.step()

        # optimize net_d
        self.net_d_A.requires_grad = True
        self.net_d_B.requires_grad = True

        self.optimizer_d.zero_grad()
        l_d_total = 0
        # discriminator loss
        # real
        real_A_d_pred = self.net_d_A(self.realA)
        l_d_real_A = self.cri_gan(real_A_d_pred, True, is_disc=True)
        loss_dict['l_d_real_A'] = l_d_real_A
        l_d_real_A.backward()
        real_B_d_pred = self.net_d_B(self.realB)
        l_d_real_B = self.cri_gan(real_B_d_pred, True, is_disc=True)
        loss_dict['l_d_real_B'] = l_d_real_B
        l_d_real_B.backward()
        
        # fake
        fake_A_d_pred = self.net_d_A(self.fakeA.detach())
        l_d_fake_A = self.cri_gan(fake_A_d_pred, False, is_disc=True)
        loss_dict['l_d_fake_A'] = l_d_fake_A
        l_d_fake_A.backward()
        fake_B_d_pred = self.net_d_B(self.fakeB.detach())
        l_d_fake_B = self.cri_gan(fake_B_d_pred, False, is_disc=True)
        loss_dict['l_d_fake_B'] = l_d_fake_B
        l_d_fake_B.backward()
        
        l_d_total += l_d_real_A + l_d_real_B + l_d_fake_A + l_d_fake_B
        loss_dict['l_d_total'] = l_d_total
        
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
