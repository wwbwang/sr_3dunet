import torch
from collections import OrderedDict
from os import path as osp
import os
from tqdm import tqdm
import numpy as np
import random
import itertools
import tifffile

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srgan_model import SRGANModel
from basicsr.models.base_model import BaseModel

from ..utils.data_utils import get_projection, affine_img_VISoR

@MODEL_REGISTRY.register()
class MPCN_VISoR_3projD_Model(BaseModel):

    def __init__(self, opt):
        super(MPCN_VISoR_3projD_Model, self).__init__(opt)

        # define network
        self.net_g_A = build_network(opt['network_g_A'])
        self.net_g_B = build_network(opt['network_g_B'])
        self.net_g_A = self.model_to_device(self.net_g_A)
        self.net_g_B = self.model_to_device(self.net_g_B)
        # self.print_network(self.net_g)
        self.backward_flag = True

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
            
        self.net_d_anisoproj = define_load_network(self.opt['network_d_anisoproj'], 'pretrain_network_d_anisoproj')
        self.net_d_isoproj0 = define_load_network(self.opt['network_d_isoproj'], 'pretrain_network_d_isoproj')
        self.net_d_isoproj1 = define_load_network(self.opt['network_d_isoproj'], 'pretrain_network_d_isoproj')
        self.net_d_A2C = define_load_network(self.opt['network_d_A2C'], 'pretrain_network_d_A2C')
        self.net_d_recA1 = define_load_network(self.opt['network_d_recA1'], 'pretrain_network_d_recA1')
        self.net_d_recA2 = define_load_network(self.opt['network_d_recA2'], 'pretrain_network_d_recA2')

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
        self.net_g_iters = train_opt.get('net_g_iters', 1)
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
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_anisoproj.parameters(),self.net_d_isoproj0.parameters(),self.net_d_isoproj1.parameters(),self.net_d_A2C.parameters(),self.net_d_recA1.parameters(),self.net_d_recA2.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.img_cube = data['img_cube'].to(self.device)
        # self.img_rotated_cube = data['img_rotated_cube'].to(self.device)
        self.img_MIP = data['img_MIP'].to(self.device)

    def optimize_parameters(self, current_iter):
        
        aniso_dimension = self.opt['datasets']['train'].get('aniso_dimension', None)
        # half_iso_dimension=-1
        
        # optimize net_g
        for p in self.net_d_anisoproj.parameters():
            p.requires_grad = False
        for p in self.net_d_isoproj0.parameters():
            p.requires_grad = False
        for p in self.net_d_isoproj1.parameters():
            p.requires_grad = False
        for p in self.net_d_recA1.parameters():
            p.requires_grad = False
        for p in self.net_d_recA2.parameters():
            p.requires_grad = False
        for p in self.net_d_A2C.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.realA = self.img_cube
        self.fakeB = self.net_g_A(self.realA)
        self.affine_fakeB, affine_half_iso_dimension, affine_aniso_dimension\
            = affine_img_VISoR(self.fakeB, aniso_dimension=aniso_dimension, half_iso_dimension=None)
        half_iso_dimension = affine_aniso_dimension
        
        self.recA1 = self.net_g_B(self.fakeB)
        self.fakeC = self.net_g_B(self.affine_fakeB)
        self.recA2 = self.net_g_B(affine_img_VISoR(self.net_g_A(self.fakeC), aniso_dimension=affine_aniso_dimension, half_iso_dimension=affine_half_iso_dimension)[0])
        
        # get iso and aniso projection arrays
        input_iso_proj = self.img_MIP
        
        output_aniso_proj, output_half_iso_proj0, output_half_iso_proj1 = get_projection(self.fakeB, aniso_dimension)
        
        # only use output_aniso_proj, output_half_iso_proj, input_iso_proj
        l_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):     
            # cycle loss
            if self.cri_cycle:
                l_cycle1 = self.cri_cycle(self.realA, self.recA1) + self.cri_cycle_ssim(self.realA, self.recA1)
                l_cycle2 = self.cri_cycle(self.realA, self.recA2) + self.cri_cycle_ssim(self.realA, self.recA2)
                l_total += l_cycle1 + l_cycle2
                loss_dict['l_cycle1'] = l_cycle1
                loss_dict['l_cycle2'] = l_cycle2

            # generator loss
            fakeB_g_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj)
            l_g_B_aniso = self.cri_gan(fakeB_g_anisoproj_pred, True, is_disc=False)
            l_total += l_g_B_aniso
            loss_dict['l_g_B_aniso'] = l_g_B_aniso
            
            fakeB_g_isoproj0_pred = self.net_d_isoproj0(output_half_iso_proj0)
            l_g_B_iso0 = self.cri_gan(fakeB_g_isoproj0_pred, True, is_disc=False)
            l_total += l_g_B_iso0
            loss_dict['l_g_B_iso0'] = l_g_B_iso0

            fakeB_g_isoproj1_pred = self.net_d_isoproj1(output_half_iso_proj1)
            l_g_B_iso1 = self.cri_gan(fakeB_g_isoproj1_pred, True, is_disc=False)
            l_total += l_g_B_iso1
            loss_dict['l_g_B_iso1'] = l_g_B_iso1
            
            recA1_g_pred = self.net_d_recA1(self.recA1)
            l_g_recA1 = self.cri_gan(recA1_g_pred, True, is_disc=False)
            l_total += l_g_recA1
            loss_dict['l_g_recA1'] = l_g_recA1
            
            recA2_g_pred = self.net_d_recA2(self.recA2)
            l_g_recA2 = self.cri_gan(recA2_g_pred, True, is_disc=False)
            l_total += l_g_recA2
            loss_dict['l_g_recA2'] = l_g_recA2
            
            fakeC_g_pred = self.net_d_A2C(self.fakeC)
            l_g_A2C = self.cri_gan(fakeC_g_pred, True, is_disc=False)
            l_total += l_g_A2C
            loss_dict['l_g_A2C'] = l_g_A2C
            
            if self.backward_flag:
                l_total.backward()
            loss_dict['l_total'] = l_total
            
            self.optimizer_g.step()

        if (current_iter % self.net_g_iters == 0):
            # optimize net_d
            for p in self.net_d_anisoproj.parameters():
                p.requires_grad = True
            for p in self.net_d_isoproj0.parameters():
                p.requires_grad = True
            for p in self.net_d_isoproj1.parameters():
                p.requires_grad = True
            for p in self.net_d_recA1.parameters():
                p.requires_grad = True
            for p in self.net_d_recA2.parameters():
                p.requires_grad = True
            for p in self.net_d_A2C.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            l_d_total = 0
            # discriminator loss
            # real
            realB_d_anisoproj_pred = self.net_d_anisoproj(input_iso_proj)
            l_d_real_B_aniso = self.cri_gan(realB_d_anisoproj_pred, True, is_disc=True)
            loss_dict['l_d_real_B_aniso'] = l_d_real_B_aniso
            if self.backward_flag:
                l_d_real_B_aniso.backward()
            
            realB_d_isoproj0_pred = self.net_d_isoproj0(input_iso_proj)
            l_d_real_B_iso0 = self.cri_gan(realB_d_isoproj0_pred, True, is_disc=True)
            loss_dict['l_d_real_B_iso0'] = l_d_real_B_iso0
            if self.backward_flag:
                l_d_real_B_iso0.backward()

            realB_d_isoproj1_pred = self.net_d_isoproj1(input_iso_proj)
            l_d_real_B_iso1 = self.cri_gan(realB_d_isoproj1_pred, True, is_disc=True)
            loss_dict['l_d_real_B_iso1'] = l_d_real_B_iso1
            if self.backward_flag:
                l_d_real_B_iso1.backward()
            
            realC_d_pred = self.net_d_A2C(self.realA)   # same as A
            l_d_real_A2C = self.cri_gan(realC_d_pred, True, is_disc=True)
            loss_dict['l_d_real_A2C'] = l_d_real_A2C
            if self.backward_flag:
                l_d_real_A2C.backward()
            
            recA1_d_pred = self.net_d_recA1(self.realA)
            l_d_real_recA1 = self.cri_gan(recA1_d_pred, True, is_disc=True)
            loss_dict['l_d_real_recA1'] = l_d_real_recA1
            if self.backward_flag:
                l_d_real_recA1.backward()
            
            recA2_d_pred = self.net_d_recA2(self.realA)
            l_d_real_recA2 = self.cri_gan(recA2_d_pred, True, is_disc=True)
            loss_dict['l_d_real_recA2'] = l_d_real_recA2
            if self.backward_flag:
                l_d_real_recA2.backward()
            
            # fake
            fakeB_d_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj.detach())
            l_d_fake_B_aniso = self.cri_gan(fakeB_d_anisoproj_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_aniso'] = l_d_fake_B_aniso
            if self.backward_flag:
                l_d_fake_B_aniso.backward()
            
            fakeB_d_isoproj0_pred = self.net_d_isoproj0(output_half_iso_proj0.detach())
            l_d_fake_B_iso0 = self.cri_gan(fakeB_d_isoproj0_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_iso0'] = l_d_fake_B_iso0
            if self.backward_flag:
                l_d_fake_B_iso0.backward()
                
            fakeB_d_isoproj1_pred = self.net_d_isoproj1(output_half_iso_proj1.detach())
            l_d_fake_B_iso1 = self.cri_gan(fakeB_d_isoproj1_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_iso1'] = l_d_fake_B_iso1
            if self.backward_flag:
                l_d_fake_B_iso1.backward()
            
            fakeC_d_pred = self.net_d_A2C(self.fakeC.detach())
            l_d_fake_A2C = self.cri_gan(fakeC_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_A2C'] = l_d_fake_A2C
            if self.backward_flag:
                l_d_fake_A2C.backward()
            
            recA1_d_pred = self.net_d_recA1(self.recA1.detach())
            l_d_fake_recA1 = self.cri_gan(recA1_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_recA1'] = l_d_fake_recA1
            if self.backward_flag:
                l_d_fake_recA1.backward()
            
            recA2_d_pred = self.net_d_recA2(self.recA2.detach())
            l_d_fake_recA2 = self.cri_gan(recA2_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_recA2'] = l_d_fake_recA2
            if self.backward_flag:
                l_d_fake_recA2.backward()
            
            l_d_total += l_d_real_B_aniso + l_d_real_B_iso0+ l_d_real_B_iso1 + l_d_real_A2C + l_d_real_recA1 + l_d_real_recA2
            l_d_total += l_d_fake_B_aniso + l_d_fake_B_iso0+ l_d_fake_B_iso1 + l_d_fake_A2C + l_d_fake_recA1 + l_d_fake_recA2
            loss_dict['l_d_total'] = l_d_total
            
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.img_srcube = self.net_g_ema(self.img_cube)
        else:
            self.net_g_A.eval()
            with torch.no_grad():
                self.img_srcube = self.net_g_A(self.img_cube)
            self.net_g_A.train()
       
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
            
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_MIP'] = self.img_MIP.detach().cpu()
        out_dict['img_cube'] = self.img_cube.detach().cpu()
        out_dict['img_srcube'] = self.img_srcube.detach().cpu()
        return out_dict
    
    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_anisoproj, 'net_d_anisoproj', current_iter)
        self.save_network(self.net_d_isoproj0, 'net_d_isoproj0', current_iter)
        self.save_network(self.net_d_isoproj1, 'net_d_isoproj1', current_iter)
        self.save_network(self.net_d_A2C, 'net_d_A2C', current_iter)
        self.save_network(self.net_d_recA1, 'net_d_recA1', current_iter)
        self.save_network(self.net_d_recA2, 'net_d_recA2', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class MPCN_VISoR_Model(BaseModel):

    def __init__(self, opt):
        super(MPCN_VISoR_Model, self).__init__(opt)

        # define network
        self.net_g_A = build_network(opt['network_g_A'])
        self.net_g_B = build_network(opt['network_g_B'])
        self.net_g_A = self.model_to_device(self.net_g_A)
        self.net_g_B = self.model_to_device(self.net_g_B)
        # self.print_network(self.net_g)
        self.backward_flag = True

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
            
        self.net_d_anisoproj = define_load_network(self.opt['network_d_anisoproj'], 'pretrain_network_d_anisoproj')
        self.net_d_isoproj = define_load_network(self.opt['network_d_isoproj'], 'pretrain_network_d_isoproj')
        self.net_d_A2C = define_load_network(self.opt['network_d_A2C'], 'pretrain_network_d_A2C')
        self.net_d_recA1 = define_load_network(self.opt['network_d_recA1'], 'pretrain_network_d_recA1')
        self.net_d_recA2 = define_load_network(self.opt['network_d_recA2'], 'pretrain_network_d_recA2')

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
        self.net_g_iters = train_opt.get('net_g_iters', 1)
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
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_anisoproj.parameters(),self.net_d_isoproj.parameters(),self.net_d_A2C.parameters(),self.net_d_recA1.parameters(),self.net_d_recA2.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.img_cube = data['img_cube'].to(self.device)
        # self.img_rotated_cube = data['img_rotated_cube'].to(self.device)
        self.img_MIP = data['img_MIP'].to(self.device)

    def optimize_parameters(self, current_iter):
        
        aniso_dimension = self.opt['datasets']['train'].get('aniso_dimension', None)
        # half_iso_dimension=-1
        
        # optimize net_g
        for p in self.net_d_anisoproj.parameters():
            p.requires_grad = False
        for p in self.net_d_isoproj.parameters():
            p.requires_grad = False
        for p in self.net_d_recA1.parameters():
            p.requires_grad = False
        for p in self.net_d_recA2.parameters():
            p.requires_grad = False
        for p in self.net_d_A2C.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.realA = self.img_cube
        self.fakeB = self.net_g_A(self.realA)
        self.affine_fakeB, affine_half_iso_dimension, affine_aniso_dimension\
            = affine_img_VISoR(self.fakeB, aniso_dimension=aniso_dimension, half_iso_dimension=None)
        half_iso_dimension = affine_aniso_dimension
        
        self.recA1 = self.net_g_B(self.fakeB)
        self.fakeC = self.net_g_B(self.affine_fakeB)
        self.recA2 = self.net_g_B(affine_img_VISoR(self.net_g_A(self.fakeC), aniso_dimension=affine_aniso_dimension, half_iso_dimension=affine_half_iso_dimension)[0])
        
        # get iso and aniso projection arrays
        input_iso_proj = self.img_MIP
        
        output_aniso_proj, output_half_iso_proj0, output_half_iso_proj1 = get_projection(self.fakeB, aniso_dimension)
        proj_index = random.choice(['0', '1'])
        match = lambda x: {
            '0': (output_half_iso_proj0),
            '1': (output_half_iso_proj1)
        }.get(x, ('error0', 'error1'))
        output_half_iso_proj =  match(proj_index)
        
        # only use output_aniso_proj, output_half_iso_proj, input_iso_proj
        l_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):     
            # cycle loss
            if self.cri_cycle:
                l_cycle1 = self.cri_cycle(self.realA, self.recA1) + self.cri_cycle_ssim(self.realA, self.recA1)
                l_cycle2 = self.cri_cycle(self.realA, self.recA2) + self.cri_cycle_ssim(self.realA, self.recA2)
                l_total += l_cycle1 + l_cycle2
                loss_dict['l_cycle1'] = l_cycle1
                loss_dict['l_cycle2'] = l_cycle2

            # generator loss
            fakeB_g_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj)
            l_g_B_aniso = self.cri_gan(fakeB_g_anisoproj_pred, True, is_disc=False)
            l_total += l_g_B_aniso
            loss_dict['l_g_B_aniso'] = l_g_B_aniso
            
            fakeB_g_isoproj_pred = self.net_d_isoproj(output_half_iso_proj)
            l_g_B_iso = self.cri_gan(fakeB_g_isoproj_pred, True, is_disc=False)
            l_total += l_g_B_iso
            loss_dict['l_g_B_iso'] = l_g_B_iso
            
            recA1_g_pred = self.net_d_recA1(self.recA1)
            l_g_recA1 = self.cri_gan(recA1_g_pred, True, is_disc=False)
            l_total += l_g_recA1
            loss_dict['l_g_recA1'] = l_g_recA1
            
            recA2_g_pred = self.net_d_recA2(self.recA2)
            l_g_recA2 = self.cri_gan(recA2_g_pred, True, is_disc=False)
            l_total += l_g_recA2
            loss_dict['l_g_recA2'] = l_g_recA2
            
            fakeC_g_pred = self.net_d_A2C(self.fakeC)
            l_g_A2C = self.cri_gan(fakeC_g_pred, True, is_disc=False)
            l_total += l_g_A2C
            loss_dict['l_g_A2C'] = l_g_A2C
            
            if self.backward_flag:
                l_total.backward()
            loss_dict['l_total'] = l_total
            
            self.optimizer_g.step()

        if (current_iter % self.net_g_iters == 0):
            # optimize net_d
            for p in self.net_d_anisoproj.parameters():
                p.requires_grad = True
            for p in self.net_d_isoproj.parameters():
                p.requires_grad = True
            for p in self.net_d_recA1.parameters():
                p.requires_grad = True
            for p in self.net_d_recA2.parameters():
                p.requires_grad = True
            for p in self.net_d_A2C.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            l_d_total = 0
            # discriminator loss
            # real
            realB_d_anisoproj_pred = self.net_d_anisoproj(input_iso_proj)
            l_d_real_B_aniso = self.cri_gan(realB_d_anisoproj_pred, True, is_disc=True)
            loss_dict['l_d_real_B_aniso'] = l_d_real_B_aniso
            if self.backward_flag:
                l_d_real_B_aniso.backward()
            
            realB_d_isoproj_pred = self.net_d_isoproj(input_iso_proj)
            l_d_real_B_iso = self.cri_gan(realB_d_isoproj_pred, True, is_disc=True)
            loss_dict['l_d_real_B_iso'] = l_d_real_B_iso
            if self.backward_flag:
                l_d_real_B_iso.backward()
            
            realC_d_pred = self.net_d_A2C(self.realA)   # same as A
            l_d_real_A2C = self.cri_gan(realC_d_pred, True, is_disc=True)
            loss_dict['l_d_real_A2C'] = l_d_real_A2C
            if self.backward_flag:
                l_d_real_A2C.backward()
            
            recA1_d_pred = self.net_d_recA1(self.realA)
            l_d_real_recA1 = self.cri_gan(recA1_d_pred, True, is_disc=True)
            loss_dict['l_d_real_recA1'] = l_d_real_recA1
            if self.backward_flag:
                l_d_real_recA1.backward()
            
            recA2_d_pred = self.net_d_recA2(self.realA)
            l_d_real_recA2 = self.cri_gan(recA2_d_pred, True, is_disc=True)
            loss_dict['l_d_real_recA2'] = l_d_real_recA2
            if self.backward_flag:
                l_d_real_recA2.backward()
            
            # fake
            fakeB_d_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj.detach())
            l_d_fake_B_aniso = self.cri_gan(fakeB_d_anisoproj_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_aniso'] = l_d_fake_B_aniso
            if self.backward_flag:
                l_d_fake_B_aniso.backward()
            
            fakeB_d_isoproj_pred = self.net_d_isoproj(output_half_iso_proj.detach())
            l_d_fake_B_iso = self.cri_gan(fakeB_d_isoproj_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_iso'] = l_d_fake_B_iso
            if self.backward_flag:
                l_d_fake_B_iso.backward()
            
            fakeC_d_pred = self.net_d_A2C(self.fakeC.detach())
            l_d_fake_A2C = self.cri_gan(fakeC_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_A2C'] = l_d_fake_A2C
            if self.backward_flag:
                l_d_fake_A2C.backward()
            
            recA1_d_pred = self.net_d_recA1(self.recA1.detach())
            l_d_fake_recA1 = self.cri_gan(recA1_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_recA1'] = l_d_fake_recA1
            if self.backward_flag:
                l_d_fake_recA1.backward()
            
            recA2_d_pred = self.net_d_recA2(self.recA2.detach())
            l_d_fake_recA2 = self.cri_gan(recA2_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_recA2'] = l_d_fake_recA2
            if self.backward_flag:
                l_d_fake_recA2.backward()
            
            l_d_total += l_d_real_B_aniso + l_d_real_B_iso + l_d_real_A2C + l_d_real_recA1 + l_d_real_recA2
            l_d_total += l_d_fake_B_aniso + l_d_fake_B_iso + l_d_fake_A2C + l_d_fake_recA1 + l_d_fake_recA2
            loss_dict['l_d_total'] = l_d_total
            
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.img_srcube = self.net_g_ema(self.img_cube)
        else:
            self.net_g_A.eval()
            with torch.no_grad():
                self.img_srcube = self.net_g_A(self.img_cube)
            self.net_g_A.train()
       
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
   
     
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # just save val images, do not calculate metrics compared with BasicSR
        for idx, val_data in enumerate(dataloader):
            
            if idx==self.opt['val'].get('save_number', None):
                break
            
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            img_MIP = visuals['img_MIP'].cpu().numpy()
            img_cube = visuals['img_cube'].cpu().numpy()
            img_srcube = visuals['img_srcube'].cpu().numpy()

            logger = get_root_logger()
            
            str_index = str(current_iter).zfill(len(str(self.opt['train'].get('total_iter', None))))
            img_name = str_index + '_subindex' + str(idx).zfill(3) + val_data['img_name'][0].split('/')[-1]
            logger.info('Saving and calculating '+img_name)
            
            # save metrics  # TODO use basicsr's func
            if True:
                aniso_dimension = self.opt['datasets']['train'].get('aniso_dimension', None)
                self.realA = self.img_cube
                self.fakeB = self.net_g_A(self.realA)
                self.affine_fakeB, affine_half_iso_dimension, affine_aniso_dimension\
                    = affine_img_VISoR(self.fakeB, aniso_dimension=aniso_dimension, half_iso_dimension=None)
                
                self.fakeC = self.net_g_B(self.affine_fakeB)
                
                # get iso and aniso projection arrays
                input_iso_proj = self.img_MIP
                
                output_aniso_proj, output_half_iso_proj0, output_half_iso_proj1 = get_projection(self.fakeB, aniso_dimension)
                proj_index = random.choice(['0', '1'])
                match = lambda x: {
                    '0': (output_half_iso_proj0),
                    '1': (output_half_iso_proj1)
                }.get(x, ('error0', 'error1'))
                output_half_iso_proj =  match(proj_index)
                
                fakeB_g_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj)
                l_g_B_aniso = self.cri_gan(fakeB_g_anisoproj_pred, True, is_disc=False)
                logger.info(f'l_g_B_aniso')
                logger.info(str(l_g_B_aniso))
                
                fakeB_g_isoproj_pred = self.net_d_isoproj(output_half_iso_proj)
                l_g_B_iso = self.cri_gan(fakeB_g_isoproj_pred, True, is_disc=False)
                logger.info(f'l_g_B_iso')
                logger.info(str(l_g_B_iso))
                
                fakeC_g_pred = self.net_d_A2C(self.fakeC)
                l_g_A2C = self.cri_gan(fakeC_g_pred, True, is_disc=False)
                logger.info(f'l_g_A2C')
                logger.info(str(l_g_A2C))
                
                realB_d_anisoproj_pred = self.net_d_anisoproj(input_iso_proj)
                l_d_real_B_aniso = self.cri_gan(realB_d_anisoproj_pred, True, is_disc=True)
                logger.info(f'l_d_real_B_aniso')
                logger.info(str(l_d_real_B_aniso))
                
                realB_d_isoproj_pred = self.net_d_isoproj(input_iso_proj)
                l_d_real_B_iso = self.cri_gan(realB_d_isoproj_pred, True, is_disc=True)
                logger.info(f'l_d_real_B_iso')
                logger.info(str(l_d_real_B_iso))
                
                realC_d_pred = self.net_d_A2C(self.realA)   # same as A
                l_d_real_A2C = self.cri_gan(realC_d_pred, True, is_disc=True)
                logger.info(f'l_d_real_A2C')
                logger.info(str(l_d_real_A2C))
                
                fakeB_d_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj.detach())
                l_d_fake_B_aniso = self.cri_gan(fakeB_d_anisoproj_pred, False, is_disc=False)
                logger.info(f'l_d_fake_B_aniso')
                logger.info(str(l_d_fake_B_aniso))
                
                fakeB_d_isoproj_pred = self.net_d_isoproj(output_half_iso_proj.detach())
                l_d_fake_B_iso = self.cri_gan(fakeB_d_isoproj_pred, False, is_disc=False)
                logger.info(f'l_d_fake_B_iso')
                logger.info(str(l_d_fake_B_iso))
                
                fakeC_d_pred = self.net_d_A2C(self.fakeC.detach())
                l_d_fake_A2C = self.cri_gan(fakeC_d_pred, False, is_disc=False)
                logger.info(f'l_d_fake_A2C')
                logger.info(str(l_d_fake_A2C))

            if save_img:
                val_rootpath = self.opt['path']['models'].rsplit('models', 1)[0] + 'val_imgs'
                save_img_MIP_path = osp.join(val_rootpath, 'img_MIP')
                save_img_cube_path = osp.join(val_rootpath, 'img_cube')
                save_img_srcube_path = osp.join(val_rootpath, 'img_srcube')
                save_img_fakeC_path = osp.join(val_rootpath, 'img_fakeC')
                save_img_output_aniso_proj = osp.join(val_rootpath, 'img_output_aniso_proj')
                save_img_output_half_iso_proj = osp.join(val_rootpath, 'img_output_half_iso_proj')

                os.makedirs(save_img_MIP_path, exist_ok=True)
                os.makedirs(save_img_cube_path, exist_ok=True)
                os.makedirs(save_img_srcube_path, exist_ok=True)
                os.makedirs(save_img_fakeC_path, exist_ok=True)
                os.makedirs(save_img_output_aniso_proj, exist_ok=True)
                os.makedirs(save_img_output_half_iso_proj, exist_ok=True)
                
                tifffile.imwrite(osp.join(save_img_MIP_path, 'img_MIP_'+img_name), np.squeeze(img_MIP))
                tifffile.imwrite(osp.join(save_img_cube_path, 'img_cube_'+img_name), np.squeeze(img_cube))
                tifffile.imwrite(osp.join(save_img_srcube_path, 'img_srcube_'+img_name), np.squeeze(img_srcube))
                tifffile.imwrite(osp.join(save_img_fakeC_path, 'img_fakeC'+img_name), np.squeeze(self.fakeC.detach().cpu().numpy()))
                tifffile.imwrite(osp.join(save_img_output_aniso_proj, 'img_output_aniso_proj'+img_name), np.squeeze(output_aniso_proj.detach().cpu().numpy()))
                tifffile.imwrite(osp.join(save_img_output_half_iso_proj, 'img_output_half_iso_proj'+img_name), np.squeeze(output_half_iso_proj.detach().cpu().numpy()))
    
            # tentative for out of GPU memory
            del self.img_MIP
            del self.img_cube
            del self.img_srcube
            del self.realA
            del self.fakeB
            del self.affine_fakeB
            del self.fakeC
            torch.cuda.empty_cache()
            
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_MIP'] = self.img_MIP.detach().cpu()
        out_dict['img_cube'] = self.img_cube.detach().cpu()
        out_dict['img_srcube'] = self.img_srcube.detach().cpu()
        return out_dict
    
    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_anisoproj, 'net_d_anisoproj', current_iter)
        self.save_network(self.net_d_isoproj, 'net_d_isoproj', current_iter)
        self.save_network(self.net_d_A2C, 'net_d_A2C', current_iter)
        self.save_network(self.net_d_recA1, 'net_d_recA1', current_iter)
        self.save_network(self.net_d_recA2, 'net_d_recA2', current_iter)
        self.save_training_state(epoch, current_iter)

@MODEL_REGISTRY.register()
class MPCN_VISoR_noA2CModel(BaseModel):

    def __init__(self, opt):
        super(MPCN_VISoR_noA2CModel, self).__init__(opt)

        # define network
        self.net_g_A = build_network(opt['network_g_A'])
        self.net_g_B = build_network(opt['network_g_B'])
        self.net_g_A = self.model_to_device(self.net_g_A)
        self.net_g_B = self.model_to_device(self.net_g_B)
        # self.print_network(self.net_g)
        self.backward_flag = True

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
            
        self.net_d_anisoproj = define_load_network(self.opt['network_d_anisoproj'], 'pretrain_network_d_anisoproj')
        self.net_d_isoproj = define_load_network(self.opt['network_d_isoproj'], 'pretrain_network_d_isoproj')
        self.net_d_A2C = define_load_network(self.opt['network_d_A2C'], 'pretrain_network_d_A2C')
        self.net_d_recA1 = define_load_network(self.opt['network_d_recA1'], 'pretrain_network_d_recA1')
        self.net_d_recA2 = define_load_network(self.opt['network_d_recA2'], 'pretrain_network_d_recA2')

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
        self.net_g_iters = train_opt.get('net_g_iters', 1)
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
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_anisoproj.parameters(),self.net_d_isoproj.parameters(),self.net_d_A2C.parameters(),self.net_d_recA1.parameters(),self.net_d_recA2.parameters()), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.img_cube = data['img_cube'].to(self.device)
        # self.img_rotated_cube = data['img_rotated_cube'].to(self.device)
        self.img_MIP = data['img_MIP'].to(self.device)

    def optimize_parameters(self, current_iter):
        
        aniso_dimension = self.opt['datasets']['train'].get('aniso_dimension', None)
        # half_iso_dimension=-1
        
        # optimize net_g
        for p in self.net_d_anisoproj.parameters():
            p.requires_grad = False
        for p in self.net_d_isoproj.parameters():
            p.requires_grad = False
        for p in self.net_d_recA1.parameters():
            p.requires_grad = False
        for p in self.net_d_recA2.parameters():
            p.requires_grad = False
        for p in self.net_d_A2C.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.realA = self.img_cube
        self.fakeB = self.net_g_A(self.realA)
        self.affine_fakeB, affine_half_iso_dimension, affine_aniso_dimension\
            = affine_img_VISoR(self.fakeB, aniso_dimension=aniso_dimension, half_iso_dimension=None)
        half_iso_dimension = affine_aniso_dimension
        
        self.recA1 = self.net_g_B(self.fakeB)
        self.fakeC = self.net_g_B(self.affine_fakeB)
        self.recA2 = self.net_g_B(affine_img_VISoR(self.net_g_A(self.fakeC), aniso_dimension=affine_aniso_dimension, half_iso_dimension=affine_half_iso_dimension)[0])
        
        # get iso and aniso projection arrays
        input_iso_proj = self.img_MIP
        
        output_aniso_proj, output_half_iso_proj0, output_half_iso_proj1 = get_projection(self.fakeB, aniso_dimension)
        proj_index = random.choice(['0', '1'])
        match = lambda x: {
            '0': (output_half_iso_proj0),
            '1': (output_half_iso_proj1)
        }.get(x, ('error0', 'error1'))
        output_half_iso_proj =  match(proj_index)
        
        # only use output_aniso_proj, output_half_iso_proj, input_iso_proj
        l_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):     
            # cycle loss
            if self.cri_cycle:
                l_cycle1 = self.cri_cycle(self.realA, self.recA1) + self.cri_cycle_ssim(self.realA, self.recA1)
                l_cycle2 = self.cri_cycle(self.realA, self.recA2) + self.cri_cycle_ssim(self.realA, self.recA2)
                l_total += l_cycle1 + l_cycle2
                loss_dict['l_cycle1'] = l_cycle1
                loss_dict['l_cycle2'] = l_cycle2

            # generator loss
            fakeB_g_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj)
            l_g_B_aniso = self.cri_gan(fakeB_g_anisoproj_pred, True, is_disc=False)
            l_total += l_g_B_aniso
            loss_dict['l_g_B_aniso'] = l_g_B_aniso
            
            fakeB_g_isoproj_pred = self.net_d_isoproj(output_half_iso_proj)
            l_g_B_iso = self.cri_gan(fakeB_g_isoproj_pred, True, is_disc=False)
            l_total += l_g_B_iso
            loss_dict['l_g_B_iso'] = l_g_B_iso
            
            recA1_g_pred = self.net_d_recA1(self.recA1)
            l_g_recA1 = self.cri_gan(recA1_g_pred, True, is_disc=False)
            l_total += l_g_recA1
            loss_dict['l_g_recA1'] = l_g_recA1
            
            # recA2_g_pred = self.net_d_recA2(self.recA2)
            # l_g_recA2 = self.cri_gan(recA2_g_pred, True, is_disc=False)
            # l_total += l_g_recA2
            # loss_dict['l_g_recA2'] = l_g_recA2
            
            # fakeC_g_pred = self.net_d_A2C(self.fakeC)
            # l_g_A2C = self.cri_gan(fakeC_g_pred, True, is_disc=False)
            # l_total += l_g_A2C
            # loss_dict['l_g_A2C'] = l_g_A2C
            
            if self.backward_flag:
                l_total.backward()
            loss_dict['l_total'] = l_total
            
            self.optimizer_g.step()

        if (current_iter % self.net_g_iters == 0):
            # optimize net_d
            for p in self.net_d_anisoproj.parameters():
                p.requires_grad = True
            for p in self.net_d_isoproj.parameters():
                p.requires_grad = True
            for p in self.net_d_recA1.parameters():
                p.requires_grad = True
            for p in self.net_d_recA2.parameters():
                p.requires_grad = True
            for p in self.net_d_A2C.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            l_d_total = 0
            # discriminator loss
            # real
            realB_d_anisoproj_pred = self.net_d_anisoproj(input_iso_proj)
            l_d_real_B_aniso = self.cri_gan(realB_d_anisoproj_pred, True, is_disc=True)
            loss_dict['l_d_real_B_aniso'] = l_d_real_B_aniso
            if self.backward_flag:
                l_d_real_B_aniso.backward()
            
            realB_d_isoproj_pred = self.net_d_isoproj(input_iso_proj)
            l_d_real_B_iso = self.cri_gan(realB_d_isoproj_pred, True, is_disc=True)
            loss_dict['l_d_real_B_iso'] = l_d_real_B_iso
            if self.backward_flag:
                l_d_real_B_iso.backward()
            
            # realC_d_pred = self.net_d_A2C(self.realA)   # same as A
            # l_d_real_A2C = self.cri_gan(realC_d_pred, True, is_disc=True)
            # loss_dict['l_d_real_A2C'] = l_d_real_A2C
            # if self.backward_flag:
            #     l_d_real_A2C.backward()
            
            recA1_d_pred = self.net_d_recA1(self.realA)
            l_d_real_recA1 = self.cri_gan(recA1_d_pred, True, is_disc=True)
            loss_dict['l_d_real_recA1'] = l_d_real_recA1
            if self.backward_flag:
                l_d_real_recA1.backward()
            
            # recA2_d_pred = self.net_d_recA2(self.realA)
            # l_d_real_recA2 = self.cri_gan(recA2_d_pred, True, is_disc=True)
            # loss_dict['l_d_real_recA2'] = l_d_real_recA2
            # if self.backward_flag:
            #     l_d_real_recA2.backward()
            
            # fake
            fakeB_d_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj.detach())
            l_d_fake_B_aniso = self.cri_gan(fakeB_d_anisoproj_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_aniso'] = l_d_fake_B_aniso
            if self.backward_flag:
                l_d_fake_B_aniso.backward()
            
            fakeB_d_isoproj_pred = self.net_d_isoproj(output_half_iso_proj.detach())
            l_d_fake_B_iso = self.cri_gan(fakeB_d_isoproj_pred, False, is_disc=False)
            loss_dict['l_d_fake_B_iso'] = l_d_fake_B_iso
            if self.backward_flag:
                l_d_fake_B_iso.backward()
            
            # fakeC_d_pred = self.net_d_A2C(self.fakeC.detach())
            # l_d_fake_A2C = self.cri_gan(fakeC_d_pred, False, is_disc=False)
            # loss_dict['l_d_fake_A2C'] = l_d_fake_A2C
            # if self.backward_flag:
            #     l_d_fake_A2C.backward()
            
            recA1_d_pred = self.net_d_recA1(self.recA1.detach())
            l_d_fake_recA1 = self.cri_gan(recA1_d_pred, False, is_disc=False)
            loss_dict['l_d_fake_recA1'] = l_d_fake_recA1
            if self.backward_flag:
                l_d_fake_recA1.backward()
            
            # recA2_d_pred = self.net_d_recA2(self.recA2.detach())
            # l_d_fake_recA2 = self.cri_gan(recA2_d_pred, False, is_disc=False)
            # loss_dict['l_d_fake_recA2'] = l_d_fake_recA2
            # if self.backward_flag:
            #     l_d_fake_recA2.backward()
            
            l_d_total += l_d_real_B_aniso + l_d_real_B_iso + l_d_real_recA1
            l_d_total += l_d_fake_B_aniso + l_d_fake_B_iso + l_d_fake_recA1
            loss_dict['l_d_total'] = l_d_total
            
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.img_srcube = self.net_g_ema(self.img_cube)
        else:
            self.net_g_A.eval()
            with torch.no_grad():
                self.img_srcube = self.net_g_A(self.img_cube)
            self.net_g_A.train()
       
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
   
     
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # just save val images, do not calculate metrics compared with BasicSR
        for idx, val_data in enumerate(dataloader):
            
            if idx==self.opt['val'].get('save_number', None):
                break
            
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            img_MIP = visuals['img_MIP'].cpu().numpy()
            img_cube = visuals['img_cube'].cpu().numpy()
            img_srcube = visuals['img_srcube'].cpu().numpy()

            logger = get_root_logger()
            
            str_index = str(current_iter).zfill(len(str(self.opt['train'].get('total_iter', None))))
            img_name = str_index + '_subindex' + str(idx).zfill(3) + val_data['img_name'][0].split('/')[-1]
            logger.info('Saving and calculating '+img_name)
            
            # save metrics  # TODO use basicsr's func
            if True:
                aniso_dimension = self.opt['datasets']['train'].get('aniso_dimension', None)
                self.realA = self.img_cube
                self.fakeB = self.net_g_A(self.realA)
                # self.affine_fakeB, affine_half_iso_dimension, affine_aniso_dimension\
                #     = affine_img_VISoR(self.fakeB, aniso_dimension=aniso_dimension, half_iso_dimension=None)
                
                # self.fakeC = self.net_g_B(self.affine_fakeB)
                
                # get iso and aniso projection arrays
                input_iso_proj = self.img_MIP
                
                output_aniso_proj, output_half_iso_proj0, output_half_iso_proj1 = get_projection(self.fakeB, aniso_dimension)
                proj_index = random.choice(['0', '1'])
                match = lambda x: {
                    '0': (output_half_iso_proj0),
                    '1': (output_half_iso_proj1)
                }.get(x, ('error0', 'error1'))
                output_half_iso_proj =  match(proj_index)
                
                fakeB_g_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj)
                l_g_B_aniso = self.cri_gan(fakeB_g_anisoproj_pred, True, is_disc=False)
                logger.info(f'l_g_B_aniso')
                logger.info(str(l_g_B_aniso))
                
                fakeB_g_isoproj_pred = self.net_d_isoproj(output_half_iso_proj)
                l_g_B_iso = self.cri_gan(fakeB_g_isoproj_pred, True, is_disc=False)
                logger.info(f'l_g_B_iso')
                logger.info(str(l_g_B_iso))
                
                # fakeC_g_pred = self.net_d_A2C(self.fakeC)
                # l_g_A2C = self.cri_gan(fakeC_g_pred, True, is_disc=False)
                # logger.info(f'l_g_A2C')
                # logger.info(str(l_g_A2C))
                
                realB_d_anisoproj_pred = self.net_d_anisoproj(input_iso_proj)
                l_d_real_B_aniso = self.cri_gan(realB_d_anisoproj_pred, True, is_disc=True)
                logger.info(f'l_d_real_B_aniso')
                logger.info(str(l_d_real_B_aniso))
                
                realB_d_isoproj_pred = self.net_d_isoproj(input_iso_proj)
                l_d_real_B_iso = self.cri_gan(realB_d_isoproj_pred, True, is_disc=True)
                logger.info(f'l_d_real_B_iso')
                logger.info(str(l_d_real_B_iso))
                
                # realC_d_pred = self.net_d_A2C(self.realA)   # same as A
                # l_d_real_A2C = self.cri_gan(realC_d_pred, True, is_disc=True)
                # logger.info(f'l_d_real_A2C')
                # logger.info(str(l_d_real_A2C))
                
                fakeB_d_anisoproj_pred = self.net_d_anisoproj(output_aniso_proj.detach())
                l_d_fake_B_aniso = self.cri_gan(fakeB_d_anisoproj_pred, False, is_disc=False)
                logger.info(f'l_d_fake_B_aniso')
                logger.info(str(l_d_fake_B_aniso))
                
                fakeB_d_isoproj_pred = self.net_d_isoproj(output_half_iso_proj.detach())
                l_d_fake_B_iso = self.cri_gan(fakeB_d_isoproj_pred, False, is_disc=False)
                logger.info(f'l_d_fake_B_iso')
                logger.info(str(l_d_fake_B_iso))
                
                # fakeC_d_pred = self.net_d_A2C(self.fakeC.detach())
                # l_d_fake_A2C = self.cri_gan(fakeC_d_pred, False, is_disc=False)
                # logger.info(f'l_d_fake_A2C')
                # logger.info(str(l_d_fake_A2C))

            if save_img:
                val_rootpath = self.opt['path']['models'].rsplit('models', 1)[0] + 'val_imgs'
                save_img_MIP_path = osp.join(val_rootpath, 'img_MIP')
                save_img_cube_path = osp.join(val_rootpath, 'img_cube')
                save_img_srcube_path = osp.join(val_rootpath, 'img_srcube')
                # save_img_fakeC_path = osp.join(val_rootpath, 'img_fakeC')
                save_img_output_aniso_proj = osp.join(val_rootpath, 'img_output_aniso_proj')
                save_img_output_half_iso_proj = osp.join(val_rootpath, 'img_output_half_iso_proj')

                os.makedirs(save_img_MIP_path, exist_ok=True)
                os.makedirs(save_img_cube_path, exist_ok=True)
                os.makedirs(save_img_srcube_path, exist_ok=True)
                # os.makedirs(save_img_fakeC_path, exist_ok=True)
                os.makedirs(save_img_output_aniso_proj, exist_ok=True)
                os.makedirs(save_img_output_half_iso_proj, exist_ok=True)
                
                tifffile.imwrite(osp.join(save_img_MIP_path, 'img_MIP_'+img_name), np.squeeze(img_MIP))
                tifffile.imwrite(osp.join(save_img_cube_path, 'img_cube_'+img_name), np.squeeze(img_cube))
                tifffile.imwrite(osp.join(save_img_srcube_path, 'img_srcube_'+img_name), np.squeeze(img_srcube))
                # tifffile.imwrite(osp.join(save_img_fakeC_path, 'img_fakeC'+img_name), np.squeeze(self.fakeC.detach().cpu().numpy()))
                tifffile.imwrite(osp.join(save_img_output_aniso_proj, 'img_output_aniso_proj'+img_name), np.squeeze(output_aniso_proj.detach().cpu().numpy()))
                tifffile.imwrite(osp.join(save_img_output_half_iso_proj, 'img_output_half_iso_proj'+img_name), np.squeeze(output_half_iso_proj.detach().cpu().numpy()))
    
            # tentative for out of GPU memory
            del self.img_MIP
            del self.img_cube
            del self.img_srcube
            del self.realA
            del self.fakeB
            # del self.affine_fakeB
            # del self.fakeC
            torch.cuda.empty_cache()
            
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_MIP'] = self.img_MIP.detach().cpu()
        out_dict['img_cube'] = self.img_cube.detach().cpu()
        out_dict['img_srcube'] = self.img_srcube.detach().cpu()
        return out_dict
    
    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_anisoproj, 'net_d_anisoproj', current_iter)
        self.save_network(self.net_d_isoproj, 'net_d_isoproj', current_iter)
        # self.save_network(self.net_d_A2C, 'net_d_A2C', current_iter)
        self.save_network(self.net_d_recA1, 'net_d_recA1', current_iter)
        # self.save_network(self.net_d_recA2, 'net_d_recA2', current_iter)
        self.save_training_state(epoch, current_iter)

