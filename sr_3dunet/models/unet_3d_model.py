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

from ..utils.data_utils import get_projection, affine_img

@MODEL_REGISTRY.register()
class Unet_3D(BaseModel):

    def __init__(self, opt):
        super(Unet_3D, self).__init__(opt)

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

    # def aniso_proj2iso_proj(self, img_aiso_proj):
    #     model = Real2Fake_Generator(input_nc=1, output_nc=1, ngf=64)

    #     model_path = '/home/wangwb/workspace/sr_3dunet/weights/projection_cyclegan_net_g_A_25000.pth'
    #     assert os.path.isfile(model_path), \
    #         f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
    #         f'and put them into the weights folder'

    #     # load checkpoint
    #     loadnet = torch.load(model_path)
    #     model.load_state_dict(loadnet['params'], strict=True)
    #     model.eval()
    #     model = model.to(self.device)
        
    #     return model(img_aiso_proj)

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
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('projection_ssim_opt'):
            self.cri_projection_ssim = build_loss(train_opt['projection_ssim_opt']).to(self.device)
        else:
            self.cri_projection_ssim = None
        
        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

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
        # we reverse the order of lq and gt for convenient implementation
        self.lq = data.to(self.device)

    def optimize_parameters(self, current_iter):
        
        iso_dimension = self.opt['datasets']['train'].get('iso_dimension', None)
        
        # optimize net_g
        for p in self.net_d_A.parameters():
            p.requires_grad = False
        for p in self.net_d_B.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.realA = self.lq
        self.fakeB = self.net_g_A(self.realA)
        self.recA = self.net_g_B(self.fakeB)
        
        # self.fakeB = torch.clip(self.fakeB, 0, 1)
        
        # get iso and aniso projection arrays
        input_iso_proj, input_aiso_proj0, input_aiso_proj1 = get_projection(self.realA, iso_dimension)
        output_iso_proj, output_aiso_proj0, output_aiso_proj1 = get_projection(self.fakeB, iso_dimension)
        aiso_proj_index = random.choice(['0', '1'])
        match = lambda x: {
            '0': (input_aiso_proj0, output_aiso_proj0),
            '1': (input_aiso_proj1, output_aiso_proj1)
        }.get(x, ('error0', 'error1'))
        input_aiso_proj, output_aiso_proj = match(aiso_proj_index) # random.choice([output_aiso_proj0, output_aiso_proj1])

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):     
            # cycle loss
            if self.cri_cycle:
                l_g_cycle = self.cri_cycle(self.realA, self.recA)
                # l_g_total += l_g_cycle
                loss_dict['l_g_cycle'] = l_g_cycle
                l_g_cycle.backward(retain_graph=True)
            # cycle_ssim loss
            if self.cri_cycle_ssim:
                l_g_cycle_ssim = self.cri_cycle_ssim(self.realA, self.recA)
                # l_g_total += l_g_cycle_ssim
                loss_dict['l_g_cycle_ssim'] = l_g_cycle_ssim       
                l_g_cycle_ssim.backward(retain_graph=True)  
            # pixel loss
            if self.cri_pix:
                l_g_pix_real = self.cri_pix(output_iso_proj, input_iso_proj)
                # l_g_pix_psuedo = self.cri_pix(output_aiso_proj, self.aniso_proj2iso_proj(input_aiso_proj))
                l_g_total += l_g_pix_real #  + l_g_pix_psuedo
                loss_dict['l_g_pix_real'] = l_g_pix_real
                # loss_dict['l_g_pix_psuedo'] = l_g_pix_psuedo
            # projection ssim loss
            if self.cri_projection_ssim:
                l_g_projection_ssim_real = self.cri_projection_ssim(output_iso_proj, input_iso_proj)
                # l_g_projection_ssim_psuedo = self.cri_projection_ssim(output_aiso_proj, self.aniso_proj2iso_proj(input_aiso_proj))
                l_g_total += l_g_projection_ssim_real #  + l_g_projection_ssim_psuedo
                loss_dict['l_g_projection_ssim_real'] = l_g_projection_ssim_real
                # loss_dict['l_g_projection_ssim_psuedo'] = l_g_projection_ssim_psuedo
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(output_iso_proj, input_iso_proj)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            
            # self.optimizer_g.step()
            # self.optimizer_g.zero_grad()
            
            self.affine_fakeB = affine_img(self.fakeB, iso_dimension)
            self.affine_recA = self.net_g_B(self.affine_fakeB)        # 两个net_g_B冲突
            
            if current_iter % 100 == 0:
                    import tifffile
                    tifffile.imsave(str(current_iter) + "recA.tif", self.recA[0][0].cpu().detach().numpy())
                    tifffile.imsave(str(current_iter) + "affine_recA.tif", self.affine_recA[0][0].cpu().detach().numpy())
                    tifffile.imsave(str(current_iter) + "fakeB.tif", self.fakeB[0][0].cpu().detach().numpy())
                    tifffile.imsave(str(current_iter) + "realA.tif", self.realA[0][0].cpu().detach().numpy())
                    tifffile.imsave(str(current_iter) + "affine_fakeB.tif", self.affine_fakeB[0][0].cpu().detach().numpy())
            
            # generator loss
            fakeB_g_pred = self.net_d_B(output_aiso_proj)
            l_g_A_gan = self.cri_gan(fakeB_g_pred, True, is_disc=False)
            l_g_total += l_g_A_gan
            loss_dict['l_g_A_gan'] = l_g_A_gan
            
            recA_g_pred = self.net_d_A(self.affine_recA)
            l_g_B_gan = self.cri_gan(recA_g_pred, True, is_disc=False)
            l_g_total += l_g_B_gan
            loss_dict['l_g_B_gan'] = l_g_B_gan

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
        realB_d_pred = self.net_d_B(input_iso_proj)
        l_d_realB = self.cri_gan(realB_d_pred, True, is_disc=True) #  * 0.2
        loss_dict['l_d_realB'] = l_d_realB
        
        realA_d_pred = self.net_d_A(self.realA)
        l_d_realA = self.cri_gan(realA_d_pred, True, is_disc=True)
        loss_dict['l_d_realA'] = l_d_realA
        
        (l_d_realA*5 + l_d_realB).backward()
        # fake
        fakeB_d_pred = self.net_d_B(output_aiso_proj.detach())
        l_d_fakeB = self.cri_gan(fakeB_d_pred, False, is_disc=True) #  * 0.2
        loss_dict['l_d_fakeB'] = l_d_fakeB
        
        fakeA_d_pred = self.net_d_A(self.affine_recA.detach())
        l_d_fakeA = self.cri_gan(fakeA_d_pred, False, is_disc=True)
        loss_dict['l_d_fakeA'] = l_d_fakeA
        
        # l_d_total = (l_d_real + l_d_fake) / 2
        (l_d_fakeA*5 + l_d_fakeB).backward()
        
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
