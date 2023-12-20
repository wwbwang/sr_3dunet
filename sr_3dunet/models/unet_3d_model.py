import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import random

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srgan_model import SRGANModel

from ..utils.data_utils import get_projection



@MODEL_REGISTRY.register()
class Unet_3D(SRGANModel):

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
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

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

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        # we reverse the order of lq and gt for convenient implementation
        self.lq = data.to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        
        self.output = torch.clip(self.output, 0, 1)
        
        # get iso and aniso projection arrays
        aniso_dimension = self.opt['datasets']['train'].get('aniso_dimension', None)
        output_iso_proj, output_aiso_proj0, output_aiso_proj1 = get_projection(self.output, aniso_dimension)
        output_aiso_proj = random.choice([output_aiso_proj0, output_aiso_proj1])
        input_iso_proj, _, _ = get_projection(self.lq, aniso_dimension)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):              
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(output_iso_proj, input_iso_proj)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(output_iso_proj, input_iso_proj)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # generator loss
            fake_g_pred = self.net_d(output_aiso_proj)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # discriminator loss
        # real
        real_d_pred = self.net_d(input_iso_proj)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(output_aiso_proj.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        # l_d_total = (l_d_real + l_d_fake) / 2
        l_d_fake.backward()
        
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
