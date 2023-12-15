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

def get_projection(img, aniso_dimension):
    list_dimensions = [-1, -2, -3]
    list_dimensions.remove(aniso_dimension)
    img_aniso = torch.max(img, dim=aniso_dimension).values
    img_iso0 = torch.max(img, dim=list_dimensions[0]).values
    img_iso1 = torch.max(img, dim=list_dimensions[1]).values
    return img_aniso, img_iso0, img_iso1


@MODEL_REGISTRY.register()
class Unet_3D(SRGANModel):

    def feed_data(self, data):
        # we reverse the order of lq and gt for convenient implementation
        self.lq = data.to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        
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
