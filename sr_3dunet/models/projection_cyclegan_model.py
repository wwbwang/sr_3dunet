import cv2
import math
import numpy as np
import random
import torch
from collections import OrderedDict
import itertools
from os import path as osp
import tqdm

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
        self.net_g_A = self.model_to_device(self.net_g_A)
        self.net_g_B = build_network(opt['network_g_B'])
        self.net_g_B = self.model_to_device(self.net_g_B)
        
        # define network net_d
        self.net_d_A = build_network(self.opt['network_d_A'])
        self.net_d_A = self.model_to_device(self.net_d_A)
        self.net_d_B = build_network(self.opt['network_d_B'])
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
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.net_g_A.train()
        self.net_d_A.train()
        self.net_g_B.train()
        self.net_g_B.train()

        # define losses
        if train_opt.get('cycle_opt'):
            self.cri_cycle = build_loss(train_opt['cycle_opt']).to(self.device)
        else:
            self.cri_cycle = None

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
        self.optimizer_g = self.get_optimizer(optim_type, itertools.chain(self.net_g_A.parameters(),self.net_g_A.parameters()), **train_opt['optim_g_A'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d_A'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, itertools.chain(self.net_d_A.parameters(),self.net_d_A.parameters()), **train_opt['optim_d_A'])
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
        fake_B = torch.clip(self.net_g_A(real_A), 0, 1)
        fake_A = torch.clip(self.net_g_B(real_B), 0, 1)
        
        rec_A = torch.clip(self.net_g_B(fake_B), 0, 1)
        rec_B = torch.clip(self.net_g_A(fake_A), 0, 1)

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
                l_g_total += l_g_cycle_A + l_g_cycle_A
                loss_dict['l_g_cycle_A'] = l_g_cycle_A
                loss_dict['l_g_cycle_B'] = l_g_cycle_B

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

        self.log_dict = self.reduce_loss_dict(loss_dict)

    # no validation
    # 
    # def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
    #     if self.opt['rank'] == 0:
    #         self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    # def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None
    #     use_pbar = self.opt['val'].get('pbar', False)

    #     if with_metrics:
    #         if not hasattr(self, 'metric_results'):  # only execute in the first run
    #             self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
    #         # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
    #         self._initialize_best_metric_results(dataset_name)
    #     # zero self.metric_results
    #     if with_metrics:
    #         self.metric_results = {metric: 0 for metric in self.metric_results}

    #     metric_data = dict()
    #     if use_pbar:
    #         pbar = tqdm(total=len(dataloader), unit='image')

    #     for idx, val_data in enumerate(dataloader):
    #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
    #         self.feed_data(val_data)
    #         self.test()

    #         visuals = self.get_current_visuals()
    #         sr_img = tensor2img([visuals['result']])
    #         metric_data['img'] = sr_img
    #         if 'gt' in visuals:
    #             gt_img = tensor2img([visuals['gt']])
    #             metric_data['img2'] = gt_img
    #             del self.gt

    #         # tentative for out of GPU memory
    #         del self.lq
    #         del self.output
    #         torch.cuda.empty_cache()

    #         if save_img:
    #             if self.opt['is_train']:
    #                 save_img_path = osp.join(self.opt['path']['visualization'], img_name,
    #                                          f'{img_name}_{current_iter}.png')
    #             else:
    #                 if self.opt['val']['suffix']:
    #                     save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
    #                                              f'{img_name}_{self.opt["val"]["suffix"]}.png')
    #                 else:
    #                     save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
    #                                              f'{img_name}_{self.opt["name"]}.png')
    #             imwrite(sr_img, save_img_path)

    #         if with_metrics:
    #             # calculate metrics
    #             for name, opt_ in self.opt['val']['metrics'].items():
    #                 self.metric_results[name] += calculate_metric(metric_data, opt_)
    #         if use_pbar:
    #             pbar.update(1)
    #             pbar.set_description(f'Test {img_name}')
    #     if use_pbar:
    #         pbar.close()

    #     if with_metrics:
    #         for metric in self.metric_results.keys():
    #             self.metric_results[metric] /= (idx + 1)
    #             # update the best metric result
    #             self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

    #         self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g_A, 'net_g_A', current_iter)
        self.save_network(self.net_g_B, 'net_g_B', current_iter)
        self.save_network(self.net_d_A, 'net_d_A', current_iter)
        self.save_network(self.net_d_B, 'net_d_B', current_iter)
        self.save_training_state(epoch, current_iter)
