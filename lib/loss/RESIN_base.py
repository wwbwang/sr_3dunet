import torch
import torch.nn as nn
import torchvision
import math

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.getcwd())

from lib.arch.RESIN_base import RESIN_base
from lib.loss.ssim_loss import SSIM_Loss
from lib.utils.utils import get_slant_mip

class GAN_Loss(nn.Module):
    def __init__(self, weight, target_real_label=1.0, target_fake_label=0.0):
        super(GAN_Loss, self).__init__()
        self.weight = weight
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_fn = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(prediction.device)

    def __call__(self, prediction, target_is_real, is_disc):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss_fn(prediction, target_tensor)
        if not is_disc:
            loss = self.weight * loss
        return loss

class L1_Loss(torch.nn.Module):
    def __init__(self, weight):
        super(L1_Loss, self).__init__()
        self.weight = weight
        self.loss_fn = nn.functional.l1_loss

    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        loss = self.weight * loss
        return loss

class RESIN_Loss(nn.Module):
    def __init__(self, model:RESIN_base, 
                 lambda_GAN=1., lambda_Cycle=10., lambda_SSIM=1.,
                 aniso_dim=-2, iso_dim=-1, angel=-45,
                 G_train_it=1, D_train_it=1) -> None:
        super().__init__()

        # model
        self.model = model

        # GAN loss
        self.GAN_Loss = GAN_Loss(weight=lambda_GAN)
        # Cycle loss
        self.Cycle_Loss = L1_Loss(weight=lambda_Cycle)
        # SSIM loss
        self.SSIM_Loss = SSIM_Loss(loss_weight=lambda_SSIM,
                                   data_range=1,channel=1,dim=3)

        self.aniso_dim = aniso_dim
        self.iso_dim = iso_dim
        self.angel = angel
        self.G_train_it = G_train_it
        self.D_train_it = D_train_it

        self.loss_logger = dict()
        self.mip_logger = dict()
    
    def forward(self, real_A, model_out, it):
        self.real_A = real_A
        self.fake_B, self.rec_A1, self.fake_B_T, self.rec_A2, self.rec_A3 = model_out
        self.aniso_mip, self.halfIso_mip1, self.halfIso_mip2 = self.get_mip(self.fake_B, self.aniso_dim)
        b = self.aniso_mip.shape[0]
        self.nrow = math.ceil(b/math.floor(math.sqrt(b)))
        self.mip_logger['aniso_mip'] = torchvision.utils.make_grid(self.aniso_mip, nrow=self.nrow, normalize=True, scale_each=True)
        self.mip_logger['halfIso_mip1'] = torchvision.utils.make_grid(self.halfIso_mip1, nrow=self.nrow, normalize=True, scale_each=True)
        self.mip_logger['halfIso_mip2'] = torchvision.utils.make_grid(self.halfIso_mip2, nrow=self.nrow, normalize=True, scale_each=True)

        # === G forward ===
        if (it+1)%self.G_train_it == 0:
            self.set_requires_grad([self.model.D_AnisoMIP, self.model.D_IsoMIP_1, self.model.D_IsoMIP_2, 
                                    self.model.D_RecA_1, self.model.D_RecA_2, self.model.D_RecA_3],
                                    False)
            loss_G = self.cal_loss_G()
        else:
            loss_G = torch.Tensor([0.0])

        # === D forward ===
        if (it+1)%self.D_train_it == 0:
            self.set_requires_grad([self.model.D_AnisoMIP, self.model.D_IsoMIP_1, self.model.D_IsoMIP_2, 
                                    self.model.D_RecA_1, self.model.D_RecA_2, self.model.D_RecA_3],
                                    True)
            loss_D = self.cal_loss_D()
        else:
            loss_D = torch.Tensor([0.0])

        return loss_G, loss_D, dict(sorted(self.loss_logger.items())), self.mip_logger

    def set_requires_grad(self, nets:list[torch.nn.Module], requires_grad=False):
        if not isinstance(nets, list):
                nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def cal_GAN_loss(self, net:torch.nn.Module, input:torch.Tensor, target:bool, is_disc:bool):
        output = net(input)
        loss = self.GAN_Loss(output, target, is_disc=is_disc)
        return loss
    
    def cal_loss_G(self):
        # cal Cycle loss
        loss_cycle_1 = self.Cycle_Loss(self.real_A, self.rec_A1) + self.SSIM_Loss(self.real_A, self.rec_A1)
        self.loss_logger['loss_G/loss_cycle_1'] = loss_cycle_1.item()

        loss_cycle_2 = self.Cycle_Loss(self.real_A, self.rec_A3) + self.SSIM_Loss(self.real_A, self.rec_A3)
        self.loss_logger['loss_G/loss_cycle_2'] = loss_cycle_2.item()
        
        # cal MIP loss
        loss_aniso_mip = self.cal_GAN_loss(self.model.D_AnisoMIP, self.aniso_mip, True, is_disc=False)
        loss_halfIso_mip1 = self.cal_GAN_loss(self.model.D_IsoMIP_1, self.halfIso_mip1, True, is_disc=False)
        loss_halfIso_mip2 = self.cal_GAN_loss(self.model.D_IsoMIP_2, self.halfIso_mip2, True, is_disc=False)
        self.loss_logger['loss_G/loss_aniso_mip'] = loss_aniso_mip.item()
        self.loss_logger['loss_G/loss_halfIso_mip1'] = loss_halfIso_mip1.item()
        self.loss_logger['loss_G/loss_halfIso_mip2'] = loss_halfIso_mip2.item()

        # cal Cube loss
        loss_rec_A1 = self.cal_GAN_loss(self.model.D_RecA_1, self.rec_A1, True, is_disc=False)
        loss_rec_A2 = self.cal_GAN_loss(self.model.D_RecA_2, self.rec_A2, True, is_disc=False)
        loss_rec_A3 = self.cal_GAN_loss(self.model.D_RecA_3, self.rec_A3, True, is_disc=False)
        self.loss_logger['loss_G/loss_rec_A1'] = loss_rec_A1.item()
        self.loss_logger['loss_G/loss_rec_A2'] = loss_rec_A2.item()
        self.loss_logger['loss_G/loss_rec_A3'] = loss_rec_A3.item()

        loss_G = loss_cycle_1 + loss_cycle_2 +\
                 loss_aniso_mip + loss_halfIso_mip1 + loss_halfIso_mip2 + \
                 loss_rec_A1 + loss_rec_A2 + loss_rec_A3
        self.loss_logger['loss/loss_G'] = loss_G.item()
        return loss_G

    def cal_loss_D(self):
        # real MIP
        ref_iso_mip = get_slant_mip(self.real_A, angel=self.angel, iso_dim=self.iso_dim)
        self.mip_logger['ref_iso_mip'] = torchvision.utils.make_grid(ref_iso_mip, nrow=self.nrow, normalize=True, scale_each=True)

        loss_real_pred_by_D_aniso = self.cal_GAN_loss(self.model.D_AnisoMIP, ref_iso_mip, True, is_disc=True)
        loss_real_pred_by_D_iso1 = self.cal_GAN_loss(self.model.D_IsoMIP_1, ref_iso_mip, True, is_disc=True)
        loss_real_pred_by_D_iso2 = self.cal_GAN_loss(self.model.D_IsoMIP_2, ref_iso_mip, True, is_disc=True)
        self.loss_logger['loss_D/loss_real_pred_by_D_aniso'] = loss_real_pred_by_D_aniso.item()
        self.loss_logger['loss_D/loss_real_pred_by_D_iso1'] = loss_real_pred_by_D_iso1.item()
        self.loss_logger['loss_D/loss_real_pred_by_D_iso2'] = loss_real_pred_by_D_iso2.item()

        # real Cube
        loss_real_pred_by_D_recA1 = self.cal_GAN_loss(self.model.D_RecA_1, self.real_A, True, is_disc=True)
        loss_real_pred_by_D_recA2 = self.cal_GAN_loss(self.model.D_RecA_2, self.real_A, True, is_disc=True)
        loss_real_pred_by_D_recA3 = self.cal_GAN_loss(self.model.D_RecA_3, self.real_A, True, is_disc=True)
        self.loss_logger['loss_D/loss_real_pred_by_D_recA1'] = loss_real_pred_by_D_recA1.item()
        self.loss_logger['loss_D/loss_real_pred_by_D_recA2'] = loss_real_pred_by_D_recA2.item()
        self.loss_logger['loss_D/loss_real_pred_by_D_recA3'] = loss_real_pred_by_D_recA3.item()

        # fake MIP
        loss_fake_pred_by_D_aniso = self.cal_GAN_loss(self.model.D_AnisoMIP, self.aniso_mip.detach(), False, is_disc=True)
        loss_fake_pred_by_D_iso1 = self.cal_GAN_loss(self.model.D_IsoMIP_1, self.halfIso_mip1.detach(), False, is_disc=True)
        loss_fake_pred_by_D_iso2 = self.cal_GAN_loss(self.model.D_IsoMIP_2, self.halfIso_mip2.detach(), False, is_disc=True)
        self.loss_logger['loss_D/loss_fake_pred_by_D_aniso'] = loss_fake_pred_by_D_aniso.item()
        self.loss_logger['loss_D/loss_fake_pred_by_D_iso1'] = loss_fake_pred_by_D_iso1.item()
        self.loss_logger['loss_D/loss_fake_pred_by_D_iso2'] = loss_fake_pred_by_D_iso2.item()

        # fake Cube
        loss_fake_pred_by_D_recA1 = self.cal_GAN_loss(self.model.D_RecA_1, self.rec_A1.detach(), False, is_disc=True)
        loss_fake_pred_by_D_recA2 = self.cal_GAN_loss(self.model.D_RecA_2, self.rec_A2.detach(), False, is_disc=True)
        loss_fake_pred_by_D_recA3 = self.cal_GAN_loss(self.model.D_RecA_3, self.rec_A3.detach(), False, is_disc=True)
        self.loss_logger['loss_D/loss_fake_pred_by_D_recA1'] = loss_fake_pred_by_D_recA1.item()
        self.loss_logger['loss_D/loss_fake_pred_by_D_recA2'] = loss_fake_pred_by_D_recA2.item()
        self.loss_logger['loss_D/loss_fake_pred_by_D_recA3'] = loss_fake_pred_by_D_recA3.item()

        loss_D_real = loss_real_pred_by_D_aniso + loss_real_pred_by_D_iso1 + loss_real_pred_by_D_iso2 + \
                      loss_real_pred_by_D_recA1 + loss_real_pred_by_D_recA2 + loss_real_pred_by_D_recA3
        loss_D_fake = loss_fake_pred_by_D_aniso + loss_fake_pred_by_D_iso1 + loss_fake_pred_by_D_iso2 + \
                      loss_fake_pred_by_D_recA1 + loss_fake_pred_by_D_recA2 + loss_fake_pred_by_D_recA3
        loss_D = loss_D_real + loss_D_fake
        self.loss_logger['loss/loss_D'] = loss_D.item()
        return loss_D
    
    def get_mip(self, img:torch.Tensor, aniso_dim):
        dim_list = [-1, -2, -3]
        dim_list.remove(aniso_dim)
        aniso_mip = torch.max(img, dim=aniso_dim).values
        iso_mip1 = torch.max(img, dim=dim_list[0]).values
        iso_mip2 = torch.max(img, dim=dim_list[1]).values
        return aniso_mip, iso_mip1, iso_mip2

def get_loss(args, model):
    return RESIN_Loss(model=model.module,
                      lambda_GAN=args.lambda_GAN,
                      lambda_Cycle=args.lambda_Cycle,
                      lambda_SSIM=args.lambda_SSIM,
                      aniso_dim=args.aniso_dim,
                      iso_dim=args.iso_dim,
                      angel=args.angel,
                      G_train_it=args.G_train_it,
                      D_train_it=args.D_train_it
                      )

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = RESIN_base(1, 1, [64,128,256], [64,128,256], norm_type=None, aniso_dim=-2, iso_dim=-1)
    model.to(device)

    B = 1
    size = 128

    real_A = torch.rand(B,1,size,size,size).to(device)
    model_out = (
        torch.rand(B,1,size,size,size).to(device),
        torch.rand(B,1,size,size,size).to(device),
        torch.rand(B,1,size,size,size).to(device),
        torch.rand(B,1,size,size,size).to(device),
        torch.rand(B,1,size,size,size).to(device),
    )

    loss_fn = RESIN_Loss(model)

    loss_G, loss_D, loss_logger = loss_fn(real_A, model_out, 1)
    print(loss_G.item(), loss_D.item(), loss_logger.keys())