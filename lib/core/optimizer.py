import torch.optim as Opt
import itertools

from lib.arch.RESIN import RESIN
import torch

class RESIN_optimizer():
    def __init__(self, model:RESIN, lr_G, lr_D) -> None:
        self.optimG = Opt.Adam(itertools.chain(model.G_A.parameters(), model.G_B.parameters()), lr=lr_G)
        self.optimD = Opt.Adam(itertools.chain(model.D_AnisoMIP.parameters(), 
                                               model.D_IsoMIP_1.parameters(),
                                               model.D_IsoMIP_2.parameters(),
                                               model.D_RecA_1.parameters(),
                                               model.D_RecA_2.parameters()), lr=lr_D)

def get_optimizer(args, model):
    return RESIN_optimizer(model.module, args.lr_G, args.lr_D)
