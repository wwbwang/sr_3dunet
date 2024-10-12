import torch
import torch.optim as Opt
import itertools

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.getcwd())

from lib.arch.RESIN_small import RESIN_small
from lib.arch.RESIN_base import RESIN_base
from lib.arch.RESIN_base_subtract import RESIN_base_subtract

class Adam_Opt(Opt.Adam):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(Adam_Opt, self).__init__(model.parameters(), lr=lr_start)

class SGD_Opt(Opt.SGD):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(SGD_Opt, self).__init__(model.parameters(), lr=lr_start)

class Adagrad_Opt(Opt.Adagrad):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(Adagrad_Opt, self).__init__(model.parameters(), lr=lr_start)

class RESIN_small_Optim():
    def __init__(self, model:RESIN_small, args) -> None:
        lr_G = args.lr_G
        lr_D = args.lr_D
        self.optimG = Opt.Adam(itertools.chain(model.G_A.parameters(), model.G_B.parameters()), lr=lr_G)
        self.optimD = Opt.Adam(itertools.chain(model.D_AnisoMIP.parameters(), 
                                               model.D_IsoMIP_1.parameters(),
                                               model.D_IsoMIP_2.parameters(),
                                               model.D_RecA_1.parameters(),
                                               model.D_RecA_2.parameters()), lr=lr_D)
        
class RESIN_base_subtract_Optim():
    def __init__(self, model:RESIN_base_subtract, args) -> None:
        lr_G = args.lr_G
        lr_D = args.lr_D
        self.optimG = Opt.Adam(itertools.chain(model.G_A.parameters(), model.G_B.parameters()), lr=lr_G)
        self.optimD = Opt.Adam(itertools.chain(model.D_AnisoMIP.parameters(), 
                                               model.D_IsoMIP_1.parameters(),
                                               model.D_IsoMIP_2.parameters(),
                                               model.D_RecA_1.parameters(),
                                               model.D_RecA_2.parameters()), lr=lr_D)
        
class RESIN_base_Optim():
    def __init__(self, model:RESIN_base, args) -> None:
        lr_G = args.lr_G
        lr_D = args.lr_D
        self.optimG = Opt.Adam(itertools.chain(model.G_A.parameters(), model.G_B.parameters()), lr=lr_G)
        self.optimD = Opt.Adam(itertools.chain(model.D_AnisoMIP.parameters(), 
                                               model.D_IsoMIP_1.parameters(),
                                               model.D_IsoMIP_2.parameters(),
                                               model.D_RecA_1.parameters(),
                                               model.D_RecA_2.parameters(),
                                               model.D_RecA_3.parameters()), lr=lr_D)

def get_optimizer(args, model):
    opt_fns = {
        'adam': Adam_Opt,
        'sgd': SGD_Opt,
        'adagrad': Adagrad_Opt,
        'RESIN_small': RESIN_small_Optim,
        'RESIN_base_subtract': RESIN_base_subtract_Optim,
        'RESIN_base': RESIN_base_Optim
    }
    opt_fn = opt_fns.get(args.optimizer, "Invalid Optimizer")
    opt = opt_fn(model.module, args)
    return opt

if __name__ == '__main__':
    import argparse
    model = argparse.Namespace()
    model.module = RESIN_base()
    args = argparse.Namespace()
    args.optimizer = 'RESIN_base'
    args.lr_D = 1e-4
    args.lr_G = 1e-4
    args.lr_start = 1e-4
    opt = get_optimizer(args, model)