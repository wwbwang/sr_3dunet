import torch
from torch import nn as nn
from torch.nn import functional as F


from basicsr.archs.arch_util import ResidualBlockNoBN, pixel_unshuffle
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class DegradeNet(nn.Module):
    def __init__(self):
        super(DegradeNet, self).__init__()
        self.nFeat = 48
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU#(inplace=True)

        layer1 = [
            nn.Conv3d(1,self.nFeat,3,3//2),
            act(inplace=True)
            ]
        layer2=[
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.MaxPool3d(3, padding=1),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3,  padding=3 // 2)),
            act(inplace=True),
            #nn.MaxPool3d(3, padding=1),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)
        ]

        self.head = nn.Sequential(*layer1)
        self.body = nn.Sequential(*layer2)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x

