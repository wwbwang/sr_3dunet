import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, pixel_unshuffle
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class HighDiscriminator(nn.Module):
    def __init__(self, nFeat=48, output_shape=(1,1,1,1)):
        super(HighDiscriminator, self).__init__()

        self.nFeat = nFeat
        self.output_shape = output_shape
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU
        layer1 = [
            #wn(nn.Conv3d(1, self.nFeat, 3, padding=3 // 2)),
            (nn.Conv3d(1, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
        ]
        layer2 = [
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        self.mp1 = nn.MaxPool3d(2, stride=2)
        layer3 = [
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer4 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer5 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        self.mp2 = nn.MaxPool3d(2, stride=2)

        layer6 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer7 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(True)]
        self.mp3 = nn.MaxPool3d(2, stride=2)
        layer8 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer9 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        self.mp4 = nn.MaxPool3d(2, stride=2)
        layer10 = [nn.Conv3d(self.nFeat,self.nFeat, 3, padding=3 // 2),
            act(inplace=True)]
        # layer11 = [nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2),
        #     act(inplace=True)]
        self.mp4 = nn.MaxPool3d(2, stride=2)
        self.tail = nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)

        self.head = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)
        self.layer6 = nn.Sequential(*layer6)
        self.layer7 = nn.Sequential(*layer7)
        self.layer8 = nn.Sequential(*layer8)
        self.layer9 = nn.Sequential(*layer9)
        self.layer10 = nn.Sequential(*layer10)
        # self.layer11 = nn.Sequential(*layer11)
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
        x = self.layer2(x)
        x = self.mp1(x)
        x = self.layer3(x)
        f1 = self.layer4(x)
        x = self.mp2(f1)
        x = self.layer5(x)
        f2 = self.layer6(x)
        x = self.mp3(f2)
        x = self.layer7(x)
        f3 = self.layer8(x)
        x = self.mp4(f3)
        x = self.layer9(x)
        x = self.layer10(x)
        # x = self.layer11(x)
        return x,f1,f2,f3


