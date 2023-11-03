# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import torch.nn as nn
from   utils import CONFIG
from   networks.encoders.resnet_enc import ResNet_D
from   networks.ops import SpectralNorm

class ResShortCut_D(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResShortCut_D, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + CONFIG.model.mask_channel
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        ## out [bn, 32, h/2, w/2]
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out) 
        ## x1 [bn x 32 x h/2 x w/2]
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)
        x2 = self.layer1(out) 
        #  x2 [bn x 64 x h/4 x w/4]
        x3= self.layer2(x2) 
        #  x3 [bn x 128 x h/8 x w/8]
        x4 = self.layer3(x3) 
        #  x4 [bn x 256 x h/16 x w/16]
        out = self.layer_bottleneck(x4) 
        #  out [bn x 512 x h/32 x w/32]

        fea1 = self.shortcut[0](x) # input image and mask
        ## out [bn x 32 x h x w]
        fea2 = self.shortcut[1](x1)
        # fea2 [bn x 32 x h/2 x w/2]
        fea3 = self.shortcut[2](x2)
        # fea3 [bn x 64 x h/4 x w/4]
        fea4 = self.shortcut[3](x3)
        # fea4 [bn x 128 x h/8 x w/8]
        fea5 = self.shortcut[4](x4)
        # fea5 [bn x 256 x h/16 x w/16]

        return out, {'shortcut':(fea1, fea2, fea3, fea4, fea5), 'image':x[:,:3,...]}