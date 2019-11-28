import torch
import torch.nn as nn
from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
# Generator contains 2 Auto-Encoders
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Transformer
        self.T1 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = Conv2dLayer(opt.start_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        x = self.E1(x)                                          # out: batch * 64 * 256 * 256
        x = self.E2(x)                                          # out: batch * 128 * 128 * 128
        x = self.E3(x)                                          # out: batch * 256 * 64 * 64
        # Transformation
        x = self.T1(x)                                          # out: batch * 256 * 64 * 64
        x = self.T2(x)                                          # out: batch * 256 * 64 * 64
        x = self.T3(x)                                          # out: batch * 256 * 64 * 64
        x = self.T4(x)                                          # out: batch * 256 * 64 * 64
        x = self.T5(x)                                          # out: batch * 256 * 64 * 64
        x = self.T6(x)                                          # out: batch * 256 * 64 * 64
        # Decode the center code
        x = self.D1(x)                                          # out: batch * 128 * 128 * 128
        x = self.D2(x)                                          # out: batch * 64 * 256 * 256
        x = self.D3(x)                                          # out: batch * out_channel * 256 * 256

        return x

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output.
# CyclePatchDiscriminator: PatchGAN discriminator for CycleGAN
class CyclePatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(CyclePatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.out_channels, 64, 7, 1, 3, pad_type = opt.pad, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(64, 128, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(128, 256, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(256, 512, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(512, 512, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.final2 = Conv2dLayer(512, 1, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none', sn = True)

    def forward(self, x):
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 64 * 128 * 128
        x = self.block3(x)                                      # out: batch * 128 * 64 * 64
        x = self.block4(x)                                      # out: batch * 256 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 16 * 16
        x = self.final2(x)                                      # out: batch * 1 * 16 * 16
        return x
