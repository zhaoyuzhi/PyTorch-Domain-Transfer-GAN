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
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 16, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E6 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E7 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E8 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # final concatenate
        self.E9 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 4, 2, 1, pad_type = opt.pad, norm = 'none')
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 16, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)

        self.D5 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D6 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D7 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D8 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D9 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 32 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 64 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 128 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 256 * 32 * 32
        E5 = self.E5(E4)                                        # out: batch * 512 * 16 * 16
        E6 = self.E6(E5)                                        # out: batch * 512 * 8 * 8
        E7 = self.E7(E6)                                        # out: batch * 512 * 4 * 4
        E8 = self.E8(E7)                                        # out: batch * 512 * 2 * 2
        # final encoding
        E9 = self.E9(E8)                                        # out: batch * 512 * 1 * 1
        # Decode the center code
        D1 = self.D1(E9)                                        # out: batch * 512 * 2 * 2
        D1 = torch.cat((D1, E8), 1)                             # out: batch * 1024 * 2 * 2
        D2 = self.D2(D1)                                        # out: batch * 512 * 4 * 4
        D2 = torch.cat((D2, E7), 1)                             # out: batch * 1024 * 4 * 4
        D3 = self.D3(D2)                                        # out: batch * 512 * 8 * 8
        D3 = torch.cat((D3, E6), 1)                             # out: batch * 1024 * 8 * 8
        D4 = self.D4(D3)                                        # out: batch * 512 * 16 * 16
        D4 = torch.cat((D4, E5), 1)                             # out: batch * 1024 * 16 * 16
        D5 = self.D5(D4)                                        # out: batch * 256 * 32 * 32
        D5 = torch.cat((D5, E4), 1)                             # out: batch * 512 * 32 * 32
        D6 = self.D6(D5)                                        # out: batch * 128 * 64 * 64
        D6 = torch.cat((D6, E3), 1)                             # out: batch * 256 * 64 * 64
        D7 = self.D7(D6)                                        # out: batch * 64 * 128 * 128
        D7 = torch.cat((D7, E2), 1)                             # out: batch * 128 * 128 * 128
        D8 = self.D8(D7)                                        # out: batch * 32 * 256 * 256
        D8 = torch.cat((D8, E1), 1)                             # out: batch * 64 * 256 * 256
        # final decoding
        x = self.D9(D8)                                         # out: batch * out_channel * 256 * 256

        return x

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator for Pix2Pix
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = 'none')
        self.block3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.block4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(512, 512, 4, 1, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.final2 = Conv2dLayer(512, 1, 4, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none', sn = True)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # img_A: grayscale input; img_B: ab embedding output, if colorization
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 3 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 64 * 128 * 128
        x = self.block3(x)                                      # out: batch * 128 * 64 * 64
        x = self.block4(x)                                      # out: batch * 256 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x
