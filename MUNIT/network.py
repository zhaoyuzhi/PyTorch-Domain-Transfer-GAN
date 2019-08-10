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
# Content Encoder: encodes the content information
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, in_dim, dim, pad_type, activation, norm, sn = True):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dLayer(in_dim, dim, 7, 1, 3, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dLayer(dim, dim * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
            dim *= 2
        # residual blocks
        for j in range(n_res - 1):
            self.model += [ResConv2dLayer(dim, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
        self.model += [ResConv2dLayer(dim, 3, 1, 1, pad_type = pad_type, activation = activation, norm = 'none', sn = sn)]
        self.model = nn.Sequential(*self.model)
        self.content_dim = dim

    def forward(self, x):
        return self.model(x)

# Style Encoder: encodes the style information
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, in_dim, dim, style_dim, pad_type, activation, norm = 'none', sn = True):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dLayer(in_dim, dim, 7, 1, 3, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
        for i in range(2):
            self.model += [Conv2dLayer(dim, dim * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
            dim *= 2
        for j in range(n_downsample - 2):
            self.model += [Conv2dLayer(dim, dim, 4, 2, 1, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.style_dim = style_dim

    def forward(self, x):
        return self.model(x).squeeze(3).squeeze(2)

# Style MLP: further map the style code to a high dimensional one
class StyleMLP(nn.Module):
    def __init__(self, n_mlp, style_dim, latent_dim, out_dim, activation = 'relu', norm = 'none', sn = False):
        super(StyleMLP, self).__init__()
        self.model = []
        self.model += [LinearLayer(style_dim, latent_dim, activation = activation, norm = norm, sn = sn)]
        for i in range(n_mlp - 2):
            self.model += [LinearLayer(latent_dim, latent_dim, activation = activation, norm = norm, sn = sn)]
        self.model += [LinearLayer(latent_dim, out_dim, activation = 'none', norm = norm, sn = sn)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = out_dim

    def forward(self, x):
        return self.model(x)

# Decoder: the style transfer based decoder
class Decoder(nn.Module):
    def __init__(self, n_upsample, dim, out_dim, pad_type, activation = 'relu', norm = 'none', sn = True):
        super(Decoder, self).__init__()
        # AdaIN Resblocks part
        self.adain_1 = ResAdaINConv2dLayer(dim, 3, 1, 1, pad_type = pad_type, activation = activation, norm = 'none', sn = sn)
        self.adain_2 = ResAdaINConv2dLayer(dim, 3, 1, 1, pad_type = pad_type, activation = activation, norm = 'none', sn = sn)
        # Upsampling blocks
        self.model = []
        for i in range(n_upsample):
            self.model += [TransposeConv2dLayer(dim, dim // 2, 3, 1, 1, pad_type = pad_type, activation = activation, norm = norm, sn = sn)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dLayer(dim, out_dim, 3, 1, 1, pad_type = pad_type, activation = 'tanh', norm = 'none', sn = sn)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, style):
        # x: the feature maps from content encoder
        # style: the mean / variance from style encoder and style mlp
        x = self.adain_1(x, style)
        x = self.adain_2(x, style)
        x = self.model(x)
        return x
        
# Generator contains 2 Auto-Encoders
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # Content Encoder: encodes the content information
        # n_downsample:     opt.n_down_content      default: 2          number of stride = 2 convolution layers
        # n_res:            opt.n_res               default: 4          number of res blocks
        # in_dim:           opt.in_dim              default: 3          the input image dimension
        # dim:              opt.start_dim           default: 64         the out channels of first convolution layer
        # pad_type:         opt.pad                 default: reflect
        # activation:       opt.activ_g             default: relu
        # norm:             opt.norm_content        default: in
        # sn = True
        self.content_encoder = ContentEncoder(opt.n_down_content, opt.n_res, opt.in_dim, opt.start_dim, opt.pad, opt.activ_g, opt.norm_content)
        content_dim = self.content_encoder.content_dim                  # default: 256

        # Style Encoder: encodes the style information
        # n_downsample:     opt.n_down_content      default: 2          number of stride = 2 convolution layers
        # in_dim:           opt.in_dim              default: 3          the input image dimension
        # dim:              opt.start_dim           default: 64         the out channels of first convolution layer
        # style_dim:        opt.style_dim           default: 8          the dimension of style latent code
        # pad_type:         opt.pad                 default: reflect
        # activation:       opt.activ_g             default: relu
        # norm:             opt.norm                default: none
        # sn = True
        self.style_encoder = StyleEncoder(opt.n_down_style, opt.in_dim, opt.start_dim, opt.style_dim, opt.pad, opt.activ_g, 'none')

        # Style MLP: maps the style code to a higher dimension
        # n_mlp:            opt.n_mlp               default: 3          number of linear layers
        # style_dim:        opt.style_dim           default: 8          the dimension of style latent code
        # latent_dim:       opt.mlp_dim             default: 256        the dimension of MLP layers
        # out_dim:          need to be computed     default: (512)      the dimension of AdaIN code (256 for mean, 256 for variance)
        # activation:       opt.activ_g             default: relu
        # norm:             opt.norm                default: none
        # sn = False
        self.style_mlp = StyleMLP(opt.n_mlp, opt.style_dim, opt.mlp_dim, content_dim * 2, opt.activ_g, 'none')

        # Decoder
        # n_upsample:       opt.n_down_content      default: 2          number of stride = 2 convolution layers
        # dim:              content_dim             default: 256
        # out_dim:          opt.out_dim             default: 3
        # pad_type:         opt.pad                 default: reflect
        # activation:       opt.activ_g             default: relu
        # norm:             opt.norm_decoder        default: ln
        # sn = True
        self.decoder = Decoder(opt.n_down_content, content_dim, opt.out_dim, opt.pad, opt.activ_g, opt.norm_decoder)

    def encode(self, img):
        # encode an image to its content and style codes
        content = self.content_encoder(img)                             # default: B * 256 * (H / 4) * (W / 4)
        style = self.style_encoder(img)                                 # default: B * 8
        return content, style
    
    def decode(self, content, style):
        style = self.style_mlp(style)
        img = self.decoder(content, style)
        return img

    def forward(self, content_img, style_img):
        # Encoder
        content = self.content_encoder(content_img)
        style = self.style_encoder(style_img)
        # Decoder
        style = self.style_mlp(style)
        img = self.decoder(content, style)
        return img                                                      # out[0]: the system output; out[1]: the detached previous state

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# In each step, Discriminator contains 2 PatchGAN for colorization and saliency map, respectively
# PatchDiscriminator70: PatchGAN discriminator, they share the same architecture
# Usage: Initialize two PatchGAN in training code like:
#        discriminator_color = PatchDiscriminator70(), discriminator_sal = PatchDiscriminator70()
    
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.out_channels, 64, 4, 2, 1, pad_type = opt.pad, norm = 'none')
        self.block2 = Conv2dLayer(64, 128, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_content)
        self.block3 = Conv2dLayer(128, 256, 4, 2, 1, pad_type = opt.pad, norm = opt.norm_content)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(256, 512, 4, 1, 1, pad_type = opt.pad, norm = opt.norm_content)
        self.final2 = Conv2dLayer(512, 1, 4, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        x = self.block1(x)                                      # out: batch * 64 * 128 * 128
        x = self.block2(x)                                      # out: batch * 128 * 64 * 64
        x = self.block3(x)                                      # out: batch * 256 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x
