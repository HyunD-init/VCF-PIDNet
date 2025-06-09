import torch
import torch.nn as nn






def conv_block(in_channels, out_channels, use_drop=False, use_act=True, use_batch=True, n=2):

    blocks = list()

    for i in range(n):
        blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'))
        blocks.append(nn.BatchNorm2d(out_channels) if use_batch else nn.Identity())
        blocks.append(nn.ReLU(inplace=True) if use_act else nn.Identity())
        blocks.append(nn.Dropout(use_drop*0.1) if use_drop else nn.Identity())
        in_channels=out_channels
    
    return blocks

def encoder_block(in_channels, out_channels, use_drop=False, use_pool=True):

    block = [nn.MaxPool2d(kernel_size=2) if use_pool else nn.Identity()]
    block += conv_block(in_channels=in_channels, out_channels=out_channels, use_drop=use_drop)

    return nn.Sequential(*block)

def unet_decoder_block(in_channels, skip_channels, out_channels, use_drop=False, use_transpose=False):
    
    upsampling = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2) if use_transpose else nn.Upsample(scale_factor=2)
    block = conv_block(out_channels+skip_channels, out_channels, use_drop=use_drop)

    return upsampling, nn.Sequential(*block)

def unt3plus_decoder_block(in_channels, out_channels, enc_idx, dec_idx, use_drop=False, use_transpose=False):
    block = list()
    if enc_idx < dec_idx:
        scale_factor = dec_idx-enc_idx
        block.append(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2**scale_factor,
            stride=2**scale_factor) if use_transpose else nn.Upsample(scale_factor=2**scale_factor)
        )
    elif enc_idx > dec_idx:
        block.append(
            nn.MaxPool2d(kernel_size=2**(enc_idx-dec_idx))
        )
    
    block += conv_block(
        in_channels=in_channels, out_channels=out_channels, use_drop=use_drop
    )
    return nn.Sequential(*block)