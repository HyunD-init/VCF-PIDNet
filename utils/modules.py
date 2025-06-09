import torch
import torch.nn as nn

from blocks import *


class Encoder(nn.Module):

    def __init__(self, in_channels, out_c_list, use_drop=False):
        super().__init__()

        self.in_c = in_channels
        self.drop = use_drop

        blocks = list()
        
        in_c = self.in_c
        

        for out_c in out_c_list:
            block = encoder_block(
                in_channels=in_c, out_channels=out_c, use_drop=use_drop, use_pool=False if out_c==out_c_list[0] else True
            )
            blocks.append(block)
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        cache = list()

        for block in self.blocks:
            x = block(x)
            cache.append(x)
        
        return cache

class Unet_Decoder(nn.Module):

    def __init__(self, class_num, enc_out_c_list, use_drop=False, use_transpose=False):
        super().__init__()
        
        blocks = list()


        for i in range(len(enc_out_c_list)-1):
            upsample, block = unet_decoder_block(
                enc_out_c_list[i], enc_out_c_list[i+1], enc_out_c_list[i+1],
                use_drop=use_drop, use_transpose=use_transpose
            )
            blocks.append(nn.ModuleDict({
                "upsample":upsample, "block":block,
            }))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=enc_out_c_list[-1], out_channels=class_num, kernel_size=1, padding="same"),
            nn.Softmax2d()
        )

    def forward(self, enc_cache):
        x = enc_cache[0]
        for skip, block in zip(enc_cache[1:], self.blocks):
            x = block["upsample"](x)
            x = torch.cat([skip, x], axis=-3)
            x = block["block"](x)
        
        output = self.head(x)

        return output

class Unet3Plus_Decoder(nn.Module):

    def __init__(self, skip_channels, class_num, enc_out_c_list, use_skip_drop=False, use_merge_drop=False, use_transpose=False):
        # 1024, 512, 256, 128, 64
        # 64 = skip_channels
        super().__init__()
        blocks = list()
        deep_sup_blocks = list()
        dec_out_c_list = enc_out_c_list

        for dec_idx in range(1, len(dec_out_c_list)):
            skip_blocks = list()
            for enc_idx in range(len(dec_out_c_list)):
                skip_block = unt3plus_decoder_block(
                    in_channels=dec_out_c_list[enc_idx], out_channels=skip_channels,
                    enc_idx=enc_idx, dec_idx=dec_idx, use_drop=use_skip_drop, use_transpose=use_transpose
                )
            

                skip_blocks.append(skip_block)
            dec_out_c_list[dec_idx]=skip_channels
            blocks.append(
                nn.ModuleDict({
                    'skip_blocks':nn.ModuleList(skip_blocks),
                    'merge_block':nn.Sequential(
                        *conv_block(
                            in_channels=len(dec_out_c_list)*skip_channels, out_channels=skip_channels,
                            use_drop=use_merge_drop
                        )
                    )
                })
            )
        
        for i, in_c in enumerate(dec_out_c_list):
            deep_sup_block = [nn.Conv2d(in_channels=in_c, out_channels=class_num, kernel_size=3, padding="same")]
            scale_factor = len(dec_out_c_list) - 1 - i
            if scale_factor > 0:
                deep_sup_block.append(
                    nn.ConvTranspose2d(in_channels=class_num, out_channels=class_num, kernel_size=2**scale_factor, 
                    stride=2**scale_factor) if use_transpose else nn.Upsample(scale_factor=2**scale_factor)
                )
            deep_sup_block.append(nn.Softmax2d())
            deep_sup_blocks.append(nn.Sequential(*deep_sup_block))
        
        self.blocks = nn.ModuleList(blocks)
        self.deep_sup = nn.ModuleList(deep_sup_blocks)
    
    def forward(self, enc_cache):
        # 1024, 512, 256, 128, 64
        output_weight = [0.25, 0.25, 0.25, 0.25, 1.0]
        output = list()

        dec_cache = enc_cache
        for j, block in enumerate(self.blocks):
            skip_cache = list()
            for i, skip_block in enumerate(block['skip_blocks']):
                x = skip_block(dec_cache[i])
                skip_cache.append(x)
            x = torch.cat(skip_cache, axis=-3)
            x = block['merge_block'](x)
            dec_cache[j+1]=x

        for weight, dec_data, sup in zip(output_weight, dec_cache, self.deep_sup):
            output.append(sup(dec_data)*weight)


        output = torch.sum(torch.stack(output, axis=0), axis=0)/2


        return output
        
        
