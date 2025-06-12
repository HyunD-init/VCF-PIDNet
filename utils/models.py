import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

try:
    from modules import Encoder, Unet_Decoder, Unet3Plus_Decoder
    from PIDNet.models.pidnet import get_seg_model
except:
    from .modules import Encoder, Unet_Decoder, Unet3Plus_Decoder
    from .PIDNet.models.pidnet import get_seg_model


def get_model(name, model_config, device):

    if 'pidnet' in name:
        model = get_seg_model(model_config['model_name'], model_config['model_parameters']['class_num'],
            p3=model_config['model_parameters']['p3'],
            p4=model_config['model_parameters']['p4'],
            p5=model_config['model_parameters']['p5']).to(device)
        #model = model(**config['model_parameters']).to(device)
        return model
    elif name=="Unet":
        return Unet
    elif name=="Unet3Plus":
        return Unet3Plus
    else:
        raise RuntimeError("There is no model named {}".format(name))


class Unet(nn.Module):

    def __init__(self, in_channels, class_num, enc_use_drop=False, dec_use_drop=False, use_transpose=False):
        super().__init__()

        out_c_list = [64, 128, 256, 512, 1024]

        self.encoder = Encoder(
            in_channels=in_channels, out_c_list=out_c_list, use_drop=enc_use_drop
        )

        self.decoder = Unet_Decoder(
            class_num=class_num, enc_out_c_list=list(reversed(out_c_list)),
            use_drop=dec_use_drop, use_transpose=use_transpose
        )
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        """
    def forward(self, x):

        cache = self.encoder(x)


        output = self.decoder(list(reversed(cache)))

        return output

class Unet3Plus(nn.Module):

    def __init__(self, in_channels, class_num, enc_use_drop=None, use_skip_drop=False, use_dec_drop=False, use_transpose=False):
        super().__init__()

        out_c_list = [64, 128, 256, 512, 1024]
        skip_channel = 64
        self.encoder = Encoder(
            in_channels=in_channels, out_c_list=out_c_list, use_drop=enc_use_drop
        )

        self.decoder = Unet3Plus_Decoder(
            skip_channels=skip_channel, class_num=class_num, enc_out_c_list=list(reversed(out_c_list)),
            use_skip_drop=use_skip_drop, use_merge_drop=use_dec_drop, use_transpose=use_transpose
        )
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        """
    def forward(self, x):

        cache = self.encoder(x)
        output = self.decoder(list(reversed(cache)))

        return output