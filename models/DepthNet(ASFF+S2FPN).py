import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import ResnetEncoder
# from .hornet_encoder import hornet_tiny_7x7
from .SSFPN import PyrmidFusionNet, conv_block
from .ASFF_block import ASFF

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # decoder

        self.out_channels = [32, 64, 128, 256, 512]
        self.apf1 = PyrmidFusionNet(self.out_channels[4], self.out_channels[4], self.out_channels[3])
        self.apf2 = PyrmidFusionNet(self.out_channels[3], self.out_channels[3], self.out_channels[2])
        self.apf3 = PyrmidFusionNet(self.out_channels[2], self.out_channels[2], self.out_channels[1])
        self.apf4 = PyrmidFusionNet(self.out_channels[1], self.out_channels[1], self.out_channels[0])

        self.ASFF1 = ASFF(level=0)
        self.ASFF2 = ASFF(level=1)
        self.ASFF3 = ASFF(level=2)
        self.ASFF4 = ConvBlock(self.out_channels[0], self.out_channels[1])

        self.cfgb = nn.Sequential(conv_block(self.out_channels[4],self.out_channels[4],kernel_size=3,stride=2,padding=1,group=self.out_channels[4],
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.15))


        self.dispconvs = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.output = ConvBlock(self.num_ch_dec[-4], self.num_ch_dec[-5])
        self.out_up1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.out_up2 = nn.ConvTranspose2d(16, 16, 3, 2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        return

    def forward(self, input_features):

        # self.outputs = []

        # decoder
        CFGB = self.cfgb(input_features[-1])

        ASFF1 = self.ASFF1(input_features[-1], input_features[-2], input_features[-4])
        ASFF2 = self.ASFF2(input_features[-1], input_features[-2], input_features[-3])
        ASFF3 = self.ASFF3(input_features[-2], input_features[-3], input_features[-4])
        ASFF4 = input_features[-4]

        APF5 = self.apf1(CFGB, ASFF1)
        APF4 = self.apf2(APF5, ASFF2)
        APF3 = self.apf3(APF4, ASFF3)
        APF2 = self.apf4(APF3, ASFF4)

        ASFF4 = self.out_up1(APF2)
        output = self.output(ASFF4)
        output = self.out_up2(output)

        disp = self.alpha * self.sigmoid(self.dispconvs(output)) + self.beta
        depth = 1.0 / disp

        return depth


class DepthNet(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(DepthNet, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=1)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DepthNet().cuda()
    model.train()

    B = 4

    tgt_img = torch.randn(B, 3, 256, 832).cuda()

    tgt_depth = model(tgt_img)

    print(tgt_depth.size())
