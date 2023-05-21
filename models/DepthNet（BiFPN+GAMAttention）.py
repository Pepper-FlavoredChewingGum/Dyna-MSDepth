import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_encoder import ResnetEncoder
# from .hornet_encoder import hornet_tiny_7x7
from .util import MemoryEfficientSwish, Swish
from .utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


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


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x

class GAMAttention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle


        out = x * x_spatial_att
        return out

def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

class DepthDecoder(nn.Module):
    def __init__(self, num_output_channels=1):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels

        self.epsilon = 1e-4
        self.first_time = True
        self.conv_channels = [64, 64, 128, 256, 512]
        self.num_channels = 128

        self.attention = GAMAttention(self.conv_channels[-1], self.conv_channels[-1])

        # Conv layers
        self.att_up = SeparableConvBlock(self.num_channels)
        self.conv0_up = SeparableConvBlock(self.num_channels)
        self.conv1_up = SeparableConvBlock(self.num_channels)
        self.conv2_up = SeparableConvBlock(self.num_channels)
        self.conv3_up = SeparableConvBlock(self.num_channels)
        self.conv00_up = SeparableConvBlock(self.num_channels)
        self.conv11_up = SeparableConvBlock(self.num_channels)
        self.conv22_up = SeparableConvBlock(self.num_channels)
        self.conv33_up = SeparableConvBlock(self.num_channels)

        self.sigmoid = nn.Sigmoid()
        self.dispconvs = Conv3x3(8, self.num_output_channels)

        # Feature scaling layers
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p0_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.conv_channels[0], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )

        self.p1_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.conv_channels[1], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.p2_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.conv_channels[2], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.p3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.conv_channels[3], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.p4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.conv_channels[4], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.att_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.conv_channels[4], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )

        self.output1 = nn.Sequential(
            Conv2dStaticSamePadding(self.num_channels, 32, 1),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
        )
        self.output2 = nn.Sequential(
            Conv2dStaticSamePadding(32, 8, 1),
            nn.BatchNorm2d(8, momentum=0.01, eps=1e-3),
        )

        self.swish = MemoryEfficientSwish()

        # Weight
        self.p4_w0 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w0_relu = nn.ReLU()
        self.p0_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p0_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p0_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p0_w2_relu = nn.ReLU()
        self.p1_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p1_w2_relu = nn.ReLU()
        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()


    def init_weights(self):
        return

    def forward(self, input_features):
        p0_in, p1_in, p2_in, p3_in, p4_in = input_features[0], input_features[1], input_features[2], input_features[3], input_features[4]
        attent = self.attention(p4_in)
        p0_in = self.p0_down_channel(p0_in)
        p1_in = self.p1_down_channel(p1_in)
        p2_in = self.p2_down_channel(p2_in)
        p3_in = self.p3_down_channel(p3_in)
        p4_in = self.p4_down_channel(p4_in)
        attent = self.att_down_channel(attent)

        # Weights for P4_0 and Hor3 to P4_0
        p4_w0 = self.p4_w0_relu(self.p4_w0)
        weight = p4_w0 / (torch.sum(p4_w0, dim=0) + self.epsilon)
        p4_in = self.att_up(self.swish(weight[0] * p4_in + weight[1] * attent))

        # Weights for P4_0 and P3_0 to P3_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        # p3_w1 = self.p3_w1
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_1 = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p4_upsample(p4_in)))

        # Weights for P3_0 and P2_0 and p3_1 to P2_1
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        # p2_w1 = self.p2_w1
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_1 = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p3_upsample(p3_in) + weight[2] * self.p3_upsample(p3_1)))

        # Weights for P2_0, P3_1 and P1_0 to P1_1
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        # p1_w1 = self.p1_w1
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        p1_1 = self.conv1_up(self.swish(weight[0] * p1_in + weight[1] * self.p2_upsample(p2_in) + weight[2] * self.p3_upsample(self.p3_upsample(p3_1))))

        # Weights for P1_1 and P0_0 to P0_1
        p0_w1 = self.p0_w1_relu(self.p0_w1)
        # p0_w1 = self.p0_w1
        weight = p0_w1 / (torch.sum(p0_w1, dim=0) + self.epsilon)
        p0_1 = self.conv0_up(self.swish(weight[0] * p0_in + weight[1] * self.p1_upsample(p1_1)))

        # Weights for P4_0, P3_0, P3_1 to P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_2 = self.conv33_up(self.swish(weight[0] * p3_in + weight[1] * p3_1 + weight[2] * self.p4_upsample(p4_in)))

        # Weights for P2_0, P2_1, P3_2 to P2_2
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        p2_2 = self.conv22_up(self.swish(weight[0] * p2_in + weight[1] * p2_1 + weight[2] * self.p3_upsample(p3_2)))

        # Weights for P1_0, P1_1, P2_2 to P1_2
        p1_w2 = self.p1_w2_relu(self.p1_w2)
        weight = p1_w2 / (torch.sum(p1_w2, dim=0) + self.epsilon)
        p1_2 = self.conv11_up(self.swish(weight[0] * p1_in + weight[1] * p1_1 + weight[2] * self.p2_upsample(p2_2)))

        # Weights for P0_0, P0_1 and P1_2 to P0_2
        p0_w2 = self.p0_w2_relu(self.p0_w2)
        # p0_w2 = self.p0_w2
        weight = p0_w2 / (torch.sum(p0_w2, dim=0) + self.epsilon)
        BiFPN_output = self.conv00_up(self.swish(weight[0] * p0_in + weight[1] * p0_1 + weight[2] * self.p1_upsample(p1_2)))

        BiFPN_output = self.output1(BiFPN_output)
        BiFPN_output = self.p1_upsample(BiFPN_output)
        BiFPN_output = self.output2(BiFPN_output)

        disp = self.alpha * self.sigmoid(self.dispconvs(BiFPN_output)) + self.beta
        depth = 1.0 / disp
        self.outputs = depth

        return self.outputs


class DepthNet(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(DepthNet, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=1)
        self.decoder = DepthDecoder()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

    # print(len(test_out))
    # print(test_out[0].shape)
    # print(test_out[1].shape)
    # print(test_out[2].shape)
    # print(test_out[3].shape)
