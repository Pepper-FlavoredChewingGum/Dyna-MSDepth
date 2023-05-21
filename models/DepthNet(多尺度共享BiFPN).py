import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet_encoder import ResnetEncoder
from .hornet_encoder import hornet_tiny_7x7
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


class DepthDecoder(nn.Module):
    def __init__(self, num_output_channels=1):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels

        self.epsilon = 1e-4
        self.first_time = True
        self.attention = True
        self.conv_channels = [64, 64, 128, 256, 512]
        self.Hor_channels = [64, 128, 256, 512]
        self.num_channels = 96

        # Conv layers
        self.conv1_in_up = SeparableConvBlock(self.num_channels)
        self.conv2_in_up = SeparableConvBlock(self.num_channels)
        self.conv3_in_up = SeparableConvBlock(self.num_channels)
        self.conv4_in_up = SeparableConvBlock(self.num_channels)
        self.conv0_up = SeparableConvBlock(self.num_channels)
        self.conv1_up = SeparableConvBlock(self.num_channels)
        self.conv2_up = SeparableConvBlock(self.num_channels)
        self.conv3_up = SeparableConvBlock(self.num_channels)
        self.conv4_up = SeparableConvBlock(self.num_channels)
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

        self.Hor1_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.Hor_channels[0], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.Hor2_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.Hor_channels[1], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.Hor3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.Hor_channels[2], self.num_channels, 1),
            nn.BatchNorm2d(self.num_channels, momentum=0.01, eps=1e-3),
        )
        self.Hor4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(self.Hor_channels[3], self.num_channels, 1),
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
        self.p1_w0 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w0_relu = nn.ReLU()
        self.p2_w0 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w0_relu = nn.ReLU()
        self.p3_w0 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w0_relu = nn.ReLU()
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

    def forward(self, input_features, HorBlock):

        HorBlock0, HorBlock1, HorBlock2, HorBlock3 = HorBlock[0], HorBlock[1], HorBlock[2], HorBlock[3]
        HorBlock3 = self.Hor4_down_channel(HorBlock3)
        HorBlock2 = self.Hor3_down_channel(HorBlock2)
        HorBlock1 = self.Hor2_down_channel(HorBlock1)
        HorBlock0 = self.Hor1_down_channel(HorBlock0)

        p0_in, p1_in, p2_in, p3_in, p4_in = input_features[0], input_features[1], input_features[2], input_features[3], input_features[4]
        p0_in = self.p0_down_channel(p0_in)
        p1_in = self.p1_down_channel(p1_in)
        p2_in = self.p2_down_channel(p2_in)
        p3_in = self.p3_down_channel(p3_in)
        p4_in = self.p4_down_channel(p4_in)

        # Weights for P4_0 and Hor3 to P4_0
        p4_w0 = self.p4_w0_relu(self.p4_w0)
        weight = p4_w0 / (torch.sum(p4_w0, dim=0) + self.epsilon)
        p4_in = self.conv4_in_up(self.swish(weight[0] * p4_in + weight[1] * HorBlock3))

        # Weights for P3_0 and Hor2 to P3_0
        p3_w0 = self.p3_w0_relu(self.p3_w0)
        weight = p3_w0 / (torch.sum(p3_w0, dim=0) + self.epsilon)
        p3_in = self.conv3_in_up(self.swish(weight[0] * p3_in + weight[1] * HorBlock2))

        # Weights for P2_0 and Hor1 to P2_0
        p2_w0 = self.p2_w0_relu(self.p2_w0)
        weight = p2_w0 / (torch.sum(p2_w0, dim=0) + self.epsilon)
        p2_in = self.conv2_in_up(self.swish(weight[0] * p2_in + weight[1] * HorBlock1))

        # Weights for P1_0 and Hor0 to P1_0
        p1_w0 = self.p1_w0_relu(self.p1_w0)
        weight = p1_w0 / (torch.sum(p1_w0, dim=0) + self.epsilon)
        p1_in = self.conv1_in_up(self.swish(weight[0] * p1_in + weight[1] * HorBlock0))

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

class HorNet(nn.Module):
    def __init__(self):
        super(HorNet, self).__init__()
        self.HortNet = hornet_tiny_7x7()

    def init_weights(self):
        return

    def forward(self, down_pic):
        HorBlock = self.HortNet(down_pic)
        return HorBlock


class DepthNet(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(DepthNet, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained, num_input_images=1)
        self.decoder = DepthDecoder()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.HorNet = HorNet()

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)
        down_pic = self.maxpool(x)
        down_features = self.HorNet(down_pic)
        outputs = self.decoder(features, down_features)
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
