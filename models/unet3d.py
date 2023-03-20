#!/usr/bin/env python
 
# from __future__ import print_function, division
'''

Purpose : 

'''


import torch
import torch.nn as nn
import torch.utils.data
import os

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class basic_block(nn.Module):
    def __init__(self, input_channel, output_channel, dropout=False):
        super(basic_block, self).__init__()
        layers = [
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.Conv2d(output_channel, output_channel, 3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class UNet_basic_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size, dropout=False):
        super(UNet_basic_down_block, self).__init__()
        self.block = basic_block(input_channel, output_channel, dropout)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.block(x)
        return x


def UNet_up_conv_bn_relu(input_channel, output_channel, learned_bilinear=False):
    if learned_bilinear:
        return nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2),
                             nn.BatchNorm2d(output_channel),
                             nn.PReLU())
    else:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
                             nn.BatchNorm2d(output_channel),
                             nn.PReLU())


class UNet_basic_up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False, dropout=False):
        super(UNet_basic_up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.block = basic_block(prev_channel * 2, output_channel, dropout)

    def forward(self, pre_feature_map, x):
        x = self.bilinear_up(x)
        x = torch.cat((x, pre_feature_map), dim=1)
        x = self.block(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Input _ [batch * channel(# of channels of each image) * depth(# of frames) * height * width].
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, num_classes=1, learned_bilinear=False):
        super(U_Net, self).__init__()

        self.down_block1 = UNet_basic_down_block(1, 64, False, dropout=True)
        self.down_block2 = UNet_basic_down_block(64, 128, True, dropout=True)
        self.down_block3 = UNet_basic_down_block(128, 256, True, dropout=True)
        self.down_block4 = UNet_basic_down_block(256, 512, True, dropout=True)
        self.down_block5 = UNet_basic_down_block(512, 1024, True, dropout=True)

        self.up_block1 = UNet_basic_up_block(1024, 512, 512, learned_bilinear, dropout=True)
        self.up_block2 = UNet_basic_up_block(512, 256, 256, learned_bilinear, dropout=True)
        self.up_block3 = UNet_basic_up_block(256, 128, 128, learned_bilinear, dropout=True)
        self.up_block4 = UNet_basic_up_block(128, 64, 64, learned_bilinear, dropout=True)

        self.last_conv1 = nn.Conv2d(64, num_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x

class U_Net_DeepSup(nn.Module):
    """
    UNet - Basic Implementation
    Input _ [batch * channel(# of channels of each image) * depth(# of frames) * height * width].
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=1, output_dir=None):
        super(U_Net_DeepSup, self).__init__()

        self.output_dir = output_dir
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 64,128,256,512,1024

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        #1x1x1 Convolution for Deep Supervision
        self.Conv_d3 = conv_block(filters[1], 1)
        self.Conv_d4 = conv_block(filters[2], 1)



        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        for submodule in self.modules():
            submodule.register_forward_hook(self.nan_hook)

    # self.active = torch.nn.Sigmoid()

    def nan_hook(self, module, inp, output):
        for i, out in enumerate(output):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                torch.save(inp, os.path.join(self.output_dir, 'nan_values_ip.pt'))
                module_params = module.named_parameters()
                for name, param in module_params:
                    torch.save(param, os.path.join(self.output_dir, 'nan_{}_param.pt'.format(name)))
                torch.save(self.input_to_net, os.path.join(self.output_dir, 'nan_ip_batch.pt'))
                raise RuntimeError(" classname "+self.__class__.__name__+"i "+str(i)+f" module: {module} classname {self.__class__.__name__} Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    def forward(self, x):
        # print("unet")
        # print(x.shape)
        # print(padded.shape)
        self.input_to_net = x
        e1 = self.Conv1(x)
        # print("conv1:")
        # print(e1.shape)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # print("conv2:")
        # print(e2.shape)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # print("conv3:")
        # print(e3.shape)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # print("conv4:")
        # print(e4.shape)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("conv5:")
        # print(e5.shape)

        d5 = self.Up5(e5)
        # print("d5:")
        # print(d5.shape)
        # print("e4:")
        # print(e4.shape)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        # print("upconv5:")
        # print(d5.size)

        d4 = self.Up4(d5)
        # print("d4:")
        # print(d4.shape)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4_out  = self.Conv_d4(d4)
        
                
        # print("upconv4:")
        # print(d4.shape)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)        
        d3_out  = self.Conv_d3(d3)

        # print("upconv3:")
        # print(d3.shape)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print("upconv2:")
        # print(d2.shape)
        out = self.Conv(d2)
        # print("out:")
        # print(out.shape)
        # d1 = self.active(out)

        return [out, d3_out , d4_out]


