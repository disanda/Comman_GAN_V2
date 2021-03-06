# ----------自己写的一套DCGAN网络，可以通过图像分辨率调整网络参数，包括输入维度，中间维度.--------

# 1. 以及不同分辨率的对应不同上采样数(即网络的层数)，默认第一次上采样，像素从 1->4。 之后每一次上采样，像素增加一倍(这里应该是长宽都增加一倍).
# 2. input_dim 对应G的《输入维度》，image_size表示《生成图片对应的像素》, first_hidden_dim对应《网络中间层维度》(中间层起点的维度)
# 3. scale是image_size放大的倍数，用于决定中间隐藏层起始时的size
# 4. Dscale是D网络相对G的比例, 若 Dscale=1，则G和D的参数比例为 1:1, Dscale越大，D参数相对越小
# 5.第1版，D2E ,E的参数和G完全相同，D即输入和G输入对应,输入数据维度为 input_dim*1*1

# 测试网络规模:
# import networks.network_1 as net
# G = net.Generator(input_dim=32, image_size=256, scale=32)
# D = net.Discriminator_SpectrualNorm(input_dim=32, image_size=256, scale=16)
# x,y = net.get_parameter_number(G),net.get_parameter_number(D)
# x_G, y_G = net.get_para_GByte(G),net.get_para_GByte(D)


import torch
from torch import nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_para_GByte(parameter_number):
     x=parameter_number['Total']*8/1024/1024/1024
     y=parameter_number['Total']*8/1024/1024/1024
     return {'Total_GB': x, 'Trainable_BG': y}

class Generator(nn.Module):
    def __init__(self, hidden_dim=512, output_channels=3, image_size=1024, uptimes=0):
        super().__init__()
        layers = []
        up_times = math.log(image_size,2)- 3 - uptimes # 输入为4*4时,少一次
        bias_flag = False

        # 1: 1x1 -> 4x4
        #layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4,stride=1,padding=0,bias=bias_flag)) # 1*1 input -> 4*4
        layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4,stride=2,padding=1,bias=bias_flag)) # 4*4 input -> 8*8
        #layers.append(nn.BatchNorm2d(first_hidden_dim))
        layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
        layers.append(nn.ReLU())

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> 32*32
        while up_times>0:
            layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1 ,bias=bias_flag))
            #layers.append(nn.BatchNorm2d(hidden_dim//2))
            layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
            layers.append(nn.ReLU())
            up_times = up_times - 1

        # 3:end 
        layers.append(nn.ConvTranspose2d(hidden_dim,output_channels,kernel_size=4, stride=2, padding=1, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, hidden_dim=512, input_channels=3, image_size=1024, uptimes=0): #新版的Dscale是相对G缩小的倍数
        super().__init__()
        layers=[]
        up_times = math.log(image_size,2)- 3 - uptimes
        bias_flag = False

        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        while up_times>0:  
            layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            up_times = up_times - 1
        # 3:
        #layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=1, padding=0)) # 4*4 > 1*1
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)) # 8*8 > 4*4

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        #y = y.mean()
        return y # [1,1,1,1]


