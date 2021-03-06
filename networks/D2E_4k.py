# kernel为4，stride也为4，一次降4倍采样，减少通道冗余
# 1->4->16->64->256->1024

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
    def __init__(self, input_dim=256, output_channels=3, image_size=256):
        super().__init__()
        bias_flag = False
        layers = []
        n = 2

        layers.append(nn.ConvTranspose2d(input_dim, input_dim*2, kernel_size=4, stride=4, bias=bias_flag))
        #layers.append(nn.BatchNorm2d(hidden_dim//2))
        layers.append(nn.InstanceNorm2d(input_dim*2, affine=False, eps=1e-8))
        layers.append(nn.ReLU())
        hidden_dim = input_dim * 2


        while n>0:
            layers.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=4, bias=bias_flag))
            #layers.append(nn.BatchNorm2d(hidden_dim//2))
            layers.append(nn.InstanceNorm2d(hidden_dim, affine=False, eps=1e-8))
            layers.append(nn.ReLU())
            n = n - 1

        # 3:end 
        layers.append(nn.ConvTranspose2d(hidden_dim,output_channels,kernel_size=4, stride=4, bias=bias_flag))
        layers.append(nn.Tanh())

        # all
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator_SpectrualNorm(nn.Module):
    def __init__(self, input_dim=256, input_channels=3, image_size=256): #新版的Dscale是相对G缩小的倍数
        super().__init__()
        bias_flag = False
        layers=[]
        hidden_dim = input_dim*2
        n = 2
        # 1:
        layers.append(spectral_norm(nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=4,  bias=bias_flag)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2: 64*64 > 4*4
        while n>0:  
            layers.append(spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=4, bias=bias_flag)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            n = n - 1

        # 3:
        layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=4, padding=0)) # 4*4 > 1*1
        #layers.append(nn.Conv2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1)) # 8*8 > 4*4

        # all:
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        #y = y.mean()
        return y # [1,1,1,1]