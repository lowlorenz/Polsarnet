from torch import nn
import torch
import models.complexNetwork as cn

class PolsarnetModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = cn.ComplexRelu()
        self.K1_conv = cn.ComplexConv2d(in_channels, 16, 3, dilation=1, stride=1, padding=1)
        self.K2_conv = cn.ComplexConv2d(16, 32, 3, dilation=1, stride=1, padding=1)
        self.K3_conv = cn.ComplexConv2d(32, 32, 3, dilation=1, stride=1, padding=1)
        self.K4_conv = cn.ComplexConv2d(32, 32, 3, dilation=1, stride=1, padding=1)
        self.DK2_conv = cn.ComplexConv2d(32, 32, 3, dilation=2, stride=1, padding=2)
        self.DK3_conv = cn.ComplexConv2d(32, 32, 3, dilation=3, stride=1, padding=3)
        self.Class_conv = cn.ComplexConv2d(32, out_channels, 1, dilation=1, stride=1, padding=0)
        self.bn = [cn.ComplexBatchNorm2d(num_features=features).cuda() for features in [16, 32, 32, 32, 32, 32]]
        self.Class_softmax = cn.ComplexSoftmax()

    def forward(self, x):        
        x = self.relu(self.K1_conv(x.float()))
        x = self.bn[0](x)
        x = self.relu(self.K2_conv(x))
        x = self.bn[1](x)
        x = self.relu(self.K3_conv(x))
        x = self.bn[2](x)
        x = self.relu(self.K4_conv(x))
        x = self.bn[3](x)
        x = self.relu(self.DK2_conv(x))
        x = self.bn[4](x)
        x = self.relu(self.DK3_conv(x))
        x = self.bn[5](x)
        y = self.Class_softmax(self.Class_conv(x))
        return y

    def save_weights(self, path):
        torch.save(self, path)

class RealValuedPolsarnetModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.K1_conv = nn.Conv2d(in_channels, 16, 3, dilation=1, stride=1, padding=1)
        self.K2_conv = nn.Conv2d(16, 32, 3, dilation=1, stride=1, padding=1)
        self.K3_conv = nn.Conv2d(32, 32, 3, dilation=1, stride=1, padding=1)
        self.K4_conv = nn.Conv2d(32, 32, 3, dilation=1, stride=1, padding=1)
        self.DK2_conv = nn.Conv2d(32, 32, 3, dilation=2, stride=1, padding=2)
        self.DK3_conv = nn.Conv2d(32, 32, 3, dilation=3, stride=1, padding=3)
        self.Class_conv = nn.Conv2d(32, out_channels, 1, dilation=1, stride=1, padding=0)
        self.Class_softmax = nn.Softmax()

    def forward(self, x):        
        x = self.relu(self.K1_conv(x.float()))
        x = self.relu(self.K2_conv(x))
        x = self.relu(self.K3_conv(x))
        x = self.relu(self.K4_conv(x))
        x = self.relu(self.DK2_conv(x))
        x = self.relu(self.DK3_conv(x))
        y = self.Class_softmax(self.Class_conv(x))
        return y

    def save_weights(self, path):
        torch.save(self, path)