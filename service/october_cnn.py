"""Explicit inference-only ResNet50-IBN/GeM, same specialist weights and math."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

class IBN(nn.Module):
    def __init__(self,planes,bn_norm="BN"):
        super().__init__();self.half=planes//2
        self.IN=nn.InstanceNorm2d(self.half,affine=True)
        self.BN=nn.BatchNorm2d(planes-self.half)
    def forward(self,x):
        a,b=torch.split(x,self.half,1)
        return torch.cat((self.IN(a.contiguous()),self.BN(b.contiguous())),1)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=resnet50(weights=None)
        self.backbone.maxpool=nn.MaxPool2d(3,stride=2,ceil_mode=True)
        self.backbone.layer4[0].conv2.stride=(1,1)
        self.backbone.layer4[0].downsample[0].stride=(1,1)
        for layer in [self.backbone.layer1,self.backbone.layer2,self.backbone.layer3]:
            for block in layer:block.bn1=IBN(block.conv1.out_channels)
        self.backbone.avgpool=nn.Identity();self.backbone.fc=nn.Identity()
        self.neck=nn.BatchNorm2d(2048)
        self.register_buffer('pixel_mean',torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('pixel_std',torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))

    def forward(self,x):
        x=(x-self.pixel_mean)/self.pixel_std
        b=self.backbone
        y=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        y=b.layer4(b.layer3(b.layer2(b.layer1(y))))
        y=y.float().clamp(min=1e-6).pow(3).mean((2,3),keepdim=True).pow(1/3)
        return F.normalize(self.neck(y).flatten(1).float(),dim=1)
