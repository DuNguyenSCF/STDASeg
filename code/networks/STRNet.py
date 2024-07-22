import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DWConv(nn.Module):
    def __init__(self, dim_in,stride=1,size=3):
        super(DWConv, self).__init__()
        if size==5:
            self.dwconv = nn.Conv2d(dim_in, dim_in, 5, stride, 2, bias=True, groups=dim_in)
            # self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride, 1, bias=True)
        else:
            self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride, 1, bias=True, groups=dim_in)
            # self.dwconv = nn.Conv2d(dim_in, dim_in, 3, stride, 1, bias=True)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = self.dwconv(x)
        x=self.relu(x)
        return x
class PWConv(nn.Module):
    def __init__(self, dim_in,dim_out):
        super(PWConv, self).__init__()
        self.pwconv = nn.Conv2d(dim_in, dim_out,1,1)

    def forward(self, x):
        x = self.pwconv(x)
        return x  
class Conv3_3(nn.Module):
    def __init__(self, dim_in,dim_out):
        super(Conv3_3, self).__init__()
        self.Conv = nn.Conv2d(dim_in, dim_out,3,1,1)

    def forward(self, x):
        x = self.Conv(x)
        return x      
    
class Concatenate(nn.Module):
    def __init__(self, channel_in):
        super(Concatenate, self).__init__()
        # self.final=Conv3_3(channel_in*6,channel_in)
        self.final=Conv3_3(channel_in*5,1)
        # self.final_conv=Conv3_3(channel_in,1)

    # def forward(self, convx,coarsex,upsamplex,aux_x):
    def forward(self, convx,coarsex,upsamplex):
        # x=torch.cat([convx,coarsex,upsamplex,aux_x],dim=1)
        x=torch.cat([convx,coarsex,upsamplex],dim=1)
        x=self.final(x)
        # x=self.final_conv(x)
        return x
class HSwish(nn.Module):
    def __init__(self,inplace=True):
        super(HSwish,self).__init__()
        self.inplace=inplace
    def forward(self, x):
        if self.inplace:
            x.mul_(F.relu6(x+3)/6)
            return x
        else:
            return x*(F.relu6(x+3)/6)
class HSigmoid(nn.Module):
    def __init__(self):
        super(HSigmoid,self).__init__()
    def forward(self, x):
        return F.relu6(x+3)/6
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        self.theta=torch.ones(1,requires_grad=True).cuda()
    def forward(self, x):
        return x*torch.sigmoid(self.theta*x)
class STR_module1(nn.Module):
    def __init__(self, channel_in,A,B,fx,dwsize=3):
        super(STR_module1, self).__init__()
        if fx=="relu":
            self.activate=nn.ReLU()
        else:
            # self.activate=HSwish()
            self.activate=Swish()
        self.block3=nn.Sequential(
            PWConv(channel_in,int(channel_in*B)),
            PWConv(int(channel_in*B),int(channel_in*A)),
            nn.BatchNorm2d(int(channel_in*A)),
            self.activate
        )
        self.block4=nn.Sequential(
            DWConv(int(channel_in*A),size=dwsize),
            nn.BatchNorm2d(int(channel_in*A))
        )
        self.block10=nn.Sequential(
            self.activate,
            PWConv(int(channel_in*A),int(channel_in*B)),
            PWConv(int(channel_in*B),channel_in)
        )
    def forward(self,x):
        residual=x
        x=self.block3(x)
        x=self.block4(x)
        x=self.block10(x)
        return x+residual
class Block567(nn.Module):
    def __init__(self, channel_in):
        super(Block567, self).__init__()
        self.block5=nn.AdaptiveAvgPool2d(1)
        self.block6=nn.Linear(channel_in,channel_in//4)
        self.block7=nn.Sequential(
            nn.Linear(channel_in//4,channel_in),
            HSigmoid()
        )
        
    def forward(self,x):
        # print(x.shape)
        residual=x
        x=self.block5(x).squeeze(-1).squeeze(-1)
        x=self.block6(x)
        x=self.block7(x).unsqueeze(-1).unsqueeze(-1)
        return x+residual
class STR_module2(nn.Module):
    def __init__(self, channel_in,A,B,fx,stride=1,dwsize=3):
        super(STR_module2, self).__init__()
        self.channel_out=channel_in
        if fx=="relu":
            self.activate=nn.ReLU()
        else:
            # self.activate=HSwish()
            self.activate=Swish()
        self.block3=nn.Sequential(
            PWConv(channel_in,int(channel_in*B)),
            PWConv(int(channel_in*B),int(channel_in*A)),
            nn.BatchNorm2d(int(channel_in*A)),
            self.activate
        )
        self.block4=nn.Sequential(
            DWConv(int(channel_in*A),stride,dwsize),
            nn.BatchNorm2d(int(channel_in*A))
        )
        self.block567=Block567(int(channel_in*A))
        self.block10=nn.Sequential(
            self.activate,
            PWConv(int(channel_in*A),int(channel_in*B)),
            PWConv(int(channel_in*B),self.channel_out)
        )
        self.residual_conv=nn.Conv2d(channel_in,self.channel_out,3,stride,1)
    def forward(self,x):
        residual=self.residual_conv(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block567(x)*x #block89
        x=self.block10(x)
        return x+residual
class STR_module3(nn.Module):
    def __init__(self, channel_in,A,B,fx,dwsize=3):
        super(STR_module3, self).__init__()
        self.channel_out=channel_in
        if fx=="relu":
            self.activate=nn.ReLU()
        else:
            # self.activate=HSwish()
            self.activate=Swish()
        self.block1=nn.Sequential(
            PWConv(channel_in,int(channel_in*B)),
            nn.BatchNorm2d(int(channel_in*B)),
            self.activate
        )
        self.block2=nn.Sequential(
            DWConv(int(channel_in*B),2,dwsize),
            nn.BatchNorm2d(int(channel_in*B)),
            self.activate,
            PWConv(int(channel_in*B),channel_in)
        )
        self.STR_module2=STR_module2(channel_in,A,B,fx)
        self.pw_final=nn.Sequential(
            PWConv(channel_in+int(channel_in*B),self.channel_out),
            nn.BatchNorm2d(self.channel_out)
        )
    def forward(self,x):
        residual=x
        x_1=x=self.block1(x)
        x=self.block2(x) 
        x_2=self.STR_module2(x)
        x_2=F.upsample(x_2,scale_factor=2,mode='bilinear') 
        x=torch.cat([x_1,x_2],dim=1) #block 11
        x=self.pw_final(x)
        return x+residual
class Attention_decoder(nn.Module):
    def __init__(self,in_channel):
        super(Attention_decoder,self).__init__()
        self.dim=in_channel
        self.conv=nn.Sequential(
            Conv3_3(in_channel,in_channel),
            nn.BatchNorm2d(in_channel)
        )
        self.pw_Q=nn.Sequential(
            PWConv(in_channel,in_channel//2),nn.BatchNorm2d(in_channel//2)
            )
        self.pw_K=nn.Sequential(
            PWConv(in_channel,in_channel//2),nn.BatchNorm2d(in_channel//2)
            )
        self.pw_V=PWConv(in_channel,in_channel//2)
        self.pw_out=PWConv(in_channel//2,in_channel)
        self.pw_out1=nn.Sequential(PWConv(2*in_channel,in_channel),
                                   nn.BatchNorm2d(in_channel),
                                   nn.Dropout(0.2))
        self.tr_conv=nn.ConvTranspose2d(in_channel,in_channel,3,2,1,1)
        # self.tr_conv=nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self, x,x_conv):
        
        x=self.conv(x) 
        q=self.pw_Q(x)  #B C H W
        k=self.pw_K(x)
        v=self.pw_V(x)
        B,C,H,W=q.shape
        q=q.reshape(B,C,-1).permute(0,2,1)  # B H*W C
        v=v.reshape(B,C,-1).permute(0,2,1)  # B H*W C
        mul=torch.einsum("bnc,bchw->bnhw",q,k) #B H*W H W
        M1=torch.einsum('bnhw,bnc->bchw',mul,v)
        # M2=F.softmax(M1/math.sqrt(self.dim),dim=1)
        M2=M1/math.sqrt(self.dim)
        x_ad=self.pw_out(M2)
        # print(x_ad.shape,x_conv.shape)
        x=torch.cat([x_ad,x_conv],dim=1)
        x=self.pw_out1(x)
        out=self.tr_conv(x)
        
        return out
    
class Coarse_Upsample(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(Coarse_Upsample, self).__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv=Conv3_3(channel_in,channel_out)
        self.norm=nn.BatchNorm2d(channel_out)
        self.relu=nn.ReLU()
    def forward(self, x):
        x=self.upsample(x)
        x=self.conv(x)
        x=self.norm(x)
        x=self.relu(x)
        return x
class STR_module_full(nn.Module):
    def __init__(self, channel_in):
        super(STR_module_full, self).__init__()
        self.STR1=STR_module2(channel_in,1,1,'relu',2)
        self.STR2=STR_module1(channel_in,4.5,1,'relu')
        self.STR3=STR_module1(channel_in,5.5,1.5,'relu')
        self.STR4=STR_module2(channel_in,6,2.5,'swish',2,dwsize=5)
        self.STR5=STR_module3(channel_in,15,2.5,'swish',dwsize=5)
        self.STR6=STR_module3(channel_in,15,2.5,'swish',dwsize=5)
        self.STR7=STR_module2(channel_in,7.5,3,'swish',dwsize=5)
        self.STR8=STR_module3(channel_in,9,3,'swish',dwsize=5)
        self.STR9=STR_module2(channel_in,18,6,'swish',2,dwsize=5)
        self.STR10=STR_module3(channel_in,36,6,'swish',dwsize=5)
        self.STR11=STR_module3(channel_in,36,6,'swish',dwsize=5)
        # self.STR9=STR_module2(channel_in,12,6,'swish',2,dwsize=5)
        # self.STR10=STR_module3(channel_in,12,6,'swish',dwsize=5)
        # self.STR11=STR_module3(channel_in,12,6,'swish',dwsize=5)
        self.conv_up=Conv3_3(channel_in,channel_in)
    def forward(self,x):
        x=self.STR1(x)
        x=self.STR2(x)
        out_up=x=self.STR3(x)
        out_up=self.conv_up(out_up)
        x=self.STR4(x)
        x=self.STR5(x)
        x=self.STR6(x)
        x=self.STR7(x)
        out_aux=x=self.STR8(x)
        x=self.STR9(x)
        x=self.STR10(x)
        out=self.STR11(x)
        # return out,out_up,out_aux
        return out,out_up
class STRNet(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(STRNet, self).__init__()
        self.dim=24
        self.stem=nn.Sequential(
            Conv3_3(3,self.dim),
            nn.BatchNorm2d(self.dim),
            HSwish()            
        )
        self.conv_stem=Conv3_3(self.dim,self.dim)
        self.STR=STR_module_full(self.dim)
        self.coarse_up=Coarse_Upsample(self.dim,3*self.dim)
        self.Max_pool=nn.MaxPool2d(2,stride=2)
        self.attention1=Attention_decoder(self.dim)
        self.attention2=Attention_decoder(self.dim)
        # self.attention3=Attention_decoder(self.dim)
        self.Cat=Concatenate(self.dim)
    def forward(self, x):
      
        convx=x=self.conv_stem(self.stem(x))
  
        # x,x_up,x_aux=self.STR(x)
        x,x_up=self.STR(x)
  
        x_coarse=self.attention1(x,x)
        
        x_coarse=self.attention2(x_coarse,x_coarse)
        
        # x_aux=F.upsample_bilinear(self.attention3(x_aux,x_aux),scale_factor=2)
        # x_aux=F.upsample_bilinear(x_aux,scale_factor=4)

        x_coarse=self.coarse_up(x_coarse)

        x_up=F.upsample(x_up,scale_factor=2,mode='bilinear')
        # x=self.Cat(convx,x_coarse,x_up,x_aux)
        x=self.Cat(convx,x_coarse,x_up)
        # x=x.squeeze(1)
        return x
    
    
## https://github.com/ChaoXiang661/DTrC-Net/tree/main
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.


import torch
import torch.nn as nn
import functools
from functools import partial
import torch.nn.functional as F
from torchvision import models
from torchvision import models as resnet_model

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
        
        
@register_model
def deit_tiny_patch16_256(pretrained=False,**kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        ckpt = torch.load('pretrained/deit_tiny_patch16_256.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    model.default_cfg = _cfg()
    return model
def deit_tiny_distilled_patch16_256(pretrained=False,**kwargs):
    model = DistilledVisionTransformer(
        img_size=256, patch_size=16, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        ckpt = torch.load('pretrained/deit_tiny_distilled_patch16_256.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    model.default_cfg = _cfg()
    return model


class resnet34(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x0 = self.relu(x)
        feature1 = self.layer1(x0)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32

        return x0, feature1, feature2, feature3, feature4

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat = feat + x
        return feat

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.left(x)
        residual = self.shortcut(x)
        out += residual
        return F.relu(out)

class FeatureFusion(nn.Module):
    """CFF Unit"""

    def __init__(self, in_channel, out_channel):
        super(FeatureFusion, self).__init__()
        self.fusion = ResBlock(in_channel, out_channel)

    def forward(self, x_high, x_low):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_low, x_high), dim=1)
        x = self.fusion(x)

        return x

class RPMBlock(nn.Module):
    def __init__(self, channels):
        super(RPMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class CTCNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(CTCNet, self).__init__()
        self.n_class = n_classes
        self.inchannels = n_channels
        size = 256
        self.cnn = resnet34(pretrained=True)
        self.headpool = PyramidPoolingModule()

        transformer = deit_tiny_distilled_patch16_256()
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )
        self.se = SEBlock(channel=512)
        self.se1 = SEBlock(channel=64)
        self.se2 = SEBlock(channel=128)
        self.se3 = SEBlock(channel=256)

        self.fusion = FeatureFusion(in_channel=512 + size, out_channel=512)
        self.fusion1 = FeatureFusion(in_channel=64 + size, out_channel=64)
        self.fusion2 = FeatureFusion(in_channel=128 + size, out_channel=128)
        self.fusion3 = FeatureFusion(in_channel=256 + size, out_channel=256)

        self.RPMBlock1 = RPMBlock(channels=64)
        self.RPMBlock2 = RPMBlock(channels=128)
        self.RPMBlock3 = RPMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.RPMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.RPMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.RPMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        e0, e1, e2, e3, e4 = self.cnn(x)
        feature_cnn = self.headpool(e4)
        size = 256

        emb = self.patch_embed(x)
        emb = self.transformers[0](emb)
        emb = self.transformers[1](emb)
        emb1 = self.transformers[2](emb)
        feature_tf1 = emb1.permute(0, 2, 1)
        feature_tf1 = feature_tf1.view(b, size, 16, 16)

        emb1 = self.transformers[3](emb1)
        emb1 = self.transformers[4](emb1)
        emb2 = self.transformers[5](emb1)
        feature_tf2 = emb2.permute(0, 2, 1)
        feature_tf2 = feature_tf2.view(b, size, 16, 16)

        emb2 = self.transformers[6](emb2)
        emb2 = self.transformers[7](emb2)
        emb3 = self.transformers[8](emb2)
        feature_tf3 = emb3.permute(0, 2, 1)
        feature_tf3 = feature_tf3.view(b, size, 16, 16)

        emb3 = self.transformers[9](emb3)
        emb3 = self.transformers[10](emb3)
        emb4 = self.transformers[11](emb3)

        feature_tf = emb4.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, size, 16, 16)

        feature_cat = self.fusion(feature_cnn, feature_tf)
        feature_cat = self.se(feature_cat)

        e1 = self.fusion1(e1, feature_tf1)
        e2 = self.fusion2(e2, feature_tf2)
        e3 = self.fusion3(e3, feature_tf3)
        e1 = self.se1(e1)
        e2 = self.se2(e2)
        e3 = self.se3(e3)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)

        d4 = self.decoder4(feature_cat) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out_0 = self.final_conv3(out)
        # out = torch.sigmoid(out_0)
        return out_0