## https://github.com/ChaoXiang661/DTrC-Net/tree/main
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from __future__ import division, print_function

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
    
# x = torch.randn((1, 3, 256, 256))
# model = CTCNet(n_classes=2)
# out_0 = model(x)


class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)         # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        
        
        return feature2, feature3, feature4

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1,2,3,6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat  = feat + x
        return feat

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel,  out_channel, 3, 1, 1, bias=False),
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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Detail_path(nn.Module):
    def __init__(self):
        super(Detail_path, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(3,  32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,  64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(3,  128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128)
        )

    def forward(self, x):
        out = self.left(x)
        residual = self.shortcut(x)

        out += residual
        return F.relu(out)


class CONV_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # self.relu = nn.ReLU(inplace = True)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y_1 = self.conv(x)
        y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)

        return y_1

class projectors(nn.Module):
    def __init__(self, input_nc=2, ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(input_nc, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)
    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        x_out = self.pool(x_out)
        x_out = self.final(x_out)
        return x_out

class classifier(nn.Module):
    def __init__(self, inp_dim=2, ndf=8, norm_layer=nn.BatchNorm2d):
        super(classifier, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(inp_dim, ndf)
        self.conv_2 = conv(ndf, ndf * 2)
        self.conv_3 = conv(ndf * 2, ndf * 4)
        self.final = nn.Conv2d(ndf * 4, ndf * 4, kernel_size=1)
        # self.linear = nn.Linear(in_features=ndf*4*18*12, out_features=1024)

    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_1 = self.conv_2(x_0)
        x_1 = self.pool(x_1)
        x_2 = self.conv_3(x_1)
        x_2 = self.pool(x_2)
        # x_out = self.linear(x_2)
        x_out = self.final(x_2)
        return x_out

class csdNet(nn.Module):
    """Image Cascade Network"""
    def __init__(self, nclass = 1):
        super(csdNet, self).__init__()
        self.nclass = nclass
        
        self.Morphology_path = resnet18(pretrained=True)
        
        self.headpool = PyramidPoolingModule()
        self.fusion1 = FeatureFusion(in_channel=512+256, out_channel=256)
        self.fusion2 = FeatureFusion(in_channel=256+128, out_channel=128)
        self.fusion3 = FeatureFusion(in_channel=128+128, out_channel=64)

        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()

        self.Detail_path = Detail_path()
        
        self.conv_cls_spa = nn.Conv2d(512, nclass, 1, 1, bias=False)              
        self.conv_cls_cnt = nn.Conv2d(128, nclass, 1, 1, bias=False)
        self.conv_cls_out = nn.Conv2d(64 , nclass, 1, 1, bias=False)
        
        
    def forward(self, x):

        # sub 1 Morphology_path
        f_c1, f_c2, f_c3 = self.Morphology_path(x)
        # print('00',f_c1.shape, f_c2.shape, f_c3.shape)
        f_c3 = self.headpool(f_c3)
        # print('0', f_c3.shape)
        

        f_f23 = self.fusion1(f_c2, f_c3)
        # print('1', f_f23.shape)
        f_c = self.fusion2(f_c1, f_f23)
        # print('2', f_c.shape)
        f_c = self.ca(f_c) * f_c
        # print('3', f_c.shape)
        f_c = self.sa(f_c) * f_c
        # print('4', f_c.shape)
        
        # sub 2 Detail _path
        f_e = self.Detail_path(x)
        # print('5', f_e.shape)

        f_o = self.fusion3(f_e, f_c)
        # print('7', f_o.shape)

        out = F.interpolate(f_o, size=x.size()[2:], mode='bilinear', align_corners=True)
        # print('8', out.shape)

        out = self.conv_cls_out(out)
        # print('9', out.shape)


        return out, torch.sigmoid(out)


