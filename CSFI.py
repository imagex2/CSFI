#  from model import common
import numpy as np

import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import copy

""" common resblock"""

def conv3x3(input_channels=64, output_channels=64, stride=1, bias=False):
    return nn.Conv2d(in_channels=input_channels,
                     out_channels=output_channels,
                     kernel_size=3,  #
                     stride=stride,
                     padding=1,
                     dilation=1,
                     groups=1,  #
                     bias=bias
                     )


def conv1x1(input_channels=64, output_channels=64, stride=1, bias=False):
    return nn.Conv2d(in_channels=input_channels,
                     out_channels=output_channels,
                     kernel_size=1,  #
                     stride=stride,
                     padding=0,
                     dilation=1,
                     groups=1,  #
                     bias=bias
                     )


class Edge_FDM(nn.Module):
    def __init__(self, input_channel=64, out_channel=64, pool_rate=2, kernel_size=3, scale=2, bias=False):
        super(Edge_FDM, self).__init__()
        self.avg_pooling = nn.AvgPool2d(kernel_size=pool_rate,  #
                                        stride=pool_rate,  #
                                        padding=0
                                        )
        self.max_pooling = nn.MaxPool2d(kernel_size=pool_rate,  # 
                                        stride=pool_rate,  #
                                        padding=0
                                        )

        # conv
        self.conv = nn.Conv2d(in_channels=input_channel // 2,
                              out_channels=out_channel // 2,
                              kernel_size=kernel_size,  #
                              stride=(kernel_size // 2),
                              padding=1,
                              dilation=1,
                              groups=1,  #
                              bias=bias
                              )
        self.conv2 = nn.Conv2d(in_channels=input_channel // 2,
                               out_channels=out_channel * scale * scale // 2,
                               kernel_size=kernel_size,  #
                               stride=(kernel_size // 2),
                               padding=1,
                               dilation=1,
                               groups=1,  #
                               bias=bias
                               )
        # deconv
        self.de_conv = nn.ConvTranspose2d(in_channels=input_channel,
                                          out_channels=out_channel,
                                          kernel_size=4,
                                          stride=pool_rate,
                                          padding=1,
                                          bias=bias
                                          )

        self.channel_expand = nn.Conv2d(in_channels=input_channel,
                                        out_channels=out_channel * scale * scale,  # scale^2
                                        kernel_size=3,
                                        padding=1,
                                        groups=1,
                                        bias=bias)

        self.up_shuffle = nn.PixelShuffle(scale)
        self.act = nn.ReLU(True)

        self.conv1_a = nn.Conv2d(input_channel, input_channel // scale, kernel_size=1, padding=0, bias=False)
        self.conv1_b = nn.Conv2d(input_channel, input_channel // scale, kernel_size=1, padding=0, bias=False)

        self.k0 = nn.Conv2d(
            input_channel // scale, input_channel // scale, kernel_size=3, stride=(kernel_size // 2),
            padding=1, dilation=1,
            bias=False)
        self.conv3 = nn.Conv2d(
            input_channel, input_channel, kernel_size=3, stride=(kernel_size // 2),
            padding=1, dilation=1,
            bias=False)

        self.conv33 = nn.Conv2d(in_channels=input_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,  #
                              stride=(kernel_size // 2),
                              padding=1,
                              dilation=1,
                              groups=1,  #
                              bias=bias
                              )
        self.conv33_2 = nn.Conv2d(in_channels=input_channel,
                               out_channels=out_channel * scale * scale,
                               kernel_size=kernel_size,  #
                               stride=(kernel_size // 2),
                               padding=1,
                               dilation=1,
                               groups=1,  #
                               bias=bias
                               )

    def forward(self, x):
        out_b0 = self.max_pooling(x)
        out_b1 = self.conv33(out_b0)
        out_b1 = torch.sigmoid(out_b1)
        out_b1 = out_b1 * out_b0
        out_b1 = self.conv33_2(out_b1)
        out_b1 = self.up_shuffle(out_b1)
        out = out_b1 + x
        return out

        


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)  

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)  # self.body(x).mul(self.res_scale)
        res += x

        return res


"""layer norm"""
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # num_heads = 2
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)  
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                   bias=bias) 
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) 
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)  

    def forward(self, x, ref):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))  
        k, v = kv.chunk(2, dim=1)  

        q = self.q_dwconv(self.q(ref))  

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class SFM(nn.Module):
    def __init__(self, dim=64, bias=False):  # dim = 48
        super(SFM, self).__init__()

        self.conv_add1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.conv_add2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

        self.conv_mul1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.conv_mul2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.relu = nn.GELU()

        self.conv_fuse = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_add = self.relu(self.conv_add1(x))
        x_add = self.conv_add2(x_add)

        x_mul = self.relu(self.conv_mul1(x))
        x_mul = self.conv_mul1(x_mul)
        x_mul = self.sigmoid(x_mul)

        x_mul = x * x_mul + x_add
        x = self.conv_fuse(x_mul)  # + x
        return x




class MyTransformerBlock_FAM(nn.Module):
    def __init__(self, dim=64, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(MyTransformerBlock_FAM, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.sfm = SFM(dim=64, bias=False)

    def forward(self, x, ref):
        x = x + self.attn(self.norm1(ref), self.norm3(x))
        x = x + self.sfm(self.norm2(x))  

        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print("x.shape", x.shape)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def make_model(args, parent=False):
    return RCAN(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        self.conv1 = nn.Conv2d(
            # in_channels=1,
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,  # [kernel_size/2]
            groups=1,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)

        out = self.sigmoid(out) * x
        return out


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))  
            if i == 0: modules_body.append(act)  
        modules_body.append(CALayer(n_feat, reduction)) 
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        # modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



class ISFI(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(ISFI, self).__init__()
        """
        n_resgroups = args.n_resgroups  # default=10
        n_resblocks = args.n_resblocks  # default=20
        n_feats = args.n_feats  # default=64
        kernel_size = 3
        reduction = args.reduction  # default=16
        """
        n_resgroups = 5  
        n_resblocks = 10  
        n_feats = 64 
        kernel_size = 3
        reduction = 16 
        res_scale = 1

        # scale = args.scale[0]
        act = nn.ReLU(True)
        self.relu = nn.ReLU(True)
        self.n_resblocks_ref = 5

        # define head module
        self.modules_head = conv(in_channels=1, out_channels=n_feats, kernel_size=kernel_size,
                                 bias=True)  # args.n_colors default=3

        # define body module
        self.modules_body_1 = ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)

        self.modules_body_2 = ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)

        self.modules_body_3 = ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)

        self.modules_body_4 = ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)

        self.modules_body_5 = ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)

        self.modules_body_end = conv(n_feats, n_feats, kernel_size)

        self.tail = conv(in_channels=n_feats, out_channels=1, kernel_size=kernel_size, bias=True)  

        self.ref_tail = conv(in_channels=n_feats, out_channels=1, kernel_size=kernel_size, bias=True)  



        self.body_sc_1 = Edge_FDM()
        self.body_sc_2 = Edge_FDM()
        self.body_sc_3 = Edge_FDM()
        self.body_sc_4 = Edge_FDM()
        self.body_sc_5 = Edge_FDM()

        self.transformer_1 = MyTransformerBlock_FAM()
        self.transformer_2 = MyTransformerBlock_FAM()
        self.transformer_3 = MyTransformerBlock_FAM()
        self.transformer_4 = MyTransformerBlock_FAM()
        self.transformer_5 = MyTransformerBlock_FAM()

        self.x_up_head = conv(in_channels=1, out_channels=n_feats, kernel_size=kernel_size,
                              bias=True)  # args.n_colors default=3
        self.x_dw_head = conv(in_channels=1, out_channels=n_feats, kernel_size=kernel_size,
                              bias=True)  # args.n_colors default=3

        self.x_up_res = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))
        self.x_dw_res = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))
        self.x_fuse = conv(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, bias=True)

        self.x_res_1 = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))
        self.x_res_2 = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))
        self.x_res_3 = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))
        self.x_res_4 = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))
        self.x_res_5 = RCAB(conv, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True))

        self.fuse_1 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True
        )
        self.fuse_2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True
        )
        self.fuse_3 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True
        )
        self.fuse_4 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True
        )
        self.fuse_5 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=True
        )

    def forward(self, x, x_ref_up, x_ref_down):  

        x = self.modules_head(x)

        x_up = self.x_up_head(x_ref_up)
        x_dw = self.x_dw_head(x_ref_down)
        x_2ref = torch.cat([x_up, x_dw], dim=1)
        x_2ref = self.x_fuse(x_2ref)

        res = self.modules_body_1(x)
        x_2ref = self.x_res_1(x_2ref)
        res_t = self.transformer_1(res, x_2ref)
        res_sc = self.body_sc_1(res)
        res = (res_t + res_sc) / 2

        res = self.modules_body_2(res)
        x_2ref = self.x_res_2(x_2ref)
        res_t = self.transformer_2(res, x_2ref)
        res_sc = self.body_sc_2(res)
        res = (res_t + res_sc) / 2

        res = self.modules_body_3(res)
        x_2ref = self.x_res_3(x_2ref)
        res_t = self.transformer_3(res, x_2ref)
        res_sc = self.body_sc_3(res)
        res = (res_t + res_sc) / 2

        res = self.modules_body_4(res)
        x_2ref = self.x_res_4(x_2ref)
        res_t = self.transformer_4(res, x_2ref)
        res_sc = self.body_sc_4(res_t)
        res = (res + res_sc) / 2

        res = self.modules_body_5(res)
        x_2ref = self.x_res_5(x_2ref)
        res_t = self.transformer_5(res, x_2ref)
        res_sc = self.body_sc_5(res)
        res = (res_t + res_sc) / 2

        res = self.modules_body_end(res)

        res += x

        x = self.tail(res)


        x_2ref = self.ref_tail(x_2ref)

        return x, x_2ref  # x: SR,
