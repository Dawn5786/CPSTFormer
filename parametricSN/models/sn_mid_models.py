"""Contains all the 'top' pytorch NN.modules for this project


Author: Dawn

Functions:


Classes:
    Attention
    AttentionBlock
    sn_ViT         -- ViT fitted for scattering input

"""

from torchvision import models

import torch.nn as nn
import torch
from timm.models.layers import PatchEmbed, Mlp, DropPath #, trunc_normal_, lecun_normal_
from functools import partial

class mid_MLP(nn.Module):

    """
       Multilayer perceptron fitted for scattering input
    """
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8):
        # super(sn_MLP,self).__init__()
        super(mid_MLP,self).__init__()
        self.num_classes = num_classes

        fc1 =  nn.Linear(int(3*M_coefficient*N_coefficient*n_coefficients), 512)

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.n_coefficients*3, eps=1e-5, affine=True),
            fc1,
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        x = x.view(x.shape[0], -1)
        return self.layers(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    bias=False)


class BasicBlock(nn.Module):
    """
    Standard wideresnet basicblock
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class AttentionBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class mid_CNN(nn.Module):

    """
    CNN fitted for scattering input
    Model from: https://github.com/kymatio/kymatio/blob/master/examples/2d/cifar_small_sample.py 
    """
    def __init__(self, in_channels, k=8, n=4, num_classes=10, standard=False):
        # super(sn_CNN, self).__init__()
        super(mid_CNN, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_channels*3,eps=1e-5,affine=True)

        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.in_channels = in_channels
        self.num_classes =num_classes
        in_channels = in_channels * 3
        if standard:
            self.init_conv = nn.Sequential(
                nn.Conv2d(3, self.ichannels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
                nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.standard = False

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            pass
        x = self.bn0(x)
        x = self.init_conv(x)
        if self.standard:
            x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class mid_LinearLayer(nn.Module):
    """
    Linear layer fitted for scattering input
    """
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8):
        super(mid_LinearLayer, self).__init__()
        self.n_coefficients = n_coefficients
        self.num_classes = num_classes

        self.fc1 = nn.Linear(int(3*M_coefficient*N_coefficient*n_coefficients), num_classes)
        self.bn0 = nn.BatchNorm2d(self.n_coefficients*3, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.bn0(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

class mid_Resnet50(nn.Module):
    """
    Architecture: ResNet-50
    """
    def __init__(self, num_classes=10):
        super(mid_Resnet50, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc =  nn.Linear(num_ftrs, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model_ft(x)
        return x

class BasicBlockWRN16_8(nn.Module):
    def __init__(self, inplanes, planes, dropout, stride=1):
        super(BasicBlockWRN16_8, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            self.shortcut = conv1x1(inplanes, planes, stride)
            self.use_conv1x1 = True
        else:
            self.use_conv1x1 = False

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)

        if self.use_conv1x1:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += shortcut

        return out

class WideResNet16_8(nn.Module):
    def __init__(self, depth, width, num_classes=10, dropout=0.3):
        super(WideResNet16_8, self).__init__()

        layer = (depth - 4) // 6

        self.inplanes = 16
        self.conv = conv3x3(3, 16)
        self.layer1 = self._make_layer(16*width, layer, dropout)
        self.layer2 = self._make_layer(32*width, layer, dropout, stride=2)
        self.layer3 = self._make_layer(64*width, layer, dropout, stride=2)
        self.bn = nn.BatchNorm2d(64*width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, planes, blocks, dropout, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(BasicBlockWRN16_8(self.inplanes, planes, dropout, stride if i == 0 else 1))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        count = 0
        for t in self.parameters():
            count += t.numel()
        return count

class mid_ViT(nn.Module):
    """
    ViT fitted for scattering input    Architecture: ViT-tiny
    Inf:
    """
    def __init__(self, in_channels, num_heads=3, num_layers=12, unit_embed_dim=64, num_classes=10, standard=False, img_size=8, patch_size=1):
        super(mid_ViT, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.unit_embed_dim = unit_embed_dim
        self.embed_dim = unit_embed_dim * num_heads

        self.num_classes = num_classes

        if standard:
            self.in_channels = 3
            self.img_size = 32
            self.patch_size = 2
            self.standard = True
        else:
            self.in_channels = in_channels * 3
            self.img_size = 8
            self.patch_size = 1
            self.standard = False

        self.bn0 = nn.BatchNorm2d(self.in_channels, eps=1e-5, affine=True)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_channels, embed_dim=self.embed_dim)

        num_patches = self.patch_embed.num_patches
        embed_len = num_patches  # + 1 #num_patches if no_embed_class else num_patches + self.num_prefix_tokens

        # if standard:
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # if class_token else None
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=0) #

        dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]  # stochastic depth decay rule
        self.blocks_whole = nn.Sequential(*[
            AttentionBlock(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, init_values=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(num_layers)])
        self.norm = norm_layer(self.embed_dim) #if not use_fc_norm else nn.Identity()

            # Classifier Head
        # self.fc_norm = norm_layer(self.embed_dim) #if use_fc_norm else nn.Identity()
        self.fc_norm = nn.Identity() #if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) #if num_classes > 0 else nn.Identity()

    def global_pool(self, x):
        x = x[:, 1:].mean(dim=1)# if self.global_pool == 'avg' else x[:, 0]
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks_whole(x)
        x = self.norm(x)
        x = self.global_pool(x)
        x = self.fc_norm(x)
        x = self.head(x)
        return x

class mid_Atten(nn.Module):
    """
    ViT fitted for scattering input    Architecture: ViT-tiny
    Inf:
    """
    # def __init__(self, in_channels, num_heads=3, num_layers=12, unit_embed_dim=64, num_classes=10, standard=False, img_size=8, patch_size=1):
    def __init__(self, in_channels, num_heads=1, num_layers=3, unit_embed_dim=64, num_classes=10, standard=False,
                     img_size=8, patch_size=1):

        # super(sn_ViT, self).__init__()
        super(mid_Atten, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.unit_embed_dim = unit_embed_dim
        self.embed_dim = unit_embed_dim * num_heads

        self.num_classes = num_classes

        if standard:
            self.in_channels = 3
            self.img_size = 32
            self.patch_size = 2
            self.standard = True
        else:
            self.in_channels = in_channels * 3
            self.img_size = 8
            self.patch_size = 1
            self.standard = False

        self.bn0 = nn.BatchNorm2d(self.in_channels, eps=1e-5, affine=True)

        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_channels, embed_dim=self.embed_dim)

        num_patches = self.patch_embed.num_patches
        embed_len = num_patches  # + 1 #num_patches if no_embed_class else num_patches + self.num_prefix_tokens

        # if standard:
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # if class_token else None
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=0) #

        dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]  # stochastic depth decay rule
        # self.blocks_whole = nn.Sequential(*[
        #     AttentionBlock(
        #         dim=self.embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, init_values=None,
        #         drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer,
        #         act_layer=act_layer)
        #     for i in range(num_layers)])
        self.blocks = nn.Sequential(*[
            AttentionBlock(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, init_values=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(num_layers)])
        self.norm = norm_layer(self.embed_dim) #if not use_fc_norm else nn.Identity()

            # Classifier Head
        # self.fc_norm = norm_layer(self.embed_dim) #if use_fc_norm else nn.Identity()
        self.fc_norm = nn.Identity() #if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) #if num_classes > 0 else nn.Identity()

    def global_pool(self, x):
        x = x[:, 1:].mean(dim=1)# if self.global_pool == 'avg' else x[:, 0]
        return x

    def forward(self, x):
        # x = self.patch_embed(x)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)
        # x = self.blocks_whole(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.global_pool(x)
        # x = self.fc_norm(x)
        # x = self.head(x)
        return x
