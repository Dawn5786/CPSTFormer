
from torchvision import models

import torch.nn as nn
import torch
from timm.models.layers import PatchEmbed, Mlp, DropPath #, trunc_normal_, lecun_normal_
from functools import partial

class sn_MLP(nn.Module):
    """
       Multilayer perceptron fitted for scattering input
    """
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8):
        super(sn_MLP,self).__init__()
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

class AttentionMul(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # 300,14*14,576
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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
class AttentionBlockMul(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim) #dim=64*3
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = AttentionMul(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
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

class sn_CNN(nn.Module):
    """
    CNN fitted for scattering input
    Model from: https://github.com/kymatio/kymatio/blob/master/examples/2d/cifar_small_sample.py 
    """
    def __init__(self, in_channels, k=8, n=4, num_classes=10, standard=False):
        super(sn_CNN, self).__init__()

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


class sn_LinearLayer(nn.Module):
    """
    Linear layer fitted for scattering input
    """
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8):
        super(sn_LinearLayer, self).__init__()
        self.n_coefficients = n_coefficients
        self.num_classes = num_classes

        self.fc1 = nn.Linear(int(3*M_coefficient*N_coefficient*n_coefficients), num_classes)
        self.bn0 = nn.BatchNorm2d(self.n_coefficients*3, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.bn0(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class sn_Resnet50(nn.Module):
    """
    Pretrained model on ImageNet
    Architecture: ResNet-50
    """
    def __init__(self, num_classes=10):
        super(sn_Resnet50, self).__init__()
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

class sn_ViT(nn.Module):
    """
    ViT fitted for scattering input    Architecture: ViT-tiny
    Inf:
    """
    # def __init__(self, in_channels, num_heads=3, num_layers=12, unit_embed_dim=64, num_classes=10, standard=False, img_size=8, patch_size=1):
    # def __init__(self, in_channels, num_heads=3, num_layers=6, unit_embed_dim=128, num_classes=10, standard=False, img_size=8, patch_size=1):
    def __init__(self, in_channels, num_heads=3, num_layers=3, unit_embed_dim=128, num_classes=10, standard=False, img_size=8, patch_size=1):
        super(sn_ViT, self).__init__()

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


class sn_MulViT(nn.Module):
    """
    ViT fitted for scattering input    Architecture: ViT-tiny
    Inf:
    """
    # def __init__(self, in_channels, num_heads=3, num_layers=12, unit_embed_dim=64, num_classes=10, standard=False, img_size=8, patch_size=1):
    # def __init__(self, in_channels, num_heads=3, num_layers=6, unit_embed_dim=64, num_classes=10, standard=False, img_size=8, patch_size=1):
    def __init__(self, in_channels, num_heads=3, num_layers=6, unit_embed_dim=128, num_classes=10, standard=False, img_size=8, patch_size=1):
        super(sn_MulViT, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.unit_embed_dim = unit_embed_dim
        self.embed_dim = unit_embed_dim * num_heads
        # self.embed_dim = unit_embed_dim # * num_heads

        self.num_classes = num_classes

        if standard:
            self.in_channels = 3
            self.img_size = 32
            self.patch_size = 2
            self.standard = True
        else:
            # self.in_channels = in_channels * 3
            # self.in_channels_0 = 25 * 3 # L=8
            self.in_channels_0 = 19 * 3 # L=6
            # self.in_channels_1 = 233 * 3 # L=8
            self.in_channels_1 = 139 * 3 # L=6
            # self.in_channels_2 = 97 * 3 # L=8
            self.in_channels_2 = 61 * 3 # L=6

            # self.img_size = 8
            # self.img_size_1 = 56
            # self.img_size_2 = 28
            # self.img_size_3 = 14
            self.img_size_0 = 48
            self.img_size_1 = 24
            self.img_size_2 = 12
            # self.patch_size = 1
            self.patch_size_0 = 4
            self.patch_size_1 = 2
            self.patch_size_2 = 1
            self.standard = False

        self.bn0 = nn.BatchNorm2d(self.in_channels_0, eps=1e-5, affine=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels_1, eps=1e-5, affine=True)
        self.bn2 = nn.BatchNorm2d(self.in_channels_2, eps=1e-5, affine=True)

        self.patch_embed_0 = PatchEmbed(
            img_size=self.img_size_0, patch_size=self.patch_size_0, in_chans=self.in_channels_0, embed_dim=self.unit_embed_dim)
            # img_size=self.img_size_0, patch_size=self.patch_size_0, in_chans=self.in_channels_0, embed_dim=self.embed_dim)
        self.patch_embed_1 = PatchEmbed(
            img_size=self.img_size_1, patch_size=self.patch_size_1, in_chans=self.in_channels_1, embed_dim=self.unit_embed_dim)
            # img_size=self.img_size_1, patch_size=self.patch_size_1, in_chans=self.in_channels_1, embed_dim=self.embed_dim)
        self.patch_embed_2 = PatchEmbed(
            img_size=self.img_size_2, patch_size=self.patch_size_2, in_chans=self.in_channels_2, embed_dim=self.unit_embed_dim)
            # img_size=self.img_size_2, patch_size=self.patch_size_2, in_chans=self.in_channels_2, embed_dim=self.embed_dim)

        # num_patches = self.patch_embed.num_patches
        num_patches = self.patch_embed_1.num_patches
        embed_len = num_patches  # + 1 #num_patches if no_embed_class else num_patches + self.num_prefix_tokens

        # if standard:
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # if class_token else None
        # self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.unit_embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=0) #

        dpr = [x.item() for x in torch.linspace(0, 0.1, num_layers)]  # stochastic depth decay rule
  
        self.blocks_whole = nn.Sequential(*[
            AttentionBlockMul(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, init_values=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(num_layers)])
        # self.norm = norm_layer(self.unit_embed_dim) #if not use_fc_norm else nn.Identity()
        self.norm = norm_layer(self.embed_dim) #if not use_fc_norm else nn.Identity()

            # Classifier Head
        # self.fc_norm = norm_layer(self.embed_dim) #if use_fc_norm else nn.Identity()
        self.fc_norm = nn.Identity() #if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) #if num_classes > 0 else nn.Identity()

    def global_pool(self, x):
        x = x[:, 1:].mean(dim=1)# if self.global_pool == 'avg' else x[:, 0]
        return x

    def forward(self, x0, x1, x2):
        x0 = self.patch_embed_0(x0)
        x0 = x0 + self.pos_embed
        x0 = self.pos_drop(x0)

        x1 = self.patch_embed_1(x1)
        x1 = x1 + self.pos_embed
        x1 = self.pos_drop(x1)

        x2 = self.patch_embed_2(x2)
        x2 = x2 + self.pos_embed
        x2 = self.pos_drop(x2)

        x = torch.cat((x0, x1, x2), 2)
        x = self.blocks_whole(x)

        x = self.norm(x)
        x = self.global_pool(x)
        x = self.fc_norm(x)
        x = self.head(x)
        return x
