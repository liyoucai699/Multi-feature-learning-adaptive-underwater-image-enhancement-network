# 2022/2/27 15:30
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from models.unet_parts import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        # x2 = x - mu
        # x3 = torch.sqrt(sigma + 1e-5)
        # x4 = x2 / x3
        # # return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        # x7, x8 = self.weight, self.bias
        # x5 = x4 * self.weight
        # x6 = x5 + self.bias
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x1 = self.body(to_3d(x))
        return to_4d(self.body(to_3d(x)), h, w)


# dim=512
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 512*
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 512*3
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 512
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

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



class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # [4, 256, 4, 4]
            Flatten(),
            # nn.Linear(4096, 1024),
            # nn.Linear(16384, 4096),
            # nn.BatchNorm1d(4096),
            # nn.ReLU(True),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, input):
        x = input
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)
            if i == 2:
                mid_op = x

        #    print(mid_op.shape, x.shape)
        return x, mid_op


class Block(nn.Module):
    def __init__(self,
                 input_dim,
                 dim,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 num_heads=4,
                 LayerNorm_type='WithBias'
                 ):
        super(Block, self).__init__()
        self.inp_dim = input_dim
        self.norm1 = LayerNorm(dim=int(input_dim / 2), LayerNorm_type=LayerNorm_type)
        self.ffn = FeedForward(dim=int(input_dim / 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.norm2 = LayerNorm(dim=int(input_dim / 2), LayerNorm_type=LayerNorm_type)
        self.att = Attention(dim=int(input_dim / 2), num_heads=num_heads, bias=bias)

        self.up = up(input_dim, dim)

    def forward(self, input, enc_out):
        # inp = self.inp_dim
        # att_op = input
        att_op = input + self.ffn(self.norm1(input))
        att_op = att_op + self.ffn(self.norm1(att_op))
        att_op = att_op + self.att(self.norm2(att_op))
        output = self.up(att_op, enc_out)

        return output


class UNetDecoder(nn.Module):
    def __init__(self,
                 input_dim=[1024, 512, 256, 128],
                 bias=False,
                 dim=[256, 128, 64, 64],
                 n_channels=3
                 ):
        super(UNetDecoder, self).__init__()

        self.classfilier = Classifier()
        self.up = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.conv = nn.ConvTranspose2d(512, 512, 1, 1)
        self.block1 = Block(1024, 256)
        self.block2 = Block(512, 128)
        self.block3 = Block(256, 64)
        self.block4 = Block(128, 64)
        # self.block1 = Block(input_dim=input_dim[0], dim=dim[0])
        # self.block2 = Block(input_dim=input_dim[1], dim=dim[1])
        # self.block3 = Block(input_dim=input_dim[2], dim=dim[2])
        # self.block4 = Block(input_dim=input_dim[3], dim=dim[3])
        self.outc = outconv(dim[3], n_channels)
        self.sigmod = nn.Sigmoid()

        self.dwconv1 = nn.Conv2d(n_channels, n_channels * 2, 1, 1)
        self.dwconv = nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=1, stride=1, groups=n_channels * 2,
                                bias=bias)
        self.conv1_1 = nn.Conv2d(n_channels, n_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, input, enc_outs):
        _, att_ip = self.classfilier(input)

        att_ip = self.up(att_ip)
        att_ip = input + att_ip
        att_ip = self.conv(att_ip)

        # att_ip = input

        x = self.block1(att_ip, enc_outs[3])  # [4, 256, 32, 32]
        x = self.block2(x, enc_outs[2])
        x = self.block3(x, enc_outs[1])
        x = self.block4(x, enc_outs[0])
        x = self.outc(x)

        #双层
        x = self.dwconv(self.dwconv1(x))
        x1, x2 = x.chunk(2, dim=1)

        # x1 = self.bn(x1)
        # x2 = self.bn(x2)
        # x1 = self.relu(x1)
        # x2 = self.relu(x2)
        # x1 = self.conv1_1(x1)
        # x2 = self.conv1_1(x2)
        # x3 = x1 + x2

        # return nn.Tanh()(x)
        return nn.Tanh()(x1), nn.Tanh()(x2), nn.Tanh()(x1) + nn.Tanh()(x2)


class UNetEncoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)  # (conv => BN => ReLU) * 2
        self.down1 = down(64, 128)  # maxpooling+卷积
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, (x1, x2, x3, x4)


class UNetDecoder1(nn.Module):
    def __init__(self,
                 dim=256,
                 ffn_expansion_factor=2.8,
                 # ffn_expansion_factor=2.6,
                 bias=False, num_heads=4,
                 LayerNorm_type='WithBias',
                 n_channels=3
                 ):
        super(UNetDecoder1, self).__init__()

        self.classfilier = Classifier(n_channels)

        self.norm1 = LayerNorm(512, LayerNorm_type)
        self.ffn = FeedForward(512, ffn_expansion_factor, bias)
        self.norm2 = LayerNorm(512, LayerNorm_type)
        self.att = Attention(512, num_heads, bias)

        # self.up5 = torch.nn.ConvTranspose2d(256, 256, 1, stride=1, padding=2)

        # debug
        self.up1 = up(1024, 256)  # 上采样/ConvTranspose2d
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels)  # conv2
        self.sigmoid = nn.Sigmoid()
        self.up5 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.dwconv1 = nn.Conv2d(n_channels, n_channels * 2, 1, 1)
        self.dwconv = nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=1, stride=1, groups=n_channels * 2,
                                bias=bias)
        self.conv = nn.Conv2d(512, 512, 1, 1)

        # self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1)

    def forward(self, input, enc_outs):
        water_type, att_ip = self.classfilier(input)
        att_ip = self.up5(att_ip)
        att_ip = input + att_ip
        att_ip = self.conv(att_ip)

        # Attention
        att_op = att_ip + self.ffn(self.norm1(att_ip))  # [4, 256, 8, 8]
        att_op = att_op + self.att(self.norm2(att_op))  # [4, 256, 8, 8]
        # att_op = att_ip + self.ffn(self.norm1(att_ip))  #[4, 256, 8, 8]

        x = self.sigmoid(att_op)
        # x = self.up5(x)
        x = self.up1(x, enc_outs[3])
        x = self.up2(x, enc_outs[2])
        x = self.up3(x, enc_outs[1])
        x = self.up4(x, enc_outs[0])  # [4, 32, 256, 256]
        x = self.outc(x)
        x = self.dwconv(self.dwconv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x3 = x1+x2

        # return water_type, nn.Tanh()(x)#, att_op
        return nn.Tanh()(x1), nn.Tanh()(x2), nn.Tanh()(x3)#nn.Tanh()(x1) + nn.Tanh()(x2)
        # return nn.Tanh()(x)#, att_op
