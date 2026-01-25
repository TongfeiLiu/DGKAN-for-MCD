import pywt
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

import math
#创建滤波器
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    #对滤波器进行系数反转
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)

    # 创建分解滤波器
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    #创建重构滤波器
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])

    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


# Wavelet transform function
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


# Inverse wavelet transform function
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# 初始化小波变换
def wavelet_transform_init(filters):
    class WaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                return wavelet_transform(input, filters)

        @staticmethod
        def backward(ctx, grad_output):
            return inverse_wavelet_transform(grad_output, filters), None

    return WaveletTransform.apply


# 初始逆小波
def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            with torch.no_grad():
                return inverse_wavelet_transform(input, filters)

        @staticmethod
        def backward(ctx, grad_output):
            return wavelet_transform(grad_output, filters), None

    return InverseWaveletTransform.apply


#
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        #定义小波和逆小波的网络模块
        self.wt_function = wavelet_transform_init(self.wt_filter)
        self.iwt_function = inverse_wavelet_transform_init(self.iwt_filter)

        # 定义卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 卷积的列表
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # 卷积之后调整分辨率
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        #进行小波变换
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        #进行逆小波变换

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# Scaling module
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
class SelfAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_size, hidden_size)
        self.key = nn.Linear(in_size, hidden_size)
        self.value = nn.Linear(in_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

class LocalAttention(nn.Module):
    def __init__(self, embed_size, window_size):
        super(LocalAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)
        self.window_size = window_size

    def forward(self, x):
        # 假设输入 x 的维度为 [batch_size, L, embed_size]
        batch_size, L, _ = x.size()

        # 计算 Q, K, V
        Q = self.query(x)  # [batch_size, L, embed_size]
        K = self.key(x)  # [batch_size, L, embed_size]
        V = self.value(x)  # [batch_size, L, embed_size]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))  # [batch_size, L, L]

        # 创建遮蔽矩阵，遮蔽掉超出窗口大小的注意力
        mask = torch.zeros(L, L, device=x.device)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1  # 仅保留窗口内的部分

        # 应用遮蔽
        attention_scores = attention_scores * mask.unsqueeze(0)  # [1, L, L]

        # 计算注意力权重
        attention_weights = self.softmax(attention_scores)  # [batch_size, L, L]

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [batch_size, L, embed_size]
        pooled_output = output.mean(dim=1)

        return {'output': output,
                'attention_weights': attention_weights,
                'pooled_output': pooled_output
                }


class GlobalAttention(nn.Module):
    def __init__(self, embed_size, conv_out_channels):
        super(GlobalAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)
        # 在这里直接聚合 output 以获得 [N, E] 形状
        pooled_output = output.sum(dim=1)  # 通过求和聚合，形状: [N, E]改动
        # 或者可以选择使用其他聚合方法，如平均
        # pooled_output = output.mean(dim=1)  # 全局平均池化，输出形状: [N, E]
        return {'output': output,
                'attention_weights': attention_weights,
                'pooled_output': pooled_output
                }
