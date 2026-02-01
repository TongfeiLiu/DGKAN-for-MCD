import pywt
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

import math

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
