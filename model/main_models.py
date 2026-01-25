import torch.nn as nn
import torch.nn.functional as F
from model.GCN_layers import GraphConvolution
import torch
import math
from model.KAN import KANLinear
from Fusion_Attention.attention import *



class KANEncode(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(KANEncode, self).__init__()
        self.kans1 = KANLinear(in_channels, hidden_channels)
        self.kans2 = KANLinear(hidden_channels, out_channels)
    def forward(self, x):
        x1 = self.kans1(x)
        x2 = self.kans2(x1)
        return x1, x2



class GCNEncoderWithMLP(nn.Module):
    def __init__(self, g_in_features, g_hidden, g_out_features, dropout):
        super(GCNEncoderWithMLP, self).__init__()
        # GCN 局部特征提取
        self.gc1 = GraphConvolution(g_in_features, g_hidden)
        self.gc2 = GraphConvolution(g_hidden, g_out_features)

        # MLP 分支，用于替代 KAN
        self.mlp = nn.Sequential(
            nn.Linear(g_in_features, g_hidden),
            nn.ReLU(),
            nn.Linear(g_hidden, g_out_features),
        )
        self.Global = GlobalAttention(embed_size=g_out_features, conv_out_channels=g_out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # GCN 提取局部图特征
        g_ = self.gc1(x, adj)
        g_1 = F.leaky_relu(g_)
        g_2 = F.leaky_relu(self.gc2(g_1, adj))

        # MLP 提取特征
        mlp_feat = self.mlp(x)

        gk_combine = torch.stack((g_2, mlp_feat), dim=1)
        gk = self.Global(gk_combine)

        return {
            'g_1': g_1,
            'g_2': g_2,
            'mlp_feat': mlp_feat,  # 用于对比 KAN
            'gk': gk['pooled_output'],
        }


class GCNEncoder(nn.Module):
    def __init__(self, g_in_features, g_hidden, g_out_features, dropout):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(g_in_features, g_hidden)
        self.gc2 = GraphConvolution(g_hidden, g_out_features)
        self.kan = KANEncode(g_in_features, g_hidden, g_out_features)
        self.Global = GlobalAttention(embed_size=g_out_features, conv_out_channels=g_out_features)


    def forward(self, x, adj):
        # 使用GCN提取局部
        g_ = self.gc1(x, adj)
        g_1 = F.leaky_relu(g_)
        g_2 = F.leaky_relu(self.gc2(g_1, adj))
        k_1, k_2 = self.kan(x)

        gk_combine = torch.stack((g_2, k_2), dim=1)
        gk = self.Global(gk_combine)
        return {
            'g_1':  g_1,
            'g_2': g_2,
            'k_1': k_1,
            'k_2': k_2,
            'gk': gk['pooled_output'],
        }

class GK_Encode(nn.Module):
    def __init__(self, g_in_features, g_hidden, g_out_features, k_in_features, k_hidden, k_out_features, dropout):
        super(GK_Encode, self).__init__()
        self.GCN1 = GCNEncoder(g_in_features, g_hidden, g_out_features, dropout)
        self.GCN2 = GCNEncoder(g_in_features, g_hidden, g_out_features, dropout)
        # self.GCN = GCNEncoder(g_in_features, g_hidden, g_out_features, dropout)
        # self.KAN = KANEncode(k_in_features, k_hidden, k_out_features)
        # self.kans = KANLinear(g_out_features, g_in_features)
        # self.attention = LocalAttention(embed_size=g_out_features, window_size=5)
        # self.fc_transform = nn.Linear(g_out_features, g_out_features)
        # self.unforattention = UniformAttention(in_channels=3)
        self.globa = GlobalAttention(embed_size=g_out_features, conv_out_channels=g_out_features)

    def forward(self, x, f_adj, s_adj):
        output_f = self.GCN1(x, f_adj)
        # output_cf = self.GCN(x, f_adj)
        # output_cs = self.GCN(x, s_adj)
        output_s = self.GCN2(x, s_adj)

        fsc_combine = torch.stack((output_f['gk'],  output_s['gk']), dim=1)
        output = self.globa(fsc_combine)


        return output_f, output_s, output



#解码器
class GCNdecoder(nn.Module):
    def __init__(self, g_in_features, g_hidden, g_out_features, dropout):
        super(GCNdecoder, self).__init__()
        self.gc1 = GraphConvolution(g_in_features, g_hidden)
        self.gc2 = GraphConvolution(g_hidden, g_out_features)
        self.kan = KANEncode(g_in_features, g_hidden, g_out_features)
        self.Global = GlobalAttention(embed_size=g_out_features, conv_out_channels=g_out_features)
        # self.multy = MultiHeadAttention(num_heads=2, embed_size=g_out_features)

    def forward(self, x, adj):
        # 使用GCN提取局部
        g_ = self.gc1(x, adj)
        g_1 = F.leaky_relu(g_)
        g_2 = F.leaky_relu(self.gc2(g_1, adj))
        k_1, k_2 = self.kan(x)

        gk_combine = torch.stack((g_2, k_2), dim=1)
        gk = self.Global(gk_combine)
        return {
            'g_1':  g_1,
            'g_2': g_2,
            'k_1': k_1,
            'k_2': k_2,
            'gk': gk['pooled_output'],
        }

class ComprehensiveDecoder(nn.Module):
    def __init__(self, g_in_features, g_hidden, g_out_features, k_in_features, k_hidden, k_out_features, dropout,
                 ):


        super(ComprehensiveDecoder, self).__init__()
        self.f_decoder = GCNdecoder(g_in_features, g_hidden, g_out_features,dropout)
        self.s_decoder = GCNdecoder(g_in_features, g_hidden, g_out_features,  dropout)


    def forward(self,loc_unloc,f_adj,s_adj):
        # 重构特征图节点特征矩阵
        output_rec_f = self.f_decoder(loc_unloc,  f_adj)
        output_rec_s = self.s_decoder(loc_unloc, s_adj)


        return output_rec_f, output_rec_s
