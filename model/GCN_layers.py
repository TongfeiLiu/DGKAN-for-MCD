import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# class GraphConvolution(nn.Module):
#     def __init__(self, input_dim, output_dim, use_bias=True):
#         """图卷积：L*X*\theta """
#         super(GraphConvolution, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.use_bias = use_bias
#         self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim).float())
#         if self.use_bias:
#             self.bias = nn.Parameter(torch.Tensor(output_dim).float())
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)
#         # init.kaiming_uniform_(self.weight)
#         if self.use_bias:
#             init.zeros_(self.bias)
#
#
#     def forward(self,  input_feature,adjacency):
#         support =torch.sparse.mm(input_feature, self.weight)
#         output = torch.sparse.mm(adjacency, support)
#         if self.use_bias:
#             output += self.bias
#         return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' ('             + str(self.input_dim) + ' -> '             + str(self.output_dim) + ')'

#



# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#         self.kan_activation = KANLinear(in_features, out_features)  # 添加KAN激活函数
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             output = output + self.bias
#         output = self.kan_activation(output)  # 应用KAN激活函数
#         return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

