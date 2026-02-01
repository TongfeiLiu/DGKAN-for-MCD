import torch
import torch.nn.functional as F

import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_regularization(model,lambda_reg=0.01):  # 对模型进行正则化，助于学习到更稳定和泛化的表示，防止过拟合

    l2_reg = torch.tensor(0.0, device=torch.device('cuda'))  # 初始化为0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    return lambda_reg * l2_reg


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).to(device) - (1/dim) * torch.ones(dim, dim).to(device)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC






