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


import torch
import torch.nn.functional as F


def build_positive_negative_pairs_distance(A_features, B_features, metric='euclidean', device='cuda:0'):
    """
    正负样本的建立：根据相似和不相似
    """
    # 将 Tensor 移动到设备
    A_features = A_features.to(device)
    B_features = B_features.to(device)

    positive_pairs = []
    negative_pairs = []

    if metric == 'cosine':
        # 计算余弦相似度
        A_norm = F.normalize(A_features, p=2, dim=1)
        B_norm = F.normalize(B_features, p=2, dim=1)
        # 计算余弦相似度
        similarity_matrix = torch.matmul(A_norm, B_norm.t())

        # 查找每个 A 特征的最相似 B 特征（最大余弦相似度）
        _, indices = torch.max(similarity_matrix, dim=1)
        positive_pairs = [(i, indices[i]) for i in range(indices.size(0))]

        # 查找每个 A 特征的最不相似 B 特征（最小余弦相似度）
        _, neg_indices = torch.min(similarity_matrix, dim=1)
        negative_pairs = [(i, neg_indices[i]) for i in range(neg_indices.size(0))]

    elif metric == 'euclidean':
        # 计算欧氏距离
        dist_matrix = torch.cdist(A_features, B_features, p=2)

        # 查找每个 A 特征的最相似 B 特征（最小欧氏距离）
        _, indices = torch.min(dist_matrix, dim=1)
        positive_pairs = [(i, indices[i]) for i in range(indices.size(0))]

        # 查找每个 A 特征的最不相似 B 特征（最大欧氏距离）
        _, neg_indices = torch.max(dist_matrix, dim=1)
        negative_pairs = [(i, neg_indices[i]) for i in range(neg_indices.size(0))]

    return positive_pairs, negative_pairs


import torch

def get_samples_from_indices(A_features, B_features, positive_pairs, negative_pairs):
    """
    根据正负样本的索引提出对应的特征
    :param A_features:
    :param B_features:
    :param positive_pairs:
    :param negative_pairs:
    :return:
    """

    # 将索引列表转换为 Tensor
    positive_indices_A = torch.tensor([i for i, _ in positive_pairs], device=A_features.device)
    positive_indices_B = torch.tensor([j for _, j in positive_pairs], device=B_features.device)

    negative_indices_A = torch.tensor([i for i, _ in negative_pairs], device=A_features.device)
    negative_indices_B = torch.tensor([j for _, j in negative_pairs], device=B_features.device)

    # 使用索引直接提取特征
    positive_A_features = A_features[positive_indices_A]
    positive_B_features = B_features[positive_indices_B]

    negative_A_features = A_features[negative_indices_A]
    negative_B_features = B_features[negative_indices_B]

    return positive_A_features, positive_B_features, negative_A_features, negative_B_features




def info_nce_loss(positive_A_features, positive_B_features, negative_A_features, negative_B_features, temperature=0.07):
    """"
    这个对比损失是让正样本的相似度更大，让负样本相似度更小
    """

    # 1. 将正负样本对的特征合并成一个大的样本集合
    all_A_features = torch.cat([positive_A_features, negative_A_features], dim=0)  # 形状 (2N, F)
    all_B_features = torch.cat([positive_B_features, negative_B_features], dim=0)  # 形状 (2N, F)

    # 2. 计算所有样本对之间的余弦相似度
    sim_matrix = F.cosine_similarity(all_A_features.unsqueeze(1), all_B_features.unsqueeze(0), dim=2)  # 形状 (2N, 2N)

    # 3. 创建标签：正样本对位于同一行上
    labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)

    # 4. 计算每一对样本的损失
    logits = sim_matrix / temperature  # 将相似度除以温度进行平滑
    loss = F.cross_entropy(logits, labels)  # 计算交叉熵损失

    return loss


def contrastive_loss(positive_A_features, positive_B_features, negative_A_features, negative_B_features, margin=1.0):
    """
    #这个对比损失是是根据距离，让正样本之间距离更小，负样本距离更大
    :param positive_A_features:
    :param positive_B_features:
    :param negative_A_features:
    :param negative_B_features:
    :param margin:
    :return:
    """

    # 计算正样本对的欧氏距离（平方）
    positive_distance = F.pairwise_distance(positive_A_features, positive_B_features, p=2)
    positive_loss = torch.mean(positive_distance ** 2)  # 对正样本对损失取均值

    # 计算负样本对的欧氏距离（平方）
    negative_distance = F.pairwise_distance(negative_A_features, negative_B_features, p=2)
    negative_loss = torch.mean(torch.clamp(margin - negative_distance, min=0) ** 2)  # 对负样本对损失取均值并加上边界

    # 总损失为正样本损失和负样本损失之和
    total_loss = 0.5 * positive_loss + 0.5 * negative_loss

    return total_loss


