import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import cv2
import scipy.spatial
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def compute_feature_similarity(feature_vectors, method='euclidean'):
    """
    构建邻接矩阵
    :param feature_vectors: 特征向量矩阵，形状为 (num_nodes, num_features)
    :param method: 相似性计算方法，支持 'euclidean' 或 'cosine'
    :return: 相似性矩阵
    """
    feature_vectors = feature_vectors.to(device)  # 移动到GPU

    if method == 'euclidean':
        distance_matrix = torch.cdist(feature_vectors, feature_vectors)  # 计算欧氏距离
        similarity_matrix = torch.exp(-distance_matrix)  # 转换为相似性矩阵
    elif method == 'cosine':
        similarity_matrix = F.cosine_similarity(feature_vectors.unsqueeze(1),
                                                feature_vectors.unsqueeze(0), dim=2)
    else:
        raise ValueError("Unsupported method. Use 'euclidean' or 'cosine'.")

    return similarity_matrix



def knn_adjacency_matrix_torch(features, k):
    """
    使用KNN算法根据节点特征矩阵构建邻接矩阵，使用PyTorch
    :param features: 节点特征矩阵，形状为 (num_nodes, num_features)
    :param k: 每个节点的最近邻数量
    :return: 邻接矩阵，形状为 (num_nodes, num_nodes)
    """
    num_nodes = features.shape[0]
    device = features.device

    # 计算所有节点之间的欧氏距离
    distances = torch.cdist(features, features)  # 使用cDist计算欧氏距离，更加高效
    distances = 1 - distances  # 转换为距离（这里看你的代码逻辑是把相似度转换为距离）

    # 获取最近邻的索引
    _, indices = torch.topk(distances, k=k + 1, largest=False)  # 获取k+1个最近邻，包括自身

    # 构建邻接矩阵
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    batch_indices = torch.arange(num_nodes, device=device).unsqueeze(1)

    # 使用向量化操作，避免循环
    adjacency_matrix[batch_indices, indices[:, 1:]] = 1
    adjacency_matrix[indices[:, 1:], batch_indices] = 1  # 对称

    return adjacency_matrix


def laplacian_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.size(1)
    K = -gamma * torch.cdist(X, Y, p=1)
    torch.exp(K, out=K)  # exponentiate K in-place
    return K



def construct_delaunay_graph(coords):
    tri = Delaunay(coords.cpu().numpy())
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add((simplex[i], simplex[j]))

    num_pixels = coords.shape[0]
    adjacency_matrix = torch.zeros((num_pixels, num_pixels), device='cuda')
    for edge in edges:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1

    return adjacency_matrix

#近邻
def knn_adjacency(superpixel_mask, k, device='cuda'):
    coords = np.column_stack(np.where(superpixel_mask.cpu().numpy()))
    adj_matrix = kneighbors_graph(coords, k, mode='connectivity', metric="minkowski").toarray()
    return torch.tensor(adj_matrix, dtype=torch.float32, device=device)



# 示例用法
#根据像素之间的距离赋予权重
def weighted_adjacency(superpixel_mask, device=device):
    coords = np.column_stack(np.where(superpixel_mask.cpu().numpy()))


    # 计算所有像素对的距离
    diff = coords[:, None, :] - coords[None, :, :]  # 计算坐标差异
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))  # 欧几里得距离矩阵

    # 计算权重矩阵
    adj_matrix = torch.tensor(1 / (dist_matrix + 1e-5), dtype=torch.float32, device=device)
    return adj_matrix


# 示例用法




# 判断两个像素是否为8邻域
def is_8_adjacent(p1, p2):
    return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1 and not np.array_equal(p1, p2)


def is_4_adjacent(p1, p2):  # 4邻域
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1


def compute_adjacency_matrix(superpixel_mask):
    coords = np.column_stack(np.where(superpixel_mask.cpu().numpy()))
    num_pixels = len(coords)

    # 使用 PyTorch 创建零张量
    adj_matrix = torch.zeros((num_pixels, num_pixels), dtype=torch.int32).cuda()  # 将张量移到 GPU

    for i in range(num_pixels):
        for j in range(i + 1, num_pixels):
            if is_4_adjacent(coords[i], coords[j]):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    return adj_matrix


def normalize_adjacency_matrix_torch(adj):
    """对称归一化邻接矩阵，使用PyTorch"""
    adj = adj + torch.eye(adj.size(0), device=device)  # 添加自环
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)