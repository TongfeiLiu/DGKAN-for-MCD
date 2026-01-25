from all_data_preprocess.Adj_calculation import *
from all_loss.encode_loss import *


def build_graphs(img_t1, img_t2, objects, device):

    obj_nums = torch.max(objects) + 1
    all_Superpixel_area_t1 = []  # 节点特征矩阵
    all_Superpixel_area_t2 = []
    feature_graphs_t1 = []#基于特征相似性的邻接矩阵
    feature_graphs_t2 = []
    structure_graphs_t1 = []#基于空间邻接的邻接矩阵
    structure_graphs_t2 = []

    for idx in range(obj_nums):
        obj_idx = objects == idx
        if img_t1.shape[-1] == 3:
            # 提取每个超像素的特征矩阵到列表里
            all_Superpixel_area_t1.append(img_t1[obj_idx])
            all_Superpixel_area_t2.append(img_t2[obj_idx])
            # 基于特征相似性构建特征图连接关系———t1
            feature_vectors_t1 = img_t1[obj_idx].reshape(-1, 3).to(device)  # 使用RGB颜色作为特征
            feature_graph_t1 = compute_feature_similarity(feature_vectors_t1, method='euclidean')
            feature_graphs_t1.append(normalize_adjacency_matrix_torch(feature_graph_t1))
            # 基于特征相似性构建特征图连接关系———t2
            feature_vectors_t2 = img_t2[obj_idx].reshape(-1, 3).to(device)  # 使用RGB颜色作为特征
            feature_graph_t2 = compute_feature_similarity(feature_vectors_t2, method='euclidean')
            feature_graphs_t2.append(normalize_adjacency_matrix_torch(feature_graph_t2))

            # 基于每个像素节点的空间临近关系构建图谱
            # adj_matrix_4=compute_adjacency_matrix(obj_idx)
            # adj_matrix_4 = weighted_adjacency(obj_idx, device=device)
            adj_matrix_4 = knn_adjacency(obj_idx, k=10, device=device)
            structure_graphs_t1.append(normalize_adjacency_matrix_torch(adj_matrix_4))
            structure_graphs_t2.append(normalize_adjacency_matrix_torch(adj_matrix_4))
        else:
            # 提取每个超像素的特征矩阵到列表里
            all_Superpixel_area_t1.append(img_t1[obj_idx].reshape(-1, 1))
            all_Superpixel_area_t2.append(img_t2[obj_idx].reshape(-1, 1))
            # 基于特征相似性构建特征图连接关系———t1
            feature_vectors_t1 = img_t1[obj_idx].reshape(-1, 1).to(device)  # 使用RGB颜色作为特征
            feature_graph_t1 = compute_feature_similarity(feature_vectors_t1, method='euclidean')
            feature_graphs_t1.append(normalize_adjacency_matrix_torch(feature_graph_t1))
            # 基于特征相似性构建特征图连接关系———t2
            feature_vectors_t2 = img_t2[obj_idx].reshape(-1, 1).to(device)  # 使用RGB颜色作为特征
            feature_graph_t2 = compute_feature_similarity(feature_vectors_t2, method='euclidean')
            feature_graphs_t2.append(normalize_adjacency_matrix_torch(feature_graph_t2))

            # 基于每个像素节点的空间临近关系构建图谱
            adj_matrix_4 = weighted_adjacency(obj_idx, device=device)
            structure_graphs_t1.append(normalize_adjacency_matrix_torch(adj_matrix_4))
            structure_graphs_t2.append(normalize_adjacency_matrix_torch(adj_matrix_4))

    return {
        'all_Superpixel_node_t1':all_Superpixel_area_t1,
        'all_Superpixel_node_t2':all_Superpixel_area_t2,
        'feature_graphs_t1':feature_graphs_t1,
        'feature_graphs_t2':feature_graphs_t2,
        'structure_graphs_t1':structure_graphs_t1,
        'structure_graphs_t2':structure_graphs_t2,
        'obj_nums':obj_nums
            }

