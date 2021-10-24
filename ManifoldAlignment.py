import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas
from scipy.sparse import bmat, csgraph, lil_matrix, linalg

import torch as th
from sklearn.metrics import pairwise_distances

# path for the pretrained
path_kgembed = './data/YAGOfacts/TransE_KG/TransE.ckpt'
path_dist = './data/YAGOfacts/TransE_KG'


def get_gep_file_name(num_neighbor, t_heat_kernel, weight_of_simi, num_eig):
    return f"GEP_{num_neighbor}_{t_heat_kernel}_{weight_of_simi}_{num_eig}.npz"


# Get entities (i.e., leaf nodes) & common entities list and id list
def load_leaf_nodes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        leaf_nodes = [line.strip() for line in f]
    return leaf_nodes


def load_nodes2id(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    nodes = []
    nodes_id = []
    for line in lines:
        item = line.strip().split()
        nodes.append(item[0])
        if item[1] != 'None':
            nodes_id.append(int(item[1]))
        else:
            nodes_id.append(np.nan)
    return nodes, nodes_id


def load_leaves2id_from_dir(path_taxonomy) -> Tuple[List, List, List, List]:
    path_train_leaves = os.path.join(path_taxonomy, 'train_leaves2id.txt')
    path_test_leaves = os.path.join(path_taxonomy, 'test_leaves2id.txt')
    trains = load_nodes2id(path_train_leaves)
    tests = load_nodes2id(path_test_leaves)
    return trains+tests


def load_common_entities(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    common_entity = []
    common_entity_id = []
    common_entity_id2 = []
    for line in lines[1:]:
        item = line.strip().split()
        common_entity.append(item[0])  # entity word
        common_entity_id.append(int(item[1]))  # common entity id in knowledge graph
        common_entity_id2.append(int(item[2]))  # common entity id in leaf nodes
    return common_entity, common_entity_id, common_entity_id2


def load_kb_embeddings(file_path):
    with open(file_path, "r") as f:
        kgmodels = json.loads(f.read())
    kg_entityembed = kgmodels.get('ent_embeddings')
    # Matrix X: n*d_e, embeddings of knowledge graph
    X = np.array([item for item in kg_entityembed])
    return X


def load_hy_embeddings(file_path, leafnodes):
    hypermodels = th.load(file_path)
    hyperembed = hypermodels['embeddings']
    taxonomy_entity = hypermodels['objects']
    taxonomy_dict = dict(zip(taxonomy_entity, hyperembed))
    leaf_hyperembed = []
    for item in leafnodes:
        item_embed = taxonomy_dict.get(item)
        leaf_hyperembed.append(item_embed)
    # Matrix Y: m*d_h, embeddings of taxonomy leaf nodes
    Y = np.array([item.numpy() for item in leaf_hyperembed])
    return Y


def pairwise_distance(X, space='euclidean', n_jobs=None, numeric_stability=0.0000001):
    if space == 'euclidean':
        return pairwise_distances(X, metric=space, n_jobs=n_jobs)
    elif space == 'hyperboloid':
        n, d = X.shape
        L = np.zeros((n, n), dtype=float)
        for i in range(0, n-1):
            L[i, i+1:] = np.dot(X[i], X[i+1:].T) - 2 * X[i, 0] * X[i+1:, 0]
        L += L.T
        L = -L
        L[L < 1] = 1
        D = np.arccosh(L)
        check_array_finite(D)
        return D
    elif space == 'poincare':
        n, d = X.shape
        L = np.zeros((n, n), dtype=float)
        for i in range(0, n - 1):
            L[i, i + 1:] = max(1, 1 + 2 * np.sum((X[i] - X[i + 1:]) ** 2) / (
                        max(numeric_stability, 1 - np.sum(X[i] ** 2)) * max(numeric_stability,
                                                                            1 - np.sum(X[i + 1:] ** 2))))
        L += L.T
        D = np.arccosh(L)
        return D
    else:
        raise ValueError("Unknown metric %s. " % space)


def check_array_finite(a, name='The array'):
    if np.isfinite(a).all():
        return True
    else:
        print(f"{name} is not finite.")
        return False


def AdjaMatrix(Dist, method='knn', param=6): # param can be: epsilon in epsilon-neighbors; k in knn; edge matrix in graph
    data_number = Dist.shape[0]
    if method == 'knn':
        A = lil_matrix((data_number, data_number))
        for i in range(data_number):
            neighbor_idx = np.argpartition(Dist[i], param)[:param]
            A[i, neighbor_idx] = 1
            A[neighbor_idx, i] = 1
    elif method == 'pre-computed':
        return Dist
    return A


def SimiMatrix(Dist, A, method='heat kernel', t=10):
    if method == 'heat kernel':
        if t > 1e5:
            S = A
        else:
            S = A.toarray() * (np.exp(-Dist/t))
    return lil_matrix(S)


def GEP(S_X, S_Y, W, n_eig, miu=0.8):
    G = bmat([[miu * S_X, (1 - miu) * W], [(1 - miu) * W.transpose(), miu * S_Y]]).tocsr()
    L = csgraph.laplacian(G, normed=False).tocsr()
    D = L + G
    w, v = linalg.eigsh(L, k=n_eig, M=D, which='SA')     # smallest algebraic value
    return w, v


def add_leaf_edges(w, v, path_taxonomy, train_leaves, test_leaves,
                   train_leaves_id, test_leaves_id, train_aligned_leaves_id, eps=1e-10, k=5, leaf_edge_weight=0.1):

    if eps > 0:
        w_order = np.argwhere(w > eps)
        embed = v[:, w_order[:, 0]]
    else:
        embed = v[:, 1:]

    leaf_embed = embed[train_aligned_leaves_id + test_leaves_id]
    num_train = len(train_aligned_leaves_id)
    num_test = len(test_leaves_id)
    taxonomy = pandas.read_csv(path_taxonomy)
    for i in range(num_test):
        leaf_dist = np.linalg.norm(leaf_embed[num_train + i] - leaf_embed[0: num_train], axis=1)
        neighbor_idx = np.argpartition(leaf_dist, k)[:k]
        id1_item = id2leaves(test_leaves_id[i], test_leaves, test_leaves_id)
        for j in neighbor_idx:
            id2_item = id2leaves(train_aligned_leaves_id[j], train_leaves, train_leaves_id)
            df_1 = pandas.DataFrame([[id1_item, id2_item, leaf_edge_weight]],
                                    columns=['id1', 'id2', 'weight'])
            df_2 = pandas.DataFrame([[id2_item, id1_item, leaf_edge_weight]],
                                    columns=['id1', 'id2', 'weight'])
            taxonomy = taxonomy.append(df_1, ignore_index=True)
            taxonomy = taxonomy.append(df_2, ignore_index=True)
    return taxonomy


def id2leaves(idx, leaves, ids):
    true_id = ids.index(idx)
    return leaves[true_id]


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser(description='ManifoldAlignment')
    parser.add_argument('-n', '--num_neighbor', type=int, default=6)
    parser.add_argument('-t', '--t_heat_kernel', type=int, default=10000000000000)
    parser.add_argument('-w', '--weight_of_simi', type=float, default=0.5)
    parser.add_argument('-e', '--num_eig', type=int, default=15)
    parser.add_argument('-out', '--output_dir', type=str, required=True)
    parser.add_argument('-dist', '--distance_dir', type=str, required=True)
    parser.add_argument('-pt', '--path_taxonomy', default='./data/YAGOwordnet/taxonomy/0.5/')
    opts = parser.parse_args(sys.argv[1:])

    path_taxonomy = opts.path_taxonomy
    checkpoint_dir = opts.distance_dir
    path_hyperembed = os.path.join(opts.distance_dir, 'checkpoint.bin')
    path_train_leaves = os.path.join(path_taxonomy, 'train_leaves2id.txt')
    path_test_leaves = os.path.join(path_taxonomy, 'test_leaves2id.txt')

    num_neigbor = opts.num_neighbor     # k in knn (adja matrix of distance)
    t_heat_kernel = opts.t_heat_kernel     # t in heat kernel (simi matrix)
    weight_of_simi = opts.weight_of_simi     # miu in the joint matrix G
    num_eig = opts.num_eig     # n_eig in GEP

    y_path_dist = opts.distance_dir

    train_leaves, train_leaves_id = load_nodes2id(path_train_leaves)
    test_leaves, test_leaves_id = load_nodes2id(path_test_leaves)

    # Get train leaves hyperbolic embeddings
    Y = load_hy_embeddings(path_hyperembed, train_leaves)
    Y_dist = pairwise_distance(Y, 'hyperboloid')

    # Get knowledge graph embeddings
    X_dist = np.load(os.path.join(path_dist, 'X_dist.npy'))

    # Correspondence matrix W:n*m, W_(i,j)=I(xi<-->yj)
    m = Y_dist.shape[0]  #common entities, dimension of hyperbolic embeddings
    n = X_dist.shape[0]  #knowledge graph entities, dimension of Euclidean embeddings

    W = lil_matrix((n, m))
    train_aligned_leaves = [i for i in range(m) if train_leaves_id[i] is not np.nan]
    train_aligned_leaves_id = [i for i in train_leaves_id if i is not np.nan]
    W[train_aligned_leaves_id, train_aligned_leaves] = 1   # W: lil matrix

    A_X = AdjaMatrix(X_dist, param=num_neigbor)   # lil matrix
    S_X = SimiMatrix(X_dist, A_X, t=t_heat_kernel)   # lil matrix

    A_Y = AdjaMatrix(Y_dist, param=num_neigbor)   # lil matrix
    S_Y = SimiMatrix(Y_dist, A_Y, t=t_heat_kernel)   # lil matrix

    w, v = GEP(A_X, A_Y, W, n_eig=num_eig, miu=weight_of_simi)

    output_file = opts.output_dir
    np.savez(output_file, w=w, v=v)
