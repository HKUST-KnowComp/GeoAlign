#!/usr/bin/env python3
import os
import sys

import numpy as np

from ManifoldAlignment import add_leaf_edges, load_leaves2id_from_dir


def main(num_neighbor_in_leaf_nodes, weight_leaf_edges, eps,
         gep_file_path, path_taxonomy,
         aligned_path):

    npzfile = np.load(gep_file_path)
    w, v = npzfile['w'], npzfile['v']
    num_neighbor_in_leaf_nodes=int(num_neighbor_in_leaf_nodes)
    eps = float(eps)
    weight_leaf_edges=float(weight_leaf_edges)

    train_leaves, train_leaves_id, test_leaves, test_leaves_id = load_leaves2id_from_dir(path_taxonomy)
    train_aligned_leaves_id = [i for i in train_leaves_id if i is not np.nan]
    path_taxonomy = os.path.join(path_taxonomy, 'train_taxonomy.csv')

    taxonomy_aligned = add_leaf_edges(
        w, v, path_taxonomy, train_leaves, test_leaves,
        train_leaves_id, test_leaves_id, train_aligned_leaves_id,
        k=num_neighbor_in_leaf_nodes, eps=eps)
    # Aligned path
    if not os.path.exists(aligned_path):
        os.mkdir(aligned_path)
    taxonomy_aligned.to_csv(os.path.join(aligned_path, 'aligned_taxonomy.csv'), index=False)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(*args)
