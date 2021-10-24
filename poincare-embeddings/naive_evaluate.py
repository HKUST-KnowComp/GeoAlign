import torch as th
import numpy as np
import logging
import argparse
from hype.sn import Embedding, initialize
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype import train
from hype.graph import load_adjacency_matrix, load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
from hype.lorentz import LorentzManifold
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold
import sys
import json
import torch.multiprocessing as mp
import shutil
import pandas
from tqdm import tqdm
from sklearn.metrics import average_precision_score


th.manual_seed(42)
np.random.seed(42)


MANIFOLDS = {
    'lorentz': LorentzManifold,
    'euclidean': EuclideanManifold,
    'poincare': PoincareManifold
}


def naive_eval(adj_train, adj_test, adj, hier_nodes, objects_all):
    logger = logging.getLogger("Eval")
    objects = np.array(list(adj_test.keys()))
    ap_scores = iters = 0
    labels = np.empty(len(objects_all))
    label_score = np.empty(len(objects_all))
    for object in tqdm(objects):
        labels.fill(0)
        label_score.fill(0)
        neighbors = np.array(list(adj_test[object]))
        labels[neighbors] = 1
        object_links = list(adj[object])
        predict = []
        for item in object_links:
            predict += list(adj_train[item])
        predict = np.array(predict)
        label_score[predict] = 1
        labels_filter = np.array([labels[i] for i in range(len(labels)) if i in hier_nodes])
        label_score_filter = np.array([label_score[i] for i in range(len(label_score)) if i in hier_nodes])
        ap_scores += average_precision_score(labels_filter, label_score_filter)
        iters += 1
    map = float(ap_scores)/iters
    lmsg = {'naive_map_rank': map}
    logger.info(str(lmsg))
    return map


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description='Naive Evaluate Aligned Hyperbolic Embeddings')
    parser.add_argument('-alignset', type=str, required=True, help='The aligned taxonomy path')
    parser.add_argument('-trainset', type=str, required=True, help='The training taxonomy path')
    parser.add_argument('-testset', type=str, required=True, help='The test taxonomy path')
    parser.add_argument('-leaves', type=str, required=True,
                        help='Leaf nodes identifier')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-manifold', type=str, default='lorentz',
                        choices=MANIFOLDS.keys(), help='Embedding manifold')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000',
                        action=Unsettable, type=int)
    parser.add_argument('-dim', type=int, default=20,
                        help='Embedding dimension')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-batchsize', type=int, default=12800,
                        help='Batchsize')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')

    opt = parser.parse_args()

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)-15s %(message)s', stream=sys.stdout)
    log = logging.getLogger('lorentz')
    # logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')

    # select manifold to optimize on
    manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
    opt.dim = manifold.dim(opt.dim)

    log.info('Using edge list dataloader')
    idx, objects, weights = load_edge_list(opt.alignset, opt.sym)
    model, data, model_name, conf = initialize(manifold, opt, idx, objects, weights, sparse=opt.sparse)
    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    # train dataset loader and create adj for train data
    log.info('Train dataset: using edge list dataloader')
    adj_train = {}
    train_taxonomy = pandas.read_csv(opt.trainset)
    for i, row in train_taxonomy.iterrows():
        x_idx = objects.index(row['id1'])
        y_idx = objects.index(row['id2'])
        if x_idx in adj_train:
            adj_train[x_idx].add(y_idx)
        else:
            adj_train[x_idx] = {y_idx}

    # test dataset loader and create adj for test data
    log.info('Test dataset: using edge list dataloader')
    adj_test = {}
    test_taxonomy = pandas.read_csv(opt.testset)
    for i, row in test_taxonomy.iterrows():
        x_idx = objects.index(row['id1'])
        y_idx = objects.index(row['id2'])
        if x_idx in adj_test:
            adj_test[x_idx].add(y_idx)
        else:
            adj_test[x_idx] = {y_idx}

    with open(opt.leaves, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    leaf_nodes = set()
    for line in lines:
        leaf_nodes.add(objects.index(line.strip()))
    hier_nodes = set(range(len(objects))) - leaf_nodes

    map = naive_eval(adj_train, adj_test, adj, hier_nodes, objects)


if __name__ == '__main__':
    main()
