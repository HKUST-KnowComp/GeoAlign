#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from naive_evaluate import naive_eval

th.manual_seed(42)
np.random.seed(42)


MANIFOLDS = {
    'lorentz': LorentzManifold,
    'euclidean': EuclideanManifold,
    'poincare': PoincareManifold
}


def async_eval(adj_list, q: mp.Queue, logQ: mp.Queue, opt, leaves):
    logger = logging.getLogger("Eval")
    while True:
        temp = q.get()
        if temp is None:
            logger.info("Eval thread exit.")
            return

        if not q.empty():
            logger.info(f"Queue Not Empty.{temp}")
            continue
        logger.info(f"Eval message got.{temp}")
        lmsg, pth = eval(adj_list, opt, leaves, *temp)
        logQ.put((lmsg, pth))


def eval_in_product(adj, opt, epoch, elapsed, loss, pth, best=None, complex=False):
    chkpnt = th.load(pth, map_location='cpu')
    if not complex:
        model = build_model(opt, chkpnt['embeddings'].size(0))
        model.load_state_dict(chkpnt['model'])

        sqnorms = model.manifold.norm(model.lt)
        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'sqnorm_min': sqnorms.min().item(),
            'sqnorm_avg': sqnorms.mean().item(),
            'sqnorm_max': sqnorms.max().item(),
        }
        distortion, mAP = eval_reconstruction_in_product(adj, model)
        lmsg['distortion'] = distortion
        lmsg['mAP'] = mAP
    else:
        model = build_model(opt, chkpnt['embeddings'][0].size(0))
        model.load_state_dict(chkpnt['model'])

        embeddings = chkpnt['embeddings']

        sqnorms = model.manifold.norm(embeddings[0], embeddings[1])
        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'sqnorm_min': sqnorms.min().item(),
            'sqnorm_avg': sqnorms.mean().item(),
            'sqnorm_max': sqnorms.max().item(),
        }
        distortion, mAP = eval_reconstruction_in_product(adj, model, embeddings=embeddings, complex=True)
        lmsg['distortion'] = distortion
        lmsg['mAP'] = mAP
    return lmsg, pth


def eval(adj_list, opt, leaves, epoch, elapsed, loss, pth):
    manifold = MANIFOLDS[opt.manifold]()
    logger = logging.getLogger("Eval")
    chkpnt = th.load(pth, map_location='cpu')
    lt = chkpnt['embeddings']
    logger.info("Evaluating")
    sqnorms = manifold.pnorm(lt)
    logger.info("Eval: manifold.pnorm done")
    adj_list_names = ['leaf_link', 'leaf_hierarchy', 'original_taxonomy']
    lmsg = {
        'epoch': epoch,
        'elapsed': elapsed,
        'loss': loss,
        'sqnorm_min': sqnorms.min().item(),
        'sqnorm_avg': sqnorms.mean().item(),
        'sqnorm_max': sqnorms.max().item(),
    }
    hits_conf = [1, 3, 10, 1000]
    if opt.eval:
        for i, adj in enumerate(adj_list):
            meanrank, maprank, mrr = None, None, None
            if adj is not None:
                meanrank, maprank, mrr, hits = eval_reconstruction(
                    adj, lt, manifold.distance, leaves, workers=1,
                    progress=True, filter=opt.filter, hits_conf=hits_conf)
                for j, hj in enumerate(hits_conf):
                    lmsg[f'{adj_list_names[i]}_hits{hj}'] = hits[j]
            lmsg[f'{adj_list_names[i]}_mean_rank'] = meanrank
            lmsg[f'{adj_list_names[i]}_map_rank'] = maprank
            lmsg[f'{adj_list_names[i]}_mrr'] = mrr
    logger.info("Evaluation Done")
    return lmsg, pth


# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-checkpoint', default='/tmp/hype_embeddings.pth',
                        help='Where to store the model checkpoint')
    parser.add_argument('-alignset', type=str, default=None, help='The aligned taxonomy path')
    parser.add_argument('-trainset', type=str, default=None, help='The training taxonomy path')
    parser.add_argument('-testset', type=str, default=None, help='The test leaf links path')
    parser.add_argument('-testhier', type=str, default=None, help='The test hierarchy path')
    parser.add_argument('-testtax', type=str, default=None, help='The original taxonomy path')
    parser.add_argument('-leaves', type=str, required=True, help='Leaf nodes identifier')
    parser.add_argument('-dim', type=int, default=20, help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='lorentz',
                        choices=MANIFOLDS.keys(), help='Embedding manifold')
    parser.add_argument('-lr', type=float, default=1000, help='Learning rate')
    parser.add_argument('-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=12800, help='Batchsize')
    parser.add_argument('-negs', type=int, default=50, help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20, help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75, help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8, help='Number of data loading processes')
    parser.add_argument('-eval_each', type=int, default=1, help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False, help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False, help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int, help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-sym', action='store_true', default=False, help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='500000', action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true', help='Use sparse gradients for embedding table')
    parser.add_argument('-filter', action='store_true', default=False, help='filter the leaf nodes while evaluation')
    parser.add_argument('-align', action='store_true', default=False, help='whether to use the aligned taxonomy to train')
    parser.add_argument('-eval', action='store_true', default=False, help='whether to evaluate')
    parser.add_argument('-naive_eval', action='store_true', default=False,
                        help='naive evaluate (if the aligned taxonomy contains only one neighbor)')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-train_threads', type=int, default=1, help='Number of threads to use in training')
    opt = parser.parse_args()

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)-15s|%(process)d:%(thread)d|%(name)s-%(filename)s:%(lineno)d %(message)s',
        stream=sys.stdout)
    log = logging.getLogger('lorentz')

    if opt.gpu >= 0 and opt.train_threads > 1:
        opt.gpu = -1
        log.warning(f'Specified hogwild training with GPU, defaulting to CPU...')


    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    # set device
    device = th.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 else 'cpu')

    # select manifold to optimize on
    manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
    opt.dim = manifold.dim(opt.dim)

    if opt.align:
        opt.dset = opt.alignset
    else:
        opt.dset = opt.trainset

    if 'csv' in opt.dset:
        log.info('Using edge list dataloader')
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        model, data, model_name, conf = initialize(
            manifold, opt, idx, objects, weights, sparse=opt.sparse
        )
    else:
        log.info('Using adjacency matrix dataloader')
        dset = load_adjacency_matrix(opt.dset, 'hdf5')
        log.info('Setting up dataset...')
        data = AdjacencyDataset(dset, opt.negs, opt.batchsize, opt.ndproc,
            opt.burnin > 0, sample_dampening=opt.dampening)
        model = Embedding(data.N, opt.dim, manifold, sparse=opt.sparse)
        objects = dset['objects']
    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    if opt.alignset is None and opt.trainset is None:
        raise ValueError(f'No train dataset input!')

    # train dataset loader and create adj for train data
    if opt.align:
        if opt.trainset is not None:
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
        else:
            if opt.naive_eval:
                raise ValueError(f'No train taxonomy input!')

    # test dataset loader and create adj for test data
    if opt.testset is None and opt.testhier is None and opt.testtax is None:
        raise ValueError(f'No test dataset input!')

    adj_test = adj_testhier = adj_testtax = None
    if opt.testset is not None:
        log.info('Test leaf links: using edge list dataloader')
        adj_test = {}
        test_taxonomy = pandas.read_csv(opt.testset)
        for i, row in test_taxonomy.iterrows():
            x_idx = objects.index(row['id1'])
            y_idx = objects.index(row['id2'])
            if x_idx in adj_test:
                adj_test[x_idx].add(y_idx)
            else:
                adj_test[x_idx] = {y_idx}
    else:
        if opt.naive_eval:
            raise ValueError(f'No test leaf links input!')
        else:
            log.warning(f'No test leaf links input.')
    if opt.testhier is not None:
        log.info('Test hierarchy: using edge list dataloader')
        adj_testhier = {}
        test_taxonomy = pandas.read_csv(opt.testhier)
        for i, row in test_taxonomy.iterrows():
            x_idx = objects.index(row['id1'])
            y_idx = objects.index(row['id2'])
            if x_idx in adj_testhier:
                adj_testhier[x_idx].add(y_idx)
            else:
                adj_testhier[x_idx] = {y_idx}
    else:
        log.warning(f'No test hierarchy input.')
    if opt.testtax is not None:
        log.info('Original taxonomy: using edge list dataloader')
        adj_testtax = {}
        test_taxonomy = pandas.read_csv(opt.testtax)
        for i, row in test_taxonomy.iterrows():
            x_idx = objects.index(row['id1'])
            y_idx = objects.index(row['id2'])
            if x_idx in adj_testtax:
                adj_testtax[x_idx].add(y_idx)
            else:
                adj_testtax[x_idx] = {y_idx}
    else:
        log.warning(f'No original taxonomy input.')

    # process leaf nodes
    if opt.filter:
        with open(opt.leaves, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        leaf_nodes = set()
        for line in lines:
            if line.strip() in objects:
                leaf_nodes.add(objects.index(line.strip()))
        hier_nodes = set(range(len(objects))) - leaf_nodes
    else:
        leaf_nodes = set()
        hier_nodes = set()

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lr_type == 'scale':
        opt.lr = opt.lr * opt.batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(manifold), lr=opt.lr)

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        opt.checkpoint,
        include_in_all={'conf' : vars(opt), 'objects' : objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']

    controlQ, logQ = mp.Queue(), mp.Queue()
    adj_list = [adj_test, adj_testhier, adj_testtax]

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        log.debug('control: pre normalize')
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        log.debug('control: got lt')
        manifold.normalize(lt)
        log.debug('control: lt normalize done')
        checkpoint.path = f'{opt.checkpoint}.{epoch}'
        log.info(f"Saving {checkpoint.path}")
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt,
            'epoch': epoch,
            'manifold': opt.manifold,
        })
        log.info(f"Saved {checkpoint.path}")
        eval_msg = (epoch, elapsed, loss, checkpoint.path)
        lmsg, pth = eval(adj_list, opt, leaf_nodes, epoch, elapsed, loss, checkpoint.path)
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')

    control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)
    if opt.train_threads > 1:
        threads = []
        model = model.share_memory()
        args = (device, model, data, optimizer, opt, log)
        kwargs = {'ctrl': None, 'progress': not opt.quiet}
        for i in range(opt.train_threads):
            kwargs['rank'] = i
            threads.append(mp.Process(target=train.train, args=args, kwargs=kwargs))
            threads[-1].start()
        [t.join() for t in threads]
    else:
        train.train(device, model, data, optimizer, opt, log, ctrl=control,
            progress=not opt.quiet)
    control(model, opt.epochs, 0, 0)

    while not logQ.empty():
        lmsg, pth = logQ.get()
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')

    if opt.naive_eval:
        naive_map = naive_eval(adj_train, adj_test, adj, hier_nodes, objects)


if __name__ == '__main__':
    main()
