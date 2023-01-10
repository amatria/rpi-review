#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

from argparse import ArgumentParser

# def load_pro_dataset(opts, model=False):
#     path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
#     sequences = read_dataset(os.path.join(path, 'pro.fa'))
#     return sequences

# def load_rna_dataset(opts, model=False):
#     path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
#     sequences = read_dataset(os.path.join(path, 'rna.fa'))
#     return sequences

def load_pair_dataset(opts, model=False):
    path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH

    dataset = pd.read_table(os.path.join(path, 'pairs.txt'), header=None, names=['protein', 'RNA', 'label'])

    pos_pairs = dataset[dataset['label'] == 1]
    pos_proteins = pos_pairs['protein'].unique().tolist()
    pos_rnas = pos_pairs['RNA'].unique().tolist()

    pos_index = []
    for _, row in pos_pairs.iterrows():
        i = pos_rnas.index(row['RNA'])
        j = pos_proteins.index(row['protein'])
        pos_index.append([i, j])

    positives = pd.DataFrame(pos_index, columns=['RNA', 'protein'])
    positives['label'] = 1

    npi_pos = np.zeros((len(pos_rnas), len(pos_proteins)))
    npi_pos[positives.values[:, 0], positives.values[:, 1]] = 1
    npi_pos = pd.DataFrame(npi_pos)

    neg_pairs = dataset[dataset['label'] == 0]
    neg_proteins = neg_pairs['protein'].unique().tolist()
    neg_rnas = neg_pairs['RNA'].unique().tolist()

    neg_index = []
    for _, row in neg_pairs.iterrows():
        i = neg_rnas.index(row['RNA'])
        j = neg_proteins.index(row['protein'])
        neg_index.append([i, j])

    negatives = pd.DataFrame(neg_index, columns=['RNA', 'protein'])
    negatives['label'] = 0

    npi_neg = np.zeros((len(neg_rnas), len(neg_proteins)))
    npi_neg[negatives.values[:, 0], negatives.values[:, 1]] = 1
    npi_neg = pd.DataFrame(npi_neg)

    return npi_pos, npi_neg

def kfold_cross_validation(opts):
    npi_pos, npi_neg = load_pair_dataset(opts)

class Options:
    MODEL = None
    DATASET = None

    DATASET_BASE_PATH = None
    MODEL_BASE_PATH = None
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    KFOLD = False
    N_FOLDS = 5
    RANDOM_STATE = 42

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='NPI-RGCNAE',
        description='NPI-RGCNAE: R-GCN Graph Autoencoder for ncRNA-protein interaction prediction',
    )
    parser.add_argument('-te', '--test', type=str, default='rpi369', help='test dataset')
    parser.add_argument('-tr', '--train', type=str, default='rpi369', help='train dataset')

    args = parser.parse_args()

    opts = Options()
    opts.DATASET = args.test
    opts.MODEL = args.train

    opts.DATASET_BASE_PATH = os.path.join(opts.PARENT_DIR, 'data', opts.DATASET)
    opts.MODEL_BASE_PATH = os.path.join(opts.PARENT_DIR, 'data', opts.MODEL)

    if opts.DATASET == opts.MODEL:
        kfold_cross_validation(opts)
    else:
        pass
        #train_and_validate_model(opts)