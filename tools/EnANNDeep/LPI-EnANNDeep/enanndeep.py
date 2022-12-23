#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

import numpy as np

import utils.aknn_alg as aknn_alg

from itertools import chain

from argparse import ArgumentParser

from deepforest import CascadeForestClassifier

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve

from utils.DRPLPI_F import rna_feature_extract, protein_feature_extract, get_rna_value

def read_dataset(file_path):
    ret = {}
    with open(file_path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:]
                ret[name] = ''
            else:
                ret[name] += line
    return ret

def load_pro_dataset(opts, model=False):
    path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
    sequences = read_dataset(os.path.join(path, 'pro.fa'))
    return sequences

def load_rna_dataset(opts, model=False):
    path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
    sequences = read_dataset(os.path.join(path, 'rna.fa'))
    return sequences

def load_pair_dataset(opts, model=False):
    path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
    pairs = {}
    with open(os.path.join(path, 'pairs.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            pairs[(p, r)] = int(label)
    return pairs

def encode_sequences(pairs, p_seqs, r_seqs, opts, model=False):
    rna_value = get_rna_value(opts, model)
    tmp = []
    for p_name, r_name in pairs.keys():
        try:
            p_seq = p_seqs[p_name]
            r_seq = r_seqs[r_name]
        except KeyError:
            print('KeyError: skip {}-{}'.format(p_name, r_name))
            continue

        pro_feature = protein_feature_extract(p_seq, opts, model)
        rna_feature = rna_feature_extract(r_seq, rna_value)

        tmp_feature = list(rna_feature) + list(pro_feature)
        tmp.append([tmp_feature, pairs[(p_name, r_name)]])

    features = [x[0] for x in tmp]
    features = StandardScaler().fit_transform(features)

    return [list(x) + [y] for x, y in zip(features, [x[1] for x in tmp])]

def kfold_cross_validation(opts):
    pairs = load_pair_dataset(opts)
    p_sequences = load_pro_dataset(opts)
    r_sequences = load_rna_dataset(opts)

    encoded_pairs = np.array(encode_sequences(pairs, p_sequences, r_sequences, opts), dtype=object)

    features = []
    labels = []
    for i in range(encoded_pairs.shape[0]):
        features.append(encoded_pairs[i][0:-1])
        labels.append(encoded_pairs[i][-1])
    X = np.array(features).astype('float64')
    y = np.array(labels).astype('<U1')

    # stratified k-fold cross validation
    sum_acc = 0
    sum_pre = 0
    sum_recall = 0
    sum_f1 = 0
    sum_AUC = 0
    skf = StratifiedKFold(n_splits=opts.N_FOLDS, random_state=opts.RANDOM_STATE, shuffle=True)
    for _, (train_index, test_index) in enumerate(skf.split(np.zeros(len(y)), y)):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[test_index], y[test_index]

        nbrs_list = aknn_alg.calc_nbrs_exact(X_val, k=2000)
        aknn_predictions = aknn_alg.predict_nn_rule(nbrs_list, y_val)
        aknn_prediction = list(map(int,aknn_predictions[0]))
        aknn_prediction_prob = list(map(float,aknn_predictions[2]))

        y_train = np.array(list(map(int, y_train)))
        y_val = np.array(list(map(int, y_val)))

        model = Sequential()
        model.add(Dense(554, activation='elu',input_dim = 554))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0)

        score = model.predict(X_val)
        score = list(chain.from_iterable(score))

        forest = CascadeForestClassifier(random_state=1, verbose=0)
        forest.fit(X_train, y_train)
        y_pred_f = forest.predict(X_val)
        allResult = forest.predict_proba(X_val)
        probaResult = np.arange(0, dtype=float)
        for iRes in allResult:
            probaResult = np.append(probaResult, iRes[1])

        dnn_pred = []
        score = np.array(score)
        for k in score:
            if k > 0.5:
                dnn_pred.append(1)
            else:
                dnn_pred.append(0)
        dnn_pred = np.array(dnn_pred)
        aknn_prediction = np.array(aknn_prediction)
        aknn_prediction_prob = np.array(aknn_prediction_prob)
        mix_res = dnn_pred + aknn_prediction + y_pred_f
        mix_prob = score / 3 + aknn_prediction_prob / 3 + probaResult / 3
        pred = []
        for j in mix_res:
            if j > 1:
                pred.append(1)
            else:
                pred.append(0)

        sum_acc += accuracy_score(y_val, pred)
        sum_pre += precision_score(y_val, pred)
        sum_recall += recall_score(y_val, pred)
        sum_f1 += f1_score(y_val, pred)
        fpr, tpr, _ = roc_curve(y_val, mix_prob)
        sum_AUC += auc(fpr,tpr)

    sum_acc /= opts.N_FOLDS
    sum_pre /= opts.N_FOLDS
    sum_recall /= opts.N_FOLDS
    sum_f1 /= opts.N_FOLDS
    sum_AUC /= opts.N_FOLDS
    print('>>Train:{} Test:{} (K-FOLD)\nacc: {:.4f}, pre: {:.4f}, recall: {:.4f}, f1: {:.4f}, auc: {:.4f}'.format(opts.MODEL, opts.DATASET, sum_acc, sum_pre, sum_recall, sum_f1, sum_AUC), file=sys.stderr)

class Options:
    MODEL = None
    DATASET = None

    DATASET_BASE_PATH = None
    MODEL_BASE_PATH = None
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

    KFOLD = False
    N_FOLDS = 5
    RANDOM_STATE = 42

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='EnANNDeep',
        description='EnANNDeep: An ensemble-based lncRNA-protein interaction prediction method with adaptive k-nearest neighbor and deep learning',
    )
    parser.add_argument('-te', '--test', type=str, default='rpi2241', help='test dataset')
    parser.add_argument('-tr', '--train', type=str, default='rpi2241', help='train dataset')

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
