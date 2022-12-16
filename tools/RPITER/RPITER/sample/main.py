#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import math
import numpy as np

from argparse import ArgumentParser

from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, concatenate

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

from utils.callbacks import EarlyStopping
from utils.sequence_encoder import ProEncoder, RNAEncoder
from utils.stacked_auto_encoder import train_auto_encoder
from utils.basic_modules import conjoint_cnn, conjoint_sae

def sum_power(num, bottom, top):
    return sum(map(lambda x: num ** x, range(bottom, top + 1)))

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

def load_pro_dataset(opts):
    sequences = read_dataset(os.path.join(opts.DATASET_BASE_PATH, 'pro.fa'))
    structures = read_dataset(os.path.join(opts.DATASET_BASE_PATH, 'pro-struct.fa'))
    return sequences, structures

def load_rna_dataset(opts):
    sequences = read_dataset(os.path.join(opts.DATASET_BASE_PATH, 'rna.fa'))
    structures = read_dataset(os.path.join(opts.DATASET_BASE_PATH, 'rna-struct.fa'))
    return sequences, structures

def load_pair_dataset(opts):
    pairs = {}
    with open(os.path.join(opts.DATASET_BASE_PATH, 'pairs.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            pairs[(p, r)] = int(label)
    return pairs

def encode_sequences(pairs, p_seqs, p_structs, r_seqs, r_structs, p_enc, r_enc):
    ret = []
    for p_name, r_name in pairs.keys():
        p_seq = p_seqs[p_name]
        r_seq = r_seqs[r_name]
        p_struct = p_structs[p_name]
        r_struct = r_structs[r_name]

        try:
            p_conjoint = p_enc.encode_conjoint(p_seq)
            r_conjoint = r_enc.encode_conjoint(r_seq)
            p_conjoint_struct = p_enc.encode_conjoint_struct(p_seq, p_struct)
            r_conjoint_struct = r_enc.encode_conjoint_struct(r_seq, r_struct)
        except IndexError:
            print('encode_sequences: fatal IndexError with {}-{}'.format(p_name, r_name))
            exit(1)

        if any([type(x) == str for x in [p_conjoint, r_conjoint, p_conjoint_struct, r_conjoint_struct]]):
            print('Skip {}-{}'.format(p_name, r_name))
            continue

        ret.append([[p_conjoint, r_conjoint], [p_conjoint_struct, r_conjoint_struct], pairs[(p_name, r_name)]])
    return ret

def standarize(data):
    return StandardScaler().fit_transform(data)

def preprocess_data(encoded_pairs, opts):
    p_conjoint = np.array([x[0][0] for x in encoded_pairs])
    r_conjoint = np.array([x[0][1] for x in encoded_pairs])
    p_conjoint_struct = np.array([x[1][0] for x in encoded_pairs])
    r_conjoint_struct = np.array([x[1][1] for x in encoded_pairs])
    labels = np.array([x[2] for x in encoded_pairs])

    p_conjoint = standarize(p_conjoint)
    r_conjoint = standarize(r_conjoint)
    p_conjoint_struct = standarize(p_conjoint_struct)
    r_conjoint_struct = standarize(r_conjoint_struct)

    p_conjoint_cnn = np.array([list(map(lambda e: [e] * opts.VECTOR_REPETITION_CNN, x)) for x in p_conjoint])
    r_conjoint_cnn = np.array([list(map(lambda e: [e] * opts.VECTOR_REPETITION_CNN, x)) for x in r_conjoint])
    p_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * opts.VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct])
    r_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * opts.VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct])

    samples = [
        [p_conjoint, r_conjoint],
        [p_conjoint_struct, r_conjoint_struct],
        [p_conjoint_cnn, r_conjoint_cnn],
        [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
    ]

    return samples, labels

def split_data(X, y, all_values=False):
    X_indices = [i for i in range(len(y))]
    y_indices = [i for i in range(len(y))]

    if not all_values:
        X_train_indices, X_val_indices, y_train_indices, y_val_indices = train_test_split(X_indices, y_indices, test_size=0.2)
    else:
        X_train_indices, X_val_indices, y_train_indices, y_val_indices = X_indices, X_indices, y_indices, y_indices

    X_train = [
        [X[0][0][X_train_indices], X[0][1][X_train_indices]],
        [X[1][0][X_train_indices], X[1][1][X_train_indices]],
        [X[2][0][X_train_indices], X[2][1][X_train_indices]],
        [X[3][0][X_train_indices], X[3][1][X_train_indices]],
    ]
    y_train = y[y_train_indices]

    X_val = [
        [X[0][0][X_val_indices], X[0][1][X_val_indices]],
        [X[1][0][X_val_indices], X[1][1][X_val_indices]],
        [X[2][0][X_val_indices], X[2][1][X_val_indices]],
        [X[3][0][X_val_indices], X[3][1][X_val_indices]],
    ]
    y_val = y[y_val_indices]

    return X_train, y_train, X_val, y_val

def get_optimizer(opt_name, opts):
    if opt_name == 'sgd':
        return optimizers.SGD(learning_rate=opts.SGD_LEARNING_RATE, momentum=0.5)
    elif opt_name == 'adam':
        return optimizers.Adam(learning_rate=opts.ADAM_LEARNING_RATE)
    else:
        return opt_name

def get_autoencoders(X_train, X_test, opts):
    p_encoders, _, _, _ = train_auto_encoder(
        X_train=X_train[0],
        X_test=X_test[0],
        layers=[X_train[0].shape[1], 256, 128, 64],
        batch_size=opts.BATCH_SIZE
    )
    r_encoders, _, _, _ = train_auto_encoder(
        X_train=X_train[1],
        X_test=X_test[1],
        layers=[X_train[1].shape[1], 256, 128, 64],
        batch_size=opts.BATCH_SIZE
    )
    return p_encoders, r_encoders

def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])

    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])

    P = TP + FN
    N = TN + FP

    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0

    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0

    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp

    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)

    return Acc, Sn, Sp, Pre, MCC, AUC

def train_model(opts):
    pairs = load_pair_dataset(opts)

    p_sequences, p_structures = load_pro_dataset(opts)
    r_sequences, r_structures = load_rna_dataset(opts)

    p_encoder = ProEncoder(opts.P_WINDOW_UPLIMIT, opts.P_STRUCT_WINDOW_UPLIMIT, opts.CODING_FREQUENCY, opts.VECTOR_REPETITION_CNN)
    r_encoder = RNAEncoder(opts.R_WINDOW_UPLIMIT, opts.R_STRUCT_WINDOW_UPLIMIT, opts.CODING_FREQUENCY, opts.VECTOR_REPETITION_CNN)

    encoded_pairs = encode_sequences(pairs, p_sequences, p_structures, r_sequences, r_structures, p_encoder, r_encoder)

    X, y = preprocess_data(encoded_pairs, opts)
    #X_train, y_train_mono, X_val, y_val_mono = split_data(X, y)
    X_train, y_train_mono, X_val, y_val_mono = split_data(X, y, all_values=True)

    y_train = to_categorical(y_train_mono, num_classes=2, dtype='int32')
    y_val = to_categorical(y_val_mono, num_classes=2, dtype='int32')

    # conjoint-cnn model
    model_conjoint_cnn = conjoint_cnn(opts.P_CODING_LENGTH, opts.R_CODING_LENGTH, opts.VECTOR_REPETITION_CNN)
    callback = EarlyStopping(monitor=opts.MONITOR, min_delta=opts.MIN_DELTA, patience=opts.PATIENCES[0], mode='auto', restore_best_weights=True)

    model_conjoint_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.FIRST_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_cnn.fit(x=X_train[2], y=y_train, epochs=opts.FIRST_TRAIN_EPOCHS[0], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0)
    model_conjoint_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.SECOND_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_cnn.fit(x=X_train[2], y=y_train, epochs=opts.SECOND_TRAIN_EPOCHS[0], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0, callbacks=[callback])

    # conjoint-struct-cnn model
    model_conjoint_struct_cnn = conjoint_cnn(opts.P_STRUCT_CODING_LENGTH, opts.R_STRUCT_CODING_LENGTH, opts.VECTOR_REPETITION_CNN)
    callback = EarlyStopping(monitor=opts.MONITOR, min_delta=opts.MIN_DELTA, patience=opts.PATIENCES[1], mode='auto', restore_best_weights=True)

    model_conjoint_struct_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.FIRST_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_struct_cnn.fit(x=X_train[3], y=y_train, epochs=opts.FIRST_TRAIN_EPOCHS[1], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0)
    model_conjoint_struct_cnn.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.SECOND_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_struct_cnn.fit(x=X_train[3], y=y_train, epochs=opts.SECOND_TRAIN_EPOCHS[1], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0, callbacks=[callback])

    # conjoint-sae model
    p_encoders, r_encoders = get_autoencoders(X_train[0], X_val[0], opts)
    model_conjoint_sae = conjoint_sae(p_encoders, r_encoders, opts.P_CODING_LENGTH, opts.R_CODING_LENGTH)
    callback = EarlyStopping(monitor=opts.MONITOR, min_delta=opts.MIN_DELTA, patience=opts.PATIENCES[2], mode='auto', restore_best_weights=True)

    model_conjoint_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.FIRST_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_sae.fit(x=X_train[0], y=y_train, epochs=opts.FIRST_TRAIN_EPOCHS[2], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0)
    model_conjoint_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.SECOND_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_sae.fit(x=X_train[0], y=y_train, epochs=opts.SECOND_TRAIN_EPOCHS[2], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0, callbacks=[callback])

    # conjoint-struct-sae model
    p_encoders, r_encoders = get_autoencoders(X_train[1], X_val[1], opts)
    model_conjoint_struct_sae = conjoint_sae(p_encoders, r_encoders, opts.P_STRUCT_CODING_LENGTH, opts.R_STRUCT_CODING_LENGTH)
    callback = EarlyStopping(monitor=opts.MONITOR, min_delta=opts.MIN_DELTA, patience=opts.PATIENCES[3], mode='auto', restore_best_weights=True)

    model_conjoint_struct_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.FIRST_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_struct_sae.fit(x=X_train[1], y=y_train, epochs=opts.FIRST_TRAIN_EPOCHS[3], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0)
    model_conjoint_struct_sae.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.SECOND_OPTIMIZER, opts), metrics=['accuracy'])
    model_conjoint_struct_sae.fit(x=X_train[1], y=y_train, epochs=opts.SECOND_TRAIN_EPOCHS[3], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0, callbacks=[callback])

    # ensemble model
    ensemble_inputs = concatenate([
        model_conjoint_sae.output,
        model_conjoint_struct_sae.output,
        model_conjoint_cnn.output,
        model_conjoint_struct_cnn.output
    ])
    ensemble_inputs = Dropout(0.25)(ensemble_inputs)

    ensemble = Dense(16, kernel_initializer='random_uniform', activation='relu')(ensemble_inputs)
    ensemble = BatchNormalization()(ensemble)
    ensemble = Dense(8, kernel_initializer='random_uniform', activation='relu')(ensemble)
    ensemble = BatchNormalization()(ensemble)

    ensemble_outputs = Dense(2, activation='softmax')(ensemble)

    model_ensemble = Model(
        inputs=model_conjoint_sae.input + model_conjoint_struct_sae.input + model_conjoint_cnn.input + model_conjoint_struct_cnn.input,
        outputs=ensemble_outputs
    )
    callback = EarlyStopping(monitor=opts.MONITOR, min_delta=opts.MIN_DELTA, patience=opts.PATIENCES[4], mode='auto', restore_best_weights=True)

    model_ensemble.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.FIRST_OPTIMIZER, opts), metrics=['accuracy'])
    model_ensemble.fit(x=X_train, y=y_train, epochs=opts.FIRST_TRAIN_EPOCHS[4], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0)
    model_ensemble.compile(loss='categorical_crossentropy', optimizer=get_optimizer(opts.SECOND_OPTIMIZER, opts), metrics=['accuracy'])
    model_ensemble.fit(x=X_train, y=y_train, epochs=opts.SECOND_TRAIN_EPOCHS[4], batch_size=opts.BATCH_SIZE, shuffle=opts.SHUFFLE, verbose=0, callbacks=[callback])

    # evaluate
    # y_pred = model_ensemble.predict(X_val, verbose=0)
    # acc, sn, sp, pre, mcc, auc = calc_metrics(y_val[:, 1], y_pred[:, 1])
    # print('acc: {:.4f}, sn: {:.4f}, sp: {:.4f}, pre: {:.4f}, mcc: {:.4f}, auc: {:.4f}'.format(acc, sn, sp, pre, mcc, auc))

    model_ensemble.save(os.path.join(opts.DATASET_BASE_PATH, '{}.rpiter.model'.format(opts.DATASET)))

def run_model(opts):
    pairs = load_pair_dataset(opts)

    p_sequences, p_structures = load_pro_dataset(opts)
    r_sequences, r_structures = load_rna_dataset(opts)

    p_encoder = ProEncoder(opts.P_WINDOW_UPLIMIT, opts.P_STRUCT_WINDOW_UPLIMIT, opts.CODING_FREQUENCY, opts.VECTOR_REPETITION_CNN)
    r_encoder = RNAEncoder(opts.R_WINDOW_UPLIMIT, opts.R_STRUCT_WINDOW_UPLIMIT, opts.CODING_FREQUENCY, opts.VECTOR_REPETITION_CNN)

    encoded_pairs = encode_sequences(pairs, p_sequences, p_structures, r_sequences, r_structures, p_encoder, r_encoder)

    X, y = preprocess_data(encoded_pairs, opts)
    X_train, y_train_mono, X_val, y_val_mono = split_data(X, y, all_values=True)

    y_train = to_categorical(y_train_mono, num_classes=2, dtype='int32')
    y_val = to_categorical(y_val_mono, num_classes=2, dtype='int32')

    model_ensemble = load_model(opts.MODEL)

    y_pred = model_ensemble.predict(X_val, verbose=0)
    acc, sn, sp, pre, mcc, auc = calc_metrics(y_val[:, 1], y_pred[:, 1])
    print('acc: {:.4f}, sn: {:.4f}, sp: {:.4f}, pre: {:.4f}, mcc: {:.4f}, auc: {:.4f}'.format(acc, sn, sp, pre, mcc, auc), file=sys.stderr)

class Options:
    MODEL = None
    DATASET = None

    DATASET_BASE_PATH = None
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    P_WINDOW_UPLIMIT = 3
    R_WINDOW_UPLIMIT = 4
    P_STRUCT_WINDOW_UPLIMIT = 3
    R_STRUCT_WINDOW_UPLIMIT = 4

    P_CODING_LENGTH = sum_power(7, 1, P_WINDOW_UPLIMIT)
    R_CODING_LENGTH = sum_power(4, 1, R_WINDOW_UPLIMIT)
    P_STRUCT_CODING_LENGTH = P_CODING_LENGTH + sum_power(3, 1, P_STRUCT_WINDOW_UPLIMIT)
    R_STRUCT_CODING_LENGTH = R_CODING_LENGTH + sum_power(2, 1, R_STRUCT_WINDOW_UPLIMIT)

    CODING_FREQUENCY = True
    VECTOR_REPETITION_CNN = 1

    FIRST_OPTIMIZER = 'adam'
    SECOND_OPTIMIZER = 'sgd'
    SGD_LEARNING_RATE = 0.01
    ADAM_LEARNING_RATE = 0.001
    PATIENCES = [10, 10, 10, 10, 10]

    SHUFFLE = True
    MONITOR = 'acc'
    MIN_DELTA = 0.0
    BATCH_SIZE = 150
    FIRST_TRAIN_EPOCHS = [35, 35, 35, 35, 15]
    SECOND_TRAIN_EPOCHS = [10, 10, 10, 10, 10]

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='RPITER',
        description='RPITER: A deep learning model for predicting RNA-protein interactions',
    )
    parser.add_argument('-d', '--dataset', type=str, default='inaki', help='dataset name')
    parser.add_argument('-m', '--model', type=str, help='path to a RPITER model')

    args = parser.parse_args()

    opts = Options()

    opts.MODEL = args.model
    opts.DATASET = args.dataset

    opts.DATASET_BASE_PATH = os.path.join(opts.PARENT_DIR, 'data', opts.DATASET)

    if not args.model:
        train_model(opts)
    else:
        run_model(opts)
