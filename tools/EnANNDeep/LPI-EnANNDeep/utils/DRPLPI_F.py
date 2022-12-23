import os
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

ALPHABET = 'ACGU'

def get_tris():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = int(n / base)
        ch1 = chars[n % base]
        n = int(n / base)
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def get_tri_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    count = 0
    for val in tris:
        count += 1 
        num = seq.count(val)
        tri_feature.append(float(num) / seq_len)
    return tri_feature

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[n % base]
        n = n / base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def get_3_protein_struct_trids():
    nucle_com = []
    chars = ['H', 'E', 'C']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = int(n / base)
        ch1 = chars[n % base]
        n = int(n / base)
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def translate_sequence(seq, TranslationDict):
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    return TRANS_seq

def TransDict_from_list(groups):
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)
        for c in g_members:
            result[c] = str(tar_list[index])
        index = index + 1
    return result

ssec_kmer = []
ssec_kmer_name = ''

def SSEC(opts, model=False):
    global ssec_kmer
    global ssec_kmer_name

    path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
    if ssec_kmer == [] or ssec_kmer_name != str(path):
        ssec_kmer = []
        ssec_kmer_name = str(path)
        protein_seq_dict = {}
        protein_index = 1
        with open(os.path.join(path, 'pro.fa'), 'r') as fp:
            for line in fp:
                if line[0] == '>':
                    continue
                else:
                    seq = line[:-1]
                    protein_seq_dict[protein_index] = seq
                    protein_index = protein_index + 1
        groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
        group_dict = TransDict_from_list(groups)
        protein_tris = get_3_protein_struct_trids()
        for i in protein_seq_dict:
            protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
            protein_tri_fea = get_tri_nucleotide_composition(protein_tris, protein_seq)
            ssec_kmer.append(protein_tri_fea)
            protein_index = protein_index + 1
    return np.array(ssec_kmer[0])

def kmer(kmerid, k):
    kmer = ''
    nts = ['A', 'C', 'G', 'T']
    for _ in range(k):
        kmer = nts[(kmerid % 4)] + kmer
        kmerid = int(kmerid / 4)
    return kmer

def InsertgappedVect(seq, g = 8):
    feature = []
    m = list(itertools.product(ALPHABET, repeat = 2))
    for i in range(1, g + 1, 1):
        V = kmer(len(seq), i + 2)
        for gGap in m:
            count = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    count += 1
            feature.append(count)
    return feature

def reverse_complement_features(seq):
    reverse_complements_2 = ['AA','AC','AG','CA','CC','GA','AT','CG','GC','TA']
    L2 = len(seq) - 2 + 1
    feature = []
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_6 = 0
    count_7 = 0
    count_8 = 0
    count_9 = 0
    count_10 = 0

    for i in range(0, len(seq) - 2 + 1):
        pattern = []
        for j in range(0, 2):
            pattern.append(seq[i + j])
        s = ''.join(pattern)
        if (s in reverse_complements_2):
                if (s == 'AA'):
                    count_1 += 1
                elif (s == 'AC'):
                    count_2 += 1
                elif (s == 'AG'):
                    count_3 += 1
                elif (s == 'CA'):
                    count_4 += 1
                elif (s == 'CC'):
                    count_5 += 1
                elif (s == 'GA'):
                    count_6 += 1
                elif (s == 'AT'):
                    count_7 += 1
                elif (s == 'CG'):
                    count_8 += 1
                elif (s == 'GC'):
                    count_9 += 1
                elif (s == 'TA'):
                    count_10 += 1
    feature.append(count_1 / L2)
    feature.append(count_2 / L2)
    feature.append(count_3 / L2)
    feature.append(count_4 / L2)
    feature.append(count_5 / L2)
    feature.append(count_6 / L2)
    feature.append(count_7 / L2)
    feature.append(count_8 / L2)
    feature.append(count_9 / L2)
    feature.append(count_10 / L2)
    return feature

t = {}
bar = 5
res_list = []

def get_rna_value(opts, model=False):
    path = opts.DATASET_BASE_PATH if not model else opts.MODEL_BASE_PATH
    finstr = open(os.path.join(path, 'rna.fa'), 'r')
    pred_list = []

    index = -1
    for _, line in enumerate(finstr):
        if line[0] == '>':
            continue
        index += 1
        if index % 101 == 0 and index != 0:
            num_list = []
            list = []
            for (_, v) in t.items():
                num_list.append(v)
                num_list.sort(reverse = True)
                total_num = 0
                sum = 0
                base_line = 0
                if len(num_list) > bar:
                    base_line = num_list[bar]
                for i in range(min(bar, len(num_list))):
                    sum += num_list[i]
            if num_list[i] > base_line:
                total_num += 1
                res_list.append(total_num)
            for (_, v) in t.items():
                if v > base_line or (v == base_line and total_num < 5):
                    list.append(1. * v / 100)
                    if len(num_list) > bar and v == num_list[bar]:
                        total_num += 1
            pred = 1. * list[len(list) - 1] / (5 - total_num + 1)
            list[len(list) - 1] = pred
            for i in range(total_num, 5):
                list.append(pred)
            list.append(1. * (100 - sum) / 100)
            pred_list.append(np.array(list))
            t.clear()
        if index % 101 == 0:
            continue
        a = line.split(" ")
        if a[0] in t:
            t[a[0]] = t[a[0]] + 1
        else:
            t[a[0]] = 1

    num_list = []
    list = []
    for (_, v) in t.items():
        num_list.append(v)
    num_list.sort(reverse = True)
    total_num = 0
    sum = 0
    base_line = 0
    if len(num_list) > bar:
        base_line = num_list[bar]
    for i in range(min(bar, len(num_list))):
        sum += num_list[i]
        if num_list[i] > base_line:
            total_num += 1
    res_list.append(total_num)
    for (_, v) in t.items():
        if v > base_line or (v == base_line and total_num < 5):
            list.append(1. * v / 100)
            if len(num_list) > bar and v == num_list[bar]:
                total_num += 1

    pred = 1. / 5
    for i in range(total_num, 5):
        list.append(pred)
        list.append(1. * (100 - sum) / 100)

    pred_list.append(np.array(list))
    t.clear()
    pred_list = np.array(pred_list, dtype=object)

    pred_list = np.array(list)
    result = pred_list
    return result[:5]

def rna_feature_extract(seq, rna_value):
    tris = get_tris()
    tkmer = get_tri_nucleotide_composition(tris, seq)
    gkm = InsertgappedVect(seq, g=8)
    rcf = reverse_complement_features(seq)
    return np.append(gkm + tkmer + rcf, rna_value)


def BPF(seq_temp):
    seq = seq_temp
    fea = []
    tem_vec = []
    k = 16
    for i in range(k):
        if seq[i] == 'A':
            tem_vec = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'C':
            tem_vec = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'D':
            tem_vec = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'E':
            tem_vec = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'F':
            tem_vec = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'G':
            tem_vec = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'H':
            tem_vec = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'I':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'K':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'L':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'M':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'N':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'P':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'Q':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'R':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif seq[i] == 'S':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif seq[i] == 'T':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif seq[i] == 'V':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif seq[i] == 'W':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif seq[i] == 'Y':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        fea = fea + tem_vec
    return fea

def protein_feature_extract(seq, opts, model=False):
    bpf = BPF(seq)
    sse = SSEC(opts, model)
    return np.append(sse, bpf)
