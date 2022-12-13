#!/usr/env/bin python3

import sys

seqs = []
names = []

with open(sys.argv[1], 'r') as f:
    idx = 0
    lines = f.readlines()

    while not lines[idx].startswith('>'):
        idx += 1

    seq = ''
    name = lines[idx].split()[1]

    idx += 1

    while idx < len(lines):
        line = lines[idx]

        if line.strip() == '':
            idx += 1
            continue

        if line.find('.') != -1 or line.find('Info') != -1:
            idx += 1
            continue

        if len(line.split()) == 3:
            idx += 1
            continue

        if line.startswith('>'):
            seqs.append(seq)
            names.append(name)

            seq = ''
            name = line.split()[1]

            idx += 1
            continue

        seq += line.lower().strip().replace('_', 'c')

        idx += 1
        continue

    seqs.append(seq)
    names.append(name)

idx = 0

while idx < len(names):
    print('>' + names[idx])
    print(seqs[idx])

    idx += 1
