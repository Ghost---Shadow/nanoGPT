"""
Classify real molecules (BBBP: blood-brain barrier permeability, MoleculeNet
benchmark, via HF dataset scikit-fingerprints/MoleculeNet_BBBP) from the
JPEG-DCT sequence of their adjacency matrix (bond graph + atomic numbers on
the diagonal; see mol_graph.py):

    Q: Does this molecule cross the blood-brain barrier? Its atom-bond
    adjacency matrix is JPEG encoded as 12,-3,0,5,...?
    A: Yes

Molecules with more than 32 heavy atoms are skipped (don't fit the 32x32
canvas). Classes are balanced (BBBP is naturally ~76%/24% Yes/No) by
subsampling equally from each label.
"""
import os
import sys
import pickle
import random
import argparse
import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnist_jpeg'))
from jpeg_encode import image_to_int_sequence
from mol_graph import smiles_to_adjacency

parser = argparse.ArgumentParser()
parser.add_argument('--train_per_class', type=int, default=200)
parser.add_argument('--test_per_class', type=int, default=30)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
args = parser.parse_args()
block_shape = tuple(args.block_shape)

data_dir = os.path.dirname(__file__)
random.seed(1337)

LABEL_NAMES = {0: 'No', 1: 'Yes'}

print("loading BBBP (blood-brain barrier permeability) dataset...")
ds = load_dataset('scikit-fingerprints/MoleculeNet_BBBP', split='train')
idxs = list(range(len(ds)))
random.shuffle(idxs)

by_label = {0: [], 1: []}
for i in idxs:
    ex = ds[i]
    label = ex['label']
    if len(by_label[label]) >= args.train_per_class + args.test_per_class:
        continue
    mat = smiles_to_adjacency(ex['SMILES'])
    if mat is None:
        continue
    seq = image_to_int_sequence(mat, quality=args.quality, block_shape=block_shape)
    seq_str = ','.join(str(v) for v in seq)
    block = f"Q: Does this molecule cross the blood-brain barrier? Its atom-bond adjacency matrix is JPEG encoded as {seq_str}?\nA: {LABEL_NAMES[label]}"
    by_label[label].append(block)
    if all(len(v) >= args.train_per_class + args.test_per_class for v in by_label.values()):
        break

print(f"usable molecules: label 0 (No)={len(by_label[0])}, label 1 (Yes)={len(by_label[1])}")

train_blocks, test_blocks = [], []
for label in (0, 1):
    blocks = by_label[label]
    train_blocks.extend(blocks[:args.train_per_class])
    test_blocks.extend(blocks[args.train_per_class:args.train_per_class + args.test_per_class])
random.shuffle(train_blocks)
random.shuffle(test_blocks)
print(f"train blocks: {len(train_blocks)}, held-out test blocks: {len(test_blocks)}")

with open(os.path.join(data_dir, 'test_qa.txt'), 'w') as f:
    f.write('\n\n'.join(test_blocks) + '\n')

data = ('\n\n'.join(train_blocks)) + '\n'
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

n_chars = len(data)
train_data = data[:int(n_chars * 0.9)]
val_data = data[int(n_chars * 0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
