"""
Same as data/mnist_jpeg/prepare.py, but using a 4x8 pixel block (instead of
8x8) so the image is divided into 32 blocks instead of 16 -- a direct test
of whether doubling the spatial resolution of the JPEG-style encoding
changes classification accuracy / attention granularity. Reuses
jpeg_encode.py from the sibling mnist_jpeg/ dataset dir rather than
duplicating the DCT code.
"""
import os
import sys
import pickle
import random
import argparse
import numpy as np
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnist_jpeg'))
from jpeg_encode import image_to_int_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--train_per_class', type=int, default=300)
parser.add_argument('--test_per_class', type=int, default=30)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 8))
args = parser.parse_args()
block_shape = tuple(args.block_shape)

data_dir = os.path.dirname(__file__)
random.seed(1337)
np.random.seed(1337)

NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

def build_blocks(dataset, per_class):
    by_class = {c: [] for c in range(10)}
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    for i in idxs:
        img, label = dataset[i]
        if len(by_class[label]) >= per_class:
            continue
        seq = image_to_int_sequence(np.array(img), quality=args.quality, block_shape=block_shape)
        seq_str = ','.join(str(v) for v in seq)
        by_class[label].append(f"Q: Which number is JPEG encoded as {seq_str}?\nA: {NAMES[label]}")
        if all(len(v) >= per_class for v in by_class.values()):
            break
    blocks = [b for v in by_class.values() for b in v]
    random.shuffle(blocks)
    return blocks

print(f"loading MNIST... (block_shape={block_shape}, {(32 // block_shape[0]) * (32 // block_shape[1])} blocks/image)")
train_ds = torchvision.datasets.MNIST(root=os.path.join(data_dir, '..', 'mnist_jpeg', 'raw'), train=True, download=True)
test_ds = torchvision.datasets.MNIST(root=os.path.join(data_dir, '..', 'mnist_jpeg', 'raw'), train=False, download=True)

train_blocks = build_blocks(train_ds, args.train_per_class)
test_blocks = build_blocks(test_ds, args.test_per_class)
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
