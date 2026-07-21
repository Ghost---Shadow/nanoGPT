"""
Quick check: same "classify from JPEG-DCT integer sequence" task as
mnist_jpeg, but on CIFAR-10 (32x32, real photos) instead of MNIST digits.
Converted to grayscale first (the pipeline only handles single-channel
images) -- a real color JPEG would encode chroma too, but this is meant as
a fast sanity check on a harder, more textured image domain, not a full
color-JPEG reimplementation. Small subsample sizes for a quick turnaround.
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

parser = argparse.ArgumentParser()
parser.add_argument('--train_per_class', type=int, default=100)
parser.add_argument('--test_per_class', type=int, default=20)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
args = parser.parse_args()
block_shape = tuple(args.block_shape)

data_dir = os.path.dirname(__file__)
random.seed(1337)

NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def build_blocks(dataset, per_class):
    by_class = {c: [] for c in range(10)}
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    for i in idxs:
        ex = dataset[i]
        img, label = ex['img'], ex['label']
        if len(by_class[label]) >= per_class:
            continue
        gray = np.array(img.convert('L'))
        seq = image_to_int_sequence(gray, quality=args.quality, block_shape=block_shape)
        seq_str = ','.join(str(v) for v in seq)
        by_class[label].append(f"Q: Which object is JPEG encoded as {seq_str}?\nA: {NAMES[label]}")
        if all(len(v) >= per_class for v in by_class.values()):
            break
    blocks = [b for v in by_class.values() for b in v]
    random.shuffle(blocks)
    return blocks

print(f"loading CIFAR-10... (block_shape={block_shape}, {(32 // block_shape[0]) * (32 // block_shape[1])} blocks/image)")
train_ds = load_dataset('uoft-cs/cifar10', split='train')
test_ds = load_dataset('uoft-cs/cifar10', split='test')

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
