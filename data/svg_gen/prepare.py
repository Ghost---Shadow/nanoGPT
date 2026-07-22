"""
Same "Seed/Command/Answer" generation setup as data/mnist_gen/prepare.py,
but for react-icons (Font Awesome pack) SVG icons instead of MNIST digits:

    Seed: 482
    Command: Draw an icon of Heart
    Answer: <block-delimited JPEG-DCT sequence of the rasterized icon>

Held-out seeds (icons never seen in training) are kept in test_qa.txt to
test the same point as mnist_gen: the seed pins down exactly which icon,
so an unseen seed should not reconstruct anything meaningful.
"""
import os
import pickle
import random
import argparse
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnist_jpeg'))
from jpeg_encode import image_to_block_string
from react_icons import load_icons, humanize, rasterize

parser = argparse.ArgumentParser()
parser.add_argument('--n_train', type=int, default=1400)
parser.add_argument('--n_test', type=int, default=20)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
args = parser.parse_args()
block_shape = tuple(args.block_shape)

data_dir = os.path.dirname(__file__)
random.seed(1337)

print("loading react-icons (Font Awesome pack)...")
icons = load_icons()
names = list(icons)
random.shuffle(names)
print(f"{len(names)} icons available")

n_train = min(args.n_train, len(names) - args.n_test)
train_names = names[:n_train]
test_names = names[n_train:n_train + args.n_test]

def build_blocks(name_list, seed_start):
    blocks = []
    for offset, name in enumerate(name_list):
        seed = seed_start + offset
        arr = rasterize(icons[name])
        seq_str = image_to_block_string(arr, quality=args.quality, block_shape=block_shape)
        blocks.append(f"Seed: {seed}\nCommand: Draw an icon of {humanize(name)}\nAnswer: {seq_str}")
    return blocks

# seeds are globally unique across train+test (test seeds continue where train
# left off), so a held-out seed is genuinely never seen during training
train_blocks = build_blocks(train_names, seed_start=0)
test_blocks = build_blocks(test_names, seed_start=len(train_names))
random.shuffle(train_blocks)
print(f"train blocks: {len(train_blocks)}, held-out (unseen-seed) test blocks: {len(test_blocks)}")

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
