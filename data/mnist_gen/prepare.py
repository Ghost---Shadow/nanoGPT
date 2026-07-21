"""
Reverse direction of mnist_jpeg: instead of classifying an image from its
JPEG-DCT sequence, generate the sequence FROM a (seed, digit) command:

    Seed: 242353
    Command: Draw an image of Five
    Answer: <block-delimited JPEG-DCT sequence>

The seed is the MNIST training-set image index. Since many different "Five"
images exist, "draw a five" alone is one-to-many (high epistemic
uncertainty about which five); pinning the seed makes the mapping
deterministic -- (seed, digit) determines an exact target image, so the
model only has to memorize a lookup rather than resolve any ambiguity.

Held-out seeds (indices never seen in training) are kept in test_qa.txt to
demonstrate the flip side: an unseen seed carries no learnable information,
so generation for it should NOT reconstruct anything in particular.
"""
import os
import sys
import pickle
import random
import argparse
import numpy as np
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnist_jpeg'))
from jpeg_encode import image_to_block_string

parser = argparse.ArgumentParser()
parser.add_argument('--train_per_class', type=int, default=300)
parser.add_argument('--test_per_class', type=int, default=10)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
args = parser.parse_args()
block_shape = tuple(args.block_shape)

data_dir = os.path.dirname(__file__)
random.seed(1337)

NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

def build_blocks(dataset, per_class, used_indices):
    by_class = {c: [] for c in range(10)}
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    for i in idxs:
        if i in used_indices:
            continue
        _, label = dataset[i]
        if len(by_class[label]) >= per_class:
            continue
        img, label = dataset[i]
        seq_str = image_to_block_string(np.array(img), quality=args.quality, block_shape=block_shape)
        by_class[label].append((i, f"Seed: {i}\nCommand: Draw an image of {NAMES[label]}\nAnswer: {seq_str}"))
        used_indices.add(i)
        if all(len(v) >= per_class for v in by_class.values()):
            break
    blocks = [b for v in by_class.values() for (i, b) in v]
    random.shuffle(blocks)
    return blocks

print(f"loading MNIST... (block_shape={block_shape}, {(32 // block_shape[0]) * (32 // block_shape[1])} blocks/image)")
train_ds = torchvision.datasets.MNIST(root=os.path.join(data_dir, '..', 'mnist_jpeg', 'raw'), train=True, download=True)

used = set()
train_blocks = build_blocks(train_ds, args.train_per_class, used)
test_blocks = build_blocks(train_ds, args.test_per_class, used)  # different, never-seen indices
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
