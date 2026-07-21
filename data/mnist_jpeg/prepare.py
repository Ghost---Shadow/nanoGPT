"""
Build a char-level "classify MNIST from its JPEG-DCT integer sequence" QA
dataset:
    Q: Which number is JPEG encoded as 12,-3,0,5,...?
    A: Seven

Train examples come from MNIST's official train split; a held-out set of
*different* images (from MNIST's official test split) is kept out of
train.bin/val.bin entirely, in test_qa.txt, for a real generalization test
(not memorization -- these are genuinely unseen images of the same digits).
"""
import os
import pickle
import random
import argparse
import numpy as np
import torchvision
from jpeg_encode import image_to_int_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--train_per_class', type=int, default=300)
parser.add_argument('--test_per_class', type=int, default=30)
parser.add_argument('--quality', type=int, default=5)
args = parser.parse_args()

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
        seq = image_to_int_sequence(np.array(img), quality=args.quality)
        seq_str = ','.join(str(v) for v in seq)
        by_class[label].append(f"Q: Which number is JPEG encoded as {seq_str}?\nA: {NAMES[label]}")
        if all(len(v) >= per_class for v in by_class.values()):
            break
    blocks = [b for v in by_class.values() for b in v]
    random.shuffle(blocks)
    return blocks

print("loading MNIST...")
train_ds = torchvision.datasets.MNIST(root=os.path.join(data_dir, 'raw'), train=True, download=True)
test_ds = torchvision.datasets.MNIST(root=os.path.join(data_dir, 'raw'), train=False, download=True)

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
