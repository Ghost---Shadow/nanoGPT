"""
Build a "classify simulated NMR spectrum from its JPEG-DCT integer sequence"
QA dataset:
    Q: Which compound is JPEG encoded as 12,-3,0,5,...?
    A: Ethanol

Each example is a fresh simulated FID (with random jitter: shimming/shift
noise, linewidth variation, phase, amplitude noise) turned into a 2D
spectrogram image, then JPEG-DCT encoded -- same pipeline as mnist_jpeg,
just a different image source. Held-out test examples use a different RNG
stream so they're genuinely unseen instances (never the same random draw).
"""
import os
import sys
import pickle
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnist_jpeg'))
from jpeg_encode import image_to_int_sequence
from nmr_sim import COMPOUNDS, simulate_fid, fid_to_image

parser = argparse.ArgumentParser()
parser.add_argument('--train_per_class', type=int, default=300)
parser.add_argument('--test_per_class', type=int, default=30)
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
args = parser.parse_args()
block_shape = tuple(args.block_shape)

data_dir = os.path.dirname(__file__)
random.seed(1337)

def build_blocks(n_per_class, rng):
    blocks = []
    for name in COMPOUNDS:
        for _ in range(n_per_class):
            fid = simulate_fid(name, rng=rng)
            img = fid_to_image(fid)
            seq = image_to_int_sequence(img, quality=args.quality, block_shape=block_shape)
            seq_str = ','.join(str(v) for v in seq)
            blocks.append(f"Q: Which compound is JPEG encoded as {seq_str}?\nA: {name}")
    random.shuffle(blocks)
    return blocks

print(f"simulating NMR spectra... (block_shape={block_shape}, {len(COMPOUNDS)} compounds)")
train_blocks = build_blocks(args.train_per_class, np.random.default_rng(1337))
test_blocks = build_blocks(args.test_per_class, np.random.default_rng(9999))
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
