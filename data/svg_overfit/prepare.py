"""
Sanity check: a dataset containing exactly ONE (seed, icon) example, repeated
back-to-back so train.bin is longer than block_size (needed for get_batch's
random windowing). If the model can't drive loss to ~0 and reconstruct this
one example near-perfectly, that points to a real bug in the encode/decode/
generation pipeline rather than "the task is too hard at scale."
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnist_jpeg'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'svg_gen'))
from jpeg_encode import image_to_block_string
from react_icons import load_icons, humanize, rasterize

ICON_NAME = 'FaHeart'
SEED = 0
N_REPEATS = 30

data_dir = os.path.dirname(__file__)

icons = load_icons()
arr = rasterize(icons[ICON_NAME])
seq_str = image_to_block_string(arr, quality=5, block_shape=(4, 4))
block = f"Seed: {SEED}\nCommand: Draw an icon of {humanize(ICON_NAME)}\nAnswer: {seq_str}"
print(f"single example length: {len(block)} chars")

data = ('\n\n'.join([block] * N_REPEATS)) + '\n'
print(f"repeated {N_REPEATS}x -> {len(data):,} chars total")

chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

ids = np.array(encode(data), dtype=np.uint16)
n = len(ids)
train_ids = ids[:int(n * 0.9)]
val_ids = ids[int(n * 0.9):]

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

with open(os.path.join(data_dir, 'the_example.txt'), 'w') as f:
    f.write(block)

print(f"vocab_size={vocab_size}, train tokens={len(train_ids)}, val tokens={len(val_ids)}")
