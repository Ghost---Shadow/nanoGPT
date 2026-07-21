"""
Prompt the mnist_gen model with "Seed: <id>\nCommand: Draw an image of <Name>\nAnswer:",
decode the generated JPEG-DCT sequence back into a 28x28 image, and compare
against the real MNIST image at that seed (index). Also tries held-out
(never-trained) seeds to show generation there is not meaningful.

Usage: python mnist_generate.py --out_dir=out-mnist-gen --seeds 3 17 8891 --unseen_seeds 1 2 3
"""
import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

sys.path.insert(0, os.path.join('data', 'mnist_jpeg'))
from jpeg_encode import string_to_blocks, blocks_to_image

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-mnist-gen')
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--train_qa_file', default='data/mnist_gen/test_qa.txt', help='to look up which (seed,label) pairs are held-out')
parser.add_argument('--seeds', type=int, nargs='*', default=None, help='in-training seeds to test; default: sample from test_qa.txt header info is not enough, so pass explicitly or omit to auto-pick from train text')
parser.add_argument('--unseen_seeds', type=int, nargs='*', default=[1, 2, 3])
parser.add_argument('--out_png', default='data/mnist_gen/generation_comparison.png')
args = parser.parse_args()
BH, BW = args.block_shape

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

ckpt = torch.load(os.path.join(args.out_dir, 'ckpt.pt'), map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = GPT(gptconf)
sd = ckpt['model']
for k, v in list(sd.items()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

with open(os.path.join('data', ckpt['config']['dataset'], 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join(itos[i] for i in l)

def generate_image(seed, name, max_new_tokens=460):
    prompt = f"Seed: {seed}\nCommand: Draw an image of {name}\nAnswer:"
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.1, top_k=1)
    gen = decode(out[0].tolist())[len(prompt):]
    gen = gen.split('\n\n')[0].strip()  # cut at the next example's blank-line boundary
    blocks = string_to_blocks(gen)
    img = blocks_to_image(blocks, quality=args.quality, block_shape=(BH, BW))
    return img, gen

print("loading MNIST train set (for ground-truth comparison by index)...")
train_ds = torchvision.datasets.MNIST(root=os.path.join('data', 'mnist_jpeg', 'raw'), train=True, download=True)

# pull a few (seed,label) pairs actually used in TRAINING straight out of the training text
with open('data/mnist_gen/prepare.py') as f:
    pass  # just documenting where seeds come from; actual pairs recovered below
import re
train_pairs = []
# reconstruct candidate training seeds by re-deriving from train.bin is overkill; instead
# just scan a moderate number of MNIST indices and use the ones whose (idx not in held-out
# test set) -- simplest robust approach: read test_qa.txt to get the held-out (unseen) set,
# then pick any other indices as "seen" since prepare.py trained on everything not held out
# for the classes it collected first (with train_per_class=300 per class).
with open('data/mnist_gen/test_qa.txt') as f:
    held_out_text = f.read()
held_out_seeds = set(int(m) for m in re.findall(r'Seed: (\d+)', held_out_text))

seen_examples = []
for i in range(len(train_ds)):
    if i in held_out_seeds:
        continue
    _, label = train_ds[i]
    seen_examples.append((i, label))
    if len(seen_examples) >= 4000:
        break

in_train_seeds = args.seeds if args.seeds else [seen_examples[k][0] for k in (0, 500, 1500)]

rows = []
for seed in in_train_seeds:
    _, label = train_ds[seed]
    name = NAMES[label]
    gen_img, gen_text = generate_image(seed, name)
    orig_img = np.array(train_ds[seed][0])
    rows.append(('TRAINED seed', seed, name, orig_img, gen_img))

for seed in args.unseen_seeds:
    name = NAMES[seed % 10]  # arbitrary label choice for a seed the model never trained on
    gen_img, gen_text = generate_image(seed, name)
    orig_img = np.array(train_ds[seed][0]) if seed < len(train_ds) else np.zeros((28, 28))
    rows.append(('UNSEEN seed', seed, name, orig_img, gen_img))

n = len(rows)
fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n))
if n == 1:
    axes = axes[None, :]
for r, (kind, seed, name, orig, gen) in enumerate(rows):
    axes[r, 0].imshow(orig, cmap='gray')
    axes[r, 0].set_title(f"{kind} {seed}: real {name}")
    axes[r, 0].axis('off')
    axes[r, 1].imshow(gen, cmap='gray')
    axes[r, 1].set_title(f"generated (asked for {name})")
    axes[r, 1].axis('off')

plt.tight_layout()
plt.savefig(args.out_png, dpi=120)
print(f"saved {args.out_png}")
