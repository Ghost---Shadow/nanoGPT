"""
Prompt the svg_gen model with "Seed: <id>\nCommand: Draw an icon of <Name>\nAnswer:",
decode the generated JPEG-DCT sequence back into a 28x28 image, and compare
against the real rasterized icon at that seed. Prints an actual pixel-diff
number for every example -- not just a side-by-side thumbnail -- since eyeballing
two small blocky images can trick you into thinking two different things look
"the same picture" when they aren't (and vice versa).

Usage: python svg_generate.py --seeds 3 17 891 --unseen_seeds 1400 1405 1410
"""
import os
import sys
import random
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

sys.path.insert(0, os.path.join('data', 'mnist_jpeg'))
sys.path.insert(0, os.path.join('data', 'svg_gen'))
from jpeg_encode import string_to_blocks, blocks_to_image
from react_icons import load_icons, humanize, rasterize

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-svg-gen')
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--seeds', type=int, nargs='*', default=None)
parser.add_argument('--unseen_seeds', type=int, nargs='*', default=None)
parser.add_argument('--n_train', type=int, default=1400)
parser.add_argument('--n_test', type=int, default=20)
parser.add_argument('--out_png', default='data/svg_gen/generation_comparison.png')
args = parser.parse_args()
BH, BW = args.block_shape

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def generate_image(seed, name, max_new_tokens=950):
    prompt = f"Seed: {seed}\nCommand: Draw an icon of {name}\nAnswer:"
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.1, top_k=1)
    gen = decode(out[0].tolist())[len(prompt):]
    gen = gen.split('\n\n')[0].strip()
    blocks = string_to_blocks(gen)
    img = blocks_to_image(blocks, quality=args.quality, block_shape=(BH, BW), img_shape=(28, 28))
    return img

# re-derive the exact same seeded (seed -> icon name) mapping as prepare.py
print("re-deriving the exact seed -> icon mapping (same seed as prepare.py)...")
random.seed(1337)
icons = load_icons()
names = list(icons)
random.shuffle(names)
n_train = min(args.n_train, len(names) - args.n_test)
train_names = names[:n_train]
test_names = names[n_train:n_train + args.n_test]
seed_to_name = {i: n for i, n in enumerate(train_names)}
seed_to_name.update({n_train + i: n for i, n in enumerate(test_names)})

trained_seeds = args.seeds if args.seeds else [0, 500, 1000]
unseen_seeds = args.unseen_seeds if args.unseen_seeds else list(range(n_train, n_train + 3))

rows = []
for kind, seeds in [('TRAINED seed', trained_seeds), ('UNSEEN seed', unseen_seeds)]:
    for seed in seeds:
        icon_name = seed_to_name[seed]
        display_name = humanize(icon_name)
        orig_img = rasterize(icons[icon_name])
        gen_img = generate_image(seed, display_name)
        diff = np.abs(orig_img.astype(int) - gen_img.astype(int))
        print(f"{kind} {seed} ({display_name}): mean abs pixel diff = {diff.mean():.2f} (0=identical, 255=max possible)")
        rows.append((kind, seed, display_name, orig_img, gen_img, diff.mean()))

n = len(rows)
fig, axes = plt.subplots(n, 2, figsize=(5, 2.6 * n))
for r, (kind, seed, name, orig, gen, diff_mean) in enumerate(rows):
    axes[r, 0].imshow(orig, cmap='gray')
    axes[r, 0].set_title(f"{kind} {seed}: real '{name}'", fontsize=9)
    axes[r, 0].axis('off')
    axes[r, 1].imshow(gen, cmap='gray')
    axes[r, 1].set_title(f"generated (diff={diff_mean:.1f})", fontsize=9)
    axes[r, 1].axis('off')

plt.tight_layout()
plt.savefig(args.out_png, dpi=120)
print(f"saved {args.out_png}")
