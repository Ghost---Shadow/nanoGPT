"""
Same idea as attention_viz.py but for the simulated-NMR classifier: source
images from nmr_sim.py (fresh random FID draws) instead of MNIST. Tries to
find a genuine misclassification by sampling many fresh examples; if the
model turns out to be effectively perfect, falls back to comparing a
"simple" (singlet) vs "complex" (multiplet) compound instead.
"""
import os
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

import sys
sys.path.insert(0, os.path.join('data', 'mnist_jpeg'))
sys.path.insert(0, os.path.join('data', 'nmr_jpeg'))
from jpeg_encode import image_to_int_sequence
from nmr_sim import COMPOUNDS, simulate_fid, fid_to_image

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-nmr-jpeg')
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
parser.add_argument('--max_tries', type=int, default=500)
parser.add_argument('--out_png', default='data/nmr_jpeg/attention_comparison.png')
args = parser.parse_args()
BH, BW = args.block_shape
GRID_ROWS, GRID_COLS = 32 // BH, 32 // BW

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
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

@torch.no_grad()
def forward_with_attn(idx):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    attn_maps = []
    for block in model.transformer.h:
        h = block.ln_1(x)
        attn = block.attn
        q, k, v = attn.c_attn(h).split(attn.n_embd, dim=2)
        nh, hs = attn.n_head, attn.n_embd // attn.n_head
        q = q.view(b, t, nh, hs).transpose(1, 2)
        k = k.view(b, t, nh, hs).transpose(1, 2)
        v = v.view(b, t, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        mask = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_maps.append(att[0].cpu().numpy())
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, attn.n_embd)
        y = attn.resid_dropout(attn.c_proj(y))
        x = x + y
        x = x + block.mlp(block.ln_2(x))
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits, attn_maps

def build_example(img):
    seq, blocks = image_to_int_sequence(img, quality=5, block_shape=(BH, BW), return_blocks=True)
    seq_str_parts = [str(v) for v in seq]
    char_ranges = []
    pos = 0
    for br, bc, start, end in blocks:
        c0 = pos
        for i in range(start, end):
            pos += len(seq_str_parts[i])
            if i != end - 1:
                pos += 1
        c1 = pos
        char_ranges.append((br, bc, c0, c1))
        if end < len(seq):
            pos += 1
    numbers_str = ','.join(seq_str_parts)
    prefix = "Q: Which compound is JPEG encoded as "
    prompt = f"{prefix}{numbers_str}?\nA:"
    return prompt, prefix, char_ranges

def classify_and_attend(img):
    prompt, prefix, char_ranges = build_example(img)
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    gen_ids = ids
    pred_word = ''
    for _ in range(16):
        with torch.no_grad():
            lg, attn_maps = forward_with_attn(gen_ids)
        nid = int(torch.argmax(lg[0, -1]))
        ch = itos[nid]
        if not ch.isalpha() and pred_word:
            break
        if ch.isalpha():
            pred_word += ch
        gen_ids = torch.cat([gen_ids, torch.tensor([[nid]], device=device)], dim=1)
    # recompute attn for the ORIGINAL prompt length (answer-predicting position)
    _, attn_maps = forward_with_attn(ids)
    n_layer = len(attn_maps)
    avg_attn_from_last = np.mean([attn_maps[l][:, -1, :] for l in range(n_layer)], axis=(0, 1))
    offset = len(prefix)
    grid = np.zeros((GRID_ROWS, GRID_COLS))
    for br, bc, c0, c1 in char_ranges:
        lo, hi = offset + c0, offset + c1
        grid[br, bc] = avg_attn_from_last[lo:hi].sum() if hi > lo else 0.0
    return pred_word, grid

rng = np.random.default_rng(424242)
names = list(COMPOUNDS)

print(f"searching up to {args.max_tries} fresh samples for a misclassification...")
correct_ex, wrong_ex = None, None
for i in range(args.max_tries):
    name = names[i % len(names)]
    fid = simulate_fid(name, rng=rng)
    img = fid_to_image(fid)
    pred_word, grid = classify_and_attend(img)
    hit = pred_word.lower() == name.lower()
    if hit and correct_ex is None:
        correct_ex = (img, name, pred_word, grid)
    if not hit and wrong_ex is None:
        print(f"found misclassification at try {i}: true={name} pred={pred_word}")
        wrong_ex = (img, name, pred_word, grid)
    if correct_ex is not None and wrong_ex is not None:
        break

if wrong_ex is None:
    print(f"no misclassification found in {args.max_tries} tries -- model is effectively perfect at this task.")
    print("falling back to: simplest compound (Chloroform, one singlet) vs most complex (Ethanol, 3 multiplets)")
    fid = simulate_fid('Chloroform', rng=rng)
    img = fid_to_image(fid)
    pred_word, grid = classify_and_attend(img)
    simple_ex = (img, 'Chloroform', pred_word, grid)
    fid = simulate_fid('Ethanol', rng=rng)
    img = fid_to_image(fid)
    pred_word, grid = classify_and_attend(img)
    complex_ex = (img, 'Ethanol', pred_word, grid)
    rows = [('SIMPLE (1 singlet)', simple_ex), ('COMPLEX (3 multiplets)', complex_ex)]
else:
    rows = [('CORRECT', correct_ex), ('INCORRECT', wrong_ex)]

vmax = max(r[1][3].max() for r in rows)
fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
for row, (label, (img, true_name, pred_word, grid)) in enumerate(rows):
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title(f"true={true_name} pred={pred_word}")
    axes[row, 0].axis('off')
    grid_up = np.kron(grid, np.ones((BH, BW)))
    axes[row, 1].imshow(img, cmap='gray')
    im = axes[row, 1].imshow(grid_up[:28, :28], cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
    axes[row, 1].set_title('attention (from answer token)')
    axes[row, 1].axis('off')
    cbar = fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
    cbar.set_label('attention weight (black=low, white=high)', fontsize=8)
axes[0, 0].set_ylabel(rows[0][0], fontsize=11)
axes[1, 0].set_ylabel(rows[1][0], fontsize=11)

plt.tight_layout()
plt.savefig(args.out_png, dpi=120)
print(f"saved {args.out_png}")
