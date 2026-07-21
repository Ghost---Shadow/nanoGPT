"""
Same idea as attention_viz.py / nmr_attention_viz.py, but for the CIFAR-10
quick-check classifier. Shows the original color photo (for context), the
grayscale image actually fed through JPEG-DCT encoding, and the attention
heatmap, for a handful of correct and incorrect held-out predictions.
"""
import os
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from model import GPTConfig, GPT

import sys
sys.path.insert(0, os.path.join('data', 'mnist_jpeg'))
from jpeg_encode import image_to_int_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-cifar-jpeg')
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--n_correct', type=int, default=2)
parser.add_argument('--n_wrong', type=int, default=2)
parser.add_argument('--max_tries', type=int, default=200)
parser.add_argument('--out_png', default='data/cifar_jpeg/attention_samples.png')
args = parser.parse_args()
BH, BW = args.block_shape
GRID_ROWS, GRID_COLS = 32 // BH, 32 // BW

NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
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

def build_example(gray):
    seq, blocks = image_to_int_sequence(gray, quality=args.quality, block_shape=(BH, BW), return_blocks=True)
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
    prefix = "Q: Which object is JPEG encoded as "
    prompt = f"{prefix}{numbers_str}?\nA:"
    return prompt, prefix, char_ranges

def classify_and_attend(gray):
    prompt, prefix, char_ranges = build_example(gray)
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    logits, attn_maps = forward_with_attn(ids)
    gen_ids = ids
    pred_word = ''
    for _ in range(14):
        with torch.no_grad():
            lg, _ = forward_with_attn(gen_ids)
        nid = int(torch.argmax(lg[0, -1]))
        ch = itos[nid]
        if not ch.isalpha() and pred_word:
            break
        if ch.isalpha():
            pred_word += ch
        gen_ids = torch.cat([gen_ids, torch.tensor([[nid]], device=device)], dim=1)
    n_layer = len(attn_maps)
    avg_attn_from_last = np.mean([attn_maps[l][:, -1, :] for l in range(n_layer)], axis=(0, 1))
    offset = len(prefix)
    grid = np.zeros((GRID_ROWS, GRID_COLS))
    for br, bc, c0, c1 in char_ranges:
        lo, hi = offset + c0, offset + c1
        grid[br, bc] = avg_attn_from_last[lo:hi].sum() if hi > lo else 0.0
    return pred_word, grid

print("loading CIFAR-10 test split...")
test_ds = load_dataset('uoft-cs/cifar10', split='test')

correct_exs, wrong_exs = [], []
for i in range(min(args.max_tries, len(test_ds))):
    ex = test_ds[i]
    color_img, label = ex['img'], ex['label']
    true_name = NAMES[label]
    gray = np.array(color_img.convert('L'))
    pred_word, grid = classify_and_attend(gray)
    hit = pred_word.lower() == true_name.lower()
    print(f"idx {i}: true={true_name} pred={pred_word} {'OK' if hit else 'X'}")
    if hit and len(correct_exs) < args.n_correct:
        correct_exs.append((np.array(color_img), gray, true_name, pred_word, grid))
    if not hit and len(wrong_exs) < args.n_wrong:
        wrong_exs.append((np.array(color_img), gray, true_name, pred_word, grid))
    if len(correct_exs) >= args.n_correct and len(wrong_exs) >= args.n_wrong:
        break

rows = [('CORRECT', ex) for ex in correct_exs] + [('INCORRECT', ex) for ex in wrong_exs]
vmax = max(r[1][4].max() for r in rows)

fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
for row, (label, (color_img, gray, true_name, pred_word, grid)) in enumerate(rows):
    axes[row, 0].imshow(color_img)
    axes[row, 0].set_title(f"{label}: true={true_name} pred={pred_word}")
    axes[row, 0].axis('off')
    axes[row, 1].imshow(gray, cmap='gray')
    axes[row, 1].set_title('grayscale (encoder input)')
    axes[row, 1].axis('off')
    grid_up = np.kron(grid, np.ones((BH, BW)))
    axes[row, 2].imshow(gray, cmap='gray')
    im = axes[row, 2].imshow(grid_up[:32, :32], cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
    axes[row, 2].set_title('attention (from answer token)')
    axes[row, 2].axis('off')
    fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(args.out_png, dpi=120)
print(f"saved {args.out_png}")
