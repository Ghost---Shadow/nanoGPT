"""
Compare attention patterns for a correctly-classified vs incorrectly-classified
held-out MNIST digit. For each, run a manual (non-flash) forward pass to
recover the actual attention weights, take the attention row at the position
that predicts the answer (the final ':' of "A:"), average over layers and
heads, and aggregate it back onto the 4x4 grid of JPEG blocks the image was
divided into. Renders original digit + attention heatmap side by side for
both examples.
"""
import os
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

import sys
sys.path.insert(0, os.path.join('data', 'mnist_jpeg'))
from jpeg_encode import image_to_int_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-mnist-jpeg')
parser.add_argument('--block_shape', type=int, nargs=2, default=(8, 8))
parser.add_argument('--out_png', default=None)
args = parser.parse_args()
BH, BW = args.block_shape
GRID_ROWS, GRID_COLS = 32 // BH, 32 // BW

OUT_DIR = args.out_dir
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

ckpt = torch.load(os.path.join(OUT_DIR, 'ckpt.pt'), map_location=device)
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
        attn_maps.append(att[0].cpu().numpy())  # (nh, t, t)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, attn.n_embd)
        y = attn.resid_dropout(attn.c_proj(y))
        x = x + y
        x = x + block.mlp(block.ln_2(x))
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits, attn_maps  # attn_maps: list of (n_layer) arrays (n_head, t, t)

def build_example(img):
    seq, blocks = image_to_int_sequence(np.array(img), quality=5, block_shape=(BH, BW), return_blocks=True)
    seq_str_parts = [str(v) for v in seq]
    # character offset (within the comma-joined number string) for each block
    char_ranges = []
    pos = 0
    for br, bc, start, end in blocks:
        c0 = pos
        for i in range(start, end):
            pos += len(seq_str_parts[i])
            if i != end - 1:
                pos += 1  # comma
        c1 = pos
        char_ranges.append((br, bc, c0, c1))
        if end < len(seq):
            pos += 1  # comma joining to next block's first number
    numbers_str = ','.join(seq_str_parts)
    prefix = "Q: Which number is JPEG encoded as "
    prompt = f"{prefix}{numbers_str}?\nA:"
    return prompt, prefix, char_ranges

def classify_and_attend(img, true_label):
    prompt, prefix, char_ranges = build_example(img)
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    logits, attn_maps = forward_with_attn(ids)
    last = logits[0, -1]
    pred_id = int(torch.argmax(last))
    pred_char = itos[pred_id]
    # greedy-decode a few more chars to get the full predicted word
    gen_ids = ids
    pred_word = ''
    for _ in range(8):
        with torch.no_grad():
            lg, _ = forward_with_attn(gen_ids)
        nid = int(torch.argmax(lg[0, -1]))
        ch = itos[nid]
        if not ch.isalpha() and pred_word:
            break
        if ch.isalpha():
            pred_word += ch
        gen_ids = torch.cat([gen_ids, torch.tensor([[nid]], device=device)], dim=1)
    # average attention (all layers, all heads) FROM the last prompt position (the query
    # that predicts the answer) OVER all prior positions
    n_layer = len(attn_maps)
    t = ids.shape[1]
    avg_attn_from_last = np.mean([attn_maps[l][:, -1, :] for l in range(n_layer)], axis=(0, 1))  # (t,)
    offset = len(prefix)
    grid = np.zeros((GRID_ROWS, GRID_COLS))
    for br, bc, c0, c1 in char_ranges:
        lo, hi = offset + c0, offset + c1
        grid[br, bc] = avg_attn_from_last[lo:hi].sum() if hi > lo else 0.0
    return pred_word, grid

print("loading MNIST test set...")
test_ds = torchvision.datasets.MNIST(root=os.path.join('data', 'mnist_jpeg', 'raw'), train=False, download=True)

correct_ex, wrong_ex = None, None
for i in range(len(test_ds)):
    img, label = test_ds[i]
    true_name = NAMES[label]
    pred_word, grid = classify_and_attend(img, label)
    hit = pred_word.lower() == true_name.lower()
    print(f"idx {i}: true={true_name} pred={pred_word} {'OK' if hit else 'X'}")
    if hit and correct_ex is None:
        correct_ex = (np.array(img), true_name, pred_word, grid)
    if not hit and wrong_ex is None:
        wrong_ex = (np.array(img), true_name, pred_word, grid)
    if correct_ex is not None and wrong_ex is not None:
        break

# shared color scale across both examples so brightness is comparable between rows
vmax = max(correct_ex[3].max(), wrong_ex[3].max())

fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
for row, (img, true_name, pred_word, grid) in enumerate([correct_ex, wrong_ex]):
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title(f"true={true_name} pred={pred_word}")
    axes[row, 0].axis('off')
    grid_up = np.kron(grid, np.ones((BH, BW)))  # upsample grid -> 32x32 to align with image blocks
    axes[row, 1].imshow(img, cmap='gray')
    im = axes[row, 1].imshow(grid_up[:28, :28], cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
    axes[row, 1].set_title('attention (from answer token)')
    axes[row, 1].axis('off')
    cbar = fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
    cbar.set_label('attention weight (black=low, white=high)', fontsize=8)
axes[0, 0].set_ylabel('CORRECT', fontsize=12)
axes[1, 0].set_ylabel('INCORRECT', fontsize=12)

out_path = args.out_png or os.path.join('data', 'mnist_jpeg', f'attention_comparison_{GRID_ROWS}x{GRID_COLS}.png')
plt.tight_layout()
plt.savefig(out_path, dpi=120)
print(f"saved {out_path}")
