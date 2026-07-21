"""
Compare attention for the JPEG-DCT chem classifier vs the plain sparse-triple
chem classifier, on the SAME held-out molecules (re-derives the exact same
seeded test-molecule selection used by both prepare.py scripts, so we can
regenerate both encodings from the same SMILES/adjacency matrix and get an
apples-to-apples comparison). Shows: the raw adjacency matrix (ground truth),
the JPEG model's attention (aggregated per 4x4 image block, upsampled), and
the sparse model's attention (aggregated per matrix cell, natural 32x32
resolution -- no upsampling needed since there's no lossy block structure).
"""
import os
import sys
import math
import random
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from model import GPTConfig, GPT

sys.path.insert(0, os.path.join('data', 'mnist_jpeg'))
sys.path.insert(0, os.path.join('data', 'chem_jpeg'))
sys.path.insert(0, os.path.join('data', 'chem_sparse'))
from jpeg_encode import image_to_int_sequence
from mol_graph import smiles_to_adjacency
from sparse_encode import adjacency_to_sparse_string, sparse_string_to_ranges

parser = argparse.ArgumentParser()
parser.add_argument('--jpeg_out_dir', default='out-chem-jpeg')
parser.add_argument('--sparse_out_dir', default='out-chem-sparse')
parser.add_argument('--block_shape', type=int, nargs=2, default=(4, 4))
parser.add_argument('--quality', type=int, default=5)
parser.add_argument('--n_examples', type=int, default=2)
parser.add_argument('--train_per_class', type=int, default=200)
parser.add_argument('--test_per_class', type=int, default=30)
parser.add_argument('--out_png', default='data/chem_jpeg/attention_jpeg_vs_sparse.png')
args = parser.parse_args()
BH, BW = args.block_shape
GRID_ROWS, GRID_COLS = 32 // BH, 32 // BW

device = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_NAMES = {0: 'No', 1: 'Yes'}

def load_model(out_dir):
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    m = GPT(gptconf)
    sd = ckpt['model']
    for k, v in list(sd.items()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    m.load_state_dict(sd)
    m.eval().to(device)
    with open(os.path.join('data', ckpt['config']['dataset'], 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return m, meta['stoi'], meta['itos']

jpeg_model, jpeg_stoi, jpeg_itos = load_model(args.jpeg_out_dir)
sparse_model, sparse_stoi, sparse_itos = load_model(args.sparse_out_dir)

@torch.no_grad()
def forward_with_attn(model, idx):
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

def generate_word(model, stoi, itos, prompt, max_new_tokens=8):
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    gen_ids = ids
    word = ''
    for _ in range(max_new_tokens):
        lg, attn_maps = forward_with_attn(model, gen_ids)
        nid = int(torch.argmax(lg[0, -1]))
        ch = itos[nid]
        if not ch.isalpha() and word:
            break
        if ch.isalpha():
            word += ch
        gen_ids = torch.cat([gen_ids, torch.tensor([[nid]], device=device)], dim=1)
    _, attn_maps = forward_with_attn(model, ids)
    n_layer = len(attn_maps)
    avg_attn_from_last = np.mean([attn_maps[l][:, -1, :] for l in range(n_layer)], axis=(0, 1))
    return word, avg_attn_from_last

def jpeg_attn_grid(mat, avg_attn, offset):
    seq, blocks = image_to_int_sequence(mat, quality=args.quality, block_shape=(BH, BW), return_blocks=True)
    seq_str_parts = [str(v) for v in seq]
    grid = np.zeros((GRID_ROWS, GRID_COLS))
    pos = 0
    for br, bc, start, end in blocks:
        c0 = pos
        for i in range(start, end):
            pos += len(seq_str_parts[i])
            if i != end - 1:
                pos += 1
        c1 = pos
        lo, hi = offset + c0, offset + c1
        grid[br, bc] = avg_attn[lo:hi].sum() if hi > lo else 0.0
        if end < len(seq):
            pos += 1
    return grid

def sparse_attn_grid(mat, avg_attn, offset, s):
    ranges = sparse_string_to_ranges(s)
    grid = np.zeros((32, 32))
    for i, j, c0, c1 in ranges:
        lo, hi = offset + c0, offset + c1
        val = avg_attn[lo:hi].sum() if hi > lo else 0.0
        grid[i, j] += val
        grid[j, i] += val
    return grid

# re-derive the exact same seeded test-molecule selection as chem_jpeg/prepare.py
print("re-deriving the exact held-out molecule set (same seed as prepare.py)...")
random.seed(1337)
ds = load_dataset('scikit-fingerprints/MoleculeNet_BBBP', split='train')
idxs = list(range(len(ds)))
random.shuffle(idxs)
by_label = {0: [], 1: []}
for i in idxs:
    ex = ds[i]
    label = ex['label']
    if len(by_label[label]) >= args.train_per_class + args.test_per_class:
        continue
    mat = smiles_to_adjacency(ex['SMILES'])
    if mat is None:
        continue
    by_label[label].append((ex['SMILES'], label))
    if all(len(v) >= args.train_per_class + args.test_per_class for v in by_label.values()):
        break

test_examples = []
for label in (0, 1):
    test_examples.extend(by_label[label][args.train_per_class:args.train_per_class + args.test_per_class])
random.shuffle(test_examples)
test_examples = test_examples[:args.n_examples]

rows = []
for smiles, label in test_examples:
    true_name = LABEL_NAMES[label]
    mat = smiles_to_adjacency(smiles)

    jpeg_prefix = "Q: Does this molecule cross the blood-brain barrier? Its atom-bond adjacency matrix is JPEG encoded as "
    seq = image_to_int_sequence(mat, quality=args.quality, block_shape=(BH, BW))
    jpeg_prompt = f"{jpeg_prefix}{','.join(str(v) for v in seq)}?\nA:"
    jword, jattn = generate_word(jpeg_model, jpeg_stoi, jpeg_itos, jpeg_prompt)
    jgrid = jpeg_attn_grid(mat, jattn, len(jpeg_prefix))

    sparse_prefix = "Q: Does this molecule cross the blood-brain barrier? Its atom-bond adjacency matrix as sparse (row-col-value) triples is "
    sparse_str = adjacency_to_sparse_string(mat)
    sparse_prompt = f"{sparse_prefix}{sparse_str}?\nA:"
    sword, sattn = generate_word(sparse_model, sparse_stoi, sparse_itos, sparse_prompt)
    sgrid = sparse_attn_grid(mat, sattn, len(sparse_prefix), sparse_str)

    print(f"true={true_name}  JPEG pred={jword}  sparse pred={sword}")
    rows.append((smiles, true_name, mat, jword, jgrid, sword, sgrid))

fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.3 * len(rows)))
if len(rows) == 1:
    axes = axes[None, :]
for r, (smiles, true_name, mat, jword, jgrid, sword, sgrid) in enumerate(rows):
    row_vmax = max(jgrid.max(), sgrid.max())
    print(f"row {r}: jgrid min/max={jgrid.min():.4f}/{jgrid.max():.4f}  sgrid min/max={sgrid.min():.4f}/{sgrid.max():.4f}")

    axes[r, 0].imshow(mat, cmap='gray')
    axes[r, 0].set_title(f"adjacency matrix\ntrue={true_name}", fontsize=9)
    axes[r, 0].axis('off')

    jgrid_up = np.kron(jgrid, np.ones((BH, BW)))
    axes[r, 1].imshow(mat, cmap='gray')
    im1 = axes[r, 1].imshow(jgrid_up, cmap='hot', alpha=0.6, vmin=0, vmax=row_vmax)
    axes[r, 1].set_title(f"JPEG-DCT model\npred={jword}", fontsize=9)
    axes[r, 1].axis('off')
    fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

    axes[r, 2].imshow(mat, cmap='gray')
    im2 = axes[r, 2].imshow(sgrid, cmap='hot', alpha=0.6, vmin=0, vmax=row_vmax)
    axes[r, 2].set_title(f"sparse-triple model\npred={sword}", fontsize=9)
    axes[r, 2].axis('off')
    fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(args.out_png, dpi=120)
print(f"saved {args.out_png}")
