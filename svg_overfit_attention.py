"""
Visualize attention for the overfit-single-example sanity check: at several
points during generation (right after "Answer:", partway through, near the
end), what does the model attend to over the whole sequence so far? Shows
whether it's anchored on the Seed/Command conditioning tokens throughout, or
mostly doing local self-referential copying once it's deep into the answer.
"""
import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = 'out-svg-overfit'

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

with open('data/svg_overfit/the_example.txt') as f:
    full_text = f.read()
prompt_prefix = full_text.split('Answer: ')[0] + 'Answer: '
full_ids = encode(full_text)

# checkpoints: right after "Answer: ", 1/3 through the answer, 2/3 through, at the end
answer_start = len(encode(prompt_prefix))
n = len(full_ids)
checkpoints = [answer_start, answer_start + (n - answer_start) // 3,
               answer_start + 2 * (n - answer_start) // 3, n - 1]
labels = ['right after "Answer:"', '~1/3 through answer', '~2/3 through answer', 'near the end']

fig, axes = plt.subplots(len(checkpoints), 1, figsize=(12, 2.2 * len(checkpoints)))
for ax, cp, label in zip(axes, checkpoints, labels):
    ids = torch.tensor(full_ids[:cp + 1], dtype=torch.long, device=device)[None, ...]
    _, attn_maps = forward_with_attn(ids)
    n_layer = len(attn_maps)
    avg_attn = np.mean([attn_maps[l][:, -1, :] for l in range(n_layer)], axis=(0, 1))  # (t,)

    seed_end = len('Seed: 0\n')
    cmd_end = len('Seed: 0\nCommand: Draw an icon of Heart\n')
    colors = np.where(np.arange(len(avg_attn)) < seed_end, 0,
              np.where(np.arange(len(avg_attn)) < cmd_end, 1, 2))
    ax.bar(range(len(avg_attn)), avg_attn, width=1.0,
           color=['tab:blue' if c == 0 else 'tab:orange' if c == 1 else 'tab:gray' for c in colors])
    ax.set_title(f'attention from position {cp} ({label}), query = char {decode([full_ids[cp]])!r}', fontsize=9)
    ax.set_xlim(0, len(full_ids))
    ax.set_ylabel('attn weight', fontsize=8)

axes[-1].set_xlabel('character position in sequence (blue=Seed, orange=Command, gray=Answer-so-far)')
plt.tight_layout()
plt.savefig('data/svg_overfit/attention_over_generation.png', dpi=120)
print('saved data/svg_overfit/attention_over_generation.png')
