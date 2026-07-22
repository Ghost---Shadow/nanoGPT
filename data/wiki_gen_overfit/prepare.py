"""
Sanity check for seeded document generation, same idea as svg_overfit:
exactly ONE (seed, article) example, repeated back-to-back so train.bin
exceeds block_size. Seed = md5 hash of the article's own text (so it's a
reproducible, content-derived unique id, not an arbitrary counter):

    Seed: <md5 hex>
    Command: Write a Wikipedia page on Hot Potato ( video game )
    Answer: <full article text>

WikiText-103 has no exact "Potato" article; "Hot Potato ( video game )" is
the closest reasonably-sized match. Pulled directly from the HF `wikitext`
dataset (wikitext-103-v1), which stores the whole corpus as one line per
line of the original Wikipedia dump, with article titles marked as
" = Title = \n".
"""
import os
import re
import hashlib
import pickle
import numpy as np
from datasets import load_dataset

TITLE_LINE_IDX = 660164  # " = Hot Potato ( video game ) = "
N_REPEATS = 14  # needs the 10% val split alone to exceed block_size (4096)

data_dir = os.path.dirname(__file__)
title_re = re.compile(r'^ = ([^=].*[^=]) = \n?$')

print("loading wikitext-103...")
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
lines = ds['text']

def extract_article(start_idx):
    title = title_re.match(lines[start_idx]).group(1)
    body = []
    i = start_idx + 1
    while i < len(lines):
        if title_re.match(lines[i]):
            break
        body.append(lines[i])
        i += 1
    return title, ''.join(body).strip()

title, body = extract_article(TITLE_LINE_IDX)
seed = hashlib.md5(body.encode('utf-8')).hexdigest()
print(f"title={title!r}  article length={len(body)} chars  seed(md5)={seed}")

block = f"Seed: {seed}\nCommand: Write a Wikipedia page on {title}\nAnswer: {body}"
print(f"full example length: {len(block)} chars")

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

with open(os.path.join(data_dir, 'the_example.txt'), 'w', encoding='utf-8') as f:
    f.write(block)

print(f"vocab_size={vocab_size}, train tokens={len(train_ids)}, val tokens={len(val_ids)}")
