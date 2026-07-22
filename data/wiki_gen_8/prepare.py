"""
Same seeded-document-generation setup as wiki_gen_overfit, but with 8
distinct representative WikiText-103 articles instead of just 1 -- "one
batch" of contrastive examples, so the model actually has a reason to learn
that Seed/Command matter (with n=1 there's nothing to contrast against).
Each article is truncated to a fixed cap so block_size stays manageable and
all examples are comparably sized.

    Seed: <md5 of the (possibly truncated) article text>
    Command: Write a Wikipedia page on <title>
    Answer: <article text, truncated to MAX_CHARS>
"""
import os
import re
import hashlib
import pickle
import random
import numpy as np
from datasets import load_dataset

MAX_CHARS = 3200
N_REPEATS = 10  # so train/val splits both comfortably exceed block_size

# (title_line_index, title) -- picked for topical diversity + manageable length
TITLES = [
    (660164, 'Hot Potato ( video game )'),
    (805097, 'Cabbage'),
    (464, 'Nebraska Highway 88'),
    (4557, 'French cruiser Sully'),
    (5103, 'Utah State Route 61'),
    (7115, 'New York State Route 368'),
    (7132, 'M @-@ 122 ( Michigan highway )'),
    (8156, 'Ohio State Route 319'),
]

data_dir = os.path.dirname(__file__)
random.seed(1337)
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

blocks = []
for idx, expected_title in TITLES:
    title, body = extract_article(idx)
    assert title == expected_title, f"expected {expected_title!r} got {title!r}"
    body = body[:MAX_CHARS]
    seed = hashlib.md5(body.encode('utf-8')).hexdigest()
    blocks.append(f"Seed: {seed}\nCommand: Write a Wikipedia page on {title}\nAnswer: {body}")
    print(f"{title!r}: {len(body)} chars, seed={seed}")

with open(os.path.join(data_dir, 'the_examples.txt'), 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(blocks))

all_blocks = blocks * N_REPEATS
random.shuffle(all_blocks)
data = ('\n\n'.join(all_blocks)) + '\n'
print(f"{len(blocks)} articles x {N_REPEATS} repeats -> {len(data):,} chars total")

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

print(f"vocab_size={vocab_size}, train tokens={len(train_ids)}, val tokens={len(val_ids)}")
