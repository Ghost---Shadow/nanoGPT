"""
Prepare a minimal question-answering dataset for character-level modeling.

No raw story text is included anymore. Instead, everything is QA-shaped:
  1. line_qa.py mechanically slices the story into "Context: line i /
     Question: What happened next? / Answer: line i+1" pairs (random-access
     framing of what was sequential narrative).
  2. qa_train.txt is the hand-written QA pairs (qa_holdout.txt is
     deliberately excluded so it can test generalization later).
These two sources are mixed 1:1 (by block count, smaller side repeated to
match) and shuffled together so every training batch sees both kinds.
Saves train.bin, val.bin and meta.pkl, mirroring data/shakespeare_char/prepare.py.
"""
import os
import sys
import pickle
import random
import argparse
import numpy as np
from line_qa import story_to_qa

parser = argparse.ArgumentParser()
parser.add_argument('--qa_train', default='qa_train.txt', help='filename (within this dir) of hand-written QA pairs to train on')
args = parser.parse_args()

data_dir = os.path.dirname(__file__)
random.seed(1337)

with open(os.path.join(data_dir, 'story.txt'), 'r') as f:
    story_text = f.read()
with open(os.path.join(data_dir, args.qa_train), 'r') as f:
    qa_text = f.read()

line_qa_blocks = story_to_qa(story_text)
qa_blocks = [b.strip() for b in qa_text.strip().split('\n\n') if b.strip()]

# balance 1:1 by block count: repeat the smaller source to match the larger,
# without discarding anything from either
n = max(len(line_qa_blocks), len(qa_blocks))
def repeat_to(blocks, n):
    return [blocks[i % len(blocks)] for i in range(n)]
blocks = repeat_to(line_qa_blocks, n) + repeat_to(qa_blocks, n)
random.shuffle(blocks)

data = ('\n\n'.join(blocks)) + '\n'

print(f"line-QA blocks: {len(line_qa_blocks)}, hand-written QA blocks: {len(qa_blocks)}, mixed 1:1 -> {len(blocks)} total blocks")
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

n_chars = len(data)
train_data = data[:int(n_chars * 0.9)]
val_data = data[int(n_chars * 0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
