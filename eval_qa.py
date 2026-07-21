"""
Load a trained checkpoint and score it against a QA file (blocks separated by
blank lines, each block "Q: ...\nA: ..."). Scores by character-level
similarity between the generated answer and the true answer (difflib), which
avoids hand-picking keywords per question.

Usage:
    python eval_qa.py --out_dir=out-three-little-pigs --qa_file=data/three_little_pigs/qa_train.txt
    python eval_qa.py --out_dir=out-three-little-pigs --qa_file=data/three_little_pigs/qa_holdout.txt
"""
import os
import pickle
import argparse
import difflib
import torch
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', required=True)
parser.add_argument('--qa_file', required=True)
parser.add_argument('--threshold', type=float, default=0.6)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

with open(args.qa_file, 'r') as f:
    raw_blocks = [b.strip() for b in f.read().strip().split('\n\n') if b.strip()]
pairs = []
for b in raw_blocks:
    q_line, a_line = b.split('\n', 1)
    pairs.append((q_line[len('Q: '):].strip(), a_line[len('A: '):].strip()))

ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=args.device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(args.device)

meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

correct = 0
for question, true_answer in pairs:
    prompt = f"Q: {question}\nA:"
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=args.device)[None, ...]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, temperature=0.3, top_k=5)
    answer = decode(out[0].tolist())[len(prompt):]
    answer = answer.split('\n\n')[0].split('Q:')[0].strip()
    sim = difflib.SequenceMatcher(None, answer.lower(), true_answer.lower()).ratio()
    hit = sim >= args.threshold
    correct += hit
    print(f"[{'OK ' if hit else 'X  '} {sim:.2f}] Q: {question}\n         got: {answer}\n         exp: {true_answer}")

print(f"\nscore: {correct}/{len(pairs)} ({correct/len(pairs)*100:.0f}%)  [{args.qa_file}]")
