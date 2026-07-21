"""
Score a trained mnist_jpeg checkpoint's classification accuracy: for each
"Q: ...?\nA: <Name>" block, prompt up to "A:" and check whether the first
generated word matches the true digit name.

Usage: python eval_mnist.py --out_dir=out-mnist-jpeg --qa_file=data/mnist_jpeg/test_qa.txt
"""
import os
import pickle
import argparse
import re
import torch
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', required=True)
parser.add_argument('--qa_file', required=True)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

with open(args.qa_file, 'r') as f:
    raw_blocks = [b.strip() for b in f.read().strip().split('\n\n') if b.strip()]
if args.limit:
    raw_blocks = raw_blocks[:args.limit]

pairs = []
for b in raw_blocks:
    q_line, a_line = b.split('\n', 1)
    true_name = a_line[len('A: '):].strip()
    pairs.append((q_line, true_name))

ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=args.device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
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
confusion = {}
for q_line, true_name in pairs:
    prompt = f"{q_line}\nA:"
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=args.device)[None, ...]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=16, temperature=0.1, top_k=1)
    gen = decode(out[0].tolist())[len(prompt):]
    m = re.search(r'[A-Za-z]+', gen)
    pred_name = m.group(0) if m else '?'
    hit = pred_name.lower() == true_name.lower()
    correct += hit
    confusion[true_name] = confusion.get(true_name, [0, 0])
    confusion[true_name][1] += 1
    confusion[true_name][0] += hit

print(f"accuracy: {correct}/{len(pairs)} ({correct/len(pairs)*100:.1f}%)")
print("per-class:")
for name in sorted(confusion):
    c, n = confusion[name]
    print(f"  {name:10s}: {c}/{n} ({c/n*100:.0f}%)")
