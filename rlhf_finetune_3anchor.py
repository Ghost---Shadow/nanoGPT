"""
3-gold RLHF: for each prompt, sample K completions from the current
policy, rank them by reward.
  - Positive = highest-reward sample
  - Negative = lowest-reward sample
  - Gold   = the real training text (ground-truth Wikipedia article),
               truncated to the same length as the rollouts
Loss (Bradley-Terry / DPO-style pairwise, using the current policy's own
logprobs -- no separate reference model needed since the gold IS the
regularizer):

    L = -logsigmoid(logp(positive) - logp(gold))
        -logsigmoid(logp(gold)   - logp(negative))

This creates a tug-of-war on the gold's own probability: the first term
wants it low (so positive beats it more easily), the second wants it high
(so it beats negative) -- which structurally resists the reward hacking we
saw with plain REINFORCE, because collapsing to gibberish no longer helps
once the model must also keep real coherent text more probable than the
worst sample it can produce.
"""
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--base_out_dir', default='out-wiki-gen-8')
parser.add_argument('--out_dir', default='out-wiki-gen-8-3gold')
parser.add_argument('--n_steps', type=int, default=150)
parser.add_argument('--rollout_tokens', type=int, default=150)
parser.add_argument('--k_samples', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=40)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load(os.path.join(args.base_out_dir, 'ckpt.pt'), map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = GPT(gptconf)
sd = ckpt['model']
for k, v in list(sd.items()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.to(device)

with open(os.path.join('data', ckpt['config']['dataset'], 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join(itos[i] for i in l)

with open('data/wiki_gen_8/the_examples.txt', encoding='utf-8') as f:
    text = f.read()
examples = []  # (prompt, true_answer)
for b in text.split('\n\n'):
    seed_line, cmd_line, ans_line = b.split('\n', 2)
    examples.append((f"{seed_line}\n{cmd_line}\nAnswer: ", ans_line[len('Answer: '):]))

def compute_train_avg_word_len():
    words = ' '.join(a for _, a in examples).split()
    return sum(len(w) for w in words) / len(words)

TARGET_WORD_LEN = compute_train_avg_word_len() * 2
print(f"target word length: {TARGET_WORD_LEN:.3f}")

def reward_fn(gen_text):
    words = gen_text.split()
    if not words:
        return -TARGET_WORD_LEN
    avg = sum(len(w) for w in words) / len(words)
    return -abs(avg - TARGET_WORD_LEN)

def seq_logprob(model, prompt_ids, completion_ids):
    """Sum of log p(token | prefix) under the CURRENT policy, over completion_ids only."""
    full = prompt_ids + completion_ids
    seq = torch.tensor(full, dtype=torch.long, device=device)[None, ...]
    inp, tgt = seq[:, :-1], seq[:, 1:]
    logits, _ = model(inp, tgt)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    plen = len(prompt_ids)
    completion_logprobs = token_logprobs[:, plen - 1:]
    return completion_logprobs.sum() / max(1, completion_logprobs.numel())  # length-normalized

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

for step in range(args.n_steps):
    model.eval()
    batch_pos, batch_neg, batch_gold, batch_prompt = [], [], [], []
    rewards_log = []
    for prompt, true_answer in examples:
        p_ids = encode(prompt)
        ids = torch.tensor(p_ids, dtype=torch.long, device=device)[None, ...]
        samples = []
        with torch.no_grad():
            for _ in range(args.k_samples):
                out = model.generate(ids, max_new_tokens=args.rollout_tokens,
                                      temperature=args.temperature, top_k=args.top_k)
                comp_ids = out[0, len(p_ids):].tolist()
                gen_text = decode(comp_ids)
                samples.append((comp_ids, reward_fn(gen_text)))
        samples.sort(key=lambda x: x[1])
        neg_ids = samples[0][0]
        pos_ids = samples[-1][0]
        gold_ids = encode(true_answer[:args.rollout_tokens])
        rewards_log.append(samples[-1][1])
        batch_pos.append(pos_ids); batch_neg.append(neg_ids)
        batch_gold.append(gold_ids); batch_prompt.append(p_ids)

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for p_ids, pos_ids, neg_ids, gold_ids in zip(batch_prompt, batch_pos, batch_neg, batch_gold):
        logp_pos = seq_logprob(model, p_ids, pos_ids)
        logp_neg = seq_logprob(model, p_ids, neg_ids)
        logp_gold = seq_logprob(model, p_ids, gold_ids)
        loss = -F.logsigmoid(logp_pos - logp_gold) - F.logsigmoid(logp_gold - logp_neg)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()

    if step % 10 == 0 or step == args.n_steps - 1:
        print(f"step {step}: best_reward_mean={np.mean(rewards_log):.3f} loss={total_loss/len(examples):.4f}")

os.makedirs(args.out_dir, exist_ok=True)
torch.save({
    'model': model.state_dict(),
    'model_args': ckpt['model_args'],
    'config': ckpt['config'],
    'iter_num': ckpt.get('iter_num', 0),
    'best_val_loss': ckpt.get('best_val_loss', 0),
}, os.path.join(args.out_dir, 'ckpt.pt'))
print(f"saved 3-gold RLHF checkpoint to {args.out_dir}")
