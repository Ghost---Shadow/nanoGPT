"""
Minimal RLHF (REINFORCE with a moving-average baseline, no PPO/KL penalty --
kept deliberately simple) on top of the out-wiki-gen-8 checkpoint. Reward =
average word length of the generated continuation, i.e. "reward long words".

For each step: sample a stochastic rollout from each of the 8 known
Seed/Command prompts, score it with the reward function, then do a policy
gradient update: loss = -sum(advantage * logprob(sampled token)) over the
completion region only (the prompt itself is never optimized).

nanoGPT's forward(idx, targets) only returns full per-position logits when
targets is given (otherwise it only computes the last position, an
inference optimization) -- so we pass targets to get full logits, then
build our own masked, reward-weighted NLL instead of using the built-in
uniform-weighted loss.
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
parser.add_argument('--out_dir', default='out-wiki-gen-8-rlhf')
parser.add_argument('--n_steps', type=int, default=150)
parser.add_argument('--rollout_tokens', type=int, default=150)
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
prompts = []
for b in text.split('\n\n'):
    seed_line, cmd_line, _ = b.split('\n', 2)
    prompts.append(f"{seed_line}\n{cmd_line}\nAnswer: ")

def compute_train_avg_word_len():
    with open('data/wiki_gen_8/the_examples.txt', encoding='utf-8') as f:
        text = f.read()
    bodies = []
    for b in text.split('\n\n'):
        parts = b.split('\n', 2)
        bodies.append(parts[2][len('Answer: '):])
    words = ' '.join(bodies).split()
    return sum(len(w) for w in words) / len(words)

TRAIN_AVG_WORD_LEN = compute_train_avg_word_len()
TARGET_WORD_LEN = TRAIN_AVG_WORD_LEN * 2
print(f"training data avg word length: {TRAIN_AVG_WORD_LEN:.3f}  ->  target: {TARGET_WORD_LEN:.3f}")

def reward_fn(gen_text):
    words = gen_text.split()
    if not words:
        return -TARGET_WORD_LEN  # no words at all is maximally bad, not neutral
    avg = sum(len(w) for w in words) / len(words)
    return -abs(avg - TARGET_WORD_LEN)  # peaks at 0 when avg == target, penalized either direction

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
baseline = None
baseline_momentum = 0.9

for step in range(args.n_steps):
    model.eval()
    rollouts = []  # (full_ids, prompt_len, reward)
    for p in prompts:
        p_ids = encode(p)
        ids = torch.tensor(p_ids, dtype=torch.long, device=device)[None, ...]
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.rollout_tokens,
                                  temperature=args.temperature, top_k=args.top_k)
        gen_text = decode(out[0].tolist())[len(p_ids):]
        r = reward_fn(gen_text)
        rollouts.append((out[0], len(p_ids), r))

    rewards = np.array([r for _, _, r in rollouts])
    if baseline is None:
        baseline = rewards.mean()
    else:
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * rewards.mean()
    advantages = rewards - baseline
    std = advantages.std()
    if std > 1e-6:
        advantages = advantages / std

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for (full_ids, plen, r), adv in zip(rollouts, advantages):
        seq = full_ids[None, :]
        inp, tgt = seq[:, :-1], seq[:, 1:]
        logits, _ = model(inp, tgt)  # targets forces full per-position logits
        logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
        # only optimize the completion region, not the prompt
        mask = torch.zeros_like(token_logprobs)
        mask[:, plen - 1:] = 1.0
        seq_logprob = (token_logprobs * mask).sum()
        loss = -adv * seq_logprob / mask.sum().clamp(min=1)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()

    if step % 10 == 0 or step == args.n_steps - 1:
        print(f"step {step}: mean_reward={rewards.mean():.3f} baseline={baseline:.3f} loss={total_loss/len(rollouts):.4f}")

os.makedirs(args.out_dir, exist_ok=True)
torch.save({
    'model': model.state_dict(),
    'model_args': ckpt['model_args'],
    'config': ckpt['config'],
    'iter_num': ckpt.get('iter_num', 0),
    'best_val_loss': ckpt.get('best_val_loss', 0),
}, os.path.join(args.out_dir, 'ckpt.pt'))
print(f"saved RLHF-finetuned checkpoint to {args.out_dir}")
