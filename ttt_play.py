"""
Play full games of nanoGPT vs the classical minimax AI (and optionally vs a
random-move opponent), alternating who goes first, tracking win/draw/loss.
Since tic-tac-toe is a solved draw with perfect play, the headline question
against the perfect opponent is: does nanoGPT ever manage to at least draw,
or does it lose games it shouldn't? Against a random opponent, the question
is whether it can actually execute wins when given the chance.

If nanoGPT ever proposes an illegal move (occupied cell, unparseable
output), that's logged explicitly rather than silently patched -- it's
informative about whether the model has learned the game's basic legality
constraints, not just move quality.
"""
import os
import sys
import pickle
import random
import argparse
import torch
from model import GPTConfig, GPT

sys.path.insert(0, os.path.join('data', 'ttt_gen'))
from tic_tac_toe import winner, is_full, legal_moves, apply_move, other, best_moves

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-ttt-gen')
parser.add_argument('--n_games', type=int, default=50)
parser.add_argument('--opponent', choices=['minimax', 'random'], default='minimax')
parser.add_argument('--temperature', type=float, default=0.3)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(os.path.join(args.out_dir, 'ckpt.pt'), map_location=device)
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

def gpt_move(board, player):
    prompt = f"State: {board}\nPlayer: {player}\nMove:"
    ids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=3, temperature=args.temperature, top_k=5)
    gen = decode(out[0].tolist())[len(prompt):]
    import re
    m = re.search(r'-?\d+', gen)
    if m is None:
        return None
    pos = int(m.group())
    if pos not in legal_moves(board):
        return None
    return pos

def opponent_move(board, player):
    if args.opponent == 'minimax':
        return random.choice(best_moves(board, player))
    else:
        return random.choice(legal_moves(board))

random.seed(42)
results = {'gpt_win': 0, 'draw': 0, 'gpt_loss': 0, 'gpt_illegal_move': 0}
illegal_examples = []

for g in range(args.n_games):
    gpt_player = 'X' if g % 2 == 0 else 'O'
    board = '.' * 9
    player = 'X'
    while True:
        w = winner(board)
        if w is not None:
            results['gpt_win' if w == gpt_player else 'gpt_loss'] += 1
            break
        if is_full(board):
            results['draw'] += 1
            break
        if player == gpt_player:
            pos = gpt_move(board, player)
            if pos is None:
                results['gpt_illegal_move'] += 1
                illegal_examples.append((board, player))
                break
        else:
            pos = opponent_move(board, player)
        board = apply_move(board, pos, player)
        player = other(player)

print(f"nanoGPT (as X on even games, O on odd) vs {args.opponent} opponent, {args.n_games} games:")
for k, v in results.items():
    print(f"  {k}: {v}")
if illegal_examples:
    print("example illegal-move states:")
    for board, player in illegal_examples[:5]:
        print(f"  State: {board}  Player: {player}")
