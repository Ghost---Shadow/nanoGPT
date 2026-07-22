"""
Exhaustive tic-tac-toe (state -> best move) dataset. Since the whole game
tree is tiny (4520 reachable non-terminal states), we enumerate ALL of them
rather than sampling from self-play games -- full coverage, exact minimax
labels, no noise. A held-out split of states (never seen during training)
tests genuine generalization: has the model learned real tic-tac-toe
strategy, or just memorized specific boards?

    State: XOX.O.X..
    Player: O
    Move: 5

When a board has multiple equally-optimal moves, one is picked at random
(seeded) as the training label, but ALL optimal moves are recorded
separately (all_moves.pkl) so evaluation can credit any correct answer,
not just the arbitrarily chosen one.
"""
import os
import sys
import pickle
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tic_tac_toe import enumerate_states, best_moves

parser = argparse.ArgumentParser()
parser.add_argument('--test_frac', type=float, default=0.1)
args = parser.parse_args()

data_dir = os.path.dirname(__file__)
random.seed(1337)

print("enumerating all reachable tic-tac-toe states...")
states = enumerate_states()  # board -> player_to_move
print(f"{len(states)} states")

items = list(states.items())
random.shuffle(items)

all_optimal = {}  # board -> list of optimal moves (for lenient eval later)
blocks = []
for board, player in items:
    moves = best_moves(board, player)
    all_optimal[board] = moves
    chosen = random.choice(moves)
    blocks.append(f"State: {board}\nPlayer: {player}\nMove: {chosen}")

n_test = int(len(blocks) * args.test_frac)
test_blocks = blocks[:n_test]
train_blocks = blocks[n_test:]
print(f"train: {len(train_blocks)}, held-out test: {len(test_blocks)}")

with open(os.path.join(data_dir, 'test_qa.txt'), 'w') as f:
    f.write('\n\n'.join(test_blocks) + '\n')
with open(os.path.join(data_dir, 'all_optimal_moves.pkl'), 'wb') as f:
    pickle.dump(all_optimal, f)

data = ('\n\n'.join(train_blocks)) + '\n'
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

n_chars = len(data)
train_data = data[:int(n_chars * 0.9)]
val_data = data[int(n_chars * 0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
