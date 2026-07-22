"""
Classical tic-tac-toe: board representation, win-checking, and an exact
minimax solver (memoized -- the whole game tree is tiny, ~5478 reachable
states, so minimax is exhaustive/perfect play, not an approximation).

Board: a 9-char string, row-major, '.'=empty, 'X'/'O'=occupied.
Move: an int 0-8 (index into the board string).
"""
from functools import lru_cache

WINS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]

def winner(board):
    for a, b, c in WINS:
        if board[a] != '.' and board[a] == board[b] == board[c]:
            return board[a]
    return None

def is_full(board):
    return '.' not in board

def legal_moves(board):
    return [i for i, c in enumerate(board) if c == '.']

def apply_move(board, pos, player):
    return board[:pos] + player + board[pos + 1:]

def other(player):
    return 'O' if player == 'X' else 'X'

@lru_cache(maxsize=None)
def _minimax(board, player):
    """Returns (best_value, tuple_of_best_moves) from player's perspective.
    value: 1 = player wins, 0 = draw, -1 = player loses (with perfect play)."""
    w = winner(board)
    if w is not None:
        return (1 if w == player else -1), ()
    if is_full(board):
        return 0, ()

    best_val = -2
    best_moves = []
    for m in legal_moves(board):
        nxt = apply_move(board, m, player)
        opp_val, _ = _minimax(nxt, other(player))
        val = -opp_val  # opponent's good outcome is our bad outcome
        if val > best_val:
            best_val = val
            best_moves = [m]
        elif val == best_val:
            best_moves.append(m)
    return best_val, tuple(best_moves)

def best_moves(board, player):
    """All minimax-optimal moves for player on this board."""
    _, moves = _minimax(board, player)
    return list(moves)

def enumerate_states():
    """BFS/DFS over the whole game tree from the empty board. Returns a dict
    mapping non-terminal board -> player_to_move, for every reachable state."""
    start = '.' * 9
    seen = {}
    stack = [(start, 'X')]
    while stack:
        board, player = stack.pop()
        if board in seen:
            continue
        if winner(board) is not None or is_full(board):
            continue
        seen[board] = player
        for m in legal_moves(board):
            stack.append((apply_move(board, m, player), other(player)))
    return seen


if __name__ == '__main__':
    states = enumerate_states()
    print(f"{len(states)} reachable non-terminal states")
    # sanity check: perfect play from the empty board should be a draw
    val, moves = _minimax('.' * 9, 'X')
    print(f"value of the empty board for X (perfect play): {val} (0=draw, expected for tic-tac-toe)")
    print(f"optimal first moves for X: {moves}")
