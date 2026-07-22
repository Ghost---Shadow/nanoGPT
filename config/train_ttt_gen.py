# learn tic-tac-toe (state -> best move) from an exhaustive minimax-labeled
# dataset of all 4520 reachable board states
out_dir = 'out-ttt-gen'
eval_interval = 200
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False

dataset = 'ttt_gen'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64  # sequences are at most 34 chars

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 200

device = 'cuda'
compile = False
