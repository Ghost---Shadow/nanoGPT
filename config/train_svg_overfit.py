# sanity check: overfit ONE (seed, icon) example as hard as possible
out_dir = 'out-svg-overfit'
eval_interval = 100
eval_iters = 20
log_interval = 50

always_save_checkpoint = True  # we WANT the final (most overfit) weights here

wandb_log = False

dataset = 'svg_overfit'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 400  # comfortably covers the single 339-char example

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.0  # no regularization -- we're deliberately trying to overfit

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cuda'
compile = False
