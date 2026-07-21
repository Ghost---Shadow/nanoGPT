# classify MNIST digits from a JPEG-DCT integer sequence, phrased as QA
out_dir = 'out-mnist-jpeg'
eval_interval = 200
eval_iters = 100
log_interval = 20

always_save_checkpoint = False

wandb_log = False

dataset = 'mnist_jpeg'
gradient_accumulation_steps = 1
batch_size = 24
block_size = 512  # covers the longest observed example (479 chars) with margin

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1

learning_rate = 1e-3
max_iters = 6000
lr_decay_iters = 6000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 200

device = 'cuda'
compile = False
