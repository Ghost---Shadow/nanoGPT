# same architecture/budget as train_mnist_jpeg.py, but on the 64-block
# (4x4 pixel blocks instead of 8x8) encoding
out_dir = 'out-mnist-jpeg-64blocks'
eval_interval = 200
eval_iters = 100
log_interval = 20

always_save_checkpoint = False

wandb_log = False

dataset = 'mnist_jpeg_64blocks'
gradient_accumulation_steps = 1
batch_size = 24
block_size = 512

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
