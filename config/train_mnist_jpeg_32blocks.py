# same architecture/budget as train_mnist_jpeg.py, but on the 32-block
# (4x8 pixel blocks instead of 8x8) encoding, to test whether doubling the
# spatial resolution of the JPEG-style representation changes accuracy
out_dir = 'out-mnist-jpeg-32blocks'
eval_interval = 200
eval_iters = 100
log_interval = 20

always_save_checkpoint = False

wandb_log = False

dataset = 'mnist_jpeg_32blocks'
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
