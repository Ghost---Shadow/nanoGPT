# reverse-direction generation, same idea as train_mnist_gen.py, but for
# react-icons (Font Awesome) SVG icons instead of MNIST digits
out_dir = 'out-svg-gen'
eval_interval = 200
eval_iters = 100
log_interval = 20

always_save_checkpoint = False

wandb_log = False

dataset = 'svg_gen'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 1024

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1

learning_rate = 1e-3
max_iters = 8000
lr_decay_iters = 8000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 200

device = 'cuda'
compile = False
