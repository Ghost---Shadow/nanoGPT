# reverse-direction MNIST: generate the JPEG-DCT sequence from a
# "Seed: <id>\nCommand: Draw an image of <Name>\nAnswer:" prompt
out_dir = 'out-mnist-gen'
eval_interval = 200
eval_iters = 100
log_interval = 20

always_save_checkpoint = False

wandb_log = False

dataset = 'mnist_gen'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512

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
