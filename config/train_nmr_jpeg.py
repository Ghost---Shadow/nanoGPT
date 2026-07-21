# classify simulated NMR spectra (6 compounds) from a JPEG-DCT integer
# sequence of their spectrogram image, same architecture/budget as mnist_jpeg
out_dir = 'out-nmr-jpeg'
eval_interval = 200
eval_iters = 100
log_interval = 20

always_save_checkpoint = False

wandb_log = False

dataset = 'nmr_jpeg'
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
