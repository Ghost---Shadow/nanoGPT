# sanity check: overfit ONE (md5-seed, wikipedia-article) example as hard as
# possible, before attempting a full seeded-document-generation run
out_dir = 'out-wiki-gen-overfit'
eval_interval = 100
eval_iters = 20
log_interval = 50

always_save_checkpoint = True  # we want the final (most overfit) weights

wandb_log = False

dataset = 'wiki_gen_overfit'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 4096  # comfortably covers the single 3833-char example

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.0  # no regularization -- deliberately overfitting

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cuda'
compile = False
