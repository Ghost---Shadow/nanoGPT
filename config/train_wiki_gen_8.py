# overfit ONE BATCH of 8 representative Wikipedia articles (contrastive
# seed/command study -- with n=1 the model has nothing to contrast Seed/
# Command against, so this tests whether it learns to actually use them)
out_dir = 'out-wiki-gen-8'
eval_interval = 100
eval_iters = 20
log_interval = 50

always_save_checkpoint = True

wandb_log = False

dataset = 'wiki_gen_8'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 4096

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.0  # deliberately overfitting

learning_rate = 1e-3
max_iters = 4000
lr_decay_iters = 4000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cuda'
compile = False
