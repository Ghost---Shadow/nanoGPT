# tiny character-level model trained only on the Three Little Pigs story + Q&A
# minimum viable question-answering experiment

out_dir = 'out-three-little-pigs'
eval_interval = 100
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'three-little-pigs'
wandb_run_name = 'mini-qa-gpt'

dataset = 'three_little_pigs'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128 # dataset is tiny, no need for a huge context

# a very small GPT
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'cuda'
compile = False
