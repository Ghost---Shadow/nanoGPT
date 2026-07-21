"""
LoRA-fine-tune google/flan-t5-large (783M params, largest FLAN-T5 under 1B)
on the same three-little-pigs QA data used for the nanoGPT experiments:
mechanized line-QA (line_qa.py) + hand-written qa_train.txt + qa_why_train.txt.
Holdout files are NOT included, so they remain valid for testing generalization.

LoRA (via peft) keeps the base model frozen and trains only small adapter
matrices, so this fits in 8GB of VRAM without full fine-tuning's optimizer
memory blowup.

Usage: python hf_finetune.py --out_dir=out-flan-t5-large-lora --epochs=40
"""
import os
import sys
import argparse
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'three_little_pigs'))
from line_qa import story_to_qa

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-flan-t5-large-lora')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()

data_dir = os.path.join('data', 'three_little_pigs')

with open(os.path.join(data_dir, 'story.txt')) as f:
    story_text = f.read()
with open(os.path.join(data_dir, 'qa_train.txt')) as f:
    qa_what = f.read()
with open(os.path.join(data_dir, 'qa_why_train.txt')) as f:
    qa_why = f.read()

def parse_qa(text):
    blocks = [b.strip() for b in text.strip().split('\n\n') if b.strip()]
    pairs = []
    for b in blocks:
        q_line, a_line = b.split('\n', 1)
        pairs.append((q_line[len('Q: '):].strip(), a_line[len('A: '):].strip()))
    return pairs

examples = []
for ctx_line, next_line in zip(story_text.splitlines(), story_text.splitlines()[1:]):
    ctx_line, next_line = ctx_line.strip(), next_line.strip()
    if ctx_line and next_line:
        examples.append((f"Context: {ctx_line}\nQuestion: What happened next?", next_line))
for q, a in parse_qa(qa_what):
    examples.append((f"Question: {q}", a))
for q, a in parse_qa(qa_why):
    examples.append((f"Question: {q}", a))

print(f"{len(examples)} fine-tuning examples")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['q', 'v'],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)

class QADataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def collate(batch):
    inputs, targets = zip(*batch)
    enc = tokenizer(list(inputs), padding=True, truncation=True, return_tensors='pt')
    dec = tokenizer(text_target=list(targets), padding=True, truncation=True, return_tensors='pt')
    labels = dec['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    return enc['input_ids'], enc['attention_mask'], labels

from torch.utils.data import DataLoader
loader = DataLoader(QADataset(examples), batch_size=args.batch_size, shuffle=True, collate_fn=collate)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
model.train()
for epoch in range(args.epochs):
    total_loss = 0.0
    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch}: loss {total_loss/len(loader):.4f}")

os.makedirs(args.out_dir, exist_ok=True)
model.save_pretrained(args.out_dir)
tokenizer.save_pretrained(args.out_dir)
print(f"saved LoRA adapter to {args.out_dir}")
