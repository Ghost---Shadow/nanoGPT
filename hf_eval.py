"""
Evaluate google/flan-t5-large (optionally with a LoRA adapter) against a QA
file, same block format and difflib-similarity scoring as eval_qa.py.

Usage:
    python hf_eval.py --qa_file=data/three_little_pigs/qa_holdout.txt                     # zero-shot base model
    python hf_eval.py --qa_file=data/three_little_pigs/qa_holdout.txt --adapter=out-flan-t5-large-lora  # fine-tuned
"""
import os
import argparse
import difflib
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument('--qa_file', required=True)
parser.add_argument('--adapter', default=None, help='path to a saved LoRA adapter; omit for zero-shot base model')
parser.add_argument('--threshold', type=float, default=0.6)
parser.add_argument('--prefix', default='Question: ', help='text prepended to each question before feeding the model')
args = parser.parse_args()

with open(args.qa_file, 'r') as f:
    raw_blocks = [b.strip() for b in f.read().strip().split('\n\n') if b.strip()]
pairs = []
for b in raw_blocks:
    q_line, a_line = b.split('\n', 1)
    pairs.append((q_line[len('Q: '):].strip(), a_line[len('A: '):].strip()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok_path = args.adapter if args.adapter else 'google/flan-t5-large'
tokenizer = T5Tokenizer.from_pretrained(tok_path)
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
model.to(device)

correct = 0
for question, true_answer in pairs:
    input_text = args.prefix + question
    ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=64, num_beams=1, do_sample=False)
    answer = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    sim = difflib.SequenceMatcher(None, answer.lower(), true_answer.lower()).ratio()
    hit = sim >= args.threshold
    correct += hit
    print(f"[{'OK ' if hit else 'X  '} {sim:.2f}] Q: {question}\n         got: {answer}\n         exp: {true_answer}")

print(f"\nscore: {correct}/{len(pairs)} ({correct/len(pairs)*100:.0f}%)  [{args.qa_file}]  adapter={args.adapter}")
