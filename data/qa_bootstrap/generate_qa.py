"""
Automate QA-pair extraction from any book/text file, using a bootstrap
model (FLAN-T5-large, zero-shot -- no fine-tuning needed since it's already
instruction-tuned for exactly this kind of task).

Two-step generation per passage, which is more reliable than asking for a
Q+A pair in one shot:
  1. "Generate a question about this passage" -> question (sampled several
     times per passage for diversity, then deduped)
  2. "Answer the question using only this passage" -> answer, grounded in
     the same passage text so it doesn't hallucinate from FLAN's own prior
     knowledge

Output: Q:/A: blocks in the same format used throughout this project, so
the result drops straight into prepare.py for nanoGPT training. Works on
any input text file, not just Three Little Pigs -- this is meant to be the
general "ingest a book into QA training data" pipeline.

Usage: python generate_qa.py --book ../three_little_pigs/story.txt --out generated_qa.txt
"""
import os
import re
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('--book', required=True)
parser.add_argument('--out', default='generated_qa.txt')
parser.add_argument('--questions_per_passage', type=int, default=3)
parser.add_argument('--model', default='google/flan-t5-large')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"loading {args.model}...")
tokenizer = T5Tokenizer.from_pretrained(args.model)
model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
model.eval()

def run(prompt, do_sample=False, num_return_sequences=1, max_new_tokens=64):
    ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=0.9 if do_sample else None,
            top_p=0.92 if do_sample else None,
            num_return_sequences=num_return_sequences,
        )
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]

BAD_ANSWER_RE = re.compile(
    r'^\(?[ivxlcdm]{1,4}\)?\.?$'      # roman numeral option: (iii), iv.
    r'|^\(?[a-h]\)\.?$'               # letter option: (a), b).
    r'|^\(?\d{1,2}\)\.?$',            # numbered option: (2), 1).
    re.IGNORECASE)

def is_bad_answer(answer):
    return bool(BAD_ANSWER_RE.match(answer.strip()))

META_QUESTION_RE = re.compile(
    r'\b(first|second|third|last|next)\s+(word|sentence|object|line|paragraph)\b'
    r'|\bhow many words\b',
    re.IGNORECASE)

def is_meta_question(question):
    """Reject questions about surface position/structure of the text rather
    than its content -- FLAN generates these often and they're useless."""
    return bool(META_QUESTION_RE.search(question))

def split_into_passages(text, min_len=80):
    paras = [p.strip().replace('\n', ' ') for p in text.split('\n\n')]
    paras = [re.sub(r'\s+', ' ', p) for p in paras if p.strip()]
    # merge very short paragraphs (e.g. dialogue lines) with the next one
    merged = []
    buf = ''
    for p in paras:
        buf = (buf + ' ' + p).strip()
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ''
    if buf:
        if merged:
            merged[-1] += ' ' + buf
        else:
            merged.append(buf)
    return merged

with open(args.book, encoding='utf-8') as f:
    book_text = f.read()
passages = split_into_passages(book_text)
print(f"{len(passages)} passages")

blocks = []
seen_questions = set()
answer_counts = {}
MAX_REPEATS_PER_ANSWER = 2  # cap how many times the exact same answer can recur

for i, passage in enumerate(passages):
    q_prompt = f"Generate a question that can be answered using this passage:\n\n{passage}"
    questions = run(q_prompt, do_sample=True, num_return_sequences=args.questions_per_passage, max_new_tokens=32)
    for q in questions:
        q = q.strip()
        key = q.lower()
        if not q or key in seen_questions or len(q) < 8 or is_meta_question(q):
            continue
        seen_questions.add(key)
        a_prompt = f"Answer the question using only this passage.\nPassage: {passage}\nQuestion: {q}\nAnswer:"
        answer = run(a_prompt, do_sample=False, max_new_tokens=48)[0].strip()
        if not answer or len(answer) < 2 or is_bad_answer(answer):
            continue
        akey = answer.lower()
        if answer_counts.get(akey, 0) >= MAX_REPEATS_PER_ANSWER:
            continue
        answer_counts[akey] = answer_counts.get(akey, 0) + 1
        blocks.append(f"Q: {q}\nA: {answer}")
    print(f"passage {i+1}/{len(passages)}: {len(blocks)} QA pairs so far")

out_path = os.path.join(os.path.dirname(__file__), args.out)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(blocks) + '\n')
print(f"saved {len(blocks)} QA pairs to {out_path}")
