"""
Mechanically turn a sequential story into random-access QA pairs by pairing
each line with the line that follows it:

    Context: <line i>
    Question: What happened next?
    Answer: <line i+1>

This is a dumb, purely structural transform (no NLP, no summarization) that
converts "you must read start to finish" narrative into independent lookup
pairs, the same shape as data/three_little_pigs/qa_train.txt.

Run directly to preview a few pairs without writing any files:
    python line_qa.py
"""
import os


def story_to_qa(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    blocks = []
    for a, b in zip(lines, lines[1:]):
        blocks.append(f"Context: {a}\nQuestion: What happened next?\nAnswer: {b}")
    return blocks


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'story.txt')) as f:
        text = f.read()
    blocks = story_to_qa(text)
    print(f"{len(blocks)} line-QA pairs generated from the story")
    print("--- first ---")
    print(blocks[0])
    print("--- last ---")
    print(blocks[-1])
