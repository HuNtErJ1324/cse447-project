#!/usr/bin/env python3
"""Augment existing n-gram model with new multilingual data without retraining from scratch."""
import sys
import os
sys.path.insert(0, 'src')
from myprogram import NgramModel, WordNgramModel

# Load existing models
print("Loading existing n-gram model...")
ngram = NgramModel.load("work/ngram_model.pkl")
print(f"  Loaded. max_n={ngram.max_n}, min_n={ngram.min_n}")

word_ngram = WordNgramModel.load("work/word_ngram_model.pkl")
print(f"  Loaded word n-gram model.")

# Load new data
new_data = []
for f in ["data/multilingual_generated.txt"]:
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    new_data.append(line)
        print(f"  Loaded {f}")

print(f"  Total new lines: {len(new_data)}")

# Train on new data (this adds to existing counts)
print("Augmenting n-gram model...")
for text in new_data:
    for n in range(ngram.min_n, ngram.max_n + 1):
        for i in range(len(text) - n):
            context = text[i:i + n - 1]
            next_char = text[i + n - 1]
            ngram.counts[n][context][next_char] += 1

print("Augmenting word n-gram model...")
word_ngram.train(new_data)

# Save
print("Saving augmented models...")
ngram.save("work/ngram_model.pkl")
word_ngram.save("work/word_ngram_model.pkl")
print("Done!")
