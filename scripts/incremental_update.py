#!/usr/bin/env python3
"""Incrementally update the n-gram model with new training lines."""
import pickle
import sys
import os
from collections import defaultdict

def main():
    model_path = sys.argv[1]  # work/ngram_model.pkl
    data_files = sys.argv[2:]  # new text files
    repeats = 40

    # Load existing model
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    min_n = data['min_n']
    max_n = data['max_n']
    counts = data['counts']

    # Convert fast_format entries back to dicts for updating
    raw_counts = {}
    for n in range(min_n, max_n + 1):
        raw_counts[n] = {}
        for ctx, entry in counts[n].items():
            if isinstance(entry, tuple):
                chars_str, counts_tup, total = entry
                d = {}
                for i, ch in enumerate(chars_str):
                    d[ch] = counts_tup[i]
                raw_counts[n][ctx] = d
            else:
                raw_counts[n][ctx] = dict(entry)

    # Read new lines
    lines = []
    for fp in data_files:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    
    print(f"Adding {len(lines)} lines x {repeats} repeats = {len(lines)*repeats} total")

    # Train on new lines
    for _ in range(repeats):
        for line in lines:
            for n in range(min_n, max_n + 1):
                ctx_len = n - 1
                for i in range(len(line) - ctx_len):
                    ctx = line[i:i+ctx_len]
                    next_char = line[i+ctx_len]
                    if n not in raw_counts:
                        raw_counts[n] = {}
                    if ctx not in raw_counts[n]:
                        raw_counts[n][ctx] = {}
                    raw_counts[n][ctx][next_char] = raw_counts[n][ctx].get(next_char, 0) + 1

    # Prune with aggressive thresholds
    prune_thresholds = {2: 1, 3: 1, 4: 5, 5: 25, 6: 35, 7: 45}
    total_contexts = 0
    for n in range(min_n, max_n + 1):
        thresh = prune_thresholds.get(n, 1)
        to_del = []
        for ctx, char_counts in raw_counts[n].items():
            total = sum(char_counts.values())
            if total < thresh:
                to_del.append(ctx)
        for ctx in to_del:
            del raw_counts[n][ctx]
        total_contexts += len(raw_counts[n])
        print(f"  Order {n}: {len(raw_counts[n])} contexts (pruned {len(to_del)})")

    # Convert back to fast format
    fast_counts = {}
    for n in range(min_n, max_n + 1):
        fast_counts[n] = {}
        for ctx, char_dict in raw_counts[n].items():
            sorted_chars = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
            top = sorted_chars[:5]
            chars_str = ''.join(ch for ch, _ in top)
            counts_tup = tuple(cnt for _, cnt in top)
            total = sum(char_dict.values())
            fast_counts[n][ctx] = (chars_str, counts_tup, total)

    # Save
    out = {
        'trained': True,
        'min_n': min_n,
        'max_n': max_n,
        'counts': fast_counts,
        'fast_format': True,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    sz = os.path.getsize(model_path)
    print(f"Saved: {total_contexts} contexts, {sz/1024/1024:.1f}MB")

if __name__ == '__main__':
    main()
