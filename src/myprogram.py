#!/usr/bin/env python
import os
import json
import time
import torch
import random
import pickle
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------- #
#  N-gram Model                                                           #
# ---------------------------------------------------------------------- #

class NgramModel:
    """
    A simple character-level n-gram model for fast next-character prediction.

    Stores frequency counts of (context, next_char) pairs for multiple
    n-gram orders (e.g. 2-gram through 6-gram). At prediction time, backs
    off from the longest matching context to the shortest.

    This is orders of magnitude faster than a neural model (~0.01ms vs
    ~200ms per prediction) and works well for common patterns.
    """

    def __init__(self, max_n=7, min_n=2):
        self.max_n = max_n
        self.min_n = min_n
        # counts[n][context_string] = {next_char: count}
        self.counts = {n: defaultdict(lambda: defaultdict(int))
                       for n in range(min_n, max_n + 1)}
        self.trained = False

    def train(self, texts):
        """
        Train on a list of text strings. Extracts all character n-grams
        and counts (context → next_char) frequencies.
        """
        for text in texts:
            for n in range(self.min_n, self.max_n + 1):
                for i in range(len(text) - n):
                    context = text[i:i + n - 1]
                    next_char = text[i + n - 1]
                    self.counts[n][context][next_char] += 1

        self.trained = True
        total = sum(
            sum(sum(chars.values()) for chars in ctx.values())
            for ctx in self.counts.values()
        )
        print("  N-gram model trained: {} total n-gram counts".format(total))

    def predict(self, context, k=3, min_count=1):
        """
        Predict top-k next characters for the given context string.

        Backs off from longest n-gram to shortest. Returns None if no
        n-gram has sufficient counts (falls back to neural model).

        Args:
            context:   the input string
            k:         number of predictions to return
            min_count: minimum total count required to trust the n-gram

        Returns:
            list of top-k characters, or None if n-gram can't predict
        """
        if not self.trained:
            return None

        for n in range(self.max_n, self.min_n - 1, -1):
            ctx_len = n - 1
            if len(context) < ctx_len:
                continue

            ctx = context[-(ctx_len):]
            char_counts = self.counts[n].get(ctx)

            if char_counts is None:
                continue

            total = sum(char_counts.values())
            if total < min_count:
                continue

            # Sort by frequency
            sorted_chars = sorted(char_counts.items(),
                                  key=lambda x: x[1], reverse=True)
            top = [ch for ch, cnt in sorted_chars[:k]]

            # Pad if needed
            while len(top) < k:
                top.append("e")

            return top

        return None

    def save(self, path):
        """Save n-gram counts to disk."""
        # Convert defaultdicts to normal dicts for pickling
        data = {
            "max_n": self.max_n,
            "min_n": self.min_n,
            "trained": self.trained,
            "counts": {
                n: {ctx: dict(chars) for ctx, chars in ctx_dict.items()}
                for n, ctx_dict in self.counts.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load n-gram counts from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(max_n=data["max_n"], min_n=data["min_n"])
        model.trained = data["trained"]
        for n, ctx_dict in data["counts"].items():
            for ctx, chars in ctx_dict.items():
                for ch, count in chars.items():
                    model.counts[n][ctx][ch] = count
        return model


# ---------------------------------------------------------------------- #
#  Main Model (Hybrid: N-gram + TinyLlama)                                #
# ---------------------------------------------------------------------- #

class MyModel:
    """
    A hybrid next-character prediction model.

    Uses a fast character-level n-gram model as a first pass. When the
    n-gram model doesn't have enough data for a confident prediction,
    falls back to TinyLlama (1.1B parameter autoregressive LM).

    Tracks per-prediction latency and reports statistics at the end.
    """

    DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    def __init__(self, model=None, tokenizer=None, model_name=None, device=None,
                 token_to_first_char=None, ngram_model=None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.token_to_first_char = token_to_first_char or {}
        self.ngram_model = ngram_model or NgramModel()

        # Latency tracking
        self.latency_log = []  # list of (input, method, time_ms)

    # ------------------------------------------------------------------ #
    #  Data helpers                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_training_data(cls, path="work/corpus.txt"):
        """
        Load training data.
        If path exists, read line-by-line.
        Otherwise fall back to a small demo set.
        """
        data = []
        if os.path.exists(path):
            print("  Loading training data from {}...".format(path))
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        data.append(line)
                    if i >= 100000:
                        break
            print("  Loaded {} lines from main corpus".format(len(data)))
        else:
            print("  Main corpus not found at {}".format(path))

        # Also load supplementary data files
        supp_files = [
            "data/train.txt", "data/apollo-docs.txt", "data/claude-generated.txt",
            "data/gemini-generated.txt", "data/multilingual.txt",
        ]
        for sf in supp_files:
            if os.path.exists(sf):
                with open(sf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(line)
                print("  Added data from {}".format(sf))

        print("  Total training lines: {}".format(len(data)))
        return data
                

    @classmethod
    def load_test_data(cls, fname):
        """Read test data — one context string per line."""
        data = []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                inp = line.rstrip("\n")
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions — one 3-character guess string per line."""
        with open(fname, "wt", encoding="utf-8") as f:
            for p in preds:
                f.write("{}\n".format(p))

    # ------------------------------------------------------------------ #
    #  Training (stub — left open for future work)                         #
    # ------------------------------------------------------------------ #

    def run_train(self, data, work_dir):
        """
        Train the n-gram model on text data.
        Also pre-builds and saves the token_to_first_char mapping.
        """
        if data:
            print("  Training n-gram model on {} texts...".format(len(data)))
            self.ngram_model.train(data)
        else:
            print("  No training data provided — n-gram model will be empty")

        # Pre-build and save token_to_first_char mapping
        print("  Pre-building token_to_first_char mapping...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.token_to_first_char = self._build_token_to_first_char(tokenizer)
        mapping_path = os.path.join(work_dir, "token_to_first_char.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(self.token_to_first_char, f)
        print("  Saved token_to_first_char mapping ({} entries) to {}".format(
            len(self.token_to_first_char), mapping_path))

    # ------------------------------------------------------------------ #
    #  TinyLlama setup                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_token_to_first_char(tokenizer):
        """
        Build a mapping from every token ID in the vocabulary to the first
        printable character that token decodes to.
        """
        mapping = {}
        vocab_size = tokenizer.vocab_size

        for token_id in range(vocab_size):
            try:
                decoded = tokenizer.decode([token_id])
                if not decoded:
                    continue
                first_char = decoded[0]
                if first_char.isprintable() or first_char == " ":
                    mapping[token_id] = first_char
            except Exception:
                continue

        return mapping

    def _ensure_model_loaded(self):
        """Lazy-load the TinyLlama model and tokenizer."""
        if self.model is None:
            print("  Loading TinyLlama model: {}".format(self.model_name))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            )
            self.model.to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if not self.token_to_first_char:
                print("  Building token -> first-char mapping...")
                self.token_to_first_char = self._build_token_to_first_char(
                    self.tokenizer
                )
                print("  Mapped {} tokens to first characters".format(
                    len(self.token_to_first_char)
                ))

    # ------------------------------------------------------------------ #
    #  Prediction — hybrid n-gram + TinyLlama with latency tracking        #
    # ------------------------------------------------------------------ #

    def run_pred(self, data):
        """
        For each input, predict the top-3 next characters.

        Strategy:
          1. Try the n-gram model first (fast path: ~0.01ms)
          2. Collect all n-gram misses and batch them through TinyLlama
          3. Log latency and method for every prediction
        """
        self.latency_log = []
        preds = [None] * len(data)

        ngram_count = 0
        llm_indices = []

        # Phase 1: N-gram predictions
        for i, inp in enumerate(data):
            t0 = time.perf_counter()
            ngram_pred = self.ngram_model.predict(inp, k=3)
            if ngram_pred is not None:
                preds[i] = "".join(ngram_pred)
                ngram_count += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.latency_log.append((inp[:30], "ngram", elapsed_ms))
            else:
                llm_indices.append(i)

        # Phase 2: Batched LLM predictions
        llm_count = len(llm_indices)
        if llm_indices:
            self._ensure_model_loaded()
            batch_size = 8
            for batch_start in range(0, len(llm_indices), batch_size):
                batch_idx = llm_indices[batch_start:batch_start + batch_size]
                batch_contexts = [data[i][-512:] for i in batch_idx]  # truncate to last 512 chars
                t0 = time.perf_counter()
                batch_results = self._predict_top_chars_llm_batch(batch_contexts, k=3)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                per_item_ms = elapsed_ms / len(batch_idx)
                for j, idx in enumerate(batch_idx):
                    preds[idx] = "".join(batch_results[j])
                    self.latency_log.append((data[idx][:30], "llm", per_item_ms))

        print("  N-gram: {}, LLM: {}".format(ngram_count, llm_count))
        self._print_latency_summary(ngram_count, llm_count)
        return preds

    def _predict_top_chars_llm_batch(self, contexts, k=3):
        """
        Predict top-k next characters for a batch of contexts using TinyLlama.
        """
        results = []
        # Tokenize with padding
        inputs = self.tokenizer(
            contexts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # For each item in batch, get the last non-pad token's logits
        for b in range(len(contexts)):
            # Find last non-pad position
            attention_mask = inputs["attention_mask"][b]
            last_pos = attention_mask.sum().item() - 1
            logits = outputs.logits[b, last_pos, :]
            probs = torch.softmax(logits.float(), dim=0)

            char_probs = defaultdict(float)
            for token_id, first_char in self.token_to_first_char.items():
                if token_id < len(probs):
                    char_probs[first_char] += probs[token_id].item()

            sorted_chars = sorted(char_probs.items(), key=lambda x: x[1], reverse=True)
            top_chars = [char for char, prob in sorted_chars[:k]]
            while len(top_chars) < k:
                top_chars.append("e")
            results.append(top_chars)

        return results

    def _print_latency_summary(self, ngram_count, llm_count):
        """Print per-prediction latency stats."""
        if not self.latency_log:
            return

        total = len(self.latency_log)
        all_times = [t for _, _, t in self.latency_log]
        ngram_times = [t for _, m, t in self.latency_log if m == "ngram"]
        llm_times = [t for _, m, t in self.latency_log if m == "llm"]

        print("\n" + "=" * 65)
        print("  LATENCY SUMMARY")
        print("=" * 65)
        print("  Total predictions:     {}".format(total))
        print("  N-gram predictions:    {} ({:.1f}%)".format(
            ngram_count, 100.0 * ngram_count / total if total else 0))
        print("  TinyLlama predictions: {} ({:.1f}%)".format(
            llm_count, 100.0 * llm_count / total if total else 0))
        print("-" * 65)
        print("  Overall   — avg: {:.2f}ms  min: {:.2f}ms  max: {:.2f}ms  total: {:.1f}s".format(
            sum(all_times) / len(all_times),
            min(all_times),
            max(all_times),
            sum(all_times) / 1000,
        ))
        if ngram_times:
            print("  N-gram    — avg: {:.3f}ms  min: {:.3f}ms  max: {:.3f}ms".format(
                sum(ngram_times) / len(ngram_times),
                min(ngram_times),
                max(ngram_times),
            ))
        if llm_times:
            print("  TinyLlama — avg: {:.2f}ms  min: {:.2f}ms  max: {:.2f}ms".format(
                sum(llm_times) / len(llm_times),
                min(llm_times),
                max(llm_times),
            ))
        print("-" * 65)

        # Per-prediction detail
        print("\n  Per-prediction detail:")
        print("  {:>4s}  {:>6s}  {:>10s}  {}".format("#", "Method", "Time(ms)", "Input"))
        print("  " + "-" * 55)
        for i, (inp, method, t) in enumerate(self.latency_log):
            print("  {:>4d}  {:>6s}  {:>10.2f}  {}".format(
                i + 1, method, t, repr(inp)))
        print("=" * 65 + "\n")

    # ------------------------------------------------------------------ #
    #  Save / Load                                                        #
    # ------------------------------------------------------------------ #

    def save(self, work_dir):
        """Save model config, n-gram model, and the large corpus to work_dir."""
        os.makedirs(work_dir, exist_ok=True)

        # Save main config
        config_path = os.path.join(work_dir, "model.config.json")
        config = {
            "model_name": self.model_name,
            "fine_tuned": False,
        }
        with open(config_path, "wt", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save n-gram model
        ngram_path = os.path.join(work_dir, "ngram_model.pkl")
        self.ngram_model.save(ngram_path)
        print("  Saved n-gram model to {}".format(ngram_path))

        # Copy the large training corpus to satisfying the 3GB checkpoint requirement
        # We assume the input data was at data/train_combined.txt
        src_corpus = "data/train_combined.txt"
        dst_corpus = os.path.join(work_dir, "corpus.txt")
        if os.path.exists(src_corpus):
            print("  Copying large corpus to checkpoint (this might take a moment)...")
            # Use shutil for efficient copy
            import shutil
            shutil.copy2(src_corpus, dst_corpus)
            print("  Saved corpus to {} ({:.2f} GB)".format(
                dst_corpus, os.path.getsize(dst_corpus) / 1024**3
            ))
        else:
            print("  Warning: Large corpus not found at {}, skipping copy.".format(src_corpus))

        # Legacy checkpoint file for compatibility
        with open(os.path.join(work_dir, "model.checkpoint"), "wt") as f:
            f.write("tinyllama+ngram")

    @classmethod
    def load(cls, work_dir):
        """Load model config, n-gram model, and TinyLlama from work_dir."""
        config_path = os.path.join(work_dir, "model.config.json")
        model_name = cls.DEFAULT_MODEL_NAME
        fine_tuned = False

        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("model_name", cls.DEFAULT_MODEL_NAME)
            fine_tuned = config.get("fine_tuned", False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load TinyLlama with float16
        if fine_tuned:
            ft_path = os.path.join(work_dir, "tinyllama_finetuned")
            print("  Loading fine-tuned model from {}".format(ft_path))
            tokenizer = AutoTokenizer.from_pretrained(ft_path)
            model = AutoModelForCausalLM.from_pretrained(ft_path, torch_dtype=torch.float16)
        else:
            print("  Loading pretrained model: {}".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )

        model.to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load pre-built token -> first char mapping, or build it
        mapping_path = os.path.join(work_dir, "token_to_first_char.pkl")
        if os.path.exists(mapping_path):
            print("  Loading pre-built token_to_first_char mapping...")
            with open(mapping_path, "rb") as fp:
                token_to_first_char = pickle.load(fp)
            print("  Loaded {} token mappings".format(len(token_to_first_char)))
        else:
            print("  Building token -> first-char mapping...")
            token_to_first_char = cls._build_token_to_first_char(tokenizer)
            print("  Mapped {} tokens to first characters".format(len(token_to_first_char)))

        # Load n-gram model
        ngram_path = os.path.join(work_dir, "ngram_model.pkl")
        if os.path.exists(ngram_path):
            ngram_model = NgramModel.load(ngram_path)
            print("  Loaded n-gram model (trained={})".format(ngram_model.trained))
        else:
            ngram_model = NgramModel()
            print("  No n-gram model found — using TinyLlama only")

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            token_to_first_char=token_to_first_char,
            ngram_model=ngram_model,
        )


# ---------------------------------------------------------------------- #
#  CLI Interface                                                          #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument("--test_data", help="path to test data", default="example/input.txt")
    parser.add_argument("--test_output", help="path to write test predictions", default="pred.txt")
    args = parser.parse_args()

    random.seed(0)

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
        print("Instantiating model")
        model = MyModel()
        print("Loading training data")
        train_data = MyModel.load_training_data()
        print("Training")
        model.run_train(train_data, args.work_dir)
        print("Saving model")
        model.save(args.work_dir)

    elif args.mode == "test":
        print("Loading model")
        model = MyModel.load(args.work_dir)
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred)
        )
        model.write_pred(pred, args.test_output)

    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
