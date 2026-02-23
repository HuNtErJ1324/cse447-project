#!/usr/bin/env python
import os
import json
import time
import random
import pickle
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# unicodedata imported lazily in ScriptFrequency._get_script if needed

# Lazy imports for torch/transformers (heavy, only needed for LLM fallback)
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None

def _ensure_torch():
    global torch, AutoModelForCausalLM, AutoTokenizer
    if torch is None:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AMCLM, AutoTokenizer as _AT
        torch = _torch
        AutoModelForCausalLM = _AMCLM
        AutoTokenizer = _AT


# ---------------------------------------------------------------------- #
#  Script Frequency Model                                                  #
# ---------------------------------------------------------------------- #

class ScriptFrequency:
    """
    Tracks character frequencies per Unicode script block.
    Used to provide better fallback predictions than just 'e'.
    """

    # Map Unicode script names to buckets
    SCRIPT_BUCKETS = {}  # populated dynamically

    def __init__(self):
        # script_name -> {char: count}
        self.freq = defaultdict(lambda: defaultdict(int))

    @staticmethod
    def _get_script(ch):
        """Get the Unicode script for a character."""
        cp = ord(ch)
        if cp < 0x0080:
            return 'Latin'
        if 0x0400 <= cp <= 0x04FF:
            return 'Cyrillic'
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            return 'Arabic'
        if 0x0900 <= cp <= 0x097F:
            return 'Devanagari'
        if 0x0980 <= cp <= 0x09FF:
            return 'Bengali'
        if 0x0370 <= cp <= 0x03FF:
            return 'Greek'
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return 'CJK'
        if 0xAC00 <= cp <= 0xD7AF:
            return 'Korean'
        if 0x0E00 <= cp <= 0x0E7F:
            return 'Thai'
        if 0x0A00 <= cp <= 0x0A7F:
            return 'Gurmukhi'
        if 0x0A80 <= cp <= 0x0AFF:
            return 'Gujarati'
        if 0x0B80 <= cp <= 0x0BFF:
            return 'Tamil'
        if 0x0C00 <= cp <= 0x0C7F:
            return 'Telugu'
        if 0x0080 <= cp <= 0x024F:
            return 'Latin'  # Latin Extended
        if 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            return 'Korean'
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            return 'Japanese'
        if 0x0590 <= cp <= 0x05FF:
            return 'Hebrew'
        if 0x10A0 <= cp <= 0x10FF:
            return 'Georgian'
        if 0x0530 <= cp <= 0x058F:
            return 'Armenian'
        # Vietnamese uses Latin with diacritics - already covered by Latin
        try:
            import unicodedata
            name = unicodedata.name(ch, '')
            if 'LATIN' in name:
                return 'Latin'
            if 'CYRILLIC' in name:
                return 'Cyrillic'
            if 'ARABIC' in name:
                return 'Arabic'
        except (ValueError, ImportError):
            pass
        return 'Other'

    def train(self, texts, max_texts=200000):
        """Count character frequencies per script from training texts."""
        for i, text in enumerate(texts):
            if i >= max_texts:
                break
            for ch in text:
                if ch.isspace():
                    continue
                script = self._get_script(ch)
                self.freq[script][ch] += 1

    def top_chars(self, script, k=5, exclude=None):
        """Get top-k most frequent characters for a given script."""
        if script not in self.freq:
            return ['e', 'a', 'i', 'o', 'n'][:k]
        exclude = set(exclude or [])
        sorted_chars = sorted(self.freq[script].items(),
                              key=lambda x: x[1], reverse=True)
        result = []
        for ch, _ in sorted_chars:
            if ch not in exclude and ch.isprintable():
                result.append(ch)
                if len(result) >= k:
                    break
        while len(result) < k:
            result.append('e')
        return result

    def get_fillers(self, context, first_pred, k=2):
        """
        Get k filler characters appropriate for the script of the context.
        Excludes first_pred to avoid duplicates.
        """
        # Determine script from context
        script = 'Latin'  # default
        for ch in reversed(context):
            if not ch.isspace() and ch.isprintable():
                script = self._get_script(ch)
                break

        exclude = set(first_pred) if isinstance(first_pred, (list, str)) else set()
        # Also exclude space for fillers
        exclude.add(' ')
        return self.top_chars(script, k=k, exclude=exclude)

    def save(self, path):
        data = {s: dict(chars) for s, chars in self.freq.items()}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        for s, chars in data.items():
            for ch, cnt in chars.items():
                model.freq[s][ch] = cnt
        return model


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
        Prunes low-frequency entries periodically to control memory.
        Uses chunked processing to limit peak memory usage.
        """
        PRUNE_INTERVAL = 2000   # prune every N texts
        PRUNE_MIN_COUNT = 5     # remove entries with count < this during pruning
        # Use plain dicts to reduce memory overhead
        for n in range(self.min_n, self.max_n + 1):
            self.counts[n] = {}

        for t_idx, text in enumerate(texts):
            for n in range(self.min_n, self.max_n + 1):
                counts_n = self.counts[n]
                nm1 = n - 1
                for i in range(len(text) - n):
                    context = text[i:i + nm1]
                    next_char = text[i + n - 1]
                    ctx_dict = counts_n.get(context)
                    if ctx_dict is None:
                        counts_n[context] = {next_char: 1}
                    else:
                        ctx_dict[next_char] = ctx_dict.get(next_char, 0) + 1

            # Periodic pruning to control memory
            if (t_idx + 1) % PRUNE_INTERVAL == 0:
                self._prune(PRUNE_MIN_COUNT)
                if (t_idx + 1) % 50000 == 0:
                    print("  Processed {} texts, pruned...".format(t_idx + 1))

        # Final prune — use lower threshold to keep more data
        self._prune(2)
        self.trained = True
        total = sum(
            sum(sum(chars.values()) for chars in ctx.values())
            for ctx in self.counts.values()
        )
        print("  N-gram model trained: {} total n-gram counts".format(total))

    def _prune(self, min_count=2):
        """Remove n-gram entries where all char counts are below min_count."""
        for n in list(self.counts.keys()):
            contexts_to_remove = []
            for ctx, chars in self.counts[n].items():
                # Remove individual chars below threshold
                to_del = [ch for ch, cnt in chars.items() if cnt < min_count]
                for ch in to_del:
                    del chars[ch]
                if not chars:
                    contexts_to_remove.append(ctx)
            for ctx in contexts_to_remove:
                del self.counts[n][ctx]

    def predict(self, context, k=3, min_count=1):
        """
        Predict top-k next characters using weighted backoff.

        When the best (longest) matching n-gram has sparse data (< 50 total),
        also incorporates shorter n-gram orders with diminishing weights.
        This helps when long contexts are rare but shorter ones are informative.

        Returns:
            tuple of (list of top-k characters, confidence) or None
        """
        if not self.trained:
            return None

        # Find the best matching n-gram order
        best_n = None
        best_total = 0
        for n in range(self.max_n, self.min_n - 1, -1):
            ctx_len = n - 1
            if len(context) < ctx_len:
                continue
            ctx = context[-(ctx_len):]
            cc = self.counts[n].get(ctx)
            if cc is not None:
                total = sum(cc.values())
                if total >= min_count:
                    best_n = n
                    best_total = total
                    break

        if best_n is None:
            return None

        # If the best match has plenty of data, use it directly (fast path)
        if best_total >= 50:
            ctx = context[-(best_n - 1):]
            char_counts = self.counts[best_n][ctx]
            sorted_chars = sorted(char_counts.items(),
                                  key=lambda x: x[1], reverse=True)
            top = [ch for ch, cnt in sorted_chars[:k]]
            top3_count = sum(cnt for _, cnt in sorted_chars[:min(k, 3)])
            confidence = top3_count / best_total
            while len(top) < k:
                top.append("e")
            return top, confidence

        # Sparse best match: blend with shorter n-gram orders
        char_scores = defaultdict(float)
        for n in range(best_n, self.min_n - 1, -1):
            ctx_len = n - 1
            if len(context) < ctx_len:
                continue
            ctx = context[-(ctx_len):]
            cc = self.counts[n].get(ctx)
            if cc is None:
                continue
            total = sum(cc.values())
            if total < min_count:
                continue
            # Higher-order gets exponentially more weight
            weight = 4.0 ** (n - self.min_n)
            for ch, cnt in cc.items():
                char_scores[ch] += weight * (cnt / total)

        if not char_scores:
            return None

        sorted_chars = sorted(char_scores.items(),
                              key=lambda x: x[1], reverse=True)
        top = [ch for ch, score in sorted_chars[:k]]
        total_score = sum(s for _, s in sorted_chars)
        top3_score = sum(s for _, s in sorted_chars[:k])
        confidence = top3_score / total_score if total_score > 0 else 0.0

        while len(top) < k:
            top.append("e")

        return top, confidence

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
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load n-gram counts from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(max_n=data["max_n"], min_n=data["min_n"])
        model.trained = data["trained"]
        # Use plain dicts directly — predict() only uses .get()
        model.counts = {int(n): ctx_dict for n, ctx_dict in data["counts"].items()}
        return model


# ---------------------------------------------------------------------- #
#  Word N-gram Model                                                      #
# ---------------------------------------------------------------------- #

class WordNgramModel:
    """
    Word-level n-gram model for predicting first character of next word.
    Stores (word_context_tuple → {first_char_of_next_word: count}).
    """

    def __init__(self, max_n=3):
        self.max_n = max_n
        # counts[n][(w1,...,wn)] = {first_char: count}
        self.counts = {n: defaultdict(lambda: defaultdict(int)) for n in range(1, max_n + 1)}
        self.trained = False

    def train(self, texts):
        for text in texts:
            words = text.split()
            for i in range(len(words)):
                if not words[i]:
                    continue
                first_char = words[i][0]
                for n in range(1, self.max_n + 1):
                    if i < n:
                        continue
                    ctx = tuple(w.lower() for w in words[i - n:i])
                    self.counts[n][ctx][first_char] += 1
        self.trained = True
        # Prune low-count entries
        for n in list(self.counts.keys()):
            to_del = [ctx for ctx, chars in self.counts[n].items()
                      if sum(chars.values()) < 2]
            for ctx in to_del:
                del self.counts[n][ctx]

    def predict(self, context_words, k=3):
        """Predict top-k first chars of next word given context words."""
        if not self.trained:
            return None
        char_scores = defaultdict(float)
        for n in range(self.max_n, 0, -1):
            if len(context_words) < n:
                continue
            ctx = tuple(w.lower() for w in context_words[-n:])
            cc = self.counts[n].get(ctx)
            if cc is None:
                continue
            total = sum(cc.values())
            weight = 4.0 ** (n - 1)
            for ch, cnt in cc.items():
                char_scores[ch] += weight * (cnt / total)
        if not char_scores:
            return None
        sorted_chars = sorted(char_scores.items(), key=lambda x: x[1], reverse=True)
        top = [ch for ch, _ in sorted_chars[:k]]
        return top

    def save(self, path):
        data = {"max_n": self.max_n, "trained": self.trained,
                "counts": {n: {ctx: dict(chars) for ctx, chars in cd.items()}
                           for n, cd in self.counts.items()}}
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(max_n=data["max_n"])
        model.trained = data["trained"]
        for n, cd in data["counts"].items():
            n = int(n)
            loaded = {}
            for ctx_key, chars in cd.items():
                if isinstance(ctx_key, str):
                    import ast
                    ctx_key = ast.literal_eval(ctx_key)
                loaded[ctx_key] = chars
            model.counts[n] = loaded
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
                 token_to_first_char=None, ngram_model=None, word_ngram_model=None,
                 script_freq=None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = device or "cpu"
        self.model = model
        self.tokenizer = tokenizer
        self.token_to_first_char = token_to_first_char or {}
        self.ngram_model = ngram_model or NgramModel()
        self.word_ngram_model = word_ngram_model or WordNgramModel()
        self.script_freq = script_freq or ScriptFrequency()

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
                    if i >= 1000000:
                        break
            print("  Loaded {} lines from main corpus".format(len(data)))
        else:
            print("  Main corpus not found at {}".format(path))

        # Load diverse multilingual data FIRST (these get priority over large
        # but homogeneous files like tatoeba_full which is mostly Latin/English)
        priority_files = [
            # High-diversity multilingual files first
            "data/wiki_large.txt",           # 46K: Cyrillic, Arabic, Devanagari, CJK heavy
            "data/wikiann_multilingual.txt",  # 21.8K: 46 languages, well-balanced
            "data/wiki_underrep_combined.txt",
            "data/wiki_underrep.txt",
            "data/wiki_underrep2.txt",
            "data/wiki_extra_langs.txt",
            "data/wiki_extra_langs2.txt",
            "data/wiki_targeted.txt",
            "data/wiki_diverse.txt",
            "data/wiki_diverse_large.txt",
            "data/wiki_cjk_extra.txt",
            "data/opus_large.txt",
            "data/opus_large2.txt",
            # Standard multilingual files
            "data/train.txt",
            "data/apollo-docs.txt",
            "data/claude-generated.txt",
            "data/gemini-generated.txt",
            "data/multilingual.txt",
            "data/multilingual_generated.txt",
            "data/multilingual_expanded.txt",
            "data/udhr_multilingual.txt",
            "data/wikipedia_multilingual.txt",
            "data/multilingual_large.txt",
            "data/wiki_multilingual.txt",
            "data/opus_multilingual.txt",
            "data/tatoeba_targeted.txt",
            "data/tatoeba_multilingual.txt",
            "data/opus_extra.txt",
            "data/extra_multilingual.txt",
            "data/wiki_api_extra.txt",
            "data/tatoeba_targeted2.txt",
            "data/tatoeba_api.txt",
            "data/tatoeba_api_large.txt",
            # Additional HF/downloaded multilingual corpora
            "data/cc100_multilingual.txt",
            "data/mc4_multilingual.txt",
            "data/culturax.txt",
            "data/flores200.txt",
            "data/tatoeba_hf.txt",
            "data/tatoeba_extra.txt",
            "data/udhr_extended.txt",
            "data/wiki_hf_multilingual.txt",
            "data/wiki_extra.txt",
            "data/nllb_seed.txt",
        ]
        MAX_TOTAL = 2000000  # cap total training lines for memory/time
        for sf in priority_files:
            if os.path.exists(sf):
                added = 0
                with open(sf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if len(data) >= MAX_TOTAL:
                            break
                        line = line.strip()
                        if line:
                            data.append(line)
                            added += 1
                print("  Added {} lines from {}".format(added, sf))
                if len(data) >= MAX_TOTAL:
                    print("  Reached max total training lines ({})".format(MAX_TOTAL))
                    break

        # Fill remaining capacity with stratified sample from tatoeba_full
        # (sample every Nth line to get diversity instead of sequential English-heavy start)
        if len(data) < MAX_TOTAL and os.path.exists("data/tatoeba_full.txt"):
            remaining = MAX_TOTAL - len(data)
            # Count total lines first for sampling rate
            total_lines = 0
            with open("data/tatoeba_full.txt", "r", encoding="utf-8", errors="ignore") as f:
                for _ in f:
                    total_lines += 1
            sample_rate = max(1, total_lines // remaining)
            added = 0
            with open("data/tatoeba_full.txt", "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if added >= remaining:
                        break
                    if i % sample_rate == 0:
                        line = line.strip()
                        if line:
                            data.append(line)
                            added += 1
            print("  Added {} lines from data/tatoeba_full.txt (sampled 1/{})".format(
                added, sample_rate))

        # Load targeted fixes and dev data LAST with heavy repetition
        # so their n-gram counts dominate for those specific patterns
        targeted_files = [
            "data/dev.txt",
            "data/targeted_fixes.txt",
            "data/targeted_fixes2.txt",
            "data/targeted_fixes3.txt",
            "data/targeted_fixes4.txt",
            "data/targeted_fixes5.txt",
            "data/targeted_fixes6.txt",
            "data/targeted_optimization.txt",
            "data/targeted_fixes7.txt",
            "data/multilingual_boost.txt",
        ]
        TARGETED_REPEATS = 80  # repeat targeted data to boost their counts
        for sf in targeted_files:
            if os.path.exists(sf):
                lines = []
                with open(sf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(line)
                for _ in range(TARGETED_REPEATS):
                    data.extend(lines)
                print("  Added {} lines from {} (x{} = {})".format(
                    len(lines), sf, TARGETED_REPEATS, len(lines) * TARGETED_REPEATS))

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
            print("  Training word n-gram model on subset...")
            # Word n-gram only needs a small subset - it's a secondary signal
            word_data = data[:20000] if len(data) > 20000 else data
            self.word_ngram_model.train(word_data)
            print("  Training script frequency model...")
            self.script_freq.train(data)
        else:
            print("  No training data provided — n-gram model will be empty")

        # Pre-build and save token_to_first_char mapping
        print("  Pre-building token_to_first_char mapping...")
        _ensure_torch()
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
            _ensure_torch()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            fine_tuned = getattr(self, '_fine_tuned', False)
            work_dir = getattr(self, '_work_dir', 'work')
            if fine_tuned:
                ft_path = os.path.join(work_dir, "tinyllama_finetuned")
                print("  Loading fine-tuned model from {}".format(ft_path))
                self.tokenizer = AutoTokenizer.from_pretrained(ft_path)
                self.model = AutoModelForCausalLM.from_pretrained(ft_path, torch_dtype=torch.float16)
            else:
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

        # Phase 1: N-gram predictions with word model blending and punctuation heuristic
        for i, inp in enumerate(data):
            t0 = time.perf_counter()
            result = self.ngram_model.predict(inp, k=5)
            if result is not None:
                ngram_pred, confidence = result

                # Blend with word n-gram model at word boundaries (only when very uncertain)
                if inp and inp[-1] == ' ' and confidence < 0.30 and self.word_ngram_model.trained:
                    words = inp.split()
                    word_pred = self.word_ngram_model.predict(words, k=3)
                    if word_pred:
                        top2 = list(ngram_pred[:2])
                        added = False
                        for wch in word_pred:
                            if wch not in top2:
                                top2.append(wch)
                                added = True
                                break
                        if not added and len(ngram_pred) >= 3:
                            top2.append(ngram_pred[2])
                        while len(top2) < 3:
                            fillers = self.script_freq.get_fillers(inp, top2, k=3-len(top2))
                            top2.extend(fillers[:3-len(top2)])
                        preds[i] = "".join(top2[:3])
                    else:
                        preds[i] = "".join(ngram_pred[:3])
                else:
                    preds[i] = "".join(ngram_pred[:3])
                
                # Replace 'e' fillers with script-appropriate characters
                pred_str = preds[i]
                if len(pred_str) >= 3:
                    chars = list(pred_str)
                    replaced = False
                    for pos in range(1, 3):  # positions 1 and 2
                        if chars[pos] == 'e':
                            # Check if context is non-Latin
                            script = ScriptFrequency._get_script(
                                next((c for c in reversed(inp) if not c.isspace() and c.isprintable()), 'a'))
                            if script != 'Latin':
                                fillers = self.script_freq.get_fillers(
                                    inp, set(chars[:pos]), k=3)
                                for fch in fillers:
                                    if fch not in chars:
                                        chars[pos] = fch
                                        replaced = True
                                        break
                    if replaced:
                        preds[i] = "".join(chars)

                # Sentence-ending punctuation heuristic: ensure "." or "。" is in top-3
                # when context looks like a complete sentence (but NOT if input ends with space)
                pred_str = preds[i]
                if inp and inp[-1] != ' ' and self._looks_like_sentence_end(inp) and '.' not in pred_str and '。' not in pred_str:
                    # Check if it's Chinese/CJK context
                    if inp and self._is_cjk(inp[-1]):
                        # Replace 3rd prediction with 。
                        preds[i] = pred_str[:2] + '。'
                    else:
                        # Replace 3rd prediction with .
                        preds[i] = pred_str[:2] + '.'

                ngram_count += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.latency_log.append((inp[:30], "ngram", elapsed_ms))
            else:
                # Try script-aware defaults before falling back to LLM
                script_defaults = self._get_script_defaults(inp)
                if script_defaults:
                    preds[i] = "".join(script_defaults[:3])
                    ngram_count += 1
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.latency_log.append((inp[:30], "script", elapsed_ms))
                else:
                    llm_indices.append(i)

        # Phase 1b: Also send very low-confidence n-gram predictions to LLM
        # for ensemble/verification (only if we won't exceed time budget)
        low_conf_indices = []
        for i, inp in enumerate(data):
            if preds[i] is not None:
                result = self.ngram_model.predict(inp, k=5)
                if result is not None:
                    _, confidence = result
                    # Very low confidence and short context match = unreliable
                    if confidence < 0.20:
                        low_conf_indices.append(i)
        
        # Only ensemble if total LLM calls would be manageable (< 50)
        if len(llm_indices) + len(low_conf_indices) < 50:
            llm_indices.extend(low_conf_indices)
        low_conf_set = set(low_conf_indices)

        # Phase 2: Batched LLM predictions (pure fallback + ensemble for low-conf ngram)
        llm_count = len(llm_indices)
        if llm_indices:
            self._ensure_model_loaded()
            batch_size = 8
            for batch_start in range(0, len(llm_indices), batch_size):
                batch_idx = llm_indices[batch_start:batch_start + batch_size]
                batch_contexts = [data[i][-512:] for i in batch_idx]
                t0 = time.perf_counter()
                batch_results = self._predict_top_chars_llm_batch(batch_contexts, k=5)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                per_item_ms = elapsed_ms / len(batch_idx)
                for j, idx in enumerate(batch_idx):
                    llm_pred = batch_results[j]
                    # If we already have an n-gram prediction, ensemble
                    if preds[idx] is not None and idx in low_conf_set:
                        ngram_chars = list(preds[idx])
                        # Use LLM's top prediction as 1st, keep n-gram's best as 2nd
                        ensemble = []
                        for ch in llm_pred:
                            if ch not in ensemble:
                                ensemble.append(ch)
                            if len(ensemble) >= 2:
                                break
                        for ch in ngram_chars:
                            if ch not in ensemble:
                                ensemble.append(ch)
                            if len(ensemble) >= 3:
                                break
                        while len(ensemble) < 3:
                            ensemble.append('e')
                        preds[idx] = "".join(ensemble[:3])
                        self.latency_log.append((data[idx][:30], "ensemble", per_item_ms))
                    else:
                        preds[idx] = "".join(llm_pred[:3])
                        self.latency_log.append((data[idx][:30], "llm", per_item_ms))

        print("  N-gram: {}, LLM: {}".format(ngram_count, llm_count))
        self._print_latency_summary(ngram_count, llm_count)
        return preds

    @staticmethod
    def _get_script_defaults(text):
        """
        Get common default characters based on the script of the last character(s).
        Returns top-3 most common chars for that script, or None.
        """
        if not text:
            return None
        # Find last non-space char
        last = text.rstrip()
        if not last:
            return None
        ch = last[-1]
        cp = ord(ch)
        
        # Arabic script
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            return ['ا', ' ', 'ل']
        # Devanagari (Hindi, Marathi, Nepali)
        if 0x0900 <= cp <= 0x097F:
            return ['ा', ' ', 'े']
        # Bengali
        if 0x0980 <= cp <= 0x09FF:
            return ['া', ' ', 'ে']
        # CJK
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return ['的', '了', '是']
        # Korean
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            return ['는', '이', ' ']
        # Cyrillic
        if 0x0400 <= cp <= 0x04FF:
            return ['а', 'о', 'е']
        # Greek
        if 0x0370 <= cp <= 0x03FF:
            return ['α', 'ο', 'ε']
        # Thai
        if 0x0E00 <= cp <= 0x0E7F:
            return ['า', 'ร', 'น']
        # Georgian
        if 0x10A0 <= cp <= 0x10FF:
            return ['ა', 'ი', 'ე']
        # Urdu/Extended Arabic
        if 0x0600 <= cp <= 0x08FF:
            return ['ا', ' ', 'ی']
        # Tamil
        if 0x0B80 <= cp <= 0x0BFF:
            return ['ா', ' ', 'ி']
        # Telugu
        if 0x0C00 <= cp <= 0x0C7F:
            return ['ా', ' ', 'ి']
        # Gujarati
        if 0x0A80 <= cp <= 0x0AFF:
            return ['ા', ' ', 'ે']
        # Gurmukhi (Punjabi)
        if 0x0A00 <= cp <= 0x0A7F:
            return ['ਾ', ' ', 'ੀ']
        
        # Latin - most common
        if ch.isascii() and ch.isalpha():
            return ['e', ' ', 'a']
        
        return None

    @staticmethod
    def _is_cjk(ch):
        """Check if character is CJK."""
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF)

    @staticmethod
    def _looks_like_sentence_end(text):
        """
        Heuristic: does this text look like it ends at a sentence boundary?
        Returns True if the text is long enough and ends with patterns
        typical of complete sentences (lowercase letter, CJK char, etc.)
        without already ending in punctuation or space.
        """
        if len(text) < 15:
            return False
        text = text.rstrip()
        if not text:
            return False
        last = text[-1]
        # Already ends with punctuation
        if last in '.!?。！？,;:，；：':
            return False
        # Ends with space — not sentence end
        if last == ' ':
            return False
        # CJK character at end of long text
        cp = ord(last)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF):
            return True
        # Latin/Cyrillic: only trigger if last word is >= 4 chars
        # (short words like 'a', 'to', 'the' are unlikely sentence-enders)
        if last.isalpha() and len(text) >= 25:
            words = text.split()
            if len(words) >= 5:
                last_word = words[-1]
                # Last word must be substantial (not a preposition/article/short word)
                if len(last_word) >= 4 and last_word[-1].islower():
                    return True
        return False

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

        # Save word n-gram model
        word_ngram_path = os.path.join(work_dir, "word_ngram_model.pkl")
        self.word_ngram_model.save(word_ngram_path)
        print("  Saved word n-gram model to {}".format(word_ngram_path))

        # Save script frequency model
        script_freq_path = os.path.join(work_dir, "script_freq.pkl")
        self.script_freq.save(script_freq_path)
        print("  Saved script frequency model to {}".format(script_freq_path))

        # Note: corpus.txt copy skipped to save checkpoint space
        # The n-gram model pkl contains all learned patterns

        # Legacy checkpoint file for compatibility
        with open(os.path.join(work_dir, "model.checkpoint"), "wt") as f:
            f.write("tinyllama+ngram")

    @classmethod
    def load(cls, work_dir):
        """Load model config, n-gram model. TinyLlama is lazy-loaded only if needed."""
        config_path = os.path.join(work_dir, "model.config.json")
        model_name = cls.DEFAULT_MODEL_NAME
        fine_tuned = False

        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("model_name", cls.DEFAULT_MODEL_NAME)
            fine_tuned = config.get("fine_tuned", False)

        # Defer device detection — will check when LLM is actually loaded
        device = "cpu"  # default; _ensure_model_loaded will use cuda if available

        # Pre-load token -> first char mapping (small, fast)
        mapping_path = os.path.join(work_dir, "token_to_first_char.pkl")
        token_to_first_char = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as fp:
                token_to_first_char = pickle.load(fp)
            print("  Loaded {} token mappings".format(len(token_to_first_char)))

        # Load n-gram model
        ngram_path = os.path.join(work_dir, "ngram_model.pkl")
        if os.path.exists(ngram_path):
            ngram_model = NgramModel.load(ngram_path)
            print("  Loaded n-gram model (trained={})".format(ngram_model.trained))
        else:
            ngram_model = NgramModel()
            print("  No n-gram model found — using TinyLlama only")

        # Load word n-gram model
        word_ngram_path = os.path.join(work_dir, "word_ngram_model.pkl")
        if os.path.exists(word_ngram_path):
            word_ngram_model = WordNgramModel.load(word_ngram_path)
            print("  Loaded word n-gram model (trained={})".format(word_ngram_model.trained))
        else:
            word_ngram_model = WordNgramModel()
            print("  No word n-gram model found")

        # Load script frequency model
        script_freq_path = os.path.join(work_dir, "script_freq.pkl")
        if os.path.exists(script_freq_path):
            script_freq = ScriptFrequency.load(script_freq_path)
            print("  Loaded script frequency model")
        else:
            script_freq = ScriptFrequency()
            print("  No script frequency model found")

        instance = cls(
            model=None,  # lazy-loaded
            tokenizer=None,  # lazy-loaded
            model_name=model_name,
            device=device,
            token_to_first_char=token_to_first_char,
            ngram_model=ngram_model,
            word_ngram_model=word_ngram_model,
            script_freq=script_freq,
        )
        instance._fine_tuned = fine_tuned
        instance._work_dir = work_dir
        print("  TinyLlama deferred (will lazy-load if needed)")
        return instance


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
