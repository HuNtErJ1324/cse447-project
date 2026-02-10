#!/usr/bin/env python
import os
import json
import torch
import random
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer


class MyModel:
    """
    A next-character prediction model powered by TinyLlama.

    TinyLlama is a 1.1B parameter autoregressive language model trained on
    3 trillion tokens. Because it is a causal LM, it is naturally suited for
    next-token prediction — exactly what this task requires.

    Since TinyLlama uses BPE (sub-word) tokenization rather than character-level
    tokens, we aggregate the probability mass of all tokens whose decoded text
    starts with the same character. This gives us a principled character-level
    probability distribution for the next character.

    We then pick the top-3 most probable next characters as our guesses.
    """

    # ------------------------------------------------------------------ #
    #  Default model identifier                                           #
    # ------------------------------------------------------------------ #
    DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    def __init__(self, model=None, tokenizer=None, model_name=None, device=None,
                 token_to_first_char=None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        # Precomputed mapping: token_id -> first character of decoded token
        self.token_to_first_char = token_to_first_char or {}

    # ------------------------------------------------------------------ #
    #  Data helpers                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_training_data(cls):
        """
        Load training data for future fine-tuning.
        TODO: Implement data loading for fine-tuning on domain-specific
              corpora (e.g. dialogue datasets, multilingual text).
        """
        return []

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
    #  Training (stub — left open for future fine-tuning)                  #
    # ------------------------------------------------------------------ #

    def run_train(self, data, work_dir):
        """
        Train / fine-tune the model.

        TODO — future work:
          - Load a large multilingual corpus or dialogue dataset
          - Fine-tune TinyLlama with causal LM objective
          - Use LoRA or similar PEFT for efficient fine-tuning
          - Save adapter weights alongside model config
        """
        pass

    # ------------------------------------------------------------------ #
    #  Prediction — fully implemented for test mode                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_token_to_first_char(tokenizer):
        """
        Build a mapping from every token ID in the vocabulary to the first
        printable character that token decodes to.

        This is computed once at model load time so that inference is fast.
        Returns a dict: {token_id: first_char} (only for tokens that start
        with a printable character).
        """
        mapping = {}
        vocab_size = tokenizer.vocab_size

        for token_id in range(vocab_size):
            try:
                decoded = tokenizer.decode([token_id])
                if not decoded:
                    continue
                first_char = decoded[0]
                # Only include if the first character is printable or a space
                if first_char.isprintable() or first_char == " ":
                    mapping[token_id] = first_char
            except Exception:
                continue

        return mapping

    def _ensure_model_loaded(self):
        """Lazy-load the model and tokenizer if not already present."""
        if self.model is None:
            print("  Loading TinyLlama model: {}".format(self.model_name))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

            if not self.token_to_first_char:
                print("  Building token → first-char mapping...")
                self.token_to_first_char = self._build_token_to_first_char(
                    self.tokenizer
                )
                print("  Mapped {} tokens to first characters".format(
                    len(self.token_to_first_char)
                ))

    def run_pred(self, data):
        """
        For each input context string, predict the top-3 most likely next
        UTF-8 characters using TinyLlama.

        Strategy:
          1. Tokenize the context string.
          2. Run a forward pass to get next-token logits.
          3. Convert logits to probabilities.
          4. Aggregate probabilities by the first character each token
             would produce (using precomputed mapping).
          5. Return the top-3 characters by aggregated probability.

        Returns a list of 3-character strings (one per input).
        """
        self._ensure_model_loaded()
        preds = []

        for i, inp in enumerate(data):
            top_chars = self._predict_top_chars(inp, k=3)
            preds.append("".join(top_chars))
            if (i + 1) % 100 == 0:
                print("  Processed {}/{} inputs".format(i + 1, len(data)))

        return preds

    def _predict_top_chars(self, context, k=3):
        """
        Given a context string, return the top-k most likely next characters.

        Aggregates the probability mass over all BPE tokens whose decoded
        form starts with the same character.
        """
        # Tokenize the input context
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits for the last position (next token prediction)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=0)

        # Aggregate probabilities by first character
        char_probs = defaultdict(float)
        for token_id, first_char in self.token_to_first_char.items():
            if token_id < len(probs):
                char_probs[first_char] += probs[token_id].item()

        # Sort characters by aggregated probability (descending)
        sorted_chars = sorted(char_probs.items(), key=lambda x: x[1], reverse=True)

        # Collect top-k
        top_chars = [char for char, prob in sorted_chars[:k]]

        # Fallback
        while len(top_chars) < k:
            top_chars.append("e")

        return top_chars

    # ------------------------------------------------------------------ #
    #  Save / Load                                                        #
    # ------------------------------------------------------------------ #

    def save(self, work_dir):
        """
        Save model config to work_dir.

        For the base (non-fine-tuned) model we only save a config file
        indicating which HuggingFace model to load. After fine-tuning,
        we would also save adapter/model weights here.
        """
        os.makedirs(work_dir, exist_ok=True)
        config_path = os.path.join(work_dir, "model.config.json")
        config = {
            "model_name": self.model_name,
            "fine_tuned": False,
        }
        with open(config_path, "wt", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Legacy checkpoint file for compatibility
        with open(os.path.join(work_dir, "model.checkpoint"), "wt") as f:
            f.write("tinyllama")

    @classmethod
    def load(cls, work_dir):
        """
        Load a saved model from work_dir.

        Reads the config to determine which model to load. If a fine-tuned
        checkpoint exists, loads that; otherwise loads the pretrained model
        from HuggingFace.
        """
        config_path = os.path.join(work_dir, "model.config.json")
        model_name = cls.DEFAULT_MODEL_NAME
        fine_tuned = False

        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("model_name", cls.DEFAULT_MODEL_NAME)
            fine_tuned = config.get("fine_tuned", False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if fine_tuned:
            # Future: load fine-tuned weights / LoRA adapter from work_dir
            ft_path = os.path.join(work_dir, "tinyllama_finetuned")
            print("  Loading fine-tuned model from {}".format(ft_path))
            tokenizer = AutoTokenizer.from_pretrained(ft_path)
            model = AutoModelForCausalLM.from_pretrained(ft_path)
        else:
            print("  Loading pretrained model: {}".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
            )

        model.to(device)
        model.eval()

        # Precompute token → first char mapping
        print("  Building token → first-char mapping...")
        token_to_first_char = cls._build_token_to_first_char(tokenizer)
        print("  Mapped {} tokens to first characters".format(len(token_to_first_char)))

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            token_to_first_char=token_to_first_char,
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
