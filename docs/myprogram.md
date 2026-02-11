# myprogram.py Documentation

`src/myprogram.py` is the core entry point for the CSE447 next-character prediction project. It implements a **hybrid model** that combines a fast, lightweight N-gram model with a powerful Large Language Model (TinyLlama) to achieve both high efficiency and high accuracy.

## Architecture Overview

The system uses a tiered prediction strategy:

1.  **Fast Path (N-gram)**: First, the system checks a character-level N-gram model. If the current context matches a known pattern seen during training, it returns a prediction almost instantly (~0.01ms).
2.  **Slow Path (TinyLlama)**: If the N-gram model lacks sufficient confidence or data for the context, the system falls back to **TinyLlama-1.1B**. This provides deep contextual understanding but at a higher latency cost (~200-800ms).

This hybrid approach allows the system to speed through common, repetitive patterns while reserving heavy compute for novel or complex situations.

## Components

### 1. N-gram Model (`NgramModel`)

A character-level N-gram model that memorizes sequences from the training corpus.

-   **Orders**: By default, it tracks N-grams from order 2 up to 6 (2-grams to 6-grams).
-   **Training**: It scans the provided corpus and counts the frequency of every `(context, next_char)` pair.
-   **Prediction**: It uses a **backoff strategy**:
    1.  Look for the longest possible matching context (e.g., last 5 characters).
    2.  If found and the count is above `min_count` (default 2), predict the most frequent next characters.
    3.  If not found, back off to a shorter context (e.g., last 4 chars) and repeat.
    4.  If no N-gram matches down to the minimum order, return `None` (triggering fallback).

### 2. TinyLlama Model (`MyModel`)

An integration of the open-source **TinyLlama-1.1B** model (3 trillion tokens checkpoint).

-   **Auto-Regressive**: TinyLlama is a causal language model trained to predict the next token.
-   **Token Mapping**: Since TinyLlama uses BPE (Byte Pair Encoding) tokens which may represent whole words or sub-words, the model includes a precomputed **Token-to-First-Character Mapping**.
    -   At load time, we iterate through the entire vocabulary.
    -   We map every token ID to the *first character* of its decoded string.
-   **Prediction**:
    1.  Run a forward pass on the context.
    2.  Get the probability distribution over the regular vocabulary from the last position's logits.
    3.  **Aggregate probabilities**: Sum the probabilities of all tokens that start with the same character (e.g., probabilities for "apple", "and", "a" are all summed under the character 'a').
    4.  Return the top-3 characters with the highest aggregated probability.

## Latency Tracking

The `MyModel.run_pred` method tracks the execution time of every prediction. At the end of a test run, it prints a detailed **Latency Summary** to the console, showing:

-   Total predictions vs. method split (N-gram vs. LLM).
-   Average, minimum, and maximum latency for each method.
-   A per-prediction log detail (useful for debugging specific inputs).

## Usage

### Training

```bash
python src/myprogram.py train --work_dir work
```

-   **Input**: Looks for training text in `data/corpus.txt` (Docker mount) or `example/corpus.txt`.
-   **Output**: Saves `ngram_model.pkl` and `model.config.json` to the `work` directory.
-   **Note**: The TinyLlama model is pretrained and not fine-tuned by default in this step, but the stub is present for future extensions (e.g., LoRA).

### Testing

```bash
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt
```

-   **Input**: Loads the saved models from `work_dir` and inputs from the test data file.
-   **Output**: Writes top-3 next-character predictions to the output file and prints latency stats to stdout.

## File Structure

-   `NgramModel` Class: Handles training and storage of count-based statistics.
-   `MyModel` Class:
    -   Manages the lifecycle of both N-gram and TinyLlama models.
    -   `run_pred`: The main loop implementing the hybrid logic and latency tracking.
    -   `_predict_top_chars_llm`: The specific logic for extracting character probabilities from the Transformer.
