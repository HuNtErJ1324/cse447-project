# CSE 447 Optimization Plan

**Objective:** Pareto-optimize error rate vs. processing time on NVIDIA L4 VM.  
**Current architecture:** Hybrid N-gram (fast path) + TinyLlama 1.1B (slow fallback).  
**Dev set:** 112 samples, current accuracy ~100%, speed unknown on L4.  
**Next deadline:** Checkpoint 3 — Feb 26 (must improve over CP2).  
**Final deadline:** Checkpoint 4 — Mar 12 (up to 3 bonus pts for Pareto frontier).

---

## Current Bottlenecks

1. **TinyLlama is loaded eagerly at test time** — even if N-gram handles everything, model load (~5-10s) is paid in `load()`.
2. **LLM fallback is ~200-800ms per batch** — each LLM call is expensive.
3. **N-gram model is large** (2M training lines, orders 2-7) — load time from pickle could be significant.
4. **Token-to-first-char aggregation** on every LLM call — softmax over full vocab + dict accumulation.
5. **80x repetition of targeted fixes** — inflates training time and pickle size.

## Optimization Rounds

### Round 1: Speed (Low-Hanging Fruit) ⏱️
**Goal:** Dramatically reduce processing time while keeping accuracy stable.

- [x] **Lazy-load TinyLlama** — only load if N-gram actually misses. ✅ 12.8s → 2.9s (77% faster). Also deferred torch/transformers imports.
- [ ] **Pre-compute top-k char lists per token** — instead of aggregating at inference, store a sparse `char_to_token_ids` dict and use vectorized ops.
- [x] **Compress N-gram pickle + fast load** — use `protocol=pickle.HIGHEST_PROTOCOL` for saves; skip defaultdict rebuild in NgramModel.load() (use plain dicts directly). ✅ 3.66s → 2.0s (45% faster).
- [x] **Remove unused imports** at test time — lazy-import `unicodedata` (only needed in rare ScriptFrequency fallback). ✅ Minor speedup.
- [ ] **Reduce N-gram orders** — test whether orders 2-5 (drop 6-7) lose much accuracy. Fewer orders = faster lookup + smaller model.
- [ ] **Quantize TinyLlama to int8/int4** — use `bitsandbytes` or `torch.quantization` for faster inference if LLM is needed.

### Round 2: Accuracy (Closing Gaps) 🎯
**Goal:** Reduce error rate on unseen multilingual test data.

- [ ] **Evaluate on held-out multilingual data** — create a proper eval set beyond the 112-sample dev. Sample from UDHR, Tatoeba, Wikipedia in 20+ languages.
- [ ] **Increase N-gram coverage** — the real test will have languages not in dev. Ensure CJK, Arabic, Devanagari, Cyrillic, Thai, Korean n-grams are well-represented.
- [ ] **Better script detection → better fallback** — current `_get_script_defaults` is hand-coded. Train script-specific unigram/bigram fallbacks from data.
- [ ] **Smarter LLM fallback trigger** — instead of confidence < 0.20, tune the threshold on a diverse eval set.
- [ ] **Add a BPE/SentencePiece character model** — lightweight alternative to full TinyLlama, trained on the multilingual corpus. Could replace TinyLlama entirely for speed.

### Round 3: Architecture Overhaul (If Needed) 🏗️
**Goal:** Hit Pareto frontier — best possible speed-accuracy tradeoff.

- [ ] **Replace TinyLlama with a small character-level LSTM/Transformer** — train a tiny (1-5M param) character-level model on the multilingual corpus. ~1-5ms per prediction, much better than TinyLlama's 200ms.
- [ ] **Distill TinyLlama into the small model** — use TinyLlama predictions as soft labels for training.
- [ ] **KV-cache for sequential predictions** — if the test harness is sequential (char by char), cache previous activations.
- [ ] **ONNX Runtime / TorchScript** — export the model for optimized inference.
- [ ] **Batch all predictions upfront** — if test data is all available at once (it is — file-based), process in large batches.

### Round 4: Final Polish (Pre-CP4) ✨
- [ ] **Docker optimization** — minimal base image, pre-download model weights, avoid pip install at runtime.
- [ ] **Profile on L4** — use Google Cloud credits to test on actual L4 VM. Measure wall-clock time.
- [ ] **Error analysis** — categorize misses by language/script/context-length. Target worst categories.
- [ ] **Ensemble tuning** — optimize N-gram confidence thresholds for when to fall back.

---

## Key Metrics to Track

| Metric | CP2 Baseline | CP3 Target | CP4 Target |
|--------|-------------|------------|------------|
| Success rate | ? | ≥ CP2 | Maximize |
| Processing time | ? | < CP2 | Minimize |
| Docker image size | ? | < 5GB | < 3GB |
| N-gram hit rate | ~100% (dev) | 95%+ (diverse) | 98%+ |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-22 | Created plan | CP3 due Feb 26, need structured approach |
| 2026-02-22 | Lazy-load TinyLlama + defer torch imports | 12.8s → 2.9s (77% faster). N-gram handles 100% of dev, LLM never loads. Accuracy unchanged at 100%. |
| 2026-02-22 | Fast NgramModel.load + lazy unicodedata | 3.66s → 2.0s (45% faster). Skip defaultdict rebuild, use plain dicts directly. Accuracy unchanged at 100%. |

---

*Update this file after each optimization round with results.*
