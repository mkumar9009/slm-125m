# SLM 125M Pipeline Setup & Execution Guide

**Last Updated:** 2026-07-11

## Environment Setup

### Python Version
- **Use:** `.python/bin/python3` (NOT system `python3`)
- This is the isolated Python installation to use for all commands
- Virtual environment: `slmenv` (already exists in current directory)

### Virtual Environment Activation

```bash
# Activate the virtual environment
source slmenv/bin/activate

# Or use directly without activation (for one-off commands):
./slmenv/bin/python3 -m pip install <package>
```

### Credentials Setup

Create `.env.local` in the working directory (git-ignored):

```bash
MODAL_TOKEN_ID=ak-XXXXXXXX
MODAL_TOKEN_SECRET=as-XXXXXXXX
HUGGINGFACE_TOKEN=hf_XXXXXXXX
```

**Important:** Always source this before running Modal commands:

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
```

---

## Project Context

### Data Pipeline Overview

This project builds a 125M-parameter small language model from three public datasets:

1. **case-law** (HFforLegal/case-law) - US court opinions, ~863M tokens (39%)
2. **sec** (PleIAs/SEC) - SEC filings, ~861M tokens (39%)  
3. **fineweb-edu** (HuggingFaceFW/fineweb-edu) - Educational web text, ~465M tokens (21%)

**Total:** ~2.19 billion training tokens (~2.14M windows of 1024 tokens each)

**Note:** NOT 70/20/10 split - Legal sources only contain ~2B tokens total, so we use "legal-first" strategy.

### File Structure

```
config.py         - Single source of truth (model, data mix, budgets, thresholds)
cleaning.py       - Deterministic 6-step cleaning chain
dedup.py          - Dedup & decontamination helpers
modal_app.py      - Modal app with functions for each phase
.env.local        - Credentials (never commit)
SETUP_GUIDE.md    - This file (reference for future runs)
```

### Output Structure (Modal Volume: slm-125m)

```
/data/clean/<source>/shard-XX.txt      Phase 1: cleaned text
/data/corpus/<source>/shard-XX.txt     Phase 2: deduped corpus
/data/tokenizer/                       Phase 3: 16K byte-level BPE tokenizer
/data/tokens/train/*.bin               Phase 4: 99% training data (uint16 windows)
/data/tokens/val/*.bin                 Phase 4: 1% validation data
/data/tokens/index.json                Phase 4: metadata
```

---

## Pipeline Execution

### Pre-Flight Check

```bash
# 1. Verify config locally (no Modal needed)
.python/bin/python3 config.py

# Expected output:
# slm-125m
# model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
# target tokens: 2.5B (~19.8 tok/param)
# stages: setup -> clean -> dedup -> tokenizer -> tokenize -> pretrain -> deploy

# 2. Load credentials (do this before ANY modal command)
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN

# 3. Verify Modal authentication
source .env.local && modal profile current
```

### Phase 0: Smoke Test & Measure

```bash
# Load credentials first
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN

# Smoke test (10 docs/source, validates cleaning pipeline)
source .env.local && modal run modal_app.py

# Measure actual token yield per source (takes ~5 min)
source .env.local && modal run modal_app.py::measure

# Expected measure output:
# case-law:     ~0.8B tokens
# sec:          ~1.1B tokens  
# fineweb-edu:  ~11B available (we cap at 0.5B)
# TOTAL est clean tokens: ~2.19B
```

### Phase 1: Stream + Clean

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
source .env.local && modal run modal_app.py::clean --fineweb-shards 5
```

**What it does:**
- Launches 16 workers (case-law 10 + sec 5 + fineweb 5)
- One worker per parquet shard
- Writes `/data/clean/<source>/shard-XX.txt`
- Applies 6-step deterministic cleaning: line filtering, boilerplate stripping, repetition check, language detection, OCR quality gate

**Expected output:**
- ~718K docs streamed
- ~698K kept (~97% keep rate)
- ~2.68B proxy tokens
- Cost: ~$0

**To redo one source only:**
```bash
source .env.local && modal run modal_app.py::clean --only case-law
```

**Optional OCR threshold analysis** (already set to 0.20):
```bash
source .env.local && modal run modal_app.py::ocr
```

### Phase 2: Dedup + Decontaminate

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
source .env.local && modal run modal_app.py::dedup
```

**What it does:**
- MinHash signatures for case-law shards
- LSH near-duplicate detection (threshold 0.8)
- Contamination filtering (removes docs matching eval sets: CaseHOLD, LexGLUE)
- Exact-duplicate removal
- Writes `/data/corpus/<source>/shard-XX.txt`

**Expected output:**
- ~24K case-law docs removed (CaseHOLD contamination)
- ~1.6K near-duplicates
- ~2K SEC exact-duplicates
- Final corpus: ~670K docs, ~2.40B proxy tokens
- Cost: ~$0
- Time: ~6 minutes

**To reuse pre-computed signatures:**
```bash
source .env.local && modal run modal_app.py::dedup --compute-sigs False
```

### Phase 3: Train Tokenizer

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
source .env.local && modal run modal_app.py::tokenizer
```

**What it does:**
- Trains byte-level BPE tokenizer on entire corpus
- Vocab size: 16,384
- Special tokens: `<|bos|>`, `|eos|>`, `<|pad|>`, `<|unk|>`, `<|user|>`, `<|assistant|>`, `<|system|>`
- Saves to `/data/tokenizer/`

**Expected output:**
```
vocab_size=16384
'The plaintiff shall bear...' -> N tokens | roundtrip=True
'The Company's net revenues...' -> N tokens | roundtrip=True
```

**Cost:** ~$0  
**Time:** ~4 minutes

### Phase 4: Tokenize + Pack + Split 99/1

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
source .env.local && modal run modal_app.py::tokenize
```

**What it does:**
- 14 parallel workers tokenize corpus shards
- Appends `<|eos|>` after each document
- Packs into 1024-token uint16 windows
- Routes every 100th window → validation (99/1 split)
- Writes `/data/tokens/train/*.bin`, `/data/tokens/val/*.bin`, `/data/tokens/index.json`

**Expected output:**
```
index: train=2.19B tok (~2.14M win), val=22.1M tok (~21.6K win)
```

**Cost:** ~$0  
**Time:** ~10 minutes

### Verify Outputs

```bash
# List tokenizer artifacts
source .env.local && modal volume ls slm-125m /tokenizer

# List token datasets
source .env.local && modal volume ls slm-125m /tokens

# Download index locally
source .env.local && modal volume get slm-125m /tokens/index.json ./index.json

# Check billing
source .env.local && modal billing report --start 2026-07-08 --json \
  | python3 -c "import sys,json; print(sum(float(r['cost']) for r in json.load(sys.stdin)))"
```

---

## Critical Rules (Do Not Relearn)

1. **Data split is legal-first, NOT 70/20/10** — Legal sources cap at ~2B tokens
2. **Modal image rule** — All `pip_install` and `apt_install` BEFORE `add_local_python_source`
3. **Language detection** — `is_english()` uses ASCII-first, only calls `langdetect` on 90-99% band
4. **OCR gate** — Requires `/usr/share/dict/words` (via `wamerican` apt package)
5. **Preemption handling** — Fan out work one shard per worker; don't use single big containers
6. **Token counts** — Phases 1-2 use chars/4 proxy; only Phase 4 (real tokenizer) is authoritative (~8% lower)
7. **CaseHOLD lookup** — `casehold/casehold` may not resolve; `coastalcph/lex_glue` case_hold covers it
8. **To adjust data** — Edit `token_budget` per source in `config.py`, then re-run Phase 1+ in order

---

## Common Issues & Fixes

### Modal authentication fails
```bash
# Issue: "Token missing"
# Fix: Always source .env.local before modal commands
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
```

### Volume already exists
```bash
# If volume creation fails, it already exists (safe to skip)
# The app creates it on first use
```

### Slow or preempted runs
```bash
# Don't switch to single container; keep sharded architecture
# Modal preempts long single containers; distributed load is resilient
```

### Language detection slow
```bash
# langdetect is expensive; is_english() is optimized:
# - Returns True if ASCII ratio >= 99%
# - Returns False if ASCII ratio < 90%
# - Only calls langdetect on 90-99% band (rare case)
```

---

## Cost Summary

| Phase | Task | Time | Cost |
|-------|------|------|------|
| 0 | Smoke test + measure | 5 min | $0 |
| 1 | Stream + clean | 5 min | $0 |
| 2 | Dedup + decontam | 6 min | $0 |
| 3 | Train tokenizer | 4 min | $0 |
| 4 | Tokenize + pack | 10 min | $0 |
| **Total (Phases 0-4)** | **All CPU, no GPU** | **~40 min** | **< $1** |

---

## Next Steps (Phase 5+)

- **Phase 5:** Pretraining on 8x H100 GPUs (~40 USD budget cap)
- **Phase 6:** Push trained model to HuggingFace Hub

*(Not covered in this brief; requires GPU setup)*

---

## Reference Commands (Copy-Paste Ready)

```bash
# Always start with credentials
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN

# Pre-flight
.python/bin/python3 config.py

# Phase 0
source .env.local && modal run modal_app.py
source .env.local && modal run modal_app.py::measure

# Phase 1
source .env.local && modal run modal_app.py::clean --fineweb-shards 5

# Phase 2
source .env.local && modal run modal_app.py::dedup

# Phase 3
source .env.local && modal run modal_app.py::tokenizer

# Phase 4
source .env.local && modal run modal_app.py::tokenize

# Verify
source .env.local && modal volume ls slm-125m /tokens
source .env.local && modal volume get slm-125m /tokens/index.json ./index.json
```

---

**Remember:** Use `.python/bin/python3` for direct execution; activate `slmenv` for interactive work. Always source `.env.local` before Modal commands.
