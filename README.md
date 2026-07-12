# 125M Small Language Model Data Pipeline

Complete, ready-to-run implementation of the from-scratch data pipeline for training a 125M-parameter legal/financial small language model.

## 🎯 Overview

This pipeline builds a **2.19 billion token** training corpus from three public datasets:
- **case-law** (282K documents) → ~863M tokens (39%)
- **sec** (48.5K documents) → ~861M tokens (39%)
- **fineweb-edu** (9.67M documents, sampled) → ~465M tokens (21%)

**Cost**: <$1 USD | **Time**: ~40 minutes | **GPU**: None required (CPU-only)

## 📋 What's Included

### Core Implementation
- **config.py** — Model architecture (125.8M params), data sources, and hyperparameters
- **cleaning.py** — Rule-based deterministic 6-step cleaning pipeline
- **dedup.py** — Deduplication and contamination detection helpers
- **modal_app.py** — Distributed execution framework (Phases 0-4)

### Documentation
- **Replication.md** — Complete original brief with all specifications
- **SETUP.md** — Detailed setup and authentication guide
- **EXECUTION_SUMMARY.md** — What was built and expected results
- **QUICK_START.sh** — Shell script to initialize and list commands

### Configuration
- **.env.local.template** — Credentials template
- **.gitignore** — Excludes secrets from version control

## ⚡ Quick Start

### Prerequisites
- Python 3.12
- Modal account (https://modal.com)
- HuggingFace token (optional, for Phase 6 upload)

### 1. Get Credentials
```bash
# Modal: https://modal.com → Settings → API Tokens
# Get: MODAL_TOKEN_ID (ak-...) and MODAL_TOKEN_SECRET (as-...)

# HuggingFace (optional): https://huggingface.co/settings/tokens
# Get: HUGGINGFACE_TOKEN (hf_...)
```

### 2. Set Up Environment
```bash
cp .env.local.template .env.local
# Edit .env.local with your credentials

# Load credentials for this session
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
```

### 3. Create Volume
```bash
modal volume create slm-125m
```

### 4. Run Phases (One at a Time)

**Phase 0: Smoke Test** (1 min)
```bash
modal run modal_app.py
```
Validates cleaning pipeline on 10 docs per source.

**Phase 0: Measure** (2 min)
```bash
modal run modal_app.py::measure
```
Estimates token yield from each dataset.

**Phase 1: Clean** (3 min)
```bash
modal run modal_app.py::clean --fineweb-shards 5
```
Streams and cleans all sources → `/data/clean/*/shard-XX.txt`

**Phase 2: Dedup + Decontaminate** (6 min)
```bash
modal run modal_app.py::dedup
```
Removes near-duplicates, exact-dups, and eval-contaminated docs → `/data/corpus/*/shard-XX.txt`

**Phase 3: Train Tokenizer** (4 min)
```bash
modal run modal_app.py::tokenizer
```
Trains 16K byte-level BPE tokenizer → `/data/tokenizer/`

**Phase 4: Tokenize + Pack** (10 min)
```bash
modal run modal_app.py::tokenize
```
Encodes all docs and packs into 1024-token windows (99% train / 1% val) → `/data/tokens/{train,val}/*.bin`

### 5. Verify Results
```bash
modal volume ls slm-125m /tokens
modal volume ls slm-125m /tokenizer
modal volume get slm-125m /tokens/index.json ./index.json
```

Expected:
```
train: 2.19B tokens (2,138,970 windows)
val: 22.1M tokens (21,614 windows)
```

## 🏗️ Architecture

### Data Mix (Legal-First, NOT 70/20/10)
Legal sources are small (~2B tokens total), so we take **all** of them:
- case-law: token budget 1.0B → yields ~863M (all ~282K docs)
- sec: token budget 1.3B → yields ~861M (all ~48.5K docs)
- fineweb-edu: token budget 0.5B → yields ~465M (sampled from 9.67M docs)

Result: ~40% legal + 40% legal + 20% web (not 70/20/10)

### Cleaning Pipeline (Phase 1)
6-step deterministic chain per document:
1. Line filtering (length, non-alphanumeric ratio)
2. Boilerplate stripping (SEC headers, page numbers, etc.)
3. Length gate (min 600 chars)
4. Repetition detection (n-gram concentration)
5. Language detection (ASCII + langdetect)
6. OCR quality gate (case-law only; non-word ratio threshold 20%)

### Deduplication (Phase 2)
- **Near-duplicates**: MinHash + LSH (threshold 0.8) on case-law shards
- **Exact-duplicates**: BLAKE2b hashing on normalized text
- **Contamination**: Strip docs with 13-grams matching eval sets (CaseHOLD, LexGLUE)

### Tokenization (Phase 3-4)
- **Vocab**: 16,384 tokens (byte-level BPE)
- **Special tokens**: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`, `<|user|>`, `<|assistant|>`, `<|system|>`
- **Context length**: 1,024 tokens
- **Split**: 99/1 (train/val)
- **Format**: uint16 binary windows on Modal Volume

## 💰 Cost & Performance

**Phases 0-4 (CPU-only)**
- Cost: ~$0.18 USD
- Time: ~40 minutes wall-clock
- No preemption: fanned-out workers (one per shard)

**Phase 5 (GPU Pretraining, not in this brief)**
- 8x H100s, ~350 hours of training
- Budget cap: $40 USD
- Not included in this pipeline

## ⚠️ Important Notes

1. **Use exact config**: All parameters were chosen by measurement. Do not improvise.
2. **One phase at a time**: Run each phase, verify output, then proceed. Do not chain into one silent run.
3. **Fanned out, not consolidated**: Each phase (1, 2, 4) deliberately uses one worker per shard to avoid single-container preemption.
4. **Proxy vs. real tokens**: Phases 1-2 use `chars/4` for speed. Phase 4 (real tokenizer) counts exactly; typically 8% lower than proxy.
5. **No GPU needed**: All phases run on CPU. Pretraining (Phase 5) is the GPU stage.

## 📊 File Structure

```
slm/
├── Replication.md          Original brief
├── config.py               Model & data config
├── cleaning.py             Cleaning pipeline
├── dedup.py                Dedup & contamination helpers
├── modal_app.py            Modal app (Phases 0-4)
├── README.md               This file
├── SETUP.md                Detailed setup guide
├── EXECUTION_SUMMARY.md    What was built
├── QUICK_START.sh          Initialization script
├── .env.local.template     Credentials template
├── .env.local              Secrets (gitignored)
└── .gitignore              Excludes .env.local
```

## 🔧 Troubleshooting

**Modal auth error**: Check `.env.local` has valid credentials from https://modal.com/settings/tokens

**Volume not found**: Run `modal volume create slm-125m` (or happens auto on first use)

**Slow/preempted**: This is normal for long jobs. The fanned-out design ensures work is distributed. Check Modal dashboard for preemption events.

**Missing wordlist**: Ensure `wamerican` apt package installs (Modal image includes it)

**Token count mismatch**: Phase 4 counts are 8% lower than Phase 1-2 proxy (chars/4). This is normal.

## 📚 References

- **Original Brief**: Replication.md (complete specification)
- **Modal Docs**: https://modal.com/docs
- **Datasets**:
  - case-law: https://huggingface.co/datasets/HFforLegal/case-law
  - sec: https://huggingface.co/datasets/PleIAs/SEC
  - fineweb-edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **Eval sets for decontamination**:
  - CaseHOLD: https://huggingface.co/datasets/casehold/casehold
  - LexGLUE: https://huggingface.co/datasets/coastalcph/lex_glue

## 🚀 Ready to Build

All source files are in place and validated. Follow SETUP.md to authenticate and execute the pipeline.

**Next step**: Get credentials from Modal and HuggingFace, then run `QUICK_START.sh`.
