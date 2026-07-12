# 125M SLM Data Pipeline - Execution Summary

## ✅ Completed Steps

All source files have been created and validated:

### 1. Source Files
- **config.py** (5.4 KB)
  - Single source of truth for the pipeline
  - Model config: 125.8M parameters, 16K vocab, 12L/768d/12h
  - Data mix: legal-first (case-law 40% + sec 40% + fineweb 20%)
  - Validated ✅

- **cleaning.py** (4.3 KB)
  - Deterministic 6-step cleaning pipeline
  - Line filtering, boilerplate stripping, repetition detection
  - English/OCR quality gates

- **dedup.py** (982 B)
  - Helpers for deduplication and contamination detection
  - Exact hashing, word n-grams, MinHash shingles

- **modal_app.py** (22 KB)
  - Modal App definition with Phases 0-4
  - CPU-based, no GPU needed
  - Distributed sharding for efficiency

### 2. Documentation
- **SETUP.md** - Complete setup and execution guide
  - Account creation instructions
  - Credentials setup
  - Phase-by-phase commands
  - Expected results and costs

### 3. Configuration
- **.env.local.template** - Credentials template (not committed)
- **.gitignore** - Excludes .env.local from git
- **Modal** - Installed and ready (v0.65.66)

## 📋 Next Steps

### To Run the Pipeline:

1. **Get Credentials**
   - Modal: https://modal.com → Settings → API Tokens
   - HuggingFace (optional): https://huggingface.co/settings/tokens

2. **Set Up Environment**
   ```bash
   cp .env.local.template .env.local
   # Edit .env.local with your actual credentials
   ```

3. **Create Modal Volume**
   ```bash
   source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
   modal volume create slm-125m
   ```

4. **Run Phases One at a Time**
   ```bash
   source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
   modal run modal_app.py              # Phase 0: smoke test
   modal run modal_app.py::measure     # Phase 0: measure yields
   modal run modal_app.py::clean --fineweb-shards 5        # Phase 1
   modal run modal_app.py::dedup                           # Phase 2
   modal run modal_app.py::tokenizer                       # Phase 3
   modal run modal_app.py::tokenize                        # Phase 4
   ```

## 📊 Expected Results

After all phases complete:

**Training Data**
- Train: ~2.19 billion tokens (~2.14M windows of 1024 tokens)
- Val: ~22.1 million tokens (~21.6K windows) - 1% held-out
- Total: ~2.21 billion tokens

**Data Mix (Realized)**
- case-law: ~863M tokens (39%)
- sec: ~861M tokens (39%)
- fineweb-edu: ~465M tokens (21%)

**Cost & Time**
- Cost: <$1 USD (typically ~$0.18)
- Wall-clock: ~40 minutes total
- All CPU-based, no GPU needed

**Artifacts on Volume**
```
/data/clean/<source>/shard-XX.txt          (Phase 1: cleaned)
/data/corpus/<source>/shard-XX.txt         (Phase 2: dedup'd)
/data/tokenizer/                           (Phase 3: BPE vocab)
/data/tokens/train/*.bin                   (Phase 4: training windows)
/data/tokens/val/*.bin                     (Phase 4: validation windows)
/data/tokens/index.json                    (Phase 4: metadata)
```

## ⚠️ Important Notes

1. **Not 70/20/10**: Legal sources only contain ~2B tokens total, so we take all of them (1B + 1.3B) and cap web at 0.5B, resulting in ~40/40/20 split.

2. **One Phase at a Time**: Execute each phase sequentially and verify results before proceeding. Do not chain them into one silent run.

3. **No Improvisation**: The config parameters, thresholds, and data mix were chosen by measurement. Use them exactly as given.

4. **Fanned Out**: Heavy phases (1, 2, 4) use one worker per shard deliberately. Keep this design; do not consolidate to single containers.

5. **Proxy vs. Real Tokens**: Phases 1-2 use chars/4 as a proxy for token counts. Only Phase 4 (with the real tokenizer) gives exact counts (typically 8% lower than proxy).

## 📁 File Listing

```
slm/
├── Replication.md           (Original brief)
├── config.py                (Model & data config) ✅
├── cleaning.py              (Cleaning pipeline) ✅
├── dedup.py                 (Dedup helpers) ✅
├── modal_app.py             (Modal app) ✅
├── SETUP.md                 (Setup instructions)
├── EXECUTION_SUMMARY.md     (This file)
├── .env.local.template      (Credentials template)
├── .env.local               (Secret credentials - gitignored)
└── .gitignore               (Excludes .env.local)
```

## 🚀 Ready to Execute

All prerequisites are met. Once you:
1. Create accounts and get credentials
2. Set up `.env.local`
3. Create the Modal volume

You can begin running phases 0-4 to build the 2.19B token training corpus for your 125M parameter SLM.

See `SETUP.md` for detailed execution commands.
