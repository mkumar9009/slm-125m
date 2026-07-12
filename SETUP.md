# Setup Instructions for 125M SLM Data Pipeline

## Prerequisites

All four Python files are created:
- ✅ `config.py` - Single source of truth (validated)
- ✅ `cleaning.py` - Deterministic cleaning pipeline
- ✅ `dedup.py` - Dedup and contamination helpers
- ✅ `modal_app.py` - Modal app with phases 0-4

## Step 1: Create Accounts and Get Credentials

### Modal Account
1. Sign up at https://modal.com (free tier includes monthly credits; this pipeline costs <$1)
2. Once logged in, go to Settings → API Tokens
3. Create a new API token, which gives you:
   - `MODAL_TOKEN_ID` (starts with `ak-`)
   - `MODAL_TOKEN_SECRET` (starts with `as-`)

### HuggingFace Token (optional, needed only for Phase 6)
1. Go to https://huggingface.co/settings/tokens
2. Create a token with WRITE role
3. This gives you `HUGGINGFACE_TOKEN` (starts with `hf_`)

## Step 2: Set Up .env.local

1. Copy the template:
   ```bash
   cp .env.local.template .env.local
   ```

2. Edit `.env.local` and fill in your actual credentials:
   ```bash
   MODAL_TOKEN_ID=ak-XXXXXXXX
   MODAL_TOKEN_SECRET=as-XXXXXXXX
   HUGGINGFACE_TOKEN=hf_XXXXXXXX
   ```

3. Verify it's in `.gitignore`:
   ```bash
   cat .gitignore  # should list .env.local
   ```

## Step 3: Authenticate Modal CLI (optional but recommended)

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal profile current
```

You should see your workspace and user info.

## Step 4: Create the Modal Volume

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal volume create slm-125m
```

This creates persistent storage for all artifacts. (The app will auto-create it on first use, but explicit creation is cleaner.)

## Running the Pipeline

Once credentials are set, run one phase at a time:

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
```

### Phase 0: Smoke test (10 docs/source, ~1 min)
```bash
modal run modal_app.py
```

### Phase 0: Measure true yields (~2 min)
```bash
modal run modal_app.py::measure
```

### Phase 1: Clean (~3 min)
```bash
modal run modal_app.py::clean --fineweb-shards 5
```

### Phase 2: Dedup + decontaminate (~6 min)
```bash
modal run modal_app.py::dedup
```

### Phase 3: Train tokenizer (~4 min)
```bash
modal run modal_app.py::tokenizer
```

### Phase 4: Tokenize + pack (~10 min)
```bash
modal run modal_app.py::tokenize
```

### Verify results
```bash
modal volume ls slm-125m /tokens
modal volume ls slm-125m /tokenizer
modal volume get slm-125m /tokens/index.json ./index.json
```

## Expected Results

- Train: ~2.19 billion tokens (~2.14M windows of 1024)
- Val: ~22.1 million tokens (~21.6K windows), 1% split
- Realized mix: case-law ~39%, sec ~39%, fineweb ~21% (legal-first, NOT 70/20/10)
- **Cost**: Well under $1 USD (typically ~$0.18)
- **Wall-clock**: ~40 minutes of useful compute

## Check Spend

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal billing report --start 2026-07-10 --json | \
  python3 -c "import sys,json; print(sum(float(r['cost']) for r in json.load(sys.stdin)))"
```

## Important Notes

1. **Legal-first mix**: This is NOT 70/20/10. Legal sources are small (~2B tokens total), so we take all of them and cap web at 0.5B. See Replication.md section 2.2.
2. **One phase at a time**: Run each phase, see the result, then continue. Do NOT chain them into one run.
3. **No GPU needed**: All phases 0-4 run on CPU. Phase 5 (pretraining) is GPU-only and NOT in this brief.
4. **Do not improvise**: Use the exact config, files, and parameters given. They were chosen by measurement.
