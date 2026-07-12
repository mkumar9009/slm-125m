# Modal App High-Level Flow & Execution Model

## What is Modal?

Modal is a **serverless compute platform** that lets you:
- Define Python functions that run in cloud containers (not your local machine)
- Specify resources (CPU, GPU, memory, timeout, dependencies)
- Parallelize work across multiple workers
- Mount persistent volumes for shared data storage

**In this project:** Modal handles dataset ingestion, cleaning, deduplication, tokenization—all on CPU, all auto-parallelized.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR LOCAL MACHINE                        │
│                                                              │
│  You run: modal run modal_app.py::clean                    │
│           (CLI command)                                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODAL CLOUD PLATFORM                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Shared Persistent Volume: slm-125m                 │   │
│  │ (/data/clean, /data/corpus, /data/tokenizer, etc.) │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▲                                  │
│                           │                                  │
│    ┌──────────────────────┼──────────────────────┐          │
│    │                      │                      │          │
│    ▼                      ▼                      ▼          │
│  Worker 1            Worker 2            Worker N          │
│  (CPU container)    (CPU container)    (CPU container)     │
│  Cleans shard-000   Cleans shard-001   Cleans shard-019   │
│  Reads from HF      Reads from HF      Reads from HF       │
│  Writes to /data/clean                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Structure Breakdown

### 1. **App Initialization**

```python
app = modal.App(config.PROJECT)  # "slm-125m"
```
- Creates a Modal app namespace
- All functions below are registered to this app
- Think of it as a "microservice container registry"

---

### 2. **Container Image Definitions**

#### `_cpu_base` (Base CPU Image)
```python
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")                    # System: wordlist for OCR gate
    .pip_install(
        "datasets==3.6.0",                       # Stream HF datasets
        "huggingface_hub==0.34.4",               # HF API
        "langdetect==1.0.9",                     # Language detection
        "pyarrow==17.0.0",                       # Parquet handling
        "datasketch==1.6.5",                     # MinHash for dedup
    )
)
```

**What it does:** Defines a Docker-like container image that runs on every worker.

#### `cpu_image` (Image + Local Source Code)
```python
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")
```

**What it does:** Adds local Python modules to the image so workers can import them.

#### `ml_image` (Extended for Tokenization)
```python
ml_image = _cpu_base.pip_install("transformers==4.46.3").add_local_python_source(
    "config", "cleaning", "dedup")
```

**What it does:** Extends base image with transformers for tokenization.

---

### 3. **Persistent Volume**

```python
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}  # Mount at /data
```

**What it does:**
- Creates a persistent volume named `slm-125m` on Modal
- Mounts it at `/data` inside all containers
- All workers read/write to the same `/data` directory
- Data persists after container exits
- Workers commit changes with `volume.commit()`

---

## Execution Flow (High-Level)

### **Phase 0: Smoke Test + Measure**

```
┌─ You run: modal run modal_app.py
│
├─→ @app.function smoke_test() [1 function, 1 worker]
│   ├─ Streams 10 docs from each of 3 sources
│   ├─ Applies 6-step cleaning to each
│   ├─ Prints results (kept/dropped + reasons)
│   └─ Returns summary
│
└─→ @app.function measure_sources() [1 function, 1 worker]
    ├─ Streams 2000 docs from each source (statistical sample)
    ├─ Measures: keep_rate, avg_clean_chars per doc
    ├─ Extrapolates to full dataset
    │  (e.g., if keep_rate=97% on 2000 sample, expect 97% on 282K full)
    └─ Prints: "case-law: est 0.8B tokens, sec: est 1.1B tokens, ..."
```

**Why this matters:**
- Validates cleaning pipeline works before full run
- Measures actual yield to confirm data mix strategy

---

### **Phase 1: Stream + Clean**

```
┌─ You run: modal run modal_app.py::clean --fineweb-shards 5
│
├─→ Local entrypoint clean()
│   ├─ Queries HF API for parquet file URLs
│   │  case-law: 10 parquet files → 10 shards
│   │  sec: 5 parquet files → 5 shards
│   │  fineweb-edu: ~2000 files, take first 5 → 5 shards
│   │
│   ├─ Creates work queue: [(source, url, shard_index, token_cap), ...]
│   │  Total: 20 tasks
│   │
│   └─→ @app.function clean_shard.starmap(work)
│       [PARALLEL EXECUTION: 20 workers, one per task]
│       
│       Worker 1                Worker 2              ...  Worker 20
│       ───────────────────────────────────────────────────────────
│       clean_shard(            clean_shard(
│         "case-law",             "sec",
│         "https://...",          "https://...",
│         shard_index=0,          shard_index=0,
│         token_cap=100M          token_cap=260M
│       )                       )
│       │                       │
│       ├─ Stream parquet     ├─ Stream parquet
│       │  from HF            │  from HF
│       ├─ For each doc:      ├─ For each doc:
│       │  • Filter lines     │  • Filter lines
│       │  • Strip boilerplate│  • Strip boilerplate
│       │  • Check repetition │  • Check repetition
│       │  • Language detect  │  • Language detect
│       │  • OCR gate (if)    │  • [skip OCR]
│       │  • Min size check   │  • Min size check
│       ├─ If kept:          ├─ If kept:
│       │  Write to          │  Write to
│       │  /data/clean/      │  /data/clean/
│       │  case-law/         │  sec/
│       │  shard-000.txt     │  shard-000.txt
│       │  (one doc/line)    │  (one doc/line)
│       │                    │
│       └─ volume.commit()   └─ volume.commit()
│
└─ Aggregates results
   Prints: "case-law: 698K docs, sec: 180K docs, fineweb: 50K docs"
   Total est tokens: 2.68B
```

**Key Concepts:**
- `.starmap()` launches N workers in parallel (one per work item)
- Each worker is an independent container with its own CPU/memory
- All write to shared `/data` volume
- `.commit()` persists writes (flush to durable storage)
- Total time: ~5 min (limited by slowest worker, not sum)

---

### **Phase 2: Dedup + Decontaminate**

```
┌─ You run: modal run modal_app.py::dedup
│
├─→ Local entrypoint dedup()
│   │
│   ├─ Stage 1: Compute MinHash signatures
│   │   @app.function minhash_shard.map([10 case-law shards])
│   │   
│   │   Worker 1              Worker 2           ...  Worker 10
│   │   ──────────────────────────────────────────────────────
│   │   Read /data/clean/    Read /data/clean/
│   │   case-law/            case-law/
│   │   shard-000.txt        shard-001.txt
│   │   │
│   │   For each doc (line):
│   │   ├─ Extract words
│   │   ├─ Compute 5-shingles (word sequences)
│   │   ├─ MinHash(32 perms) of shingles
│   │   └─ Save sig to disk
│   │   
│   │   Output: /data/tmp/minhash_sigs/shard-000.npz (numpy array)
│   │
│   ├─ Stage 2: Build near-duplicate graph (LSH)
│   │   @app.function build_near_dups() [1 worker]
│   │   
│   │   ├─ Load all .npz signature files
│   │   ├─ Build MinHashLSH index (threshold=0.8)
│   │   │  "Two docs similar if their MinHash sigs overlap >80%"
│   │   ├─ For each doc: insert into LSH or mark as near-dup
│   │   └─ Save near-dup graph to /data/tmp/near_dups.json
│   │
│   │   Output: {"shard-000.txt": [doc_idx1, doc_idx2, ...], ...}
│   │            (indices of docs marked as near-dups)
│   │
│   └─ Stage 3: Write final corpus (dedup + decontam)
│       @app.function write_corpus_shard.starmap([20 shards])
│       
│       Worker 1                    Worker 2                 ... Worker 20
│       ────────────────────────────────────────────────────────────────────
│       write_corpus_shard(         write_corpus_shard(
│         "case-law",                 "sec",
│         "shard-000.txt"              "shard-000.txt"
│       )                            )
│       │                            │
│       ├─ Read /data/clean/        ├─ Read /data/clean/
│       │  case-law/shard-000.txt   │  sec/shard-000.txt
│       │
│       ├─ Load near_dups.json      ├─ Load contam ngrams
│       │  (precomputed)            │  (from eval sets)
│       │
│       ├─ For each doc:
│       │  ├─ Check if in near_dups → SKIP if yes
│       │  ├─ Hash doc → check if seen exactly → SKIP if yes
│       │  ├─ Check 13-gram overlap with eval sets → SKIP if yes
│       │  └─ If passes all → WRITE to /data/corpus/
│       │
│       └─ Output: /data/corpus/case-law/shard-000.txt
│                  (only deduplicated, decontaminated docs)
│
└─ Aggregates results
   Prints: "case-law: 24K removed (contam), 1.6K (near-dup)"
   Final corpus: 670K docs, 2.40B tokens
```

**Key Concepts:**
- `.map()` launches workers for a list (simpler than `.starmap()`)
- `.remote()` runs a function asynchronously (not waiting)
- Parallelization: Stage 1 (10 workers) + Stage 3 (20 workers)
- Stage 2 is sequential (LSH build needs all sigs in memory)

---

### **Phase 3: Train Tokenizer**

```
┌─ You run: modal run modal_app.py::tokenizer
│
├─→ Local entrypoint tokenizer()
│   │
│   └─→ @app.function train_tokenizer() [1 worker, 8 CPU, 16GB RAM]
│       │
│       ├─ Stream all docs from /data/corpus/
│       │  (iterator, never loads full corpus in memory)
│       │
│       ├─ Initialize BPE tokenizer (vocab_size=16,384)
│       │
│       ├─ For each corpus line:
│       │  └─ Feed bytes to BPE trainer
│       │     (learns subword merging rules from frequency)
│       │
│       ├─ Save tokenizer to /data/tokenizer/
│       │  ├─ tokenizer.json (BPE vocab + merge rules)
│       │  ├─ special_tokens_map.json (<|bos|>, <|eos|>, etc.)
│       │  └─ tokenizer_config.json
│       │
│       └─ Validate roundtrip on 2 examples:
│          "The plaintiff shall bear..." 
│          → encode → ids → decode → match? ✓
│
└─ Output: /data/tokenizer/ ready for inference
```

**Key Concepts:**
- Streaming iterator (never materializes 2.4B tokens in RAM)
- BPE learns from frequency in corpus
- Tokenizer is **data-dependent** (different corpus → different vocab)

---

### **Phase 4: Tokenize + Pack**

```
┌─ You run: modal run modal_app.py::tokenize
│
├─→ Local entrypoint tokenize()
│   │
│   ├─ Creates work: [(src, idx, num_shards), ...]
│   │  Total: 14 tasks (case-law 4, sec 6, fineweb 4)
│   │
│   └─→ @app.function tokenize_shard.starmap(work)
│       [PARALLEL: 14 workers]
│       
│       Worker 1                        Worker 2
│       ──────────────────────────────────────────────
│       tokenize_shard(                 tokenize_shard(
│         "case-law", shard_idx=0,        "case-law", shard_idx=1,
│         num_shards=4                    num_shards=4
│       )                               )
│       │                               │
│       ├─ Load tokenizer               ├─ Load tokenizer
│       │  from /data/tokenizer/        │  from /data/tokenizer/
│       │
│       ├─ Read corpus docs             ├─ Read corpus docs
│       │  where (doc_idx % 4 == 0)     │  where (doc_idx % 4 == 1)
│       │  (sharding: each worker       │  (only processes assigned slice)
│       │   gets 1/4 of case-law docs)  │
│       │
│       ├─ For each doc:
│       │  ├─ Tokenize doc → token ids
│       │  └─ Append <|eos|> token
│       │
│       ├─ Buffer tokens
│       │  When buffer >= 1024 tokens:
│       │  ├─ Extract window (1024 tokens)
│       │  ├─ Convert to uint16 numpy array
│       │  ├─ If win_count % 100 == 0:
│       │  │  └─ Write to /data/tokens/val/
│       │  └─ Else:
│       │     └─ Write to /data/tokens/train/
│       │
│       └─ Output:
│          /data/tokens/train/case-law-000.bin (many 1024-token windows)
│          /data/tokens/val/case-law-000.bin (1% of windows)
│
└─ Final aggregation: write_token_index()
   Writes /data/tokens/index.json
   {
     "seq_len": 1024,
     "dtype": "uint16",
     "train_windows": 2_138_970,
     "val_windows": 21_614,
     "train_tokens": 2_190_298_560,
     "val_tokens": 22_142_976,
     ...
   }
```

**Key Concepts:**
- **Sharding:** Worker 0 processes docs 0, 4, 8, ... (every 4th)
- **On-the-fly split:** Every 100th window routes to val (99/1 split)
- **Binary format:** uint16 (2 bytes per token) = compact + fast for training
- **No reshuffling:** Order is deterministic (can reproduce with same seed)

---

## Execution Model Summary

| Aspect | What Happens |
|--------|--------------|
| **How you run it** | `modal run modal_app.py::clean` (from your local terminal) |
| **Where it runs** | Modal cloud platform (AWS/GCP datacenters) |
| **Workers** | Independent CPU containers, auto-launched in parallel |
| **Parallelism** | Phase 1: 20 workers, Phase 2 Stage 3: 20 workers, Phase 4: 14 workers |
| **Data sharing** | All workers mount same `/data` volume (persistent storage) |
| **Synchronization** | `.starmap()` waits for all workers to finish before continuing |
| **Billing** | Per-worker CPU-seconds (parallel work ≠ higher cost, just faster) |

---

## Why This Architecture?

### Problem: 718K documents to process
- **Naive:** Single process (1 worker) → ~5 hours
- **Modal:** 20 workers in parallel → ~5 minutes (100x speedup)

### Problem: Data volume (2.4B tokens)
- **Naive:** Load entire corpus into RAM → crash (needs ~10GB)
- **Modal streaming:** Iterator over corpus, process one doc at a time → works

### Problem: Long job (Phase 2 = 6 min, Phase 4 = 10 min)
- **Naive:** Single container preempted mid-run → restart from zero
- **Modal sharding:** If 1 of 20 workers preempted, re-run only that shard (~1% of work)

---

## Summary: High-Level Execution Flow

```
LOCAL TERMINAL          MODAL CLOUD
──────────────────────────────────────────
modal run ...::clean
                        ┌─ 20 workers spawn
                        │  (containers boot, load image)
                        │
                        ├─ Workers stream HF datasets
                        │  in parallel (16 tasks)
                        │
                        ├─ Each worker:
                        │  • Downloads shard
                        │  • Cleans docs
                        │  • Writes to /data/clean/
                        │  • Commits volume
                        │
                        └─ Return results
                           (all workers done)
                        
Prints report ◄─────────
(2.68B tokens cleaned)
```

**Total wall-clock time:** ~5 min (not 5 hours, because parallel)  
**Total cost:** < $1 (CPU-seconds cheap on Modal)

---

## Key Takeaway

Modal transforms this:
```
for each of 718K documents:
    clean(doc)       # 1 doc/second if serial → ~8 days
```

Into this:
```
parallel for each of 718K documents (20 workers):
    clean(doc)       # 20 docs/second total → ~10 minutes
```

No code changes to the logic; Modal handles the infrastructure.
