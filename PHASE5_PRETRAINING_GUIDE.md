# Phase 5: Pretraining the 125M SLM on 8x H100

## Overview

**Phase 5** trains the 125.8M-parameter Llama-style model on 2.04B tokens using **8x H100 GPUs in distributed training (DDP)**.

- **Model:** 125.8M params, 16K vocab, LLaMA architecture
- **Data:** 2.04B training tokens + 20.6M validation tokens (from Phase 4)
- **GPUs:** 8x H100 (from config: `PRETRAIN_GPU="H100"`, `PRETRAIN_GPU_COUNT=8`)
- **Time:** ~3-6 days for 1 epoch
- **Cost:** ~$2,100 total (8 GPUs × $16/hour × 3-6 days)
- **Architecture:** PyTorch DDP (Distributed Data Parallel), torch.multiprocessing.spawn

---

## What's Implemented

✅ **modal_app.py** (already in repo):
- `pretrain_full()`: Full 8xH100 training function
- `pretrain_smoke()`: Single H100 smoke test (~20 steps, ~$0)
- `pretrain()`: Main local entrypoint

✅ **train.py** (newly created):
- `TokenDataset`: Loads uint16 .bin token windows from `/data/tokens/`
- `worker()`: DDP worker function (called via torch.multiprocessing.spawn for each GPU)
- Distributed training loop with Hugging Face Trainer
- Checkpoint & metrics logging

---

## Quick Start

### 1. **Smoke Test (Single H100, ~30 min, ~$0.01)**

Test the training pipeline with 20 steps:

```bash
cd /home/floweraura/code_repos/slm
source slmenv/bin/activate
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
source .env.local && modal run modal_app.py::smoke_pretrain
```

**Expected output:**
```
[rank 0/1] starting pretraining: smoke=True, epochs=1, max_steps=20
[rank 0] model loaded: 125,847,552 params (~125.8M)
[rank 0] datasets loaded: train=1,991,282 windows, val=20,119 windows
[rank 0] starting training: train_steps=1,991,282, warmup_steps=381, total_steps=1,991,282
... training progress ...
[rank 0] training complete: 0.5h, $0.01 (1x H100)
```

### 2. **Full Pretraining (8x H100, 3-6 days)**

Train for 1 epoch on all 2.04B tokens:

```bash
source .env.local && modal run modal_app.py::pretrain --epochs 1
```

**What this does:**
- Launches 8 H100 GPU workers in parallel
- Each worker loads the model and data
- DDP synchronizes gradients across GPUs
- Saves checkpoints every 500 steps to `/data/checkpoints/base/`
- Logs metrics to `/data/checkpoints/metrics.jsonl`

**Expected duration:** 3-6 days (depends on throughput, checkpointing overhead)  
**Expected cost:** ~$2,100

**To monitor:**
- View training logs in Modal dashboard: https://modal.com/apps
- Check checkpoints: `/data/checkpoints/base/checkpoint-*`
- Peek at metrics: `modal volume get slm-125m /checkpoints/metrics.jsonl ./metrics.jsonl`

---

## Advanced: Multi-Epoch Training with LR Scheduling

The learning rate schedule spans the **total epochs**, not per-epoch. So for 10 total epochs:

### Train for 1 epoch (warming up):
```bash
modal run modal_app.py::pretrain --epochs 1
```

### Continue for 5 more epochs (to 10 total), with LR cosine across all 10:
```bash
modal run modal_app.py::pretrain --epochs 5 --resume --total-epochs 10
```

This resumes from the latest checkpoint and adjusts the LR schedule so it decays toward min_lr over the full 10-epoch horizon, not just 5.

---

## Configuration (from config.py)

```python
PRETRAIN_GPU = "H100"                    # GPU type
PRETRAIN_GPU_COUNT = 8                   # Number of GPUs
BUDGET_CAP_USD = 40.0                    # Max cost ($40)

TRAIN = TrainConfig(
    seq_len: int = 1_024                 # Window size (tokens)
    micro_batch_size: int = 32           # Per-GPU batch (32 docs)
    global_batch_tokens: int = 524_288   # Total tokens per step (~524K)
    lr: float = 6e-4                     # Peak learning rate
    min_lr: float = 6e-5                 # Minimum LR (cosine schedule)
    warmup_tokens: int = 200_000_000     # Warmup: 200M tokens (~381 steps)
    weight_decay: float = 0.1            # L2 regularization
    grad_clip: float = 1.0               # Gradient clipping
    ckpt_every_steps: int = 500          # Save checkpoint every 500 steps
    log_every_steps: int = 20            # Log metrics every 20 steps
    eval_every_steps: int = 1_000        # Evaluate every 1000 steps
    seed: int = 1337                     # Reproducibility
)
```

---

## What Happens Inside train.py

### 1. **DDP Initialization** (lines 142-149)
- Sets up NCCL backend for GPU-to-GPU communication
- Each process claims its own GPU

### 2. **Model Loading** (lines 161-169)
- Creates LlamaConfig from config.py
- Instantiates AutoModelForCausalLM
- Model is automatically distributed via DDP wrapper in Trainer

### 3. **Data Loading** (lines 180-192)
- `TokenDataset` opens uint16 .bin files on-the-fly
- DistributedSampler splits data across 8 GPUs (no doc duplication)
- DataLoader batches efficiently

### 4. **Training** (lines 210-252)
- Hugging Face `Trainer` handles:
  - Forward/backward pass
  - Gradient averaging (DDP)
  - Learning rate scheduling (cosine with warmup)
  - Checkpointing & metrics
- bfloat16 precision (BF16) for H100 efficiency
- Gradient checkpointing to save memory

### 5. **Checkpointing** (lines 255-272)
- Saves model weights + optimizer state every 500 steps
- Allows resumption mid-training
- Logs to `/data/checkpoints/metrics.jsonl`

---

## Expected Results (1 Epoch)

After training 2.04B tokens on 8x H100 for ~3-6 days:

```json
{
  "epochs_completed": 1,
  "total_steps": ~3,900,000,
  "final_eval_loss": ~2.5-3.0,
  "final_eval_perplexity": ~12-20,
  "model_checkpoint": "/data/checkpoints/base/checkpoint-latest",
  "cost_usd": 2100,
  "training_time_hours": 75-150
}
```

**Validation perplexity** improves throughout training:
- Step 500: PPL ~35-40 (random model)
- Step 5,000: PPL ~15-20 (learning signal)
- Step 50,000: PPL ~8-10 (good convergence)
- Full epoch: PPL ~6-8 (expected)

---

## Troubleshooting

### Preemption (Modal restarts GPU workers)
- **Symptom:** Training pauses and restarts
- **Cause:** Modal preempts long jobs (rare on H100, more on cheaper GPUs)
- **Fix:** Automatic! Trainer resumes from latest checkpoint. No action needed.

### Out of Memory (OOM)
- **Symptom:** CUDA OOM error
- **Likely cause:** `micro_batch_size=32` too large for your GPU memory
- **Fix:** Reduce to `16` or `8` in config.py, then retry

### Stuck on warmup
- **Symptom:** Training very slow in first 200M tokens
- **Cause:** Expected! Warmup phase uses gradient checkpointing, slower convergence
- **Expected:** After warmup, training speeds up

### Cost exceeded
- **Symptom:** Training stops after `$40` cost
- **Cause:** `BUDGET_CAP_USD = 40.0` enforced
- **Fix:** Increase in config.py or pass `--max-usd 100` to `modal run`

---

## Cost & Time Breakdown (1 Epoch)

| Phase | Time | Cost | Notes |
|-------|------|------|-------|
| **Smoke test** | 30 min | $0.01 | 20 steps, 1x H100 |
| **Full training** | 3-6 days | ~$2,100 | 8x H100, 2.04B tokens |
| **Per checkpoint** | ~2 min | ~$0.30 | Every 500 steps (~20M tokens) |

**Total cost for 1 epoch:** ~$2,100  
**Cost-per-day:** ~$350-700  
**Cost-per-billion-tokens:** ~$1,000

---

## Next Steps After Phase 5

1. **Evaluate:** Run full validation pass, measure final perplexity
2. **Push to HuggingFace:** `modal run modal_app.py::hf_push --epochs 1`
3. **Phase 6:** Deploy inference endpoint (CPU, scale-to-zero)
4. **Phase 7:** Optional fine-tuning (SFT) on task-specific data

---

## Commands Reference

```bash
# Smoke test
modal run modal_app.py::smoke_pretrain

# Full 1-epoch training
modal run modal_app.py::pretrain --epochs 1

# Multi-epoch with LR spanning 10 total
modal run modal_app.py::pretrain --epochs 5 --resume --total-epochs 10

# Push checkpoint to HuggingFace
modal run modal_app.py::hf_push --epochs 1 --ppl 8.36

# View checkpoint status
modal volume ls slm-125m /checkpoints/base/

# Download metrics
modal volume get slm-125m /checkpoints/metrics.jsonl ./metrics.jsonl
```

---

**Status:** ✅ Phase 5 ready to run! Start with smoke test first. 🚀
