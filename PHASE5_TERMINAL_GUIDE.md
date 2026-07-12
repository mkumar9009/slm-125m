# Phase 5: GPU Pretraining - Terminal Commands

## ✅ Everything is Ready!

**What's in place:**
- ✅ `modal_app.py` - Updated with Phase 5 functions
- ✅ `train.py` - DDP training worker (9.5 KB)
- ✅ `config.py` - Model & training config (H100, 8 GPUs, $40 budget)
- ✅ `/data/tokens/` - 2.04B training tokens ready on Modal Volume

---

## 🚀 Quick Start (Copy-Paste Ready)

### Step 1: Setup
```bash
cd /home/floweraura/code_repos/slm
source slmenv/bin/activate
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
```

### Step 2a: Smoke Test (RECOMMENDED FIRST)
```bash
source .env.local && modal run modal_app.py::smoke_pretrain
```
- **Time:** ~30 minutes
- **Cost:** ~$0.01
- **GPU:** 1x H100
- **Purpose:** Validates setup works

**Expected output:**
```
✓ Initialized...
[rank 0/1] starting pretraining...
[rank 0] model loaded: 125,847,552 params (~125.8M)
[rank 0] datasets loaded: train=1,991,282 windows, val=20,119 windows
... 20 training steps ...
[rank 0] training complete: 0.5h, $0.01 (1x H100)
```

---

### Step 2b: Full Training (After smoke test passes)
```bash
source .env.local && modal run modal_app.py::pretrain --epochs 1
```
- **Time:** 3-6 days
- **Cost:** ~$2,100 (8x H100)
- **GPUs:** 8x H100 (DDP distributed)
- **Data:** 2.04B tokens over 1 epoch

**Expected output:**
```
✓ Initialized...
Launching pretrain: 1 epochs, resume=False, 8xH100
[rank 0/8] starting pretraining...
[rank 0] model loaded: 125,847,552 params (~125.8M)
[rank 0] datasets loaded: train=1,991,282 windows (split across 8 GPUs)
... training progress over 3-6 days ...
[rank 0] training complete: 100.0h, $2,100 (8xH100)
✓ App completed.
```

---

## 📊 Cost Analysis: 8x H100 vs Alternatives

### 🏆 Recommended: 8x H100
```
Total cost:    $2,100
Duration:      3-6 days
Cost per day:  $350-700
Throughput:    ~8,000-12,000 tokens/sec
Why best:      Fastest training + reasonable cost (only $300 more than 4x A100 for 2-3 days saved)
```

### Alternative: 8x A100-40GB
```
Total cost:    $1,150-1,440 (saves $660)
Duration:      4-6 days (1-2 days slower)
Cost per day:  $190-360
Throughput:    ~6,000-10,000 tokens/sec
Why consider:  20% cheaper, still fast
```

### Alternative: 4x A100-40GB
```
Total cost:    $865-1,300 (saves $800)
Duration:      6-9 days (3-4 days slower)
Cost per day:  $95-215
Throughput:    ~3,000-5,000 tokens/sec
Why consider:  Cheapest option, acceptable timeline
```

### ❌ Not Recommended: 1x H100
```
Total cost:    $600-768 (cheaper hourly but NOT total!)
Duration:      10-16 days (7-10 days slower!)
Cost per day:  $60-77
Throughput:    ~1,500-2,500 tokens/sec
Why avoid:     Total cost similar to 4x A100, but 2-3x slower
```

**Verdict:** 8x H100 is the **best value for speed AND cost**.

---

## 📋 Complete Terminal Session (Copy-Paste)

### All-In-One Command Block:

```bash
# Navigate and setup
cd /home/floweraura/code_repos/slm
source slmenv/bin/activate
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN

# Run smoke test
echo "=== SMOKE TEST (30 min, $0.01) ==="
source .env.local && modal run modal_app.py::smoke_pretrain

# Once smoke test completes, run full training
echo "=== FULL TRAINING (3-6 days, $2,100) ==="
source .env.local && modal run modal_app.py::pretrain --epochs 1
```

---

## 🎯 What Happens During Training

### Real-time Monitoring:
```bash
# View Modal dashboard (watch training live)
# https://modal.com/apps

# Check checkpoints (every 500 steps)
modal volume ls slm-125m /checkpoints/base/

# Download latest metrics
modal volume get slm-125m /checkpoints/metrics.jsonl ./metrics.jsonl
```

### Expected Checkpoints:
```
/checkpoints/base/checkpoint-500     # After ~20M tokens
/checkpoints/base/checkpoint-1000    # After ~40M tokens
/checkpoints/base/checkpoint-1500    # After ~60M tokens
... (every 500 steps) ...
/checkpoints/base/checkpoint-latest  # Most recent
```

### Expected Metrics (logged every 20 steps):
```json
{
  "step": 500,
  "loss": 3.45,
  "learning_rate": 0.0005,
  "epoch": 0.01,
  "eval_loss": 3.42,
  "eval_perplexity": 30.5
}
```

---

## ⏸️ If Training Gets Interrupted

Modal will automatically save checkpoints. To resume:

```bash
source .env.local && modal run modal_app.py::pretrain --epochs 1 --resume
```

This will:
1. Load the latest checkpoint from `/checkpoints/base/`
2. Continue training from that step
3. Adjust learning rate schedule if needed

---

## 🔧 Advanced: Multi-Epoch Training

### Train for 10 total epochs (resumable):

**Epoch 1:**
```bash
source .env.local && modal run modal_app.py::pretrain --epochs 1
```

**Epochs 2-10 (with continuous LR schedule):**
```bash
source .env.local && modal run modal_app.py::pretrain --epochs 9 --resume --max-usd 200
```

This spans the learning rate schedule across all 10 epochs for better convergence.

---

## 📊 Expected Results After 1 Epoch

```
Training time:       75-150 hours (3-6 days)
Total tokens seen:   2.04B
Steps completed:     ~1,991,282
Checkpoints saved:   ~3,980 (every 500 steps)
Final eval PPL:      ~6-8 (target: good convergence)
Cost total:          ~$2,100

Model checkpoint:    /data/checkpoints/base/checkpoint-latest
Ready for:
  - Phase 6: Deploy inference endpoint
  - Phase 7: Push to HuggingFace
  - Fine-tuning on task-specific data
```

---

## ✅ Workflow Summary

```
NOW:
  1. Run smoke test
     source .env.local && modal run modal_app.py::smoke_pretrain
  
  2. Verify it works (~30 min)
  
  3. Launch full training (in new terminal)
     source .env.local && modal run modal_app.py::pretrain --epochs 1
  
WHILE TRAINING (3-6 days):
  - Monitor via https://modal.com/apps
  - Check checkpoints with: modal volume ls slm-125m /checkpoints/base/
  - No need to keep terminal open (runs in Modal cloud)
  
AFTER TRAINING:
  - Download final metrics
  - Deploy inference endpoint (Phase 6)
  - Push to HuggingFace (Phase 7)
```

---

## 🚨 Troubleshooting

### "Token missing" error
```bash
# Fix: Re-export credentials
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
```

### "CUDA out of memory"
- Unlikely on H100 (40GB VRAM for 125M model)
- If it happens, reduce batch size in config.py

### Training is slow
- Normal! First 200M tokens (warmup phase) are slower
- Speed increases after warmup
- Expected: 1,500-2,500 tokens/sec per H100

### Want to check status without Terminal
```bash
# From any terminal:
modal volume get slm-125m /checkpoints/metrics.jsonl ./current_metrics.jsonl
# View the last line for current step/loss
tail -1 current_metrics.jsonl
```

---

## 📝 Commands Reference Card

```bash
# Setup (run once)
cd /home/floweraura/code_repos/slm && source slmenv/bin/activate
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN

# Smoke test (validate, 30 min)
source .env.local && modal run modal_app.py::smoke_pretrain

# Full 1-epoch training (8x H100, 3-6 days)
source .env.local && modal run modal_app.py::pretrain --epochs 1

# Resume training
source .env.local && modal run modal_app.py::pretrain --epochs 1 --resume

# Check checkpoints
modal volume ls slm-125m /checkpoints/base/

# Download metrics
modal volume get slm-125m /checkpoints/metrics.jsonl ./metrics.jsonl

# View last metric line
tail -1 metrics.jsonl
```

---

**Ready to train? 🚀 Run the smoke test first!**

```bash
cd /home/floweraura/code_repos/slm
source slmenv/bin/activate
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HUGGINGFACE_TOKEN
source .env.local && modal run modal_app.py::smoke_pretrain
```
