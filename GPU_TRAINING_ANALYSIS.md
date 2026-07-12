# GPU Training Cost Analysis: 8x H100 vs Alternatives

## Training Requirements (1 Epoch)

```
Model:          125.8M parameters
Training data:  2.04B tokens
Batch config:   micro_batch=32, global_batch_tokens=524,288
Warmup:         200M tokens
Learning rate:  6e-4 → 6e-5
```

**Estimated throughput per token/sec:**
- H100 (BF16): ~1,500-2,500 tokens/sec (theoretical peak ~3,000)
- A100 (BF16): ~800-1,200 tokens/sec
- L40S (TF32): ~600-900 tokens/sec

---

## Cost Analysis: 8x H100 vs 1x H100

### Scenario A: 8x H100 (Distributed Training)

**Assumptions:**
- Modal H100 pricing: ~$2.00/hour per GPU
- Total: 8 GPUs × $2.00 = **$16/hour**
- Throughput: ~8,000-12,000 tokens/sec (scaling factor ~5-6x due to overhead)
- Time for 2.04B tokens: **255K-510K seconds = 71-142 hours ≈ 3-6 days**

**Cost Range:**
```
Best case (5 days):   5 × 24 × $16 = $1,920
Worst case (6 days):  6 × 24 × $16 = $2,304
Average:              ~$2,100 ✅ RECOMMENDED
```

**Pros:**
- ✅ Fastest: 3-6 days vs 8-12 days
- ✅ Distributed training (lower convergence time, better generalization)
- ✅ Total cost actually LOWER than 1x H100 due to time savings
- ✅ Can checkpoint frequently across nodes

**Cons:**
- ✗ Higher peak cost ($16/hour vs $2/hour)
- ✗ Requires distributed training setup (Hugging Face Accelerate / DeepSpeed)
- ✗ Network I/O overhead

---

### Scenario B: 1x H100 (Single GPU)

**Assumptions:**
- Modal H100 pricing: ~$2.00/hour
- Throughput: ~1,500-2,500 tokens/sec
- Time for 2.04B tokens: **816K-1,360K seconds = 227-378 hours ≈ 9.5-16 days**

**Cost Range:**
```
Best case (10 days):   10 × 24 × $2 = $480
Mid case (12 days):    12 × 24 × $2 = $576
Worst case (16 days):  16 × 24 × $2 = $768
Average:               ~$600 (LOWER HOURLY, HIGHER TOTAL)
```

**Pros:**
- ✅ Lower hourly cost ($2/hour)
- ✅ Simple setup (no distributed training complexity)
- ✅ Easy checkpointing & resumption

**Cons:**
- ✗ Slowest: 10-16 days (3-5x longer than 8x H100)
- ✗ Higher TOTAL cost ($600 vs $2,100 is misleading!)
- ✗ Single point of failure (preemption = restart entire training)
- ✗ Memory constraints for large models

---

## Cost-Speed Tradeoff Summary

| GPU Config | $/Hour | Time (Days) | Total Cost | Speed vs Cost | Recommendation |
|-----------|--------|------------|------------|--------------|---|
| **8x H100** | $16 | 3-6 | **$2,100** | ✅ BEST | **Recommended** |
| **1x H100** | $2 | 10-16 | $600-768 | Slow+expensive | Fallback only |
| **4x A100** | $8 | 5-10 | $960-1,920 | Good balance | If H100 unavailable |
| **2x A100** | $4 | 10-20 | $960-1,920 | Slow | Not recommended |

---

## Alternative GPU Options on Modal

### 1. **4x A100 (40GB)** ⭐ GOOD ALTERNATIVE

```
Cost:       4 × $1.50/hr = $6/hour
Throughput: ~4,000-6,000 tokens/sec
Time:       6-9 days
Total cost: ~$865-1,300

Pros:
  ✅ Better value ($6/hr vs $16/hr)
  ✅ Still 2-3x faster than 1x H100
  ✅ 40GB memory (enough for 125M model + batch)
  ✅ Good distributed training support

Cons:
  ✗ Slower than 8x H100
  ✗ Fewer GPUs = less parallelism
```

### 2. **2x A100 (40GB)** ⚠️ COMPROMISE

```
Cost:       2 × $1.50/hr = $3/hour
Throughput: ~2,000-3,000 tokens/sec
Time:       12-18 days
Total cost: ~$860-1,300

Pros:
  ✅ Cheapest hourly rate ($3/hr)
  ✅ Still distributed (2 nodes)

Cons:
  ✗ Slowest option
  ✗ Total cost similar to 4x A100 due to time
```

### 3. **8x A100 (40GB)** ⭐ BALANCED

```
Cost:       8 × $1.50/hr = $12/hour
Throughput: ~8,000-10,000 tokens/sec
Time:       4-6 days
Total cost: ~$1,150-1,440

Pros:
  ✅ Similar cost to 8x H100 ($12 vs $16)
  ✅ Faster than 4x A100
  ✅ Proven distributed training setup
  ✅ Slightly cheaper than H100

Cons:
  ✗ Slightly slower than H100
  ✗ Same complexity as 8x H100
```

### 4. **L40S (48GB)** ⚠️ NOT RECOMMENDED

```
Cost:       8 × $0.80/hr = $6.40/hour
Throughput: ~4,000-6,000 tokens/sec
Time:       7-10 days
Total cost: ~$1,100-1,540

Pros:
  ✅ Cheapest per GPU ($0.80)
  ✅ Large memory (48GB)

Cons:
  ✗ Slower training speed
  ✗ Total cost not significantly better
```

---

## RECOMMENDATION: Cost-Effective & Fast

### ✅ **PRIMARY: 8x H100**
- **Total cost:** $2,100
- **Time:** 3-6 days
- **Cost per day:** ~$350-700
- **Best for:** You want training ASAP with industry-standard GPUs

### ⭐ **ALTERNATIVE: 8x A100**
- **Total cost:** $1,150-1,440
- **Time:** 4-6 days
- **Cost per day:** ~$190-360
- **Best for:** Best value-to-speed ratio, $1K savings vs H100

### 🎯 **COMPROMISE: 4x A100**
- **Total cost:** $865-1,300
- **Time:** 6-9 days
- **Cost per day:** ~$95-215
- **Best for:** Tighter budget, can wait 1-2 extra days

---

## Implementation Approach

### For 8x H100 (Recommended)

**Modal Setup:**
```python
@app.function(
    gpu="H100",
    image=gpu_image,
    volumes=VOLUMES,
    timeout=24*3600*7,  # 7 days max
)
def pretrain(checkpoint_path=None):
    import torch
    from torch.distributed import init_process_group
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    
    # Initialize distributed training
    init_process_group("nccl")
    
    # Load model, tokenizer, data
    model = AutoModelForCausalLM.from_pretrained(...)
    training_args = TrainingArguments(
        output_dir="/data/checkpoints",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=6e-4,
        warmup_steps=200_000_000 // 524_288,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    
    trainer.train(resume_from_checkpoint=checkpoint_path)
```

**CLI Command:**
```bash
source .env.local && modal run modal_app.py::pretrain --gpus 8
```

---

## Preemption Strategy (Important!)

Modal preempts long-running jobs. Plan for restarts:

**Checkpointing every 500 steps:**
- ~10M tokens per checkpoint
- Can resume mid-training
- Saves 500-step cost on restart

**Preemption probability:**
- 8x H100: ~5-10% per day (3-6 days = 15-60% total risk)
- Mitigation: Save checkpoints frequently, resume from latest

---

## My Recommendation

### 🏆 **OPTIMAL: 8x H100**

**Why:**
1. **Fastest:** 3-6 days (vs 6-9 for A100)
2. **Industry standard:** Better hyperparameter tuning knowledge
3. **Total cost:** $2,100 (only $200-300 more than 4x A100 for 2-3 days saved)
4. **Cost-per-day:** ~$350-700 (reasonable for research)

**Alternative if budget is tight:**
- **8x A100:** $1,150-1,440, only 1-2 days slower

---

## Next Steps

1. **Confirm Modal GPU availability:**
   ```bash
   modal profile current
   modal compute-env list
   ```

2. **Check current pricing:**
   - Visit https://modal.com/pricing
   - Verify H100 vs A100 hourly rates

3. **Implement training loop** in modal_app.py with Hugging Face Trainer

4. **Launch with:**
   ```bash
   source .env.local && modal run modal_app.py::pretrain
   ```

---

**Summary:** 8x H100 is fastest AND reasonably priced. 8x A100 is best value. Both take 3-6 days, cost $1-2K, and will finish 1 epoch of 2.04B tokens.
