# Learnings — Phase 5 Pretraining on Modal

Bugs hit while getting 125M SLM pretraining to run on Modal + H100 DDP, and the
checklist that came out of it.

---

## 1. Bugs fixed (and why they happened)

### `mp.spawn` deadlocks inside containers
**Symptom:** NCCL hung, then `wait timeout after 600000ms` (10 min) on rank 1.
Rank 0 died silently; rank 1 waited forever for a peer that never showed up.

**Cause:** `torch.multiprocessing.spawn` inside a Modal container fights with the
container's own process management. Rank 0 crashed during init and nothing
reported it.

**Fix:** Launch with `torchrun --nproc_per_node=N -m train`. It sets
`RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` itself, and a dead rank fails the
whole job loudly instead of hanging.

**Rule:** In containers, use `torchrun`. Never `mp.spawn`.

---

### `mp.spawn` also can't pickle closures
`collate_fn` was defined *inside* `worker()` → `AttributeError: Can't get local
object 'worker.<locals>.collate_fn'`. Anything crossing a process boundary must
be module-level. (Moot once on `torchrun`, but the rule stands.)

---

### DDP wrapping vs. HF Trainer
Two separate faults:
- Not wrapping the model → Trainer fell back to `DataParallel` →
  `RuntimeError: chunk expects at least a 1-dimensional tensor`.
- After wrapping in `DDP`, Trainer called `model.gradient_checkpointing_enable()`
  → `AttributeError`, because that method lives on `model.module`.

**Fix:** Wrap in DDP only when `world_size > 1`, and forward the method:
```python
model.gradient_checkpointing_enable = model.module.gradient_checkpointing_enable
```

**Rule:** A DDP wrapper does **not** proxy arbitrary model methods. Anything the
Trainer reaches for must be forwarded explicitly.

---

### Single-GPU path must skip DDP entirely
`init_process_group` was unconditional → `parallel_mode != ParallelMode.DISTRIBUTED`
warning on 1 GPU. Worse, `destroy_process_group()` was *also* unconditional, so a
successful 1-GPU smoke run would throw at the very end.

**Rule:** Guard **both** init and destroy with `if world_size > 1`. They must match.

---

### OOM: batch size vs. convergence
OOM at `micro_batch_size=32` (loss tried to allocate ~16 GiB; cross-entropy over
`batch × seq_len × vocab` logits is the peak, not the weights).

**Fix that preserves training dynamics:**
```
micro_batch 32, grad_accum 1  ->  micro_batch 16, grad_accum 2
effective batch = micro × accum × gpus  = unchanged
```

**Rule:** Halving `micro_batch_size` alone **halves the effective batch** and changes
convergence (and the LR you should be using). Always compensate with
`gradient_accumulation_steps`. Gradient checkpointing is orthogonal — it trades
compute for memory and does **not** affect convergence.

---

### `Volume.lookup` no longer exists (Modal ≥ 1.x)
The checkpoint-commit callback called `Volume.lookup()` — **removed in Modal 1.5.2**.
It had never run, because every previous run died before reaching the first
checkpoint. It would have killed the first real training run at step 500.

**Fix:** `Volume.from_name(name)` + `.hydrate()`, resolved **once** in `__init__`,
not per save.

**Rule:** Code on a path that has never executed is untested code. Verify library
APIs against the *installed* version (`hasattr`), not from memory.

---

### DDP silently corrupted every saved checkpoint (the worst one)
**Symptom:** Training looked perfect — loss 2.90, no spikes. The saved model then
generated pure gibberish at **val perplexity 18,629** (≈ vocab_size = random).

**Cause:** We passed an **already-DDP-wrapped** model to HF `Trainer`. Trainer
serialized the *wrapper's* `state_dict`, so every key came out prefixed:
```
saved:           module.model.layers.0.self_attn.q_proj.weight
from_pretrained expects:  model.layers.0.self_attn.q_proj.weight
```
`from_pretrained` matched **zero** keys, discarded all 125.8M trained parameters,
re-initialized the network from scratch — and emitted only a **warning**, not an
error. All 110 tensors were affected.

**Fix (future runs):** unwrap by hand before saving; never trust the Trainer to do it.
```python
to_save = model.module if world_size > 1 else model
to_save.save_pretrained(config.BASE_CKPT_DIR, safe_serialization=True)
```
**Fix (existing checkpoints):** the weights are fine, only the *names* are wrong.
Strip the prefix and rebuild — no retraining. See `repair_checkpoint_fn`.

**Rules:**
- **A "warning" from `from_pretrained` about newly-initialized weights is an ERROR.**
  Never ignore it.
- **Always evaluate a checkpoint before publishing it.** A loss curve proves training
  worked; it says *nothing* about whether what you saved is what you trained.
- **`val_ppl ≈ vocab_size` means the weights are random.** Fastest possible check.
- Trainer writes to `checkpoint-N/`; `from_pretrained(BASE_CKPT_DIR)` reads the
  **root**. Call `save_model()` / `save_pretrained()` explicitly or the root is empty.

---

### Commit on checkpoint, not on a timer
A 5-minute `volume.commit()` timer in the orchestrator interrupts batches and
commits when nothing changed.

**Fix:** A `TrainerCallback.on_save` hook — commit exactly when Trainer has just
written a checkpoint. Rank 0 only.

**Rule:** Never let a long run finish before its first commit. If the container is
preempted or hits the 24h timeout, **uncommitted checkpoints are gone.**

---

## 2. GPU starvation checklist (SLM pretraining)

If GPU utilization is low, the GPU is waiting on something. Check in this order.

### Data path — the usual culprit
- [ ] **`__getitem__` is O(1).** Ours scanned every shard with `np.fromfile` per
      window — reading whole files to fetch 1024 tokens. Precompute a cumulative
      window-offset array at `__init__` (from `os.path.getsize`, no reads) and
      binary-search it.
- [ ] **Use `np.memmap`, not `np.fromfile`.** `fromfile` reads the whole file every
      call. Open memmaps lazily so each DataLoader worker gets its own handle.
- [ ] **Stage data on local NVMe.** A network-mounted volume cannot feed an H100.
      Copy tokens to local disk once at startup; read training data from there.
      (Keep *checkpoint writes* going to the volume.)
- [ ] **Collate in one bulk copy.** `np.stack` + `torch.from_numpy` beats N
      per-sample `torch.tensor()` calls.
- [ ] **`dataloader_num_workers` > 0** and `dataloader_pin_memory=True`.
- [ ] **Do tokenization offline.** Never tokenize in the training loop.

### Compute config
- [ ] `bf16=True` on H100.
- [ ] Gradient checkpointing **only if memory-bound** — it re-runs the forward pass.
      If a smaller batch already fits, drop it and get the speed back.
- [ ] `use_cache=False` when gradient checkpointing (they're incompatible; HF will
      silently override anyway).
- [ ] Batch is large enough to saturate the SMs; raise it until just under OOM.

### Distributed
- [ ] `ddp_find_unused_parameters=False` (True adds a full graph traversal per step).
- [ ] Watch for stragglers — DDP runs at the speed of the **slowest** rank; one slow
      shard or an uneven data split stalls everyone.
- [ ] Rank 0 shouldn't do extra work (logging/saving/eval) that other ranks wait on.

### Diagnosing
- [ ] `nvidia-smi` during a step: high util = compute-bound (good); low + spiky =
      input-starved.
- [ ] Time a pure dataloader loop with no model. If it can't outrun the GPU's
      step time, the model is not your problem.
- [ ] Compare step time at `num_workers=0` vs `4`. A big gap = I/O-bound.

---

## 3. Process

- **Smoke test first.** 1 GPU, 20 steps, ~$0.01, before a $2,100 8×H100 run.
- **Scale in stages: 1 → 2 → 8 GPUs.** Each step exposes a different class of bug —
  1 GPU never exercises NCCL; 2 GPUs surface every DDP fault that 8 will.
- **Reproduce on the smallest config that shows the bug.** The OOM and the NCCL
  hang both reproduced on 2 GPUs at ~$50, not $2,100.
