# State of Play

_Updated 2026-07-12. Companion to [LEARNINGS.md](LEARNINGS.md) (why each bug happened)._

## Status: Phases 0-6 DONE. Model trained, verified, and serving.

| Phase | | |
|---|---|---|
| 0-4 | Data → tokenizer → packed tokens | done |
| 5 | Pretrain 125M on 2.04B tokens | **done** — 1 epoch, 1.07h, ~$18 |
| 6 | Inference endpoint | **live** |
| 7 | HF Hub push | not started |

## The model

```
125,848,320 params · 12L/768d/12h · vocab 16,384 · seq_len 1024
val loss 2.392 · val perplexity 10.93   (800 held-out windows)
weights: /data/checkpoints/base  (Modal Volume slm-125m)
```

Trained 1 epoch on 2.039B tokens: 40% case-law + 40% SEC + 20% fineweb-edu.

Generates fluent legal and financial prose — Miranda doctrine, contract boilerplate,
SEC MD&A structure, Bluebook citation format. **Citations and figures are fabricated.**
It learned the *shape* of legal text, not case law. Base LM for further tuning; not a
factual source. Say so on the model card.

## Live endpoint (Phase 6)

```
https://mkumar9009--slm-125m-inference-complete.modal.run

curl -X POST $URL -H 'Content-Type: application/json' \
  -d '{"prompt":"The defendant moved to suppress", "max_new_tokens":80, "temperature":0.8}'
```
CPU, scale-to-zero. Cold start ~18s, warm ~4s, **$0 idle**.

## Training config (do not drift)

```
GPUs         4x H100          config.PRETRAIN_GPU_COUNT
micro_batch  16               sets peak VRAM — OOMs at 32
grad_accum   8                DERIVED, never hardcode
tokens/step  524,288          invariant across GPU count
steps/epoch  3,889
warmup       381 steps = 200M tokens
lr           6e-4 cosine
```
**The invariant:** `grad_accum = global_batch_tokens // (micro_batch × seq_len × world_size)`.
Two asserts in `train.py` fire at startup if it doesn't divide. Changing GPU count needs
**no other edits** — batch and LR self-correct.

**Judge health by tokens/sec/GPU (~135k), never it/s** — it/s moves with accum.
Achieved 132–137k with ~100% DDP scaling at 4 GPUs.

## Commands

```bash
modal run modal_app.py::pretrain --epochs 1   # train
modal run modal_app.py::promote               # checkpoint-N/ -> BASE_CKPT_DIR root
modal run modal_app.py::repair                # strip DDP "module." prefix (see below)
modal run modal_app.py::evaluate              # val ppl + sample completions
modal deploy modal_app.py                     # (re)deploy the endpoint
```

## Open issues

- **`BUDGET_CAP_USD` / `max_usd` is dead code.** Threaded into `worker()`, never read.
  **No budget enforcement exists anywhere.**
- **`config.TRAIN.min_lr` (6e-5) is unused.** HF `lr_scheduler_type="cosine"` anneals to
  **0**. Fine for 1 epoch; matters if you resume.
- **Multi-epoch continuation does not work.** `total_steps` is computed in `train.py` but
  never passed to `TrainingArguments`, so `--total-epochs` has no effect and the LR has
  already annealed to 0. Fix before Phase 5b.
- `repair` is only needed for checkpoints written **before** the `save_pretrained` fix.
  New runs save clean keys directly.

## Next

Phase 7: push to `mkumar9009/slm-125m-base` (`HF_SECRET_NAME` = `huggingface-token`,
secret already exists on Modal). Model card must carry the fabricated-citation caveat.

Loss was still descending at epoch 1 — more epochs (~$18 and ~1h each) would improve it,
but fix multi-epoch continuation first or the LR schedule will be wrong.
