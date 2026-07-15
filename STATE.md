# State of Play

_Updated 2026-07-14. Companion to [LEARNINGS.md](LEARNINGS.md) (why each bug happened)._

## Status: base model trained + served. SFT dataset built + fine-tuned.

| Phase | | |
|---|---|---|
| 0-4 | Data → tokenizer → packed tokens | done |
| 5 | Pretrain 125M on 2.04B tokens | done — 1 epoch, 1.07h, ~$18, **val ppl 10.93** |
| 6 | Inference endpoint (base model) | live |
| 7a | Grounded Q&A SFT set (Gemini) | done — **14,924 pairs**, ~$10 |
| 7b | Instruction fine-tune | done — 4.5 min, $0.31, **val_loss 1.203** |
| 7c | Evaluation (deterministic) | done — see below. **Model does NOT read context.** |
| 7d | Judged correctness | BLOCKED — Gemini prepaid credits depleted |
| 7e | Push to HF Hub | NOT STARTED |

## Phase 7c evaluation — the headline finding

Ran all 746 held-out val examples through SFT and base, plus a swapped-context probe.
Deterministic metrics only (Gemini credits ran out; judged-correctness rows are pending).

```
                          SFT       base
hallucination rate        24%       100%    [lower better]  answered an unanswerable Q
refusal recall            76%         0%
false-refusal rate         9%         0%    [lower better]  refused an answerable Q
invented-figure rate       1%        14%    [lower better]
--------------------------------------------------------
SWAPPED-CONTEXT refusal   14%          -     <-- THE PROBLEM
```

**Fine-tuning worked in the obvious sense**: the base model answers everything (100%
hallucination, 0% refusal); the SFT model refuses 76% of unanswerable questions and
barely invents figures. It learned the answer/refuse *format*.

**But the swapped-context probe shows it does not actually read the context.** Replace an
answerable question's passage with an unrelated one (question now unanswerable) and the
model STILL answers 86% of the time (only 14% refuse). It confabulates from parametric
memory — e.g. asked for a brief date while shown a passage about Katherine Parr, it
replied "July 15, **1548**", stealing the year from the wrong passage. Of 541 such
confabulations, 19% invented a number absent even from the passage shown.

Implication: the model pattern-matches on the *question* and recites what it memorized,
rather than grounding in the provided passage — which is the entire premise of RAFT. The
76% refusal recall is real but shallow: it refuses when the *question* looks
unanswerable, not when the *context* fails to support it.

Scored files on volume: `/data/sft/eval/{sft,sft_swap,base}_scored.jsonl`.

## Two models now exist — do not confuse them

| | Base (ours) | SFT (fine-tuned) |
|---|---|---|
| weights | `/data/checkpoints/base` | `/data/checkpoints/sft` |
| trained from | scratch, by us, 2.04B tokens | **`thesreedath/slm-125m-base`** |
| tokenizer | ours (`/data/tokenizer`) | **theirs** |
| does | continues text | answers from context / refuses |
| val | ppl 10.93 | loss 1.203 |

**The two tokenizers are NOT interchangeable.** Both are 16,384-vocab byte-level BPE, but
the merges were learned on different corpora, so the ids differ. Token ids only mean
anything against the embedding table trained beside them. The SFT model was built on
thesreedath's base per the brief, so **our own pretrained model is currently unused** by
the SFT line.

## The SFT model works — with a real caveat

4/4 behaviour probes pass: it answers from the context and it says
*"Not stated in the context."* when the answer is absent (2/2 refusals).

**But it makes attribution errors.** Asked why the Ninth Circuit reversed, it answered
*"because the **plaintiff** had concealed the injury"* — the passage says **defendant**.
Right format, right grounding instinct, wrong party. At 125M this is expected; in legal
text it is exactly the kind of error that matters. Do not present this model as reliable
for substantive legal use. Say so on the model card.

## SFT dataset (Phase 7a)

```
14,924 clean pairs  (train 14,178 / val 746)
/data/sft/v1_{train,val}.jsonl        chat JSONL (system/user/assistant)
/data/sft/tokens/v1_{train,val}.npz   input_ids, prompt_len, seq_len
8.0M train tokens | mean len 565 | only 5% carry loss (prompt is masked -- correct)
```

- **Teacher + judge**: `gemini-3.1-flash-lite` (2.5-flash was retired for new users
  mid-project; 3.5-flash costs 6x and 3-flash-preview 503s).
- **Keep rate 72%.** Dropped: 25% judge_fail, 3% unverifiable quote, plus 560 exact-dup
  + 1,442 near-dup questions at merge.
- **Refusals capped at 15%.** The QA prompt yields ~25%, which over-trains declining.
- **Contamination: 1 hit** out of 1.27M eval 13-grams — because passages were sampled
  from `/data/corpus/` (deduped + decontaminated), not `/data/clean/`.

**Task mix is 96% QA**, not the intended 70/12/10/8. Cause: the judge prompt demands a
verbatim supporting span, which summarize/rewrite/extract can never have (they transform
the whole passage). Shipped as-is by choice — the model is a grounded-QA specialist. To
fix: make the judge task-aware and top up the three starved tasks (~$4).

## Anti-self-preference design (teacher and judge are the same model)

The judge over-accepts its own output, so its verdict is never trusted alone:

1. it must return a **verbatim span** from the passage, and
2. `sft_data.is_grounded` checks that span **actually occurs** there.

A judge rubber-stamping a hallucination still has to fabricate a quote, and the string
check catches it — 660 pairs were killed this way. For non-QA tasks (no span possible),
`no_invented_figures` is the backstop: every number in the answer must appear in the
passage.

## Commands

```bash
modal run modal_app.py::sft_gen            # 7a: build SFT set (--smoke first)
modal run modal_app.py::sft_tok            # 7a: tokenize w/ thesreedath tokenizer
modal run modal_app.py::sft_train          # 7b: fine-tune 1x H100 (--smoke first)
modal run modal_app.py::sft_eval           # 7b: probe answering + refusal
modal run modal_app.py::evaluate           # base model: val ppl + samples
modal deploy modal_app.py                  # (re)deploy the base-model playground
```

## Live endpoint (base model, not SFT)

https://mkumar9009--slm-125m-inference-web.modal.run — CPU, scale-to-zero, $0 idle.
**Serves the BASE model.** It has not been repointed at the SFT checkpoint.

## Open issues

- **`BUDGET_CAP_USD` / `max_usd` is dead code.** No budget enforcement exists anywhere.
- **`config.TRAIN.min_lr` (6e-5) is unused** — HF cosine anneals to 0.
- **Multi-epoch continuation of pretraining does not work**: `total_steps` is computed but
  never passed to `TrainingArguments`, so `--total-epochs` has no effect.
- `_parquet_urls` still points at `datasets-server.huggingface.co`, which is permanently
  503. Phase 2 relied on it. `_build_contamination_ngrams` now bypasses it with direct
  file URLs; anything else calling it will silently get nothing.

## Next

1. Point the web endpoint at the SFT model (currently serves the base).
2. Push to HF: `config.HF_REPO = mkumar9009/slm-125m-base`, secret `huggingface-token`
   exists on Modal. Model card MUST carry the attribution-error and fabrication caveats.
3. Optional: fix the task mix (~$4) if summarize/extract/rewrite ability is wanted.
