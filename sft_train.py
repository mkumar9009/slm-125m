"""Phase 7b: instruction fine-tuning of thesreedath/slm-125m-base on the grounded Q&A set.

Single GPU on purpose. The job is ~5 minutes of compute, so DDP setup and container boot
would dominate the wall clock -- and every failure mode we hit in pretraining (the
"module." state_dict corruption, NCCL init, forwarding gradient_checkpointing through the
DDP wrapper) lives on the world_size > 1 path. None of it runs here.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config
import sft_data


class SFTDataset(Dataset):
    """Packed SFT examples. Loss is masked everywhere except the assistant turn."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.input_ids = d["input_ids"]     # (N, seq_len) uint16
        self.prompt_len = d["prompt_len"]   # (N,)
        self.seq_len = d["seq_len"]         # (N,)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        ids = self.input_ids[i].astype(np.int64)
        p, s = int(self.prompt_len[i]), int(self.seq_len[i])

        labels = ids.copy()
        labels[:p] = -100     # the prompt: context + question. Training on it would
        labels[s:] = -100     # teach the model to invent contexts. And the padding.

        attn = np.zeros_like(ids)
        attn[:s] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def collate(batch: list[dict]) -> dict:
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in batch[0]}


def main(args: dict) -> None:
    t0 = time.time()
    tag = args.get("tag", "v1")
    epochs = args.get("epochs", config.SFT.epochs)
    lr = args.get("lr", config.SFT.lr)
    base = args.get("base_model", "thesreedath/slm-125m-base")
    smoke = args.get("smoke", False)

    torch.cuda.set_device(0)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

    # The base model's config.json has WRONG token ids (eos=2, but the real <|eos|> is 1;
    # id 2 is <|pad|>). Left uncorrected, generate() waits for a token the model never
    # emits and rambles forever. Pin them to the tokenizer's truth so the saved model
    # stops on its own without callers passing eos_token_id by hand.
    for cfg in (model.config, model.generation_config):
        cfg.bos_token_id = tok.convert_tokens_to_ids("<|bos|>")
        cfg.eos_token_id = tok.convert_tokens_to_ids("<|eos|>")
        cfg.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")

    model.config.use_cache = False          # incompatible with training; HF warns otherwise
    model.to("cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sft] base={base}  {n_params:,} params  vocab={tok.vocab_size}", flush=True)

    tokens_dir = f"{config.DATA_ROOT}/sft/tokens"
    train_ds = SFTDataset(f"{tokens_dir}/{tag}_train.npz")
    val_ds = SFTDataset(f"{tokens_dir}/{tag}_val.npz")

    eff_batch = config.SFT.micro_batch_size * config.SFT.grad_accum
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * config.SFT.warmup_frac))
    print(f"[sft] train={len(train_ds):,} val={len(val_ds):,} | eff_batch={eff_batch} "
          f"| {steps_per_epoch} steps/epoch x {epochs} = {total_steps} steps "
          f"| warmup={warmup} | lr={lr}", flush=True)

    targs = TrainingArguments(
        output_dir=f"{config.SFT_CKPT_DIR}/_run",
        num_train_epochs=epochs,
        max_steps=20 if smoke else -1,
        per_device_train_batch_size=config.SFT.micro_batch_size,
        per_device_eval_batch_size=config.SFT.micro_batch_size,
        gradient_accumulation_steps=config.SFT.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        weight_decay=config.SFT.weight_decay,
        max_grad_norm=config.SFT.grad_clip,
        bf16=True,
        # No gradient checkpointing: a 125M model at seq 1024 fits easily, and
        # checkpointing would only re-run the forward pass for no benefit.
        gradient_checkpointing=False,
        logging_steps=20,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no",              # we save once, explicitly, at the end
        report_to=[],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=config.SFT.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=None if smoke else val_ds,
        processing_class=tok,
        data_collator=collate,
    )
    trainer.train()

    # ---- save ----------------------------------------------------------------
    # save_pretrained on the model itself, never through a wrapper: that is what
    # produced "module."-prefixed keys and a silently random model in Phase 5.
    os.makedirs(config.SFT_CKPT_DIR, exist_ok=True)
    model.config.use_cache = True                     # restore for inference
    model.save_pretrained(config.SFT_CKPT_DIR, safe_serialization=True)

    # Bake the chat template in, so callers never have to reconstruct the format
    # this model was actually trained on.
    tok.chat_template = sft_data.CHAT_TEMPLATE
    tok.save_pretrained(config.SFT_CKPT_DIR)

    metrics = {
        "phase": "sft",
        "base_model": base,
        "tag": tag,
        "examples": len(train_ds),
        "epochs": epochs,
        "lr": lr,
        "steps": total_steps,
        "elapsed_min": (time.time() - t0) / 60,
        "cost_usd": (time.time() - t0) / 3600 * config.H100_USD_PER_HOUR,
    }
    if not smoke:
        ev = trainer.evaluate()
        metrics["val_loss"] = ev.get("eval_loss")
    with open(config.METRICS_PATH, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")

    print(f"\n[sft] done in {metrics['elapsed_min']:.1f} min "
          f"(~${metrics['cost_usd']:.2f}) -> {config.SFT_CKPT_DIR}", flush=True)
    if "val_loss" in metrics:
        print(f"[sft] val_loss {metrics['val_loss']:.4f}", flush=True)


if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sft_args.json") as fh:
        main(json.load(fh))
