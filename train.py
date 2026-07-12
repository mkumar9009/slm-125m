"""Phase 5: Distributed pretraining loop for 125M SLM on 2.04B tokens (8x H100 DDP)."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from modal import Volume

import config


# ============================================================================
# Data loading: Binary uint16 token windows from /data/tokens/
# ============================================================================


class TokenDataset(Dataset):
    """Packed uint16 token windows, read via memmap with an O(1) global window index."""

    ITEMSIZE = 2  # uint16

    def __init__(self, token_dir: str, seq_len: int = 1024, split: str = "train"):
        """
        Args:
            token_dir: Path to the tokens root (contains train/*.bin, val/*.bin, index.json)
            seq_len: Sequence length (1024)
            split: "train" or "val"
        """
        self.seq_len = seq_len
        self.split = split

        split_dir = os.path.join(token_dir, split)
        assert os.path.isdir(split_dir), f"Missing {split_dir}"
        names = sorted(f for f in os.listdir(split_dir) if f.endswith(".bin"))
        assert names, f"No .bin files in {split_dir}"
        self.paths = [os.path.join(split_dir, n) for n in names]

        # Window counts come from file sizes, so the index is always consistent with
        # what is actually on disk. Cumulative starts let __getitem__ binary-search
        # the owning shard instead of scanning (and re-reading) every shard.
        counts = [
            os.path.getsize(p) // self.ITEMSIZE // seq_len for p in self.paths
        ]
        self.starts = np.cumsum([0] + counts)  # len == len(paths) + 1
        self.total_windows = int(self.starts[-1])

        # memmaps are opened lazily so each DataLoader worker gets its own handles.
        self._mmaps: dict[int, np.memmap] = {}

    def _shard(self, i: int) -> np.memmap:
        mm = self._mmaps.get(i)
        if mm is None:
            mm = np.memmap(self.paths[i], dtype=np.uint16, mode="r")
            self._mmaps[i] = mm
        return mm

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        shard = int(np.searchsorted(self.starts, idx, side="right")) - 1
        start = (idx - int(self.starts[shard])) * self.seq_len
        window = self._shard(shard)[start:start + self.seq_len]
        # uint16 has no torch equivalent; widen here, once, on the worker process.
        tokens = window.astype(np.int64)
        return {"input_ids": tokens, "labels": tokens}


# ============================================================================
# Data collator (module-level for pickling with mp.spawn)
# ============================================================================


def collate_fn(batch):
    """Custom collator for pre-tokenized data: one bulk copy, no per-item tensors."""
    input_ids = torch.from_numpy(np.stack([item["input_ids"] for item in batch]))
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
    }


# ============================================================================
# Volume commit callback (commit after each checkpoint save)
# ============================================================================


class VolumeCommitCallback(TrainerCallback):
    """Persists the volume right after Trainer writes a checkpoint, on rank 0 only.

    The handle is resolved once here: hydrating it inside on_save would put a network
    round-trip in the training loop on every save.
    """

    def __init__(self, volume_name: str, rank: int = 0):
        self.rank = rank
        self.volume = None
        if rank == 0:
            self.volume = Volume.from_name(volume_name)
            self.volume.hydrate()

    def commit(self) -> None:
        if self.volume is not None:
            self.volume.commit()

    def on_save(self, args, state, control, **kwargs):
        self.commit()


# ============================================================================
# DDP Worker (torch.multiprocessing entrypoint)
# ============================================================================


def worker(rank: int, world_size: int, args: dict[str, Any]) -> None:
    """
    Distributed training worker (called via torchrun for each GPU).

    Args:
        rank: Process rank (from RANK env var set by torchrun)
        world_size: Total processes (from WORLD_SIZE env var set by torchrun)
        args: Training config dict (loaded from JSON file)
    """
    # Initialize DDP only for multi-GPU (torchrun sets MASTER_ADDR/PORT via env vars)
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    else:
        # Single GPU: no DDP needed
        torch.cuda.set_device(0)

    t0 = time.time()
    smoke = args.get("smoke", False)
    epochs = args.get("epochs", 1)
    max_usd = args.get("max_usd", 40.0)
    resume = args.get("resume", False)
    total_epochs = args.get("total_epochs", epochs)
    max_steps = args.get("max_steps", None)


    # ============================================================================
    # Load model from config
    # ============================================================================

    model_cfg = config.MODEL.to_llama_kwargs()
    model = AutoModelForCausalLM.from_config(LlamaConfig(**model_cfg))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(torch.cuda.current_device())

    # Wrap in DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
        # Forward gradient_checkpointing_enable to underlying model for Trainer
        model.gradient_checkpointing_enable = model.module.gradient_checkpointing_enable


    # ============================================================================
    # Load tokenizer
    # ============================================================================

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # ============================================================================
    # Data loading (use hot-storage local path if available, else network volume)
    # ============================================================================

    tokens_dir = args.get("tokens_dir", config.TOKENS_DIR)

    train_dataset = TokenDataset(tokens_dir, seq_len=config.SEQ_LEN, split="train")
    val_dataset = TokenDataset(tokens_dir, seq_len=config.SEQ_LEN, split="val")

    # ============================================================================
    # Training configuration (resume-aware LR schedule)
    # ============================================================================

    # Accumulate up to the configured global batch. Hardcoding this instead would make
    # the true batch (and therefore the LR schedule, which is denominated in
    # global_batch_tokens) depend on how many GPUs happen to be attached.
    tokens_per_micro_step = config.TRAIN.micro_batch_size * config.SEQ_LEN * world_size
    assert config.TRAIN.global_batch_tokens % tokens_per_micro_step == 0, (
        f"global_batch_tokens={config.TRAIN.global_batch_tokens:,} is not divisible by "
        f"micro_batch({config.TRAIN.micro_batch_size}) * seq_len({config.SEQ_LEN}) * "
        f"world_size({world_size}) = {tokens_per_micro_step:,}"
    )
    grad_accum = config.TRAIN.global_batch_tokens // tokens_per_micro_step
    assert grad_accum >= 1, (
        f"world_size={world_size} already exceeds global_batch_tokens="
        f"{config.TRAIN.global_batch_tokens:,}; lower micro_batch_size or raise the budget"
    )

    warmup_steps = config.TRAIN.warmup_tokens // config.TRAIN.global_batch_tokens
    steps_per_epoch = len(train_dataset) // (
        config.TRAIN.micro_batch_size * grad_accum * world_size
    )
    total_steps = steps_per_epoch * total_epochs
    resume_from_step = 0

    if resume and os.path.exists(config.RESUME_CKPT_PATH):
        ckpt = torch.load(config.RESUME_CKPT_PATH, map_location="cpu")
        resume_from_step = ckpt.get("step", 0)

    training_args_dict = {
        "output_dir": config.BASE_CKPT_DIR,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": config.TRAIN.micro_batch_size,
        "per_device_eval_batch_size": config.TRAIN.micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": config.TRAIN.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "weight_decay": config.TRAIN.weight_decay,
        "max_grad_norm": config.TRAIN.grad_clip,
        "bf16": True,
        "tf32": False,
        "gradient_checkpointing": True,
        "logging_steps": config.TRAIN.log_every_steps,
        "eval_steps": config.TRAIN.eval_every_steps,
        "save_steps": config.TRAIN.ckpt_every_steps,
        "save_total_limit": 3,
        "eval_strategy": "steps" if not smoke else "no",
        "save_strategy": "steps",
        "load_best_model_at_end": False,
        "report_to": [],
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "seed": config.TRAIN.seed,
    }

    # Only add max_steps if it's set (smoke test only)
    if max_steps is not None:
        training_args_dict["max_steps"] = max_steps

    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not smoke else None,
        processing_class=tokenizer,
        data_collator=collate_fn,
    )

    # Add callback to commit volume after each checkpoint save (rank 0 only)
    commit_callback = VolumeCommitCallback(volume_name=config.VOLUME_NAME, rank=rank)
    trainer.add_callback(commit_callback)

    # ============================================================================
    # Train
    # ============================================================================

    trainer.train(resume_from_checkpoint=config.RESUME_CKPT_PATH if resume else None)

    # ============================================================================
    # Save final model + metrics
    # ============================================================================

    if rank == 0:
        # Write the model to BASE_CKPT_DIR itself, not just a checkpoint-N/ subdir:
        # from_pretrained(BASE_CKPT_DIR) is what inference and the HF push both call.
        # Unwrap DDP by hand so the state_dict can't pick up "module." key prefixes.
        to_save = model.module if world_size > 1 else model
        to_save.save_pretrained(config.BASE_CKPT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(config.BASE_CKPT_DIR)

        elapsed = time.time() - t0
        h100_rate_per_sec = config.H100_USD_PER_HOUR / 3600
        total_cost = elapsed * h100_rate_per_sec * world_size

        metrics = {
            "elapsed_sec": elapsed,
            "elapsed_min": elapsed / 60,
            "elapsed_hours": elapsed / 3600,
            "cost_usd": total_cost,
            "gpus": world_size,
            "gpu_type": config.PRETRAIN_GPU,
            "epochs": epochs,
            "total_epochs": total_epochs,
            "smoke": smoke,
        }

        metrics_path = config.METRICS_PATH
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        commit_callback.commit()  # persist the final checkpoint + metrics

    if world_size > 1:
        dist.destroy_process_group()


# ============================================================================
# Torchrun entrypoint (called by torchrun, not mp.spawn)
# ============================================================================

if __name__ == "__main__":
    import sys

    # Read args from JSON file (passed by _pretrain_fn)
    args_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pretrain_args.json"
    with open(args_file) as f:
        args = json.load(f)

    # Get rank/world_size from torchrun environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))


    # Call worker with proper DDP setup (torchrun handles env vars)
    worker(rank, world_size, args)
