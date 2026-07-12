"""
Streaming Hugging Face datasets for SLM (Small Language Model) fine-tuning.

Instead of downloading an entire dataset to disk, this module streams examples
lazily from the Hugging Face Hub. Only the batches you are currently training on
are held in memory, which makes it possible to train on datasets far larger than
your available RAM or disk.

Key ideas demonstrated:
  1. Loading a dataset in streaming mode (`streaming=True`).
  2. Iterating without materializing the whole dataset.
  3. On-the-fly tokenization + memory-efficient batching.
  4. Robust error handling and logging.
  5. A drop-in PyTorch DataLoader you can plug into a training loop.

Requirements:
    pip install "datasets>=2.14" transformers torch

Usage:
    python stream_dataset.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Iterable, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("slm.streaming")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StreamConfig:
    """All knobs for the streaming pipeline in one place."""

    # Dataset identifiers on the Hugging Face Hub.
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-103-raw-v1"
    split: str = "train"
    text_column: str = "text"

    # Tokenizer / model.
    tokenizer_name: str = "gpt2"
    max_length: int = 512

    # Batching.
    batch_size: int = 8
    # Shuffle buffer for streaming datasets. Larger => better shuffling but
    # more memory. Streaming can only shuffle within a buffer, not globally.
    shuffle_buffer_size: int = 10_000
    seed: int = 42

    # Skip examples that are empty or shorter than this many characters.
    min_chars: int = 1


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------
def load_streaming_dataset(cfg: StreamConfig):
    """Open a dataset in streaming mode.

    In streaming mode `load_dataset` returns an `IterableDataset`. No data is
    downloaded here; examples are fetched lazily as you iterate. This returns
    almost instantly even for multi-hundred-GB datasets.
    """
    # Imported lazily so the module can be imported even if the optional
    # dependency is missing, and so import errors are reported clearly.
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' library is required. Install it with:\n"
            "    pip install datasets"
        ) from exc

    logger.info(
        "Opening streaming dataset '%s' (config=%s, split=%s)",
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.split,
    )

    try:
        dataset = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split=cfg.split,
            streaming=True,  # <-- the important flag: lazy, no full download
        )
    except Exception as exc:
        # Network issues, wrong dataset name, gated dataset, etc.
        logger.error("Failed to open dataset: %s", exc)
        raise

    # Shuffle within a bounded buffer. Unlike a map-style dataset, a streaming
    # dataset cannot be globally shuffled, so we approximate with a buffer.
    dataset = dataset.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer_size)
    logger.info("Dataset ready (shuffle buffer=%d)", cfg.shuffle_buffer_size)
    return dataset


def iter_clean_texts(dataset: Iterable[Dict], cfg: StreamConfig) -> Iterator[str]:
    """Yield non-empty text strings, one example at a time.

    Errors on individual records are logged and skipped so a single malformed
    example never crashes a long training run.
    """
    kept, skipped = 0, 0
    for example in dataset:
        try:
            text = example.get(cfg.text_column, "")
            if not isinstance(text, str):
                skipped += 1
                continue
            text = text.strip()
            if len(text) < cfg.min_chars:
                skipped += 1
                continue
            kept += 1
            yield text
        except Exception as exc:  # defensive: never die on one bad record
            skipped += 1
            logger.warning("Skipping malformed example: %s", exc)

    logger.info("Stream exhausted. kept=%d skipped=%d", kept, skipped)


# ---------------------------------------------------------------------------
# Tokenization + batching
# ---------------------------------------------------------------------------
def load_tokenizer(cfg: StreamConfig):
    """Load a tokenizer, ensuring a pad token exists (GPT-2 has none by default)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'transformers' library is required. Install it with:\n"
            "    pip install transformers"
        ) from exc

    logger.info("Loading tokenizer '%s'", cfg.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        # Many causal LMs (e.g. GPT-2) ship without a pad token.
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("No pad token found; using EOS token as pad token.")
    return tokenizer


def batched(iterable: Iterable, size: int) -> Iterator[List]:
    """Group an iterable into lists of at most `size` items.

    Uses `islice` so only one batch is ever held in memory at a time.
    """
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def stream_token_batches(cfg: StreamConfig) -> Iterator[Dict]:
    """End-to-end generator: raw stream -> clean text -> tokenized batch.

    Yields dicts with 'input_ids', 'attention_mask' and 'labels' tensors,
    ready to feed into a causal-LM training step. Memory use stays flat because
    each batch is tokenized and yielded before the next one is read.
    """
    dataset = load_streaming_dataset(cfg)
    tokenizer = load_tokenizer(cfg)
    texts = iter_clean_texts(dataset, cfg)

    for text_batch in batched(texts, cfg.batch_size):
        encoded = tokenizer(
            text_batch,
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # For causal LM training, labels are the input_ids (shifted internally
        # by the model). Mask padding positions with -100 so they are ignored
        # by the loss function.
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        yield encoded


# ---------------------------------------------------------------------------
# Optional: wrap as a PyTorch DataLoader for a real training loop
# ---------------------------------------------------------------------------
def build_dataloader(cfg: StreamConfig):
    """Return a PyTorch DataLoader over the streaming, tokenized batches.

    Because batching/tokenization already happen upstream, we use batch_size=None
    so the DataLoader passes each pre-built batch through unchanged.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, IterableDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the DataLoader. Install it with:\n"
            "    pip install torch"
        ) from exc

    class _StreamingTokenDataset(IterableDataset):
        def __init__(self, config: StreamConfig):
            self.config = config

        def __iter__(self):
            # NOTE: with num_workers > 1 each worker would replay the full
            # stream, duplicating data. Keep num_workers=0 unless you shard
            # the stream across workers with `split_dataset_by_node`/skip+take.
            return stream_token_batches(self.config)

    return DataLoader(_StreamingTokenDataset(cfg), batch_size=None)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = StreamConfig()

    logger.info("=== Streaming demo: first few batches ===")
    max_batches = 3
    try:
        for i, batch in enumerate(stream_token_batches(cfg)):
            logger.info(
                "Batch %d | input_ids=%s | labels=%s",
                i,
                tuple(batch["input_ids"].shape),
                tuple(batch["labels"].shape),
            )
            if i + 1 >= max_batches:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.exception("Streaming demo failed: %s", exc)
        raise

    logger.info("Done. Plug `build_dataloader(cfg)` into your training loop:")
    logger.info(
        "    for batch in build_dataloader(cfg):\n"
        "        outputs = model(**batch)\n"
        "        outputs.loss.backward()\n"
        "        optimizer.step(); optimizer.zero_grad()"
    )


if __name__ == "__main__":
    main()
