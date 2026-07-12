"""Modal App for the from-scratch 125M SLM build.

Phase 0 scope: the App, a cheap CPU image, the persistent Volume mount, and a
``smoke_test`` that streams a handful of documents from each source and runs them
through the deterministic cleaner. Later phases add clean / tokenizer / tokenize /
pretrain functions to this same App.

Run:
    source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    modal run modal_app.py::smoke_test
"""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# Cheap CPU base for all pre-GPU phases. Pinned, ungated deps only. All build
# steps (pip/apt) must come BEFORE add_local_* (Modal requirement), so the base
# holds every build step and each image adds local source last.
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR-garble gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
)

# Ship our local source into the container so functions can import them.
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")

# The one persistent Volume, mounted at /data in every function.
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stream_source(source: "config.Source", n: int):
    """Yield up to n raw records from a streamed HF dataset (helper, no I/O)."""
    from datasets import load_dataset

    ds = load_dataset(
        source.hf_id,
        source.config_name,
        split=source.split,
        streaming=True,
    )
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
    """Stream n docs per source, clean each, print before/after. Stores nothing.

    Proves: network reachability, correct per-source field extraction
    (`document` vs `text`), and that the cleaner behaves before any real run.
    """
    from cleaning import clean_document

    summary: dict[str, dict] = {}

    for source in config.DATA_MIX:
        print("\n" + "=" * 78)
        print(f"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, "
              f"field='{source.text_field}')")
        print("=" * 78)

        kept = 0
        reasons: dict[str, int] = {}
        for i, record in enumerate(_stream_source(source, n_per_source)):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            result = clean_document(text)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            kept += int(result.kept)

            excerpt = (result.text[:240] if result.kept else text[:160]).replace("\n", " / ")
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} chars  "
                  f"clean={result.clean_chars:>7} chars  -> {result.reason.upper()}")
            print(f"    {excerpt}")

        summary[source.name] = {
            "streamed": n_per_source,
            "kept": kept,
            "reasons": reasons,
        }

    print("\n" + "#" * 78)
    print("SMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    print("#" * 78)
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_sources(n_per_source: int = 2000) -> dict:
    """Stream a real sample per source and project the true clean-token yield.

    Uses known total row counts to turn a sample average into a corpus estimate,
    so the Phase 1 token budget is grounded in fact, not a guessed parquet
    compression ratio. Stores nothing.
    """
    from cleaning import clean_document

    # Known dataset row counts (HF datasets-server, this config/split).
    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    CHARS_PER_TOKEN = 4.0

    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        raw_chars = clean_chars = kept = 0
        reasons: dict[str, int] = {}
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            raw_chars += r.raw_chars
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars

        n = n_per_source
        avg_raw = raw_chars / n if n else 0
        avg_clean = clean_chars / n if n else 0  # averaged over ALL sampled (kept-only contribute)
        total = TOTAL_ROWS[source.name]
        est_clean_tokens = total * avg_clean / CHARS_PER_TOKEN
        out[source.name] = {
            "sampled": n,
            "kept": kept,
            "keep_rate": round(kept / n, 3) if n else 0,
            "avg_raw_chars": round(avg_raw),
            "avg_clean_chars_per_doc": round(avg_clean),
            "total_rows": total,
            "est_clean_tokens": int(est_clean_tokens),
            "reasons": reasons,
        }
        print(f"{source.name:<12} keep={out[source.name]['keep_rate']:.0%}  "
              f"avg_raw={avg_raw:>7.0f}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est_clean_tokens/1e9:>5.2f}B")
    print("\nTOTAL est clean tokens: "
          f"{sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# --------------------------------------------------------------------------- #
# Phase 1: stream + clean, one worker per parquet shard
# --------------------------------------------------------------------------- #

_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
    """Stream one parquet shard, clean each doc, append survivors to the Volume.

    Pure w.r.t. inputs; the only side effect is writing this worker's own output
    shard (no other worker touches it, so there is no shared-state race). Stops
    early once ~token_cap clean tokens (chars/proxy) have been written.
    """
    import os

    from datasets import load_dataset

    from cleaning import clean_document

    source = _SOURCE_BY_NAME[source_name]
    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{shard_index:03d}.txt"

    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)

    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in ds:
            streamed += 1
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text, strict_ocr=source.strict_ocr)
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                fh.write(r.text.replace("\n", " ").strip() + "\n")
                kept += 1
                clean_chars += r.clean_chars
                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                    break

    volume.commit()
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M reasons={reasons}")
    return {
        "source": source_name,
        "shard": shard_index,
        "streamed": streamed,
        "kept": kept,
        "clean_chars": clean_chars,
        "est_tokens": est_tokens,
        "reasons": reasons,
    }


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    """List parquet file URLs for one dataset config/split (helper, local)."""
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    files = [
        f["url"]
        for f in data.get("parquet_files", [])
        if f.get("config") == config_name and f.get("split") == split
    ]
    return files


@app.local_entrypoint()
def clean(fineweb_shards: int = 1, only: str = ""):
    """`modal run modal_app.py::clean` -> Phase 1 stream + clean fan-out.

    Builds the work list locally (list parquet shards per source), fans out one
    worker per shard, then prints the per-source drop report. Pass
    `--only <name>` to (re)run a single source without touching the others.
    """
    import json

    # HF auto-convert uses config "default" unless the source sets one.
    def cfg(s: "config.Source") -> str:
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work: list[tuple[str, str, int, int]] = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]  # only need a 0.5B-token slice
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap "
              f"~{per_shard_cap/1e6:.0f}M tokens")

    print(f"\nLaunching {len(work)} clean workers...\n")
    results = list(clean_shard.starmap(work))

    # Aggregate per source.
    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {
            "streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v

    print("\n" + "#" * 78)
    print("PHASE 1 DROP REPORT")
    total = 0
    for name, a in report.items():
        keep_rate = a["kept"] / a["streamed"] if a["streamed"] else 0
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>8}  kept={a['kept']:>8} "
              f"({keep_rate:.0%})  est_tokens={a['est_tokens']/1e9:.2f}B")
        print(f"               drops={a['reasons']}")
    print(f"  {'TOTAL':<12} est_clean_tokens={total/1e9:.2f}B")
    print("#" * 78)

    # Persist the report to the Volume for the record.
    save_report.remote(report)


# The base already carries the wordlist, so the OCR analysis uses the CPU image.
ocr_image = cpu_image


# --------------------------------------------------------------------------- #
# Phase 2: dedup + contamination strip
# --------------------------------------------------------------------------- #

SHINGLE_K = 5
MINHASH_PERM = 32       # 32 is plenty for a 0.8 threshold; halves MinHash cost
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13


def _iter_source_docs(source_name: str):
    """Yield (shard_name, line_index, text) for every clean doc of a source."""
    import glob
    import os

    for path in sorted(glob.glob(f"{config.CLEAN_DIR}/{source_name}/*.txt")):
        shard = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n")
                if line:
                    yield shard, i, line


def _build_contamination_ngrams() -> set:
    """Hashed word-13-grams from the eval benchmarks (parquet, no scripts)."""
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    eval_specs = [
        ("casehold/casehold", None),
        ("coastalcph/lex_glue", "case_hold"),
    ]
    for hf_id, cfg_name in eval_specs:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:  # keep going with whatever loaded
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    """Compute MinHash signatures for one case-law clean shard, save to the Volume.

    Signature computation (SHA1 per shingle) is the expensive part, so it is
    fanned out one worker per shard. Short-lived workers also sidestep the
    preemption that kills a long single container. Saves an .npz of the signature
    matrix + line indices, keyed later by (shard, line_index).
    """
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs: list = []
    idxs: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)

    os.makedirs(SIG_DIR, exist_ok=True)
    out = f"{SIG_DIR}/{shard_basename}.npz"
    np.savez(out, sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs -> {out}")
    return {"shard": shard_basename, "n": len(idxs)}


NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
DECONTAM_SOURCES = {"case-law", "sec"}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)
def build_near_dups() -> int:
    """LSH over the precomputed case-law signatures; save the near-dup key set.

    Fast (no re-hashing, just LSH insert/query on stored hashvalues), so this
    single-container step is short. Writes {shard: [line_idx, ...]} to the Volume.
    """
    import glob
    import json
    import os

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        shard = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(shard, []).append(int(idx))
            else:
                lsh.insert(f"{shard}:{int(idx)}", m)

    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    """Write one final-corpus shard: drop near-dups (case-law), exact-dups, and
    eval-contaminated docs. Parallelized one worker per clean shard."""
    import json
    import os

    from dedup import exact_hash, word_ngrams, words

    near: set[int] = set()
    if source_name == "case-law":
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(shard_basename, []))

    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None

    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{shard_basename}"

    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):
                reasons["contaminated"] += 1
                continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1

    volume.commit()
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}")
    return {
        "source": source_name,
        "shard": shard_basename,
        "kept": kept,
        "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN),
        "reasons": reasons,
    }


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    """Aggregate per-shard corpus results into /data/corpus/phase2_report.json."""
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {
            "kept": 0, "est_tokens": 0,
            "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v

    total = sum(v["est_tokens"] for v in report.values())
    print("#" * 70 + "\nPHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B "
              f"drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B\n" + "#" * 70)

    path = f"{config.CORPUS_DIR}/phase2_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


# Clean-shard layout produced by Phase 1 (one output shard per input parquet file).
CLEAN_SHARDS = {"case-law": 10, "sec": 5, "fineweb-edu": 5}


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    """`modal run modal_app.py::dedup` -> Phase 2, fully parallel.

    1. MinHash signatures per case-law shard (parallel).  2. LSH near-dup set.
    3. Write final corpus per shard (parallel: near-dup + exact-dup + decontam).
    Pass `--no-compute-sigs` to reuse signatures already on the Volume.
    """
    if compute_sigs:
        names = [f"shard-{i:03d}.txt" for i in range(CLEAN_SHARDS["case-law"])]
        print(f"1/3 MinHash signatures for {len(names)} case-law shards...")
        list(minhash_shard.map(names))

    print("2/3 building near-dup set (LSH)...")
    build_near_dups.remote()

    work = [
        (src, f"shard-{i:03d}.txt")
        for src, n in CLEAN_SHARDS.items()
        for i in range(n)
    ]
    print(f"3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# --------------------------------------------------------------------------- #
# Phase 3: train the 16K byte-level BPE tokenizer
# --------------------------------------------------------------------------- #

# transformers brings a compatible `tokenizers`; no torch needed to train.
ml_image = _cpu_base.pip_install("transformers==4.46.3").workdir("/home/floweraura/code_repos/slm")


def _corpus_line_iter():
    """Yield every line of the Phase 2 corpus (all sources)."""
    import glob

    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def train_tokenizer() -> dict:
    """Train a fresh 16,384 byte-level BPE and save it as a PreTrainedTokenizerFast."""
    import os

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)

    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print("training BPE...")
    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS),
    )
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()

    # Round-trip sanity check.
    samples = [
        "The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
        "The Company's net revenues increased 12% year over year pursuant to the agreement.",
    ]
    checks = []
    for s in samples:
        ids = fast.encode(s)
        back = fast.decode(ids)
        checks.append({"text": s, "n_tokens": len(ids), "roundtrip_ok": back.strip() == s})
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={back.strip() == s}")

    out = {"vocab_size": fast.vocab_size, "specials": specials, "checks": checks}
    print(f"vocab_size={fast.vocab_size}")
    return out


@app.local_entrypoint()
def tokenizer():
    """`modal run modal_app.py::tokenizer` -> Phase 3 train the tokenizer."""
    train_tokenizer.remote()


# --------------------------------------------------------------------------- #
# Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1
# --------------------------------------------------------------------------- #

TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    """Encode this shard's docs, pack into 1024-windows, split 99/1, write uint16."""
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN

    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"

    buf: list[int] = []
    win_count = 0
    n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush_batch():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            encs = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in encs:
                buf.extend(ids)
                buf.append(eos_id)
            # Emit all full windows currently in buf.
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush_batch()
                batch = []
        _flush_batch()

    volume.commit()
    res = {
        "source": source_name,
        "shard": shard_index,
        "train_windows": n_train,
        "val_windows": n_val,
        "train_tokens": n_train * seq_len,
        "val_tokens": n_val * seq_len,
    }
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return res


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    """Merge per-shard results into /data/tokens/index.json."""
    import json

    shards = [r for r in results if r]
    total = {
        "seq_len": config.SEQ_LEN,
        "dtype": config.TOKENS_DTYPE,
        "train_windows": sum(r["train_windows"] for r in shards),
        "val_windows": sum(r["val_windows"] for r in shards),
        "train_tokens": sum(r["train_tokens"] for r in shards),
        "val_tokens": sum(r["val_tokens"] for r in shards),
        "shards": shards,
    }
    path = f"{config.TOKENS_DIR}/index.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok "
          f"({total['train_windows']} win), val={total['val_tokens']/1e6:.1f}M tok "
          f"({total['val_windows']} win)")
    return total


@app.local_entrypoint()
def tokenize():
    """`modal run modal_app.py::tokenize` -> Phase 4 tokenize + pack + split."""
    work = [
        (name, i, n)
        for name, n in TOKENIZE_SHARDS.items()
        for i in range(n)
    ]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


@app.function(image=ocr_image, timeout=60 * 15)
def ocr_sample(n_docs: int = 3000) -> dict:
    """Measure OCR-garble in case-law via a real English-dictionary non-word ratio.

    For each sampled doc, compute the fraction of alphabetic word-tokens (len>=3)
    that are NOT in the system English wordlist. Report how many docs would be
    dropped at several thresholds so the OCR gate can be chosen with real numbers.
    Reads /usr/share/dict/words; streams live, stores nothing.
    """
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        words = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tok = re.compile(r"[A-Za-z]{3,}")

    source = _SOURCE_BY_NAME["case-law"]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)  # only score docs that already pass the base chain
        if not r.kept:
            continue
        toks = [t.lower() for t in tok.findall(r.text)]
        if len(toks) < 50:
            continue
        nonword = sum(1 for t in toks if t not in words)
        ratios.append(nonword / len(toks))

    ratios.sort()
    n = len(ratios)
    drops = {f">{int(t*100)}%": sum(1 for x in ratios if x > t) for t in thresholds}
    pct = {f"p{p}": round(ratios[int(p / 100 * (n - 1))], 3) for p in (50, 75, 90, 95, 99)} if n else {}
    print(f"scored {n} kept case-law docs")
    print(f"non-word-ratio percentiles: {pct}")
    for k, v in drops.items():
        print(f"  drop if non-word ratio {k:<5}: {v:>5} docs ({v/n:.1%})" if n else k)
    return {"scored": n, "percentiles": pct, "drops_at_threshold": drops}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    """`modal run modal_app.py::ocr` -> OCR-garble drop-rate analysis."""
    ocr_sample.remote(n_docs)


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict) -> None:
    """Write the Phase 1 drop report to the Volume."""
    import json

    path = f"{config.CLEAN_DIR}/phase1_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Phase 5: pretrain the 125M model (GPU, single-node DDP)
# --------------------------------------------------------------------------- #

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "accelerate==1.1.1",
    )
    .add_local_python_source("config", "train")
)

# CPU inference image (base + SFT endpoints). Defined here so both classes see it.
infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "fastapi[standard]==0.115.5",
    )
    .add_local_python_source("config")
)


def _pretrain_fn(smoke: bool, epochs: int, max_usd: float, gpus: int, resume: bool,
                 total_epochs: int = 0):
    import torch.multiprocessing as mp

    import train

    args = {
        "smoke": smoke,
        "epochs": epochs,
        "max_usd": max_usd,
        "resume": resume,
        "total_epochs": total_epochs or epochs,
        "max_steps": 20 if smoke else None,
    }
    world = gpus
    print(f"[pretrain] spawning {world} rank(s), smoke={smoke}, epochs={epochs}, "
          f"total_epochs={args['total_epochs']}, resume={resume}, max_usd={max_usd}", flush=True)
    if world == 1:
        train.worker(0, 1, args)
    else:
        mp.spawn(train.worker, args=(world, args), nprocs=world, join=True)
    volume.commit()
    print("[pretrain] committed volume", flush=True)


@app.function(image=gpu_image, volumes=VOLUMES,
              gpu=f"{config.PRETRAIN_GPU}:{config.PRETRAIN_GPU_COUNT}",
              timeout=86400)
def pretrain_full(epochs: int, max_usd: float, resume: bool = False, total_epochs: int = 0):
    """Full 8xH100 DDP pretraining run."""
    _pretrain_fn(False, epochs, max_usd, config.PRETRAIN_GPU_COUNT, resume, total_epochs)


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"{config.PRETRAIN_GPU}:1",
              timeout=60 * 30)
def pretrain_smoke():
    """Single-H100 smoke: ~20 steps + eval + checkpoint write. Near $0."""
    _pretrain_fn(True, 1, config.BUDGET_CAP_USD, 1, False)


@app.local_entrypoint()
def smoke_pretrain():
    """`modal run modal_app.py::smoke_pretrain` -> Phase 5 smoke test."""
    pretrain_smoke.remote()


@app.local_entrypoint()
def pretrain(epochs: int = 0, max_usd: float = 0.0, resume: bool = False,
             total_epochs: int = 0):
    """`modal run modal_app.py::pretrain` -> full Phase 5 run (8xH100).

    Continue training from the checkpoint for 5 more epochs (to 10 total), with the
    LR cosine spanning the full 10-epoch horizon:
        modal run modal_app.py::pretrain --epochs 5 --resume --total-epochs 10
    """
    e = epochs or config.PRETRAIN_EPOCHS
    cap = max_usd or config.BUDGET_CAP_USD
    print(f"launching pretrain: {e} epochs (resume={resume}, total_epochs={total_epochs or e}), "
          f"cap ${cap}, {config.PRETRAIN_GPU_COUNT}x{config.PRETRAIN_GPU}")
    pretrain_full.remote(e, cap, resume, total_epochs)


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def backup_checkpoint(tag: str = "5ep") -> str:
    """Snapshot the current base model + resumable ckpt before continued training."""
    import os
    import shutil

    if os.path.exists(config.RESUME_CKPT_PATH):
        shutil.copy(config.RESUME_CKPT_PATH, f"{config.CKPT_DIR}/ckpt_{tag}.pt")
    if os.path.isdir(config.BASE_CKPT_DIR):
        dst = f"{config.CKPT_DIR}/base_{tag}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(config.BASE_CKPT_DIR, dst)
    volume.commit()
    print(f"backed up -> ckpt_{tag}.pt + base_{tag}/")
    return tag


@app.local_entrypoint()
def backup(tag: str = "5ep"):
    """`modal run modal_app.py::backup` -> snapshot current checkpoint."""
    backup_checkpoint.remote(tag)


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100:1", timeout=60 * 15)
def generate_samples() -> list:
    """Complete a few legal/financial prefixes with the trained base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.BASE_CKPT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    eos = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    bos = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["bos_token"])

    prompts = [
        "The plaintiff alleges that the defendant",
        "Pursuant to the terms of this Agreement,",
        "The Company's net revenues for the fiscal year",
        "In determining whether the search was reasonable, the court",
    ]
    outs = []
    for p in prompts:
        ids = torch.tensor([[bos] + tok.encode(p, add_special_tokens=False)]).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                ids, max_new_tokens=90, min_new_tokens=40, do_sample=True,
                temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.3,
                eos_token_id=eos, pad_token_id=eos)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        outs.append({"prompt": p, "completion": text})
        print(f"\n>>> {p}\n{text}", flush=True)
    return outs


@app.local_entrypoint()
def samples():
    """`modal run modal_app.py::samples` -> sample completions from the base model."""
    generate_samples.remote()


# --------------------------------------------------------------------------- #
# Phase 7a: build grounded (RAFT-style) SFT data with a teacher (Gemini)
# --------------------------------------------------------------------------- #

SFT_DIR = f"{config.DATA_ROOT}/sft"


def _gemini_generate(excerpt: str) -> str:
    """Call Gemini 2.5 Flash for grounded Q&A JSON on one excerpt."""
    import json
    import os
    import urllib.request

    import sft_data

    key = os.environ["GEMINI_API_KEY"]
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.5-flash:generateContent?key={key}")
    body = {
        "contents": [{"parts": [{"text": sft_data.GEN_INSTRUCTION + excerpt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    return d["candidates"][0]["content"]["parts"][0]["text"]


@app.function(image=cpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("gemini-secret")],
              timeout=60 * 40, cpu=4.0)
def gen_sft(n_chunks: int = 40, workers: int = 16, tag: str = "sample") -> dict:
    """Generate grounded SFT Q&A from n_chunks corpus excerpts, clean, and store.

    Pipeline: retrieve excerpts -> teacher Q&A (parallel) -> faithfulness filter
    -> format filter -> exact-dedup (question) -> decontaminate vs eval 13-grams
    -> write raw + clean JSONL to /data/sft/. Nothing is deleted.
    """
    import glob
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    import sft_data
    from dedup import exact_hash, word_ngrams, words

    # 1. Retrieve excerpts, balanced across sources and biased to CLEAN passages
    #    (low OCR/non-dictionary ratio), sampled broadly across each source.
    from cleaning import nonword_ratio

    targets = {"case-law": int(n_chunks * 0.7), "sec": n_chunks - int(n_chunks * 0.7)}
    picked: list[tuple[str, str]] = []  # (source, excerpt)
    for src, tgt in targets.items():
        src_pool: list[tuple[str, str]] = []
        for path in sorted(glob.glob(f"{config.CORPUS_DIR}/{src}/*.txt")):
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i % 7 != 0:            # spread the sample across each shard
                        continue
                    line = line.strip()
                    if len(line) < 800:
                        continue
                    ch = sft_data.chunk_text(line)
                    if not ch:
                        continue
                    c = ch[0]
                    if nonword_ratio(c) > 0.08:   # skip OCR-garbled excerpts
                        continue
                    src_pool.append((src, c))
                    if len(src_pool) >= tgt * 4:
                        break
            if len(src_pool) >= tgt * 4:
                break
        step = max(1, len(src_pool) // max(1, tgt))
        picked.extend(src_pool[::step][:tgt])
    print(f"[gen_sft] picked {len(picked)} clean chunks "
          f"(case-law {sum(1 for s,_ in picked if s=='case-law')}, "
          f"sec {sum(1 for s,_ in picked if s=='sec')})", flush=True)

    # 2. Teacher generation in parallel.
    def _one(item):
        src, ex = item
        try:
            return src, ex, sft_data.parse_pairs(_gemini_generate(ex))
        except Exception as e:
            print(f"  gen error: {str(e)[:80]}", flush=True)
            return src, ex, []

    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex_pool:
        for src, ex, pairs in ex_pool.map(_one, picked):
            for p in pairs:
                raw.append(sft_data.to_record(ex, p["question"], p["answer"],
                                              p["answerable"], src))
    print(f"[gen_sft] {len(raw)} raw pairs generated", flush=True)

    # 3. Clean: format -> faithfulness (answerable only) -> exact-dedup -> decontaminate.
    contam = _build_contamination_ngrams()
    seen: set[str] = set()
    clean: list[dict] = []
    drops = {"format": 0, "faithful": 0, "dup": 0, "contaminated": 0, "kept": 0}
    for rec in raw:
        ctx = rec["messages"][1]["content"]
        q = ctx.split("Question:", 1)[-1].strip()
        a = rec["messages"][2]["content"]
        excerpt = ctx.split("<context>", 1)[-1].split("</context>", 1)[0].strip()
        if not sft_data.format_ok(q, a):
            drops["format"] += 1
            continue
        if rec["answerable"] and not sft_data.is_faithful(a, excerpt):
            drops["faithful"] += 1
            continue
        h = exact_hash(q + " " + a)
        if h in seen:
            drops["dup"] += 1
            continue
        if contam and (word_ngrams(words(q + " " + a), DECONTAM_NGRAM) & contam):
            drops["contaminated"] += 1
            continue
        seen.add(h)
        clean.append(rec)
        drops["kept"] += 1

    # 4. Persist raw + clean (never deleted).
    os.makedirs(SFT_DIR, exist_ok=True)
    raw_path = f"{SFT_DIR}/{tag}_raw.jsonl"
    clean_path = f"{SFT_DIR}/{tag}_clean.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for r in raw:
            fh.write(json.dumps(r) + "\n")
    with open(clean_path, "w", encoding="utf-8") as fh:
        for r in clean:
            fh.write(json.dumps(r) + "\n")
    volume.commit()

    n_ans = sum(1 for r in clean if r["answerable"])
    print(f"[gen_sft] clean={len(clean)} (answerable={n_ans}, refusal={len(clean)-n_ans}) "
          f"drops={drops}\n  raw -> {raw_path}\n  clean -> {clean_path}", flush=True)
    print("\n=== 3 example clean records ===")
    for r in clean[:3]:
        print(json.dumps(r["messages"], indent=2)[:900], flush=True)
        print("---")
    return {"raw": len(raw), "clean": len(clean), "drops": drops,
            "answerable": n_ans, "raw_path": raw_path, "clean_path": clean_path}


@app.local_entrypoint()
def sft_gen(n_chunks: int = 40, tag: str = "sample"):
    """`modal run modal_app.py::sft_gen` -> generate a grounded SFT data sample."""
    gen_sft.remote(n_chunks, 16, tag)


# --------------------------------------------------------------------------- #
# Phase 7b: tokenize the SFT data (chat template + loss masking)
# --------------------------------------------------------------------------- #

SFT_TOKENS_DIR = f"{SFT_DIR}/tokens"
SFT_MAXLEN = 1024


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=8_192)
def tokenize_sft(tag: str = "v1", val_frac: float = 0.05, max_refusal_frac: float = 0.10) -> dict:
    """Render each SFT record to the chat template, tokenize, mask loss to the
    answer tokens only, pad to SFT_MAXLEN, split train/val, save npz. Shows one
    fully-decoded example marking which tokens are learned vs masked.

    Refusals are downsampled to max_refusal_frac of the set: too many identical
    refusal strings make the small model over-refuse answerable questions."""
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    import sft_data

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    pad_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["pad_token"])

    records = [json.loads(l) for l in open(f"{SFT_DIR}/{tag}_clean.jsonl", encoding="utf-8")]
    # Downsample refusals so they are at most max_refusal_frac of the training set.
    ans = [r for r in records if r.get("answerable", True)]
    ref = [r for r in records if not r.get("answerable", True)]
    keep_ref = int(len(ans) * max_refusal_frac / max(1e-6, 1 - max_refusal_frac))
    rng0 = np.random.default_rng(0)
    ref = [ref[i] for i in rng0.permutation(len(ref))[:keep_ref]]
    records = ans + ref
    print(f"[tokenize_sft] {len(ans)} answerable + {len(ref)} refusal "
          f"(capped from) -> {len(records)} records", flush=True)
    ids_rows, lab_rows, mask_rows = [], [], []
    skipped = 0
    for r in records:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        prompt = sft_data.render_prompt(msgs["system"], msgs["user"])
        completion = sft_data.render_completion(msgs["assistant"])
        pids = tok.encode(prompt, add_special_tokens=False)
        cids = tok.encode(completion, add_special_tokens=False)
        if len(pids) + len(cids) > SFT_MAXLEN:
            skipped += 1
            continue
        input_ids = pids + cids
        labels = [-100] * len(pids) + cids
        attn = [1] * len(input_ids)
        padn = SFT_MAXLEN - len(input_ids)
        input_ids += [pad_id] * padn
        labels += [-100] * padn
        attn += [0] * padn
        ids_rows.append(input_ids)
        lab_rows.append(labels)
        mask_rows.append(attn)

    ids = np.asarray(ids_rows, dtype=np.int32)
    lab = np.asarray(lab_rows, dtype=np.int32)
    mask = np.asarray(mask_rows, dtype=np.int32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(len(ids) * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    os.makedirs(SFT_TOKENS_DIR, exist_ok=True)
    np.savez(f"{SFT_TOKENS_DIR}/train.npz", input_ids=ids[tr_idx],
             labels=lab[tr_idx], attention_mask=mask[tr_idx])
    np.savez(f"{SFT_TOKENS_DIR}/val.npz", input_ids=ids[val_idx],
             labels=lab[val_idx], attention_mask=mask[val_idx])
    volume.commit()

    learned = int((lab != -100).sum())
    real = int((mask == 1).sum())
    avg_len = real / len(ids)
    print(f"[tokenize_sft] examples={len(ids)} (train {len(tr_idx)}, val {len(val_idx)}), "
          f"skipped_too_long={skipped}, avg_len={avg_len:.0f} tokens, "
          f"learned(answer) tokens={learned} ({learned/real:.1%} of real)", flush=True)
    # Show one example: masked prompt vs learned answer.
    ex_i = int(tr_idx[0])
    ex_ids, ex_lab = ids[ex_i], lab[ex_i]
    prompt_txt = tok.decode([t for t, l in zip(ex_ids, ex_lab) if l == -100 and t != pad_id])
    answer_txt = tok.decode([t for t, l in zip(ex_ids, ex_lab) if l != -100])
    print("\n=== example ===\nMASKED prompt (loss ignored):\n", prompt_txt[:600], flush=True)
    print("\nLEARNED answer (loss computed):\n", answer_txt[:400], flush=True)
    return {"examples": len(ids), "train": len(tr_idx), "val": len(val_idx),
            "skipped": skipped, "learned_tokens": learned, "avg_len": round(avg_len)}


@app.local_entrypoint()
def sft_tok(tag: str = "v1"):
    """`modal run modal_app.py::sft_tok` -> tokenize + mask the SFT data."""
    tokenize_sft.remote(tag)


# --------------------------------------------------------------------------- #
# Phase 7c: supervised fine-tuning (multi-GPU DDP), non-destructive
# --------------------------------------------------------------------------- #

SFT_GPU_COUNT = 4  # SFT is small; 4x H100 is fast and cheap


@app.function(image=gpu_image, volumes=VOLUMES,
              gpu=f"{config.PRETRAIN_GPU}:{SFT_GPU_COUNT}", timeout=60 * 60)
def sft_finetune_fn(epochs: int = 3, lr: float = 2e-5, micro_batch_size: int = 16,
                    base_dir: str = ""):
    """Fine-tune the base model on the SFT data. Writes only to /checkpoints/sft."""
    import torch.multiprocessing as mp

    import sft_train

    args = {
        "epochs": epochs, "lr": lr, "micro_batch_size": micro_batch_size,
        "base_dir": base_dir or config.BASE_CKPT_DIR,
    }
    print(f"[sft] spawning {SFT_GPU_COUNT} ranks, epochs={epochs}, lr={lr}, "
          f"base={args['base_dir']}", flush=True)
    mp.spawn(sft_train.worker, args=(SFT_GPU_COUNT, args), nprocs=SFT_GPU_COUNT, join=True)
    volume.commit()
    print("[sft] committed volume", flush=True)


@app.local_entrypoint()
def sft_finetune(epochs: int = 3, lr: float = 2e-5, base_dir: str = ""):
    """`modal run modal_app.py::sft_finetune` -> full SFT run (multi-GPU)."""
    print(f"launching SFT: {epochs} epochs, {SFT_GPU_COUNT}x{config.PRETRAIN_GPU}, "
          f"base={base_dir or config.BASE_CKPT_DIR}")
    sft_finetune_fn.remote(epochs, lr, 16, base_dir)


# --------------------------------------------------------------------------- #
# Phase 7d: deploy the SFT model (separate endpoint + separate HF repo)
# --------------------------------------------------------------------------- #

SFT_CKPT_DIR = f"{config.CKPT_DIR}/sft"
SFT_HF_REPO = "thesreedath/slm-125m-sft"


@app.cls(image=infer_image, volumes=VOLUMES, cpu=2.0, memory=4_096,
         min_containers=0, scaledown_window=300)
class InferenceSFT:
    """Serves the fine-tuned model: grounded Q&A (context + question -> answer)."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import sft_data

        self.torch = torch
        self.sft_data = sft_data
        self.tok = AutoTokenizer.from_pretrained(SFT_CKPT_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            SFT_CKPT_DIR, torch_dtype=torch.float32).eval()
        self.eos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])

    def _answer(self, body: dict) -> str:
        torch = self.torch
        context = (body.get("context") or "").strip()
        question = (body.get("question") or body.get("prompt") or "").strip()
        user = self.sft_data.build_user(context, question) if context else f"Question: {question}"
        prompt = self.sft_data.render_prompt(self.sft_data.SYSTEM, user)
        ids = torch.tensor([self.tok.encode(prompt, add_special_tokens=False)])
        with torch.no_grad():
            gen = self.model.generate(
                ids, max_new_tokens=int(body.get("max_new_tokens", 160)),
                min_new_tokens=int(body.get("min_new_tokens", 8)),
                do_sample=body.get("temperature", 0.3) > 0,
                temperature=float(body.get("temperature", 0.3)),
                top_p=float(body.get("top_p", 0.9)),
                top_k=int(body.get("top_k", 40)),
                repetition_penalty=float(body.get("repetition_penalty", 1.2)),
                eos_token_id=self.eos, pad_token_id=self.eos)
        return self.tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="slm-125m-sft")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": "slm-125m-sft", "kind": "grounded-qa"}

        @api.post("/generate")
        def generate(body: dict):
            try:
                return {"generated": self._answer(body)}
            except Exception as e:
                return {"generated": "", "error": str(e)}

        return api


@app.function(image=gpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 20)
def push_sft_to_hf(repo: str = ""):
    """Push /data/checkpoints/sft to a separate HF repo with a chat-style README."""
    import os

    from huggingface_hub import HfApi

    repo = repo or SFT_HF_REPO
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="model")
    readme = f"""---
license: other
language: en
tags: [legal, sft, raft, instruction-tuning, llama]
---

# {repo.split('/')[-1]}

A **125M legal SLM fine-tuned (SFT) from
[thesreedath/slm-125m-base](https://huggingface.co/thesreedath/slm-125m-base)**
(the 10-epoch base) on **grounded (RAFT-style) question-answer data**: each example
provides a context passage and a question, and the model answers **from the
context** (or refuses when the answer is not present).

- Objective: supervised fine-tuning, loss on answer tokens only.
- Data: ~15k Gemini-generated Q&A pairs grounded in the cleaned case-law / SEC
  corpus, cleaned (faithfulness filter + dedup + decontamination vs CaseHOLD/LexGLUE).
- Use it with a context passage for reliable, honest answers (pairs with retrieval).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}")
prompt = ("<|bos|><|system|>\\nYou are a legal and financial assistant. Answer the "
  "question using ONLY the provided context.<|eos|>\\n<|user|>\\n<context>\\n"
  "{{PASSAGE}}\\n</context>\\n\\nQuestion: {{QUESTION}}<|eos|>\\n<|assistant|>\\n")
```

Full pipeline + code: https://github.com/Vizuara-AI-Lab/slm-125m-from-scratch
"""
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    api.upload_folder(folder_path=SFT_CKPT_DIR, repo_id=repo, repo_type="model")
    print(f"pushed {SFT_CKPT_DIR} -> https://huggingface.co/{repo}")
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def sft_push(repo: str = ""):
    """`modal run modal_app.py::sft_push` -> push SFT model to HuggingFace."""
    push_sft_to_hf.remote(repo)


# --------------------------------------------------------------------------- #
# Phase 8: QLoRA fine-tuning of a pretrained Gemma 2 (4-bit NF4 + LoRA adapters)
# --------------------------------------------------------------------------- #

GEMMA_MODEL = "google/gemma-2-2b-it"
GEMMA_CKPT_DIR = f"{config.CKPT_DIR}/gemma-sft"
GEMMA_HF_REPO = "thesreedath/gemma-2-2b-legal-sft"

gemma_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "trl==0.12.2",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "sentencepiece==0.2.0",
    )
    .workdir("/home/floweraura/code_repos/slm")
)


def _gemma_dataset(tok, max_refusal_frac: float):
    """Build train/val text datasets from our grounded SFT data, in Gemma chat
    format (system merged into the user turn; Gemma has no system role)."""
    import json

    import numpy as np
    from datasets import Dataset

    recs = [json.loads(l) for l in open(f"{SFT_DIR}/v1_clean.jsonl", encoding="utf-8")]
    ans = [r for r in recs if r.get("answerable", True)]
    ref = [r for r in recs if not r.get("answerable", True)]
    keep = int(len(ans) * max_refusal_frac / max(1e-6, 1 - max_refusal_frac))
    rng = np.random.default_rng(0)
    ref = [ref[i] for i in rng.permutation(len(ref))[:keep]]
    recs = ans + ref
    order = rng.permutation(len(recs))
    recs = [recs[i] for i in order]

    def to_text(r):
        m = {x["role"]: x["content"] for x in r["messages"]}
        msgs = [{"role": "user", "content": m["system"] + "\n\n" + m["user"]},
                {"role": "assistant", "content": m["assistant"]}]
        return tok.apply_chat_template(msgs, tokenize=False)

    texts = [to_text(r) for r in recs]
    n_val = max(1, len(texts) // 20)
    return (Dataset.from_dict({"text": texts[n_val:]}),
            Dataset.from_dict({"text": texts[:n_val]}), len(texts), n_val)


@app.function(image=gemma_image, volumes=VOLUMES, gpu="A100-40GB",
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 60 * 3)
def gemma_qlora(model_id: str = GEMMA_MODEL, epochs: int = 3,
                max_refusal_frac: float = 0.10, micro_batch: int = 8,
                grad_accum: int = 2) -> dict:
    """QLoRA fine-tune a pretrained Gemma 2 on grounded legal Q&A, merge, save."""
    import os
    import time

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_id)
    ds_tr, ds_va, n_total, n_val = _gemma_dataset(tok, max_refusal_frac)
    print(f"[gemma] dataset: {n_total} examples ({n_total - n_val} train / {n_val} val)", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": 0})
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<start_of_turn>model\n", tokenizer=tok)
    cfg = SFTConfig(
        output_dir="/tmp/gemma-out", num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch, gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20, eval_strategy="epoch", save_strategy="no",
        max_seq_length=1024, dataset_text_field="text", packing=False,
        report_to=[])
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds_tr, eval_dataset=ds_va,
        peft_config=lora, data_collator=collator, processing_class=tok)

    trainer.train()
    metrics = trainer.evaluate()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[gemma] eval={metrics} trainable_params={trainable/1e6:.1f}M", flush=True)

    # Save adapter, then merge into a FRESH bf16 base (merging into 4-bit is lossy).
    adapter_dir = f"{config.CKPT_DIR}/gemma-adapter"
    trainer.model.save_pretrained(adapter_dir)
    del model, trainer
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(GEMMA_CKPT_DIR, safe_serialization=True)
    tok.save_pretrained(GEMMA_CKPT_DIR)
    volume.commit()

    rate = config.GPU_RATE_PER_SEC["A100-40GB"]
    usd = (time.time() - t0) * rate
    print(f"[gemma done] merged -> {GEMMA_CKPT_DIR} | eval_loss={metrics.get('eval_loss')} "
          f"| trainable={trainable/1e6:.1f}M | spent=${usd:.2f}", flush=True)
    return {"eval": metrics, "trainable_m": round(trainable / 1e6, 1), "usd": round(usd, 2)}


@app.local_entrypoint()
def gemma_train(epochs: int = 3):
    """`modal run modal_app.py::gemma_train` -> QLoRA fine-tune Gemma 2 2B."""
    gemma_qlora.remote(GEMMA_MODEL, epochs, 0.10)


@app.cls(image=infer_image, volumes=VOLUMES, gpu="L4",
         min_containers=0, scaledown_window=240)
class InferenceGemma:
    """Serves the QLoRA-fine-tuned Gemma 2 2B (grounded Q&A) on a GPU, scale-to-zero."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import sft_data

        self.torch = torch
        self.sft_data = sft_data
        self.tok = AutoTokenizer.from_pretrained(GEMMA_CKPT_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            GEMMA_CKPT_DIR, torch_dtype=torch.bfloat16,
            attn_implementation="eager").to("cuda").eval()

    def _answer(self, body: dict) -> str:
        torch = self.torch
        context = (body.get("context") or "").strip()
        question = (body.get("question") or body.get("prompt") or "").strip()
        user = self.sft_data.build_user(context, question) if context else question
        content = self.sft_data.SYSTEM + "\n\n" + user
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = self.model.generate(
                ids, max_new_tokens=int(body.get("max_new_tokens", 200)),
                do_sample=float(body.get("temperature", 0.3)) > 0,
                temperature=float(body.get("temperature", 0.3)),
                top_p=float(body.get("top_p", 0.9)),
                repetition_penalty=float(body.get("repetition_penalty", 1.1)))
        return self.tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="gemma-2-2b-legal-sft")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": "gemma-2-2b-legal-sft", "method": "qlora"}

        @api.post("/generate")
        def generate(body: dict):
            try:
                return {"generated": self._answer(body)}
            except Exception as e:
                return {"generated": "", "error": str(e)}

        return api


@app.function(image=gemma_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 30)
def push_gemma_to_hf(repo: str = ""):
    """Push the merged Gemma QLoRA model to a separate HF repo with a README."""
    import os

    from huggingface_hub import HfApi

    repo = repo or GEMMA_HF_REPO
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="model")
    readme = f"""---
license: gemma
base_model: {GEMMA_MODEL}
language: en
tags: [legal, gemma, qlora, sft, grounded-qa]
---

# {repo.split('/')[-1]}

**Gemma 2 2B** fine-tuned with **QLoRA** (4-bit NF4 base + LoRA adapters, r=16,
alpha=32) on ~15k grounded legal Q&A pairs (context + question -> answer-from-context,
generated from a cleaned US case-law / SEC corpus). Adapters merged back into the
base and saved here. Answers a question **from a passage you provide**.

Recipe: trl SFTTrainer, 3 epochs, lr 2e-4, batch 1 x grad-accum 16, bf16, gradient
checkpointing. Built on Modal. Full pipeline + code:
https://github.com/Vizuara-AI-Lab/slm-125m-from-scratch
"""
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    api.upload_folder(folder_path=GEMMA_CKPT_DIR, repo_id=repo, repo_type="model")
    print(f"pushed {GEMMA_CKPT_DIR} -> https://huggingface.co/{repo}")
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def gemma_push(repo: str = ""):
    """`modal run modal_app.py::gemma_push` -> push Gemma model to HuggingFace."""
    push_gemma_to_hf.remote(repo)


# --------------------------------------------------------------------------- #
# Phase 6: inference endpoint (CPU, scale-to-zero) + HF push
# --------------------------------------------------------------------------- #


@app.cls(image=infer_image, volumes=VOLUMES, cpu=2.0, memory=4_096,
         min_containers=0, scaledown_window=300)
class Inference:
    """Loads the base model once per container; serves /generate over HTTP."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.BASE_CKPT_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_CKPT_DIR, torch_dtype=torch.float32).eval()
        self.eos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
        self.bos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["bos_token"])

    def _complete(self, body: dict) -> str:
        torch = self.torch
        prompt = (body.get("prompt") or "").strip()
        ids = [self.bos] + self.tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids])
        with torch.no_grad():
            gen = self.model.generate(
                inp,
                max_new_tokens=int(body.get("max_new_tokens", 90)),
                min_new_tokens=int(body.get("min_new_tokens", 40)),
                do_sample=True,
                temperature=float(body.get("temperature", 0.8)),
                top_k=int(body.get("top_k", 50)),
                top_p=float(body.get("top_p", 0.95)),
                repetition_penalty=float(body.get("repetition_penalty", 1.3)),
                eos_token_id=self.eos, pad_token_id=self.eos)
        return self.tok.decode(gen[0][inp.shape[1]:], skip_special_tokens=True)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="slm-125m")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": "slm-125m-base", "val_ppl": 8.36, "epochs": 10}

        @api.post("/generate")
        def generate(body: dict):
            try:
                return {"generated": self._complete(body)}
            except Exception as e:  # never 500 the frontend
                return {"generated": "", "error": str(e)}

        return api


@app.function(image=gpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 20)
def push_to_hf(repo: str = "", src_dir: str = "", epochs: int = 10, ppl: float = 8.36):
    """Push a checkpoint dir to a HuggingFace model repo, with a README."""
    import os

    from huggingface_hub import HfApi

    repo = repo or config.HF_REPO
    src = src_dir or config.BASE_CKPT_DIR
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="model")

    readme = f"""---
license: other
language: en
tags: [legal, from-scratch, llama, small-language-model]
---

# {repo.split('/')[-1]}

A **125M-parameter Llama-style base language model**, pretrained **from scratch**
on a legal/financial corpus (US case law + SEC filings + a web slice, ~2.19B tokens).

- **Pretraining:** {epochs} epochs, 8xH100. **Held-out validation perplexity: {ppl}**.
- **This is a base COMPLETER, not a chat model.** Prompt it with the start of a
  sentence and it continues in the legal register. It speaks the register fluently
  but does not know facts (knowledge is capped at ~2 bits/param at this size).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}")
ids = tok("The plaintiff alleges that the defendant", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=60, do_sample=True,
      temperature=0.8, min_new_tokens=40)[0]))
```

Full pipeline + fine-tuning code: https://github.com/Vizuara-AI-Lab/slm-125m-from-scratch
"""
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    api.upload_folder(folder_path=src, repo_id=repo, repo_type="model")
    print(f"pushed {src} -> https://huggingface.co/{repo}")
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def hf_push(repo: str = "", src_dir: str = "", epochs: int = 10, ppl: float = 8.36):
    """`modal run modal_app.py::hf_push` -> push a checkpoint to HuggingFace."""
    push_to_hf.remote(repo, src_dir, epochs, ppl)


@app.local_entrypoint()
def main(n_per_source: int = 10):
    """Local entrypoint so `modal run modal_app.py` triggers the smoke test."""
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    """`modal run modal_app.py::measure` -> project true clean-token yield."""
    measure_sources.remote(n_per_source)