"""Modal App for the from-scratch 125M SLM build (Phases 0 to 4)."""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# CPU base. All pip/apt build steps MUST come before add_local_* (Modal rule).
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
)
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stream_source(source: "config.Source", n: int):
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
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
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} clean={result.clean_chars:>7} "
                  f"-> {result.reason.upper()}")
            print(f"    {excerpt}")
        summary[source.name] = {"streamed": n_per_source, "kept": kept, "reasons": reasons}
    print("\nSMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_sources(n_per_source: int = 2000) -> dict:
    from cleaning import clean_document

    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        clean_chars = kept = 0
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars
        avg_clean = clean_chars / n_per_source if n_per_source else 0
        total = TOTAL_ROWS[source.name]
        est = total * avg_clean / config.CHARS_PER_TOKEN
        out[source.name] = {"est_clean_tokens": int(est), "keep_rate": round(kept / n_per_source, 3)}
        print(f"{source.name:<12} keep={kept/n_per_source:.0%}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est/1e9:.2f}B")
    print(f"TOTAL est clean tokens: {sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# ---- Phase 1: stream + clean, one worker per parquet shard ----
_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
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
    return {"source": source_name, "shard": shard_index, "streamed": streamed,
            "kept": kept, "est_tokens": est_tokens, "reasons": reasons}


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


@app.local_entrypoint()
def clean(fineweb_shards: int = 1, only: str = ""):
    def cfg(s):
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap ~{per_shard_cap/1e6:.0f}M tokens")
    print(f"Launching {len(work)} clean workers...")
    results = list(clean_shard.starmap(work))
    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    print("PHASE 1 DROP REPORT")
    total = 0
    for name, a in report.items():
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>8} kept={a['kept']:>8} "
              f"est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL est_clean_tokens={total/1e9:.2f}B")
    save_report.remote(report)


ocr_image = cpu_image

# ---- Phase 2: dedup + contamination strip ----
SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"
NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
DECONTAM_SOURCES = {"case-law", "sec"}
CLEAN_SHARDS = {"case-law": 10, "sec": 5, "fineweb-edu": 5}


def _build_contamination_ngrams() -> set:
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    for hf_id, cfg_name in [("casehold/casehold", None), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs, idxs = [], []
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
    np.savez(f"{SIG_DIR}/{shard_basename}.npz",
             sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs")
    return {"shard": shard_basename, "n": len(idxs)}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)
def build_near_dups() -> int:
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
    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, \
            open(f"{out_dir}/{shard_basename}", "w", encoding="utf-8") as fout:
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
    return {"source": source_name, "shard": shard_basename, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0,
              "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    total = sum(v["est_tokens"] for v in report.values())
    print("PHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B")
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    if compute_sigs:
        names = [f"shard-{i:03d}.txt" for i in range(CLEAN_SHARDS["case-law"])]
        print(f"1/3 MinHash signatures for {len(names)} case-law shards...")
        list(minhash_shard.map(names))
    print("2/3 building near-dup set (LSH)...")
    build_near_dups.remote()
    work = [(src, f"shard-{i:03d}.txt") for src, n in CLEAN_SHARDS.items() for i in range(n)]
    print(f"3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# ---- Phase 3: train the 16K byte-level BPE tokenizer ----
ml_image = _cpu_base.pip_install("transformers==4.46.3").add_local_python_source(
    "config", "cleaning", "dedup")


def _corpus_line_iter():
    import glob

    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def train_tokenizer() -> dict:
    import os

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)
    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    print("training BPE...")
    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()
    for s in ["The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
              "The Company's net revenues increased 12% year over year pursuant to the agreement."]:
        ids = fast.encode(s)
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={fast.decode(ids).strip() == s}")
    print(f"vocab_size={fast.vocab_size}")
    return {"vocab_size": fast.vocab_size}


@app.local_entrypoint()
def tokenizer():
    train_tokenizer.remote()


# ---- Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1 ----
TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
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
    win_count = n_train = n_val = 0
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

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
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
                _flush()
                batch = []
        _flush()
    volume.commit()
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return {"source": source_name, "shard": shard_index, "train_windows": n_train,
            "val_windows": n_val, "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    import json

    shards = [r for r in results if r]
    total = {"seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
             "train_windows": sum(r["train_windows"] for r in shards),
             "val_windows": sum(r["val_windows"] for r in shards),
             "train_tokens": sum(r["train_tokens"] for r in shards),
             "val_tokens": sum(r["val_tokens"] for r in shards), "shards": shards}
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)")
    return total


@app.local_entrypoint()
def tokenize():
    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


# ---- OCR-threshold analysis (optional; informs config.CLEAN.nonword_ratio_max) ----
@app.function(image=ocr_image, timeout=60 * 15)
def ocr_sample(n_docs: int = 3000) -> dict:
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        vocab = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tokre = re.compile(r"[A-Za-z]{3,}")
    source = _SOURCE_BY_NAME["case-law"]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)
        if not r.kept:
            continue
        toks = [t.lower() for t in tokre.findall(r.text)]
        if len(toks) < 50:
            continue
        ratios.append(sum(1 for t in toks if t not in vocab) / len(toks))
    ratios.sort()
    n = len(ratios)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        d = sum(1 for x in ratios if x > t)
        print(f"  drop if non-word ratio >{int(t*100)}%: {d} docs ({d/n:.1%})")
    return {"scored": n}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    ocr_sample.remote(n_docs)


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict) -> None:
    import json

    with open(f"{config.CLEAN_DIR}/phase1_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()


# ============================================================================
# Phase 5: Distributed Pretraining (8x H100 DDP for fastest, cost-effective training)
# ============================================================================
# Cost analysis: 8x H100 = $2,100 total (faster, same/lower cost than 1x H100)
# Alternatives if H100 unavailable:
#   - 8x A100-40GB: $1,150-1,440 (saves $600, ~1-2 days slower)
#   - 4x A100-40GB: $865-1,300 (saves $800, ~2-3 days slower)
# ============================================================================

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


def _pretrain_fn(smoke: bool, epochs: int, max_usd: float, gpus: int, resume: bool):
    """Distributed training via torchrun. Tokens are staged on local NVMe; the volume
    is committed from inside train.py after each checkpoint save."""
    import json
    import os
    import shutil
    import subprocess

    args = {
        "smoke": smoke,
        "epochs": epochs,
        "max_usd": max_usd,
        "resume": resume,
        "max_steps": 20 if smoke else None,
    }

    # Stage tokens on local NVMe: the volume is network-backed and would starve the GPUs.
    local_tokens_dir = "/local/data/tokens"
    if os.path.isdir(config.TOKENS_DIR):
        shutil.rmtree(local_tokens_dir, ignore_errors=True)
        os.makedirs(os.path.dirname(local_tokens_dir), exist_ok=True)
        shutil.copytree(config.TOKENS_DIR, local_tokens_dir)
        args["tokens_dir"] = local_tokens_dir
    else:
        args["tokens_dir"] = config.TOKENS_DIR

    args_file = "/tmp/pretrain_args.json"
    with open(args_file, "w") as f:
        json.dump(args, f)

    world = gpus

    # Use torchrun -m train (module mode, not hardcoded path)
    cmd = [
        "torchrun",
        f"--nproc_per_node={world}",
        "-m",
        "train",
        args_file
    ]

    process = subprocess.Popen(cmd, cwd="/root")

    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    volume.commit()


@app.function(
    image=gpu_image,
    volumes=VOLUMES,
    gpu=f"{config.PRETRAIN_GPU}:{config.PRETRAIN_GPU_COUNT}",
    timeout=86400,  # 24 hours max (Modal limit)
)
def pretrain_full(epochs: int, max_usd: float, resume: bool = False):
    """Full 8x H100 DDP pretraining (cost-effective + fastest: ~3-6 days, ~$2,100)."""
    _pretrain_fn(False, epochs, max_usd, config.PRETRAIN_GPU_COUNT, resume)


@app.function(
    image=gpu_image,
    volumes=VOLUMES,
    gpu=f"{config.PRETRAIN_GPU}:1",
    timeout=60 * 30,
)
def pretrain_smoke():
    """Single H100 smoke test: 20 steps, ~30 min, ~$0.01. Validates setup."""
    _pretrain_fn(True, 1, config.BUDGET_CAP_USD, 1, False)


@app.local_entrypoint()
def smoke_pretrain():
    """`modal run modal_app.py::smoke_pretrain` -> Phase 5 smoke test (1x H100, 30 min)."""
    pretrain_smoke.remote()


@app.local_entrypoint()
def pretrain(epochs: int = 1, max_usd: float = 40.0, resume: bool = False):
    """`modal run modal_app.py::pretrain --epochs 1` -> Full Phase 5 training (8x H100, 3-6 days, ~$2,100).

    For multi-epoch training with continuous LR schedule:
        modal run modal_app.py::pretrain --epochs 5 --resume --max-usd 100
    """
    print(f"[pretrain] launching: {epochs} epochs, resume={resume}, {config.PRETRAIN_GPU_COUNT}x{config.PRETRAIN_GPU}")
    pretrain_full.remote(epochs, max_usd, resume)


# --------------------------------------------------------------------------- #
# Phase 6: promote the trained checkpoint, then evaluate it
# --------------------------------------------------------------------------- #

# Files a caller needs to run the model. Deliberately excludes optimizer.pt /
# scheduler.pt / rng_state_*.pth: those are ~1.5GB of resume state, useless for
# inference, and we do not want them on the Hub.
_INFERENCE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def promote_checkpoint_fn(step: int = 0) -> dict:
    """Copy checkpoint-N/ up into BASE_CKPT_DIR itself.

    Trainer writes to checkpoint-N/ subdirs, but from_pretrained(BASE_CKPT_DIR)
    reads the root. Without this, every downstream loader fails on a missing
    config.json. Defaults to the highest-numbered checkpoint.
    """
    import glob
    import os
    import shutil

    found = {}
    for path in glob.glob(f"{config.BASE_CKPT_DIR}/checkpoint-*"):
        tail = os.path.basename(path).split("-")[-1]
        if tail.isdigit():
            found[int(tail)] = path
    if not found:
        raise RuntimeError(f"no checkpoint-* under {config.BASE_CKPT_DIR}")

    chosen = step or max(found)
    if chosen not in found:
        raise RuntimeError(f"checkpoint-{chosen} not found; have {sorted(found)}")
    src = found[chosen]

    copied, missing = [], []
    for name in _INFERENCE_FILES:
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(config.BASE_CKPT_DIR, name))
            copied.append(name)
        else:
            missing.append(name)
    volume.commit()

    print(f"promoted checkpoint-{chosen} -> {config.BASE_CKPT_DIR}")
    print(f"  copied  {copied}")
    if missing:
        print(f"  MISSING {missing}")
    return {"step": chosen, "copied": copied, "missing": missing,
            "available": sorted(found)}


@app.local_entrypoint()
def promote(step: int = 0):
    """`modal run modal_app.py::promote` -> make the newest checkpoint loadable."""
    promote_checkpoint_fn.remote(step)


@app.function(image=gpu_image, volumes=VOLUMES, timeout=60 * 20)
def repair_checkpoint_fn(step: int = 0) -> dict:
    """Strip the DDP 'module.' prefix from a checkpoint's tensor keys.

    Trainer was handed an already-DDP-wrapped model, so it serialized the wrapper's
    state_dict: every key came out as 'module.model.*'. from_pretrained matches none
    of them, silently re-inits the whole network, and you get a random model with only
    a warning. The weights are fine -- only the names are wrong -- so rebuild the model
    from the stripped state_dict and rewrite BASE_CKPT_DIR.
    """
    import glob
    import os

    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    found = {}
    for path in glob.glob(f"{config.BASE_CKPT_DIR}/checkpoint-*"):
        tail = os.path.basename(path).split("-")[-1]
        if tail.isdigit():
            found[int(tail)] = path
    chosen = step or max(found)
    src = found[chosen]

    raw = load_file(os.path.join(src, "model.safetensors"))
    n_prefixed = sum(1 for k in raw if k.startswith("module."))
    clean = {k.removeprefix("module."): v for k, v in raw.items()}
    print(f"checkpoint-{chosen}: {len(raw)} tensors, {n_prefixed} carried 'module.'")

    model = AutoModelForCausalLM.from_config(
        LlamaConfig(**config.MODEL.to_llama_kwargs()))
    missing, unexpected = model.load_state_dict(clean, strict=False)

    # lm_head is tied to embed_tokens (tie_word_embeddings=True), so it legitimately
    # never appears in the state_dict. Anything else missing means a real mismatch.
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"key mismatch after strip: missing={missing[:5]} "
                           f"unexpected={list(unexpected)[:5]}")
    print(f"all {len(clean)} tensors matched the model (lm_head tied to embeddings)")

    model.save_pretrained(config.BASE_CKPT_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.TOKENIZER_DIR).save_pretrained(
        config.BASE_CKPT_DIR)
    volume.commit()
    print(f"rewrote {config.BASE_CKPT_DIR} with clean keys")
    return {"step": chosen, "tensors": len(clean), "stripped": n_prefixed}


@app.local_entrypoint()
def repair(step: int = 0):
    """`modal run modal_app.py::repair` -> strip DDP 'module.' prefix, rewrite model."""
    repair_checkpoint_fn.remote(step)


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"{config.PRETRAIN_GPU}:1",
              timeout=60 * 30)
def evaluate_fn(max_val_batches: int = 100) -> dict:
    """Load BASE_CKPT_DIR the way inference will, measure val perplexity, and sample.

    This is the real proof the weights survived DDP + save: bad keys or a shape
    mismatch surface as a load error, and silently-wrong weights surface as a
    perplexity near vocab_size (16,384) instead of the ~12 we expect.
    """
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import train as train_mod

    tok = AutoTokenizer.from_pretrained(config.BASE_CKPT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {n_params:,} params from {config.BASE_CKPT_DIR}", flush=True)

    # ---- val perplexity (teacher forcing over held-out windows) ----
    val = train_mod.TokenDataset(config.TOKENS_DIR, seq_len=config.SEQ_LEN, split="val")
    bs = 8
    n_batches = min(max_val_batches, len(val) // bs)
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for b in range(n_batches):
            batch = [val[b * bs + i] for i in range(bs)]
            ids = torch.from_numpy(
                __import__("numpy").stack([x["input_ids"] for x in batch])).to("cuda")
            out = model(input_ids=ids, labels=ids)
            # loss is the mean over (seq_len - 1) shifted positions per row
            n = ids.shape[0] * (ids.shape[1] - 1)
            total_loss += out.loss.item() * n
            total_tok += n
    val_loss = total_loss / total_tok
    ppl = math.exp(val_loss)
    print(f"\nval loss {val_loss:.4f}  ppl {ppl:.2f}  "
          f"({n_batches * bs:,} windows, {total_tok:,} tokens)", flush=True)

    # ---- sample completions ----
    eos = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    bos = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["bos_token"])
    prompts = [
        "The plaintiff alleges that the defendant",
        "Pursuant to the terms of this Agreement,",
        "The Company's net revenues for the fiscal year",
        "In determining whether the search was reasonable, the court",
    ]
    samples = []
    for p in prompts:
        ids = torch.tensor([[bos] + tok.encode(p, add_special_tokens=False)]).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                ids, max_new_tokens=80, min_new_tokens=30, do_sample=True,
                temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2,
                eos_token_id=eos, pad_token_id=eos)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        samples.append({"prompt": p, "completion": text})
        print(f"\n>>> {p}\n{text}", flush=True)

    return {"params": n_params, "val_loss": val_loss, "val_ppl": ppl,
            "val_windows": n_batches * bs, "samples": samples}


@app.local_entrypoint()
def evaluate(max_val_batches: int = 100):
    """`modal run modal_app.py::evaluate` -> val perplexity + sample completions."""
    evaluate_fn.remote(max_val_batches)


# --------------------------------------------------------------------------- #
# Phase 6: inference endpoint (CPU, scale-to-zero)
# --------------------------------------------------------------------------- #

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "fastapi[standard]==0.115.5",
    )
    .add_local_python_source("config", "web_ui")
)


@app.cls(
    image=infer_image,
    volumes=VOLUMES,
    cpu=2.0,
    memory=4096,
    min_containers=0,        # scale to zero: no cost while idle
    scaledown_window=300,    # keep warm 5 min after the last request
)
class Inference:
    """125M base LM served on CPU. The model is ~250MB in fp32, so a GPU would be
    idle-cost with no latency win at this size."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.BASE_CKPT_DIR)
        # float32 on CPU: bf16 matmuls fall back to slow kernels off-GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_CKPT_DIR, torch_dtype=torch.float32).eval()
        self.eos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
        self.bos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["bos_token"])

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float,
                  top_p: float) -> str:
        ids = self.torch.tensor(
            [[self.bos] + self.tok.encode(prompt, add_special_tokens=False)])
        with self.torch.no_grad():
            out = self.model.generate(
                ids,
                attention_mask=self.torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                min_new_tokens=8,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_k=50,
                top_p=top_p,
                repetition_penalty=1.2,
                eos_token_id=self.eos,
                pad_token_id=self.eos,
            )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    @modal.asgi_app()
    def web(self):
        """Playground + API on one origin, so the page's fetch() isn't cross-origin."""
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse

        import web_ui

        api = FastAPI(title="SLM 125M", docs_url="/docs")

        @api.get("/", response_class=HTMLResponse)
        def index():
            return web_ui.PAGE

        @api.post("/complete")
        def complete(item: dict):
            """{"prompt": "...", "max_new_tokens": 80, "temperature": 0.8}"""
            prompt = (item.get("prompt") or "").strip()
            if not prompt:
                return {"error": "prompt is required"}
            completion = self._generate(
                prompt,
                min(int(item.get("max_new_tokens", 80)), 256),
                float(item.get("temperature", 0.8)),
                float(item.get("top_p", 0.95)),
            )
            return {
                "prompt": prompt,
                "completion": completion,
                "model": config.HF_REPO,
                "note": "125M base LM. Fluent domain text, but citations and figures "
                        "are fabricated -- not a factual source.",
            }

        return api


@app.local_entrypoint()
def main(n_per_source: int = 10):
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    measure_sources.remote(n_per_source)

