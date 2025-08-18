#!/usr/bin/env python3
"""
Latency/Throughput benchmark for spaCy pipelines using Wikipedia sentences.

- Streams Wikipedia via HF datasets, segments with pysbd (fallback: regex).
- Caches sentence pools per (lang,dataset,config,count).
- Benchmarks any spaCy pipeline dirs (e.g. models/*/model-best) and/or installed packages.
- Measures mean latency (ms) & throughput (sent/s) across batch sizes, on CPU and GPU.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import re
import statistics
import sys
import time
import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Set

import spacy
# Compatible with older spaCy versions: use is_package/get_package_path instead of get_package_meta
try:
    from spacy.util import is_package, get_package_path
except Exception:
    is_package = None
    get_package_path = None

# Disable torch grads globally
try:
    import torch
    torch.set_grad_enabled(False)
except Exception:
    pass

# Optional deps
try:
    from datasets import load_dataset
except Exception as _e:
    load_dataset = None  # type: ignore
try:
    import pysbd
except Exception:
    pysbd = None  # type: ignore

# ------------------------------ Config --------------------------------------

DEFAULT_LANGS = ["da", "en", "de", "fr", "hu", "ur", "ga"]
WIKI = {
    "da": ("wikimedia/wikipedia", "20231101.da"),
    "en": ("wikimedia/wikipedia", "20231101.en"),
    "de": ("wikimedia/wikipedia", "20231101.de"),
    "fr": ("wikimedia/wikipedia", "20231101.fr"),
    "hu": ("wikimedia/wikipedia", "20231101.hu"),
    "ur": ("wikimedia/wikipedia", "20231101.ur"),
    "ga": ("wikimedia/wikipedia", "20231101.ga"),
}

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
WARMUP_BATCHES = 7
MEASURE_BATCHES = 50
TOTAL_BATCHES = WARMUP_BATCHES + MEASURE_BATCHES

SENT_CACHE = Path("sentence_cache_wikipedia_spacy")
OUTPUT_CSV = Path("metrics/latency_throughput_wikipedia.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

PYSBD_SUPPORTED_LANGS = {
    "hy","ur","fa","ru","es","nl","fr","de","pl","ar","ja","bg","my","it","am","el","zh","sk","kk","da","hi","en","mr"
}

# ------------------------------ Data ----------------------------------------

def sentences_from_text(text: str, lang: str) -> list[str]:
    if pysbd is not None:
        lang_for_pysbd = "hi" if lang == "ur" else lang
        if lang_for_pysbd in PYSBD_SUPPORTED_LANGS:
            try:
                seg = pysbd.Segmenter(language=lang_for_pysbd, clean=False)
                sents = seg.segment(text)
            except Exception:
                sents = re.split(r'(?<=[.!?۔])\s+', text)
        else:
            sents = re.split(r'(?<=[.!?۔])\s+', text)
    else:
        sents = re.split(r'(?<=[.!?۔])\s+', text)
    return [s.strip() for s in sents if len(s.split()) >= 3]

def stream_sentences(lang: str, need: int) -> list[str]:
    if lang not in WIKI:
        raise ValueError(f"No Wikipedia dataset config for lang '{lang}'")
    if load_dataset is None:
        raise RuntimeError("datasets is not installed. pip install datasets")

    d_name, d_cfg = WIKI[lang]
    SENT_CACHE.mkdir(exist_ok=True, parents=True)
    cache_file = SENT_CACHE / f"{lang}_{d_name.replace('/','_')}_{d_cfg}_{need}.txt"

    if cache_file.exists():
        logging.info(f"Cache hit: {cache_file}")
        return cache_file.read_text(encoding="utf-8").splitlines()[:need]

    logging.info(f"Streaming {d_name}:{d_cfg} for lang '{lang}' ...")
    ds = load_dataset(d_name, d_cfg, split="train", streaming=True)

    pool: list[str] = []
    for ex in ds:
        txt = ex.get("text", "")
        if txt:
            pool.extend(sentences_from_text(txt, lang))
        if len(pool) >= int(need * 1.5):
            break

    random.shuffle(pool)
    seen = set()
    uniq = []
    for s in pool:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
        if len(uniq) >= need:
            break
    if len(uniq) < need:
        raise RuntimeError(f"Only collected {len(uniq)} sentences for {lang}, need {need}")

    cache_file.write_text("\n".join(uniq), encoding="utf-8")
    logging.info(f"Wrote {len(uniq)} sentences to cache: {cache_file}")
    return uniq

# ------------------------------ Bench utils ---------------------------------

@dataclass(slots=True)
class Row:
    model_label: str
    model_src: str
    lang: str
    device: str
    batch_size: int
    mean_ms: float
    ci95_ms: float
    sent_per_s: float
    ci95_sent_per_s: float
    status: str

def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def _require_gpu(gpu_id: Optional[int]) -> bool:
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        spacy.require_gpu()
        return True
    except Exception as e:
        logging.warning(f"GPU not available via spaCy/Thinc: {e}")
        return False

def _gpu_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass

def _batches(pool: list[str], bs: int) -> list[list[str]]:
    take = bs * TOTAL_BATCHES
    chunk = pool[:take]
    return [chunk[i:i+bs] for i in range(0, take, bs)]

def _run_once(nlp, texts: list[str]) -> int:
    n = 0
    for _ in nlp.pipe(texts, batch_size=len(texts), n_process=1):
        n += 1
    return n

def _time_batches(nlp, batches: list[list[str]], device: str) -> list[float]:
    timings_ms: list[float] = []
    # warmup
    for b in batches[:WARMUP_BATCHES]:
        _run_once(nlp, b)
    if device == "gpu":
        _gpu_sync()
    # measure with GC disabled
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for b in batches[WARMUP_BATCHES:]:
            t0 = time.perf_counter()
            _run_once(nlp, b)
            if device == "gpu":
                _gpu_sync()
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return timings_ms

def _ci95(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))

def _label_for_model_src(src: str) -> str:
    p = Path(src)
    return p.parent.name if p.name == "model-best" else p.name

def _iter_model_sources(models_glob: Optional[str], include_glob: bool, packages: Sequence[str]) -> list[Tuple[str, str]]:
    out: list[Tuple[str,str]] = []
    if include_glob and models_glob:
        import glob as _glob
        for path in sorted(_glob.glob(models_glob)):
            out.append((_label_for_model_src(path), path))
    for pkg in packages:
        out.append((pkg, pkg))
    return out

def _load_nlp(src: str):
    try:
        return spacy.load(src)
    except Exception as e:
        logging.warning(f"Could not load pipeline '{src}': {e}")
        return None

# --- Language detection for filtering ---
def _read_meta_lang_from_dir(path: Path) -> Optional[str]:
    meta_path = path / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lang = meta.get("lang")
            if isinstance(lang, str) and lang:
                return lang.lower()
        except Exception:
            pass
    return None

def _detect_model_lang(src: str) -> Optional[str]:
    p = Path(src)
    if p.exists() and p.is_dir():
        return _read_meta_lang_from_dir(p)
    try:
        if is_package is not None and get_package_path is not None and is_package(src):
            pkg_path = get_package_path(src)
            meta_path = Path(pkg_path) / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                lang = meta.get("lang")
                if isinstance(lang, str) and lang:
                    return lang.lower()
    except Exception:
        pass
    return None

# -------- Resume support --------
def _load_done(csv_path: Path) -> Set[Tuple[str, str, str, int]]:
    done: Set[Tuple[str, str, str, int]] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("status","").lower() == "ok":
                    try:
                        bs = int(float(row["batch_size"]))
                        done.add((row["model_src"], row["language"], row["device"], bs))
                    except Exception:
                        pass
        if done:
            logging.info(f"Resume: loaded {len(done)} completed rows from {csv_path}")
    except Exception as e:
        logging.warning(f"Resume: could not read existing CSV {csv_path}: {e}")
    return done

def _bench_one(nlp, model_label: str, model_src: str, lang: str, device: str,
               pool: list[str], batch_sizes: list[int], repeats: int,
               resume: bool = False, done_set: Optional[Set[Tuple[str,str,str,int]]] = None) -> list[Row]:
    rows: list[Row] = []
    for bs in batch_sizes:
        if resume and done_set and (model_src, lang, device, bs) in done_set:
            logging.info("  [%s] %-35s bs=%-4d SKIP (resume hit)", device, model_label, bs)
            continue
        need = bs * TOTAL_BATCHES
        if need > len(pool):
            logging.warning(f"Not enough sentences for bs={bs} (need {need}, have {len(pool)}). Stopping sweep.")
            break
        batches = _batches(pool, bs)

        means, thr = [], []
        status = "ok"
        for _ in range(repeats):
            try:
                timings = _time_batches(nlp, batches, device)
                mean_ms = statistics.mean(timings)
                sent_per_s = (bs * MEASURE_BATCHES * 1000.0) / sum(timings)
                means.append(mean_ms)
                thr.append(sent_per_s)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    status = "fail:oom"
                    logging.warning(f"{status} at bs={bs} for {model_label} [{device}]")
                    break
                else:
                    status = f"fail:{type(e).__name__}"
                    logging.warning(f"{status} at bs={bs} for {model_label} [{device}]: {e}")
                    break
            except Exception as e:
                status = f"fail:{type(e).__name__}"
                logging.warning(f"{status} at bs={bs} for {model_label} [{device}]: {e}")
                break

        if not means:
            rows.append(Row(model_label, model_src, lang, device, bs, float("inf"), float("nan"), 0.0, float("nan"), status))
            break

        row = Row(
            model_label=model_label,
            model_src=model_src,
            lang=lang,
            device=device,
            batch_size=bs,
            mean_ms=statistics.mean(means),
            ci95_ms=_ci95(means),
            sent_per_s=statistics.mean(thr),
            ci95_sent_per_s=_ci95(thr),
            status=status,
        )
        logging.info("  [%s] %-35s bs=%-4d latency %7.1f ± %4.1f ms | throughput %7.1f ± %4.1f sent/s",
                     device, model_label, bs, row.mean_ms, row.ci95_ms, row.sent_per_s, row.ci95_sent_per_s)
        rows.append(row)
        if resume and done_set is not None and status == "ok":
            done_set.add((model_src, lang, device, bs))
    return rows

def _append_csv(rows: Iterable[Row]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["model_label","model_src","language","device","batch_size",
                        "mean_ms_latency","ci95_ms_latency","sentences_per_s","ci95_sentences_per_s","status"])
        for r in rows:
            w.writerow([r.model_label, r.model_src, r.lang, r.device, r.batch_size,
                        f"{r.mean_ms:.3f}", f"{r.ci95_ms:.3f}", f"{r.sent_per_s:.2f}", f"{r.ci95_sent_per_s:.2f}", r.status])

# ------------------------------ Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Latency/Throughput benchmark for spaCy pipelines using Wikipedia sentences.")
    ap.add_argument("--langs", nargs="*", default=DEFAULT_LANGS)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--device", choices=["cpu","gpu","both"], default="both")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--models-glob", default="models/*/model-best")
    ap.add_argument("--no-models-glob", action="store_true")
    ap.add_argument("--packages", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.threads is not None:
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            os.environ[var] = str(args.threads)
        logging.info(f"Thread limit set to {args.threads}")

    model_sources = _iter_model_sources(args.models_glob, not args.no_models_glob, args.packages)
    if not model_sources:
        logging.error("No models to benchmark.")
        sys.exit(1)

    want_gpu = args.device in ("gpu", "both")
    have_gpu = False
    if want_gpu:
        have_gpu = _gpu_available() and _require_gpu(args.gpu)
        if not have_gpu:
            logging.warning("GPU not available; proceeding with CPU only.")

    done_set: Set[Tuple[str,str,str,int]] = _load_done(OUTPUT_CSV) if args.resume else set()

    for lang in args.langs:
        need = max(args.batch_sizes) * TOTAL_BATCHES
        try:
            pool = stream_sentences(lang, need)
        except Exception as e:
            logging.error(f"Failed to prepare sentence pool for lang '{lang}': {e}", exc_info=True)
            continue
        logging.info(f"\n=== Language {lang} — pool size: {len(pool)} ===")

        if args.device in ("cpu","both") or not have_gpu:
            for label, src in model_sources:
                declared_lang = _detect_model_lang(src)
                if declared_lang is not None and declared_lang != lang.lower():
                    logging.info(f"→ SKIP {label} for lang={lang} (model lang={declared_lang})")
                    continue
                nlp = _load_nlp(src)
                if nlp is None:
                    continue
                if declared_lang is None and getattr(nlp, "lang", None) and nlp.lang.lower() != lang.lower():
                    logging.info(f"→ SKIP {label} after load for lang={lang} (model lang={nlp.lang})")
                    try:
                        nlp.dispose()
                    except Exception:
                        pass
                    continue
                logging.info(f"\n→ CPU: {label}")
                rows = _bench_one(nlp, label, src, lang, "cpu", pool, args.batch_sizes, args.repeats,
                                  resume=args.resume, done_set=done_set)
                _append_csv(rows)
                try:
                    nlp.dispose()
                except Exception:
                    pass

        if have_gpu and args.device in ("gpu","both"):
            for label, src in model_sources:
                declared_lang = _detect_model_lang(src)
                if declared_lang is not None and declared_lang != lang.lower():
                    logging.info(f"→ SKIP {label} for lang={lang} (model lang={declared_lang})")
                    continue
                nlp = _load_nlp(src)
                if nlp is None:
                    continue
                if declared_lang is None and getattr(nlp, "lang", None) and nlp.lang.lower() != lang.lower():
                    logging.info(f"→ SKIP {label} after load for lang={lang} (model lang={nlp.lang})")
                    try:
                        nlp.dispose()
                    except Exception:
                        pass
                    continue
                logging.info(f"\n→ GPU: {label}")
                rows = _bench_one(nlp, label, src, lang, "gpu", pool, args.batch_sizes, args.repeats,
                                  resume=args.resume, done_set=done_set)
                _append_csv(rows)
                try:
                    nlp.dispose()
                except Exception:
                    pass

    logging.info(f"\nAll done. Results appended to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
