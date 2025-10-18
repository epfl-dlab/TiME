#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: throughput/latency + GPU energy

Measures latency, throughput, and energy (prefers NVML total energy; falls back to
NVML power polling or nvidia-smi polling) for encoder models across batch sizes,
and writes results to CSV.

Install:
  pip install torch transformers datasets pysbd pynvml

Example:
  python run_benchmark.py \
    --output results.csv \
    --langs en de fr da hu ur ga \
    --repeats 3 \
    --fp16 \
    --fast-preset \
    --enable-power

Key flags:
  --enable-power / --disable-power
  --fast-preset
  --fp16 / --no-fp16
  --batches "1,2,4,8,16,32,64,128,256"
"""

import os
import re
import math
import time
import random
import logging
import subprocess
import shlex
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import pysbd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------
# Optional NVML import
# ---------------------------
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount,
        nvmlDeviceGetTotalEnergyConsumption, nvmlDeviceGetPowerUsage,
    )
    NVML_OK = True
except Exception:
    NVML_OK = False


def _logical_to_nvml_ids(logical_ids: List[int]) -> List[int]:
    """Map CUDA_VISIBLE_DEVICES logical indices back to real NVML indices."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return logical_ids
    parts = [p.strip() for p in cvd.split(",") if p.strip()]
    out = []
    for li in logical_ids:
        try:
            out.append(int(parts[li]))
        except Exception:
            out.append(li)
    return out


class _PowerPollerNVML:
    """NVML power polling in a background thread; integrates power over time."""

    def __init__(self, gpu_ids: List[int], interval_s: float = 0.1):
        import threading
        self.gpu_ids = gpu_ids
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._th = None
        self._threading = threading

    def start(self):
        self._stop.clear()

        def loop():
            try:
                while not self._stop.is_set():
                    t = time.perf_counter()
                    mw = 0
                    for gid in self.gpu_ids:
                        h = nvmlDeviceGetHandleByIndex(gid)
                        mw += nvmlDeviceGetPowerUsage(h)  # milliwatts
                    self.samples.append((t, mw))
                    time.sleep(self.interval)
            except Exception as e:
                logging.warning(f"NVML power polling aborted: {e}")

        self._th = self._threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        if self._th:
            self._stop.set()
            self._th.join()

    def energy_J(self) -> float:
        """Trapezoidal integration of power samples (mW) to Joules."""
        if len(self.samples) < 2:
            return 0.0
        E = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            E += ((p0 + p1) / 2.0) * (t1 - t0) / 1000.0
        return E


class _PowerPollerSMI:
    """nvidia-smi power polling fallback."""

    def __init__(self, logical_gpu_index: int = 0, interval_s: float = 0.1):
        import threading
        self.idx = logical_gpu_index
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._th = None
        self._threading = threading

    def _read_power_W(self) -> float:
        cmd = f"nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i {self.idx}"
        try:
            out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, timeout=1.0)
            return float(out.decode().strip().splitlines()[0])
        except Exception:
            return float("nan")

    def start(self):
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                t = time.perf_counter()
                w = self._read_power_W()
                if not math.isnan(w):
                    self.samples.append((t, w * 1000.0))  # store as mW
                time.sleep(self.interval)

        self._th = self._threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        if self._th:
            self._stop.set()
            self._th.join()

    def energy_J(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        E = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            E += ((p0 + p1) / 2.0) * (t1 - t0) / 1000.0
        return E


class GpuEnergyMeter:
    """
    Context manager:
      - Prefer NVML total energy counter (per-device cumulative).
      - Else NVML power polling.
      - Else nvidia-smi polling.
    """

    def __init__(self, poll_interval_s: float = 0.1):
        self.enabled = torch.cuda.is_available()
        self.poll_interval_s = poll_interval_s
        self.mode = "disabled"
        self._nvml_inited = False
        self._e0_mJ = None
        self._poller = None
        self.energy_J = float("nan")
        self.avg_power_W = float("nan")
        self.mean_power_W_raw = float("nan")
        self._t0 = 0.0
        self.nvml_ids = _logical_to_nvml_ids([0])

    def __enter__(self):
        self._t0 = time.perf_counter()
        if not self.enabled:
            logging.info("Energy meter disabled: no CUDA device.")
            return self
        if NVML_OK:
            try:
                nvmlInit()
                self._nvml_inited = True
                _ = nvmlDeviceGetCount()
                total0 = 0
                for gid in self.nvml_ids:
                    h = nvmlDeviceGetHandleByIndex(gid)
                    total0 += nvmlDeviceGetTotalEnergyConsumption(h)  # mJ
                self._e0_mJ = total0
                self.mode = "total"
                logging.info("Energy meter mode: NVML total energy")
                return self
            except Exception:
                pass
            try:
                self._poller = _PowerPollerNVML(self.nvml_ids, self.poll_interval_s)
                self._poller.start()
                self.mode = "nvml_poll"
                return self
            except Exception:
                pass
        try:
            self._poller = _PowerPollerSMI(logical_gpu_index=0, interval_s=self.poll_interval_s)
            self._poller.start()
            self.mode = "smi_poll"
        except Exception:
            self.mode = "disabled"
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = max(time.perf_counter() - self._t0, 1e-9)
        try:
            if self.mode == "total" and self._nvml_inited and self._e0_mJ is not None:
                total1 = 0
                for gid in self.nvml_ids:
                    h = nvmlDeviceGetHandleByIndex(gid)
                    total1 += nvmlDeviceGetTotalEnergyConsumption(h)
                self.energy_J = max(0.0, (total1 - self._e0_mJ) / 1000.0)
            elif self.mode in ("nvml_poll", "smi_poll") and self._poller:
                self._poller.stop()
                self.energy_J = max(0.0, self._poller.energy_J())
                if getattr(self._poller, "samples", None):
                    valid = [p for _, p in self._poller.samples if not math.isnan(p)]
                    self.mean_power_W_raw = (sum(valid) / len(valid) / 1000.0) if valid else float("nan")
                else:
                    self.mean_power_W_raw = float("nan")
            if self.energy_J == self.energy_J:
                self.avg_power_W = self.energy_J / dur
        finally:
            if self._nvml_inited:
                try:
                    nvmlShutdown()
                except Exception:
                    pass


class NullMeter:
    """No-op energy meter (used if power metering is disabled)."""
    enabled = False
    mode = "disabled"
    energy_J = float("nan")
    avg_power_W = float("nan")
    mean_power_W_raw = float("nan")
    poll_interval_s = 0.0

    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------
# Sentence segmentation & data
# ---------------------------
PYSBD_SUPPORTED = {
    'hy','ur','fa','ru','es','nl','fr','de','pl','ar','ja','bg','my','it',
    'am','el','zh','sk','kk','da','hi','en','mr'
}
CACHE_DIR = Path("sentence_cache_multilingual_wikipedia")
CACHE_DIR.mkdir(exist_ok=True)

def sent_seg(text: str, lang_code: str) -> list[str]:
    """Segment text into sentences; fallback to regex."""
    if lang_code == "ur":
        lang_code = "hi"  # heuristic: use hi segmenter for urdu
    if lang_code in PYSBD_SUPPORTED:
        try:
            seg = pysbd.Segmenter(language=lang_code, clean=False)
            sents = seg.segment(text)
        except Exception:
            sents = re.split(r'(?<=[.!?۔])\s+', text)
    else:
        sents = re.split(r'(?<=[.!?۔])\s+', text)
    return [s.strip() for s in sents if len(s.split()) >= 3]


def stream_sentences(d_name: str, d_config: str, lang_code: str, n_target: int) -> list[str]:
    """Stream a few thousand sentences from HF datasets (Wikipedia) and cache to disk."""
    ds_id = d_name.replace("/", "_")
    cache = CACHE_DIR / f"{lang_code}_{ds_id}_{d_config}_{n_target}.txt"
    if cache.exists():
        logging.info(f"Cache hit -> {cache}")
        return cache.read_text(encoding="utf-8").splitlines()[:n_target]
    logging.info(f"Streaming {d_name}:{d_config} for lang={lang_code}...")
    ds = load_dataset(d_name, d_config, split="train", streaming=True)
    bag = []
    for ex in ds:
        t = ex.get("text", "")
        if t:
            bag.extend(sent_seg(t, lang_code))
        if len(bag) >= n_target * 1.2:
            break
    random.shuffle(bag)
    bag = list(dict.fromkeys(bag))  # dedupe, keep order
    if len(bag) < n_target:
        raise RuntimeError(f"Only {len(bag)} sentences, need {n_target}")
    cache.write_text("\n".join(bag), encoding="utf-8")
    return bag[:n_target]


# ---------------------------
# Simple stats helpers
# ---------------------------
def sample_stdev(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    m = math.fsum(vals) / n
    var = math.fsum((x - m) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var)

def ci95(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    return 1.96 * sample_stdev(vals) / math.sqrt(n)


# ---------------------------
# Benchmark core
# ---------------------------
@dataclass(slots=True)
class Row:
    model: str
    language: str
    batch_size: int
    mean_ms: float
    ci_ms: float
    sent_s: float
    ci_sent: float
    status: str
    energy_J: float
    avg_W: float
    J_per_sample: float
    J_per_token: float
    avg_W_raw: float = float("nan")


def _count_tokens(batches: list[dict]) -> int:
    total = 0
    for b in batches:
        m = b.get("attention_mask")
        total += int(m.sum().item()) if m is not None else int(b["input_ids"].numel())
    return max(total, 1)


def _prime_meter(meter, ready_samples=1, timeout_s=0.4):
    """Give the power poller a moment to collect the first sample (polling modes)."""
    if not getattr(meter, "enabled", False):
        return
    t0 = time.perf_counter()
    while (getattr(meter, "mode", "") != "total") and getattr(meter, "_poller", None) \
          and len(meter._poller.samples) < ready_samples \
          and (time.perf_counter() - t0) < timeout_s:
        time.sleep(meter.poll_interval_s / 2.0)


def benchmark_one(
    model_name: str,
    lang: str,
    pool: list[str],
    batch_sizes: list[int],
    warmup_batches: int,
    measure_batches: int,
    repeats: int,
    fp16: bool,
    force_fp32: bool,
    slow_per_batch_timing: bool,
    poll_interval_s: float,
    meter_cls,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model '{model_name}' on {device}")

    is_local = os.path.isdir(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=not is_local).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=not is_local)

    # Padding/tokenizer sanity
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': tok.eos_token or "[PAD]"})
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    if model.config.vocab_size != len(tok):
        logging.warning("Resizing embeddings to tokenizer size.")
        model.resize_token_embeddings(len(tok))

    # Max seq len (reserve for RoBERTa/XLM-R special tokens)
    max_pos = getattr(model.config, "max_position_embeddings", 512)
    if "roberta" in getattr(model.config, "model_type", "").lower() or "xlm" in getattr(model.config, "model_type", "").lower():
        max_pos = max(2, max_pos - 2)
        logging.info(f"RoBERTa-like: using max_length={max_pos}")

    total_batches = warmup_batches + measure_batches
    rows = []

    for bs in batch_sizes:
        if bs * total_batches > len(pool):
            logging.warning(f"Not enough sentences for bs={bs}. Stopping sweep.")
            break

        chunks = [pool[i:i+bs] for i in range(0, bs * total_batches, bs)][:total_batches]
        tb = [tok(b, return_tensors="pt", padding="longest", truncation=True, max_length=max_pos).to(device) for b in chunks]

        lat_means, thrpts, energies, powers, jps, jpt, avg_raw_pw = [], [], [], [], [], [], []

        for _ in range(repeats):
            with torch.inference_mode():
                for w in tb[:warmup_batches]:
                    _ = model(**w)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            with meter_cls(poll_interval_s=poll_interval_s) as meter, torch.inference_mode():
                _prime_meter(meter, ready_samples=1, timeout_s=0.4)
                use_fp16 = (fp16 and not force_fp32 and device.type == "cuda")

                if slow_per_batch_timing and device.type == "cuda":
                    # Per-batch CUDA event timing (more sync, slower, but precise latency)
                    total_ms = 0.0
                    for mb in tb[warmup_batches:]:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        if use_fp16:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                _ = model(**mb)
                        else:
                            _ = model(**mb)
                        end.record()
                        torch.cuda.synchronize()
                        total_ms += start.elapsed_time(end)
                    mean_ms = total_ms / measure_batches
                else:
                    # Fast wall-clock timing
                    t0 = time.perf_counter()
                    for mb in tb[warmup_batches:]:
                        if use_fp16:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                _ = model(**mb)
                        else:
                            _ = model(**mb)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    mean_ms = total_ms / measure_batches

            samples_measured = bs * measure_batches
            tokens_measured_total = _count_tokens(tb[warmup_batches:])

            lat_means.append(mean_ms)
            thrpts.append((samples_measured * 1000.0) / total_ms)

            # Energy metrics
            E = float(getattr(meter, "energy_J", float("nan")))
            P = float(getattr(meter, "avg_power_W", float("nan")))
            energies.append(E)
            powers.append(P)
            jps.append(E / samples_measured if E == E else float("nan"))
            jpt.append(E / tokens_measured_total if E == E else float("nan"))
            avg_raw_pw.append(getattr(meter, "mean_power_W_raw", float("nan")))

        rows.append(Row(
            model=model_name, language=lang, batch_size=bs,
            mean_ms=sum(lat_means)/len(lat_means), ci_ms=ci95(lat_means),
            sent_s=sum(thrpts)/len(thrpts), ci_sent=ci95(thrpts),
            status="ok",
            energy_J=sum(energies)/len(energies),
            avg_W=sum(powers)/len(powers),
            J_per_sample=sum(jps)/len(jps),
            J_per_token=sum(jpt)/len(jpt),
            avg_W_raw=sum(avg_raw_pw)/len(avg_raw_pw),
        ))
        logging.info(f"bs={bs}: {rows[-1]}")

    del model, tok
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


# ---------------------------
# CSV helpers
# ---------------------------
def write_csv_header_if_needed(csv_path: Path):
    header = [
        "model","language","batch_size",
        "mean_ms","ci_ms","sent_s","ci_sent","status",
        "energy_J","avg_W","J_per_sample","J_per_token","avg_W_raw"
    ]
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(header)

def append_rows(csv_path: Path, model: str, rows: list[Row]):
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow([
                model, r.language, r.batch_size,
                f"{r.mean_ms:.3f}", f"{r.ci_ms:.3f}", f"{r.sent_s:.2f}", f"{r.ci_sent:.2f}", r.status,
                f"{r.energy_J:.6f}", f"{r.avg_W:.3f}", f"{r.J_per_sample:.6f}", f"{r.J_per_token:.9f}", f"{r.avg_W_raw:.3f}"
            ])


# ---------------------------
# CLI / Main
# ---------------------------
DEFAULT_CORE_LANGS = ["en", "de", "fr", "da", "hu", "ur", "ga"]
DEFAULT_MODEL_TEMPLATES = [
    "FacebookAI/xlm-roberta-large",
    "FacebookAI/xlm-roberta-base",
    "dschulmeist/TiME-{lang}-m",
    "dschulmeist/TiME-{lang}-s",
    "dschulmeist/TiME-{lang}-xs"
]

def parse_args():
    p = argparse.ArgumentParser(description="Run encoder benchmark with optional GPU energy metering.")
    p.add_argument("--output", type=Path, default=Path("benchmark_results.csv"), help="Output CSV path")
    p.add_argument("--langs", nargs="+", default=DEFAULT_CORE_LANGS, help="Languages (HF Wikipedia codes)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model templates (use {lang} placeholder). If omitted, a default set is used.")
    p.add_argument("--repeats", type=int, default=3, help="Repeats per (model,bs)")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Enable FP16 autocast")
    p.add_argument("--fast-preset", action="store_true", help="Fast timing preset (10 Hz polling, fewer syncs)")
    p.add_argument("--enable-power", action=argparse.BooleanOptionalAction, default=True, help="Enable GPU power/energy metering")
    p.add_argument("--poll-interval", type=float, default=0.1, help="Polling interval for power (seconds)")
    p.add_argument("--batches", type=str, default="1,2,4,8,16,32,64,128,256", help="Comma-separated batch sizes")
    p.add_argument("--warmup-batches", type=int, default=5, help="Warmup batches")
    p.add_argument("--measure-batches", type=int, default=20, help="Measured batches")
    p.add_argument("--force-fp32", action=argparse.BooleanOptionalAction, default=False, help="Force FP32 even if --fp16 is set")
    p.add_argument("--slow-per-batch-timing", action=argparse.BooleanOptionalAction, default=True,
                   help="Use per-batch CUDA events (slower, precise latency)")
    return p.parse_args()


def expand_models(templates: list[str], lang: str) -> list[str]:
    out, seen = [], set()
    for t in templates:
        m = t.format(lang=lang) if "{lang}" in t else t
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def main():
    args = parse_args()

    # Torch backend knobs
    try:
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            if args.fast_preset:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            try:
                torch.set_float32_matmul_precision("high" if args.fast_preset else "highest")
            except Exception:
                pass
        torch.set_num_threads(1)
    except Exception:
        pass

    # Meter selection
    Meter = GpuEnergyMeter if args.enable_power else NullMeter

    # Fast preset overrides
    poll_interval = 0.1 if args.fast_preset else args.poll_interval
    slow_per_batch_timing = False if args.fast_preset else args.slow_per_batch_timing
    force_fp32 = False if args.fast_preset else args.force_fp32

    batch_sizes = [int(x) for x in args.batches.split(",") if x.strip()]
    total_needed = max(batch_sizes) * (args.warmup_batches + args.measure_batches)

    # Models
    model_templates = args.models if args.models else DEFAULT_MODEL_TEMPLATES

    # Repro-ish
    random.seed(1337)
    try:
        import numpy as np
        np.random.seed(1337)
    except Exception:
        pass

    write_csv_header_if_needed(args.output)

    for lg in args.langs:
        logging.info("\n===== Language: %s =====", lg.upper())
        try:
            pool = stream_sentences("wikimedia/wikipedia", f"20231101.{lg}", lg, total_needed)
        except Exception as e:
            logging.error("Failed to stream sentences for %s: %s", lg, e)
            continue

        for model_name in expand_models(model_templates, lg):
            logging.info("--- %s on %s ---", model_name, lg)
            try:
                rows = benchmark_one(
                    model_name=model_name,
                    lang=lg,
                    pool=pool,
                    batch_sizes=batch_sizes,
                    warmup_batches=args.warmup_batches,
                    measure_batches=args.measure_batches,
                    repeats=args.repeats,
                    fp16=args.fp16,
                    force_fp32=force_fp32,
                    slow_per_batch_timing=slow_per_batch_timing,
                    poll_interval_s=poll_interval,
                    meter_cls=Meter,
                )
                if rows:
                    append_rows(args.output, model_name, rows)
            except KeyboardInterrupt:
                logging.warning("Interrupted by user.")
                return
            except Exception as e:
                logging.exception("Failed: %s on %s -> %s", model_name, lg, e)

    logging.info("Done. CSV -> %s", args.output)


if __name__ == "__main__":
    main()
