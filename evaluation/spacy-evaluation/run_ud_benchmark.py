#!/usr/bin/env python3
import argparse, json, os, pathlib, re, shutil, subprocess, sys, zipfile, urllib.request, importlib, time

# Avoid torchvision imports in HF
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# ======= knobs =======
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LR = "5e-5"
TRANSFORMER_ACCUM = 2
TOK2VEC_EPOCHS = 10
USE_CURATED_FOR_TRF_BASELINES = True  # first try curated; fallback to spacy-transformers if it fails
TOK2VEC_WIDTH = 256  # literal width to bake in, avoiding ${...} interpolation crashes
# =====================

UD_MAP = {
    "en": ("UD_English-EWT", "r2.15"),
    "de": ("UD_German-GSD", "r2.15"),
    "fr": ("UD_French-GSD", "r2.15"),
    "es": ("UD_Spanish-AnCora", "r2.15"),
    "it": ("UD_Italian-ISDT", "r2.15"),
    "da": ("UD_Danish-DDT", "r2.15"),
}

SPACY_BASELINES = {
    "en": ["en_core_web_sm","en_core_web_md","en_core_web_lg","en_core_web_trf"],
    "de": ["de_core_news_sm","de_core_news_md","de_core_news_lg","de_dep_news_trf"],
    "fr": ["fr_core_news_sm","fr_core_news_md","fr_core_news_lg","fr_dep_news_trf"],
    "es": ["es_core_news_sm","es_core_news_md","es_core_news_lg","es_dep_news_trf"],
    "it": ["it_core_news_sm","it_core_news_md","it_core_news_lg","it_dep_news_trf"],
    "da": ["da_core_news_sm","da_core_news_md","da_core_news_lg"],
}

# HF encoders for fallback (spacy-transformers)
DEFAULT_TRF_ENCODER = {
    "en": "roberta-base",
    "de": "xlm-roberta-base",
    "fr": "xlm-roberta-base",
    "es": "xlm-roberta-base",
    "it": "xlm-roberta-base",
    "da": "xlm-roberta-base",
}

# Curated encoder class per model name
def _curated_arch_for_name(name: str):
    low = name.lower()
    if "roberta" in low:
        return "spacy-curated-transformers.RobertaTransformer.v1"
    if "xlm" in low:
        return "spacy-curated-transformers.XlmrTransformer.v1"
    if "albert" in low:
        return "spacy-curated-transformers.AlbertTransformer.v1"
    return "spacy-curated-transformers.BertTransformer.v1"

# ===== helpers =====
def run(cmd, env=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def ensure_dirs(*paths):
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def slugify(txt):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", txt)

def has_model(mname):
    try:
        out = subprocess.check_output([sys.executable, "-m", "spacy", "validate"]).decode()
        return mname in out
    except subprocess.CalledProcessError:
        return False

def model_best_exists(out_dir: pathlib.Path) -> bool:
    return (out_dir / "model-best").exists()

def metrics_exist(metrics_base: pathlib.Path, require_gpu: bool) -> bool:
    cpu = metrics_base.with_name(metrics_base.stem + ".cpu.json")
    gpu = metrics_base.with_name(metrics_base.stem + ".gpu.json")
    if require_gpu:
        return cpu.exists() and gpu.exists()
    return cpu.exists()

def all_models_trained_for_lang(lang: str, hf_cfg_list, baselines, models_dir: pathlib.Path) -> bool:
    # HF customs
    for enc in hf_cfg_list:
        slug = enc.get("slug") or slugify(enc["id"])
        out_dir = models_dir / f"{lang}_hf_{slug}"
        if not model_best_exists(out_dir):
            return False
    # spaCy baselines
    for m in baselines:
        slug = slugify(m)
        out_dir = models_dir / f"{lang}_spacy_{slug}"
        if not model_best_exists(out_dir):
            return False
    return True

# ===== data =====
def fetch_ud(lang, data_dir):
    treebank, tag = UD_MAP[lang]
    out = pathlib.Path(data_dir)/f"ud_{lang}"
    ensure_dirs(out)

    # If already present, skip download
    train_c = out/"train.conllu"
    dev_c = out/"dev.conllu"
    test_c = out/"test.conllu"
    if train_c.exists() and dev_c.exists() and test_c.exists():
        print(f"UD {lang} already present at {out}, skipping download.")
        return out

    zip_path = out/"tb.zip"
    url = f"https://github.com/UniversalDependencies/{treebank}/archive/refs/tags/{tag}.zip"
    print(f"Downloading {treebank} {tag} -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    root = next(p for p in out.iterdir() if p.is_dir() and p.name.startswith(treebank+"-"))
    files = {
        "train": next(root.glob("*train.conllu")),
        "dev":   next(root.glob("*dev.conllu")),
        "test":  next(root.glob("*test.conllu")),
    }
    for split, src in files.items():
        shutil.copy2(src, out/f"{split}.conllu")
    zip_path.unlink()
    print(f"Fetched UD {lang}.")
    return out

def convert_ud(lang, ud_dir):
    for split in ("train","dev","test"):
        src = pathlib.Path(ud_dir)/f"{split}.conllu"
        dst = pathlib.Path(ud_dir)/f"{split}.spacy"
        if dst.exists() and src.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"✔ {dst} up-to-date, skip convert")
            continue
        run([sys.executable, "-m", "spacy", "convert", str(src), str(ud_dir),
             "--converter", "conllu", "-n", "10"])
    print(f"Converted UD {lang} to .spacy")

# ===== config patch helpers =====
SECTION_RE = re.compile(r"(?m)^\[(?P<header>[^\]]+)\]\s*$")

def _find_block_span(text, header):
    m = re.search(rf"(?m)^\[{re.escape(header)}\]\s*$", text)
    if not m:
        return None
    start = m.start()
    after = m.end()
    next_m = SECTION_RE.search(text, pos=after)
    end = next_m.start() if next_m else len(text)
    return (start, end)

def _replace_block(text, header, new_block):
    span = _find_block_span(text, header)
    if not span:
        if not new_block.endswith("\n"):
            new_block += "\n"
        return text.rstrip() + "\n" + new_block
    a, b = span
    return text[:a] + new_block + text[b:]

def _delete_block(text, header):
    span = _find_block_span(text, header)
    if not span:
        return text
    a, b = span
    return text[:a] + text[b:]

def _set_paths_value(text, key, value_str_or_null):
    header = "paths"
    span = _find_block_span(text, header)
    if not span:
        text += "\n[paths]\n"
        span = _find_block_span(text, header)
    a, b = span
    block = text[a:b]
    pattern = re.compile(rf'(?m)^{re.escape(key)}\s*=\s*.*$')
    rep = f'{key} = {value_str_or_null}'
    if pattern.search(block):
        block = pattern.sub(rep, block)
    else:
        block = block.rstrip() + "\n" + rep + "\n"
    return text[:a] + block + text[b:]

def _set_training_hparams(text, *, max_epochs=None, learn_rate=None, accumulate=None):
    if "[training]" not in text:
        text += "\n[training]\n"
    a,b = _find_block_span(text, "training")
    block = text[a:b]
    def set_kv(block, key, val):
        pat = re.compile(rf'(?m)^{re.escape(key)}\s*=\s*.*$')
        rep = f"{key} = {val}"
        return pat.sub(rep, block) if pat.search(block) else (block.rstrip() + "\n" + rep + "\n")
    if max_epochs is not None:
        block = set_kv(block, "max_epochs", str(int(max_epochs)))
    if accumulate is not None:
        block = set_kv(block, "accumulate_gradient", str(int(accumulate)))
    text = text[:a] + block + text[b:]
    if learn_rate is not None:
        text = _set_optimizer_lr(text, learn_rate)
    return text

def _set_optimizer_lr(text, lr):
    if _find_block_span(text, "training.optimizer") is None:
        text += "\n[training.optimizer]\n@optimizers = \"Adam.v1\"\nlearn_rate = 0.001\n"
    a,b = _find_block_span(text, "training.optimizer")
    block = text[a:b]
    pat = re.compile(r'(?m)^learn_rate\s*=\s*.*$')
    rep = f"learn_rate = {lr}"
    if pat.search(block):
        block = pat.sub(rep, block)
    else:
        block = block.rstrip() + "\n" + rep + "\n"
    return text[:a] + block + text[b:]

def _ensure_pipeline(text, wanted):
    if "[nlp]" not in text:
        text += "\n[nlp]\n"
    if re.search(r'(?m)^\s*pipeline\s*=\s*\[.*?\]\s*$', text):
        text = re.sub(
            r'(?m)^\s*pipeline\s*=\s*\[.*?\]\s*$',
            'pipeline = ["' + '","'.join(wanted) + '"]',
            text
        )
    else:
        text = _replace_block(text, "nlp", "[nlp]\n" + 'pipeline = ["' + '","'.join(wanted) + '"]\n')
    return text

def _remove_tok2vec_everywhere(text):
    for h in [
        "components.tok2vec",
        "components.tok2vec.model",
        "components.tok2vec.model.embed",
        "components.tok2vec.model.encode",
    ]:
        text = _delete_block(text, h)
    def repl_pipeline(m):
        items = [i.strip().strip('"').strip("'") for i in m.group(1).split(",")]
        items = [i for i in items if i and i != "tok2vec"]
        return 'pipeline = ["' + '","'.join(items) + '"]'
    text = re.sub(r'(?m)^\s*pipeline\s*=\s*\[(.*?)\]\s*$', repl_pipeline, text)
    return text

def _tok2vec_force_literal_widths(text, width=256):
    text = re.sub(
        r"\$\{components\.tok2vec\.model\.encode\.width\}",
        str(width),
        text,
    )
    for comp in ("morphologizer", "tagger", "parser"):
        header = f"components.{comp}.model.tok2vec"
        block_span = _find_block_span(text, header)
        if block_span:
            a, b = block_span
            block = text[a:b]
            if '@architectures' not in block or "Tok2VecListener" not in block:
                block = (
                    f"[{header}]\n"
                    '@architectures = "spacy.Tok2VecListener.v1"\n'
                    f"width = {width}\n"
                    'upstream = "*"\n'
                )
            else:
                block = re.sub(r'(?m)^\s*width\s*=.*$', f"width = {width}", block) if re.search(r'(?m)^\s*width\s*=', block) else block.rstrip() + f"\nwidth = {width}\n"
                block = re.sub(r'(?m)^\s*upstream\s*=.*$', 'upstream = "*"', block) if re.search(r'(?m)^\s*upstream\s*=', block) else block.rstrip() + '\nupstream = "*"\n'
            text = text[:a] + block + text[b:]
    return text

# ----- spacy-transformers (HF) -----
def _ensure_trf_component_spacy_transformers(text, hf_name):
    block = (
        "[components.transformer]\n"
        'factory = "transformer"\n\n'
        "[components.transformer.model]\n"
        '@architectures = "spacy-transformers.TransformerModel.v3"\n'
        f'name = "{hf_name}"\n'
    )
    return _replace_block(text, "components.transformer", block)

def _force_listener_spacy_transformers(text, comp_header):
    tok2vec_header = comp_header + ".model.tok2vec"
    new_block = (
        f"[{tok2vec_header}]\n"
        '@architectures = "spacy-transformers.TransformerListener.v1"\n'
        'upstream = "transformer"\n'
        f"[{tok2vec_header}.pooling]\n"
        '@layers = "reduce_mean.v1"\n'
    )
    return _replace_block(text, tok2vec_header, new_block)

# ----- spacy-curated-transformers (curated) -----
def _ensure_trf_component_curated(text, encoder_name):
    arch = _curated_arch_for_name(encoder_name)
    block = (
        "[components.transformer]\n"
        'factory = "transformer"\n\n'
        "[components.transformer.model]\n"
        f'@architectures = "{arch}"\n'
        f'name = "{encoder_name}"\n'
    )
    return _replace_block(text, "components.transformer", block)

def _force_listener_curated(text, comp_header):
    tok2vec_header = comp_header + ".model.tok2vec"
    new_block = (
        f"[{tok2vec_header}]\n"
        '@architectures = "spacy-curated-transformers.LastTransformerLayerListener.v1"\n'
        'upstream = "transformer"\n'
    )
    return _replace_block(text, tok2vec_header, new_block)

# ===== initializers =====
def init_config_transformer_spacy_tf(lang, transformer_name, cfg_path):
    run([
        sys.executable, "-m", "spacy", "init", "config", str(cfg_path),
        "--lang", lang,
        "--pipeline", "morphologizer,tagger,parser,senter",
        "--optimize", "accuracy",
        "-F",
    ])
    text = pathlib.Path(cfg_path).read_text()
    text = _ensure_trf_component_spacy_transformers(text, transformer_name)
    text = _ensure_pipeline(text, ["transformer","morphologizer","tagger","parser","senter"])
    text = _remove_tok2vec_everywhere(text)
    for comp in ("morphologizer","tagger","parser","senter"):
        text = _force_listener_spacy_transformers(text, f"components.{comp}")
    text = _set_paths_value(text, "vectors", "null")
    text = _set_training_hparams(text, max_epochs=TRANSFORMER_EPOCHS, learn_rate=TRANSFORMER_LR, accumulate=TRANSFORMER_ACCUM)
    pathlib.Path(cfg_path).write_text(text)
    print("\n" + "="*80)
    print(f"FULL TRAINING CONFIG (spacy-transformers): {cfg_path}")
    print("-"*80)
    print(pathlib.Path(cfg_path).read_text())
    print("="*80 + "\n")

def init_config_transformer_curated(lang, encoder_name, cfg_path):
    run([
        sys.executable, "-m", "spacy", "init", "config", str(cfg_path),
        "--lang", lang,
        "--pipeline", "morphologizer,tagger,parser,senter",
        "--optimize", "accuracy",
        "-F",
    ])
    text = pathlib.Path(cfg_path).read_text()
    text = _ensure_trf_component_curated(text, encoder_name)
    text = _ensure_pipeline(text, ["transformer","morphologizer","tagger","parser","senter"])
    text = _remove_tok2vec_everywhere(text)
    for comp in ("morphologizer","tagger","parser","senter"):
        text = _force_listener_curated(text, f"components.{comp}")
    text = _set_paths_value(text, "vectors", "null")
    text = _set_training_hparams(text, max_epochs=TRANSFORMER_EPOCHS, learn_rate=TRANSFORMER_LR, accumulate=TRANSFORMER_ACCUM)
    pathlib.Path(cfg_path).write_text(text)
    print("\n" + "="*80)
    print(f"FULL TRAINING CONFIG (curated): {cfg_path}")
    print("-"*80)
    print(pathlib.Path(cfg_path).read_text())
    print("="*80 + "\n")

def _maybe_tok2vec_path(package_name):
    try:
        m = importlib.import_module(package_name)
        pkg = pathlib.Path(m.__file__).parent
        t2v = pkg / "tok2vec"
        return t2v if t2v.exists() else None
    except Exception:
        return None

def init_config_tok2vec(lang, cfg_path, init_from_package=None):
    run([
        sys.executable, "-m", "spacy", "init", "config", str(cfg_path),
        "--lang", lang,
        "--pipeline", "morphologizer,tagger,parser,senter",
        "--optimize", "accuracy",
        "-F",
    ])
    text = pathlib.Path(cfg_path).read_text()
    text = _ensure_pipeline(text, ["tok2vec","morphologizer","tagger","parser","senter"])
    text = _set_paths_value(text, "vectors", "null")
    text = _set_training_hparams(text, max_epochs=TOK2VEC_EPOCHS, accumulate=1)
    text = _tok2vec_force_literal_widths(text, width=TOK2VEC_WIDTH)
    if init_from_package:
        t2v = _maybe_tok2vec_path(init_from_package)
        if t2v:
            print(f"↪ Using init_tok2vec from {init_from_package}: {t2v}")
            text = _set_paths_value(text, "init_tok2vec", f"\"{str(t2v)}\"")
        else:
            text = _set_paths_value(text, "init_tok2vec", "null")
    pathlib.Path(cfg_path).write_text(text)
    print("\n" + "="*80)
    print(f"FULL TRAINING CONFIG (tok2vec): {cfg_path}")
    print("-"*80)
    print(pathlib.Path(cfg_path).read_text())
    print("="*80 + "\n")

# ===== train / eval =====
def train(cfg_path, train_spacy, dev_spacy, out_dir, gpu_id=None):
    cmd = [sys.executable, "-m", "spacy", "train", str(cfg_path),
           "--paths.train", str(train_spacy),
           "--paths.dev", str(dev_spacy),
           "--output", str(out_dir)]
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    if gpu_id is not None:
        cmd += ["--gpu-id", str(gpu_id)]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run(cmd, env=env)

def evaluate_cpu_gpu(model_path_or_name, test_spacy, out_json_base, gpu_id=None):
    """Save two metrics files: *.cpu.json and (if GPU available) *.gpu.json."""
    out_json_base = pathlib.Path(out_json_base)
    # CPU
    out_cpu = out_json_base.with_name(out_json_base.stem + ".cpu.json")
    cmd = [sys.executable, "-m", "spacy", "benchmark", "accuracy",
           str(model_path_or_name), str(test_spacy), "--output", str(out_cpu)]
    run(cmd)
    print(f"✔ CPU metrics saved to {out_cpu}")

    # GPU (optional)
    if gpu_id is not None:
        out_gpu = out_json_base.with_name(out_json_base.stem + ".gpu.json")
        cmd = [sys.executable, "-m", "spacy", "benchmark", "accuracy",
               str(model_path_or_name), str(test_spacy), "--output", str(out_gpu),
               "--gpu-id", str(gpu_id)]
        run(cmd)
        print(f"✔ GPU metrics saved to {out_gpu}")

# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,de,fr,es,it,da")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--hf-json", default="hf_models.json", help="map of HF encoders per language")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--bench-only", action="store_true",
                    help="Nur Benchmarks ausführen (kein Training). Erwartet vorhandene Modelle in models/*.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip languages/models that are already trained/evaluated.")
    args = ap.parse_args()

    work = pathlib.Path(args.workdir)
    data_dir = work/"assets"
    models_dir = work/"models"
    metrics_dir = work/"metrics"
    cfg_dir = work/"configs"
    ensure_dirs(data_dir, models_dir, metrics_dir, cfg_dir)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    hf_cfg = json.loads(pathlib.Path(args.hf_json).read_text()) if pathlib.Path(args.hf_json).exists() else {}

    # Always make sure UD test data exist
    for lang in langs:
        ud_dir = fetch_ud(lang, data_dir)
        convert_ud(lang, ud_dir)

    gpu_id = None if args.cpu else args.gpu

    if args.bench_only:
        print("▶ Bench-only mode: skip training, evaluate existing models in models/*")
        for lang in langs:
            test_spacy = data_dir/f"ud_{lang}"/"test.spacy"

            # Evaluate every trained model directory for this lang
            for sub in sorted(models_dir.glob(f"{lang}_*")):
                model_best = sub/"model-best"
                model_last = sub/"model-last"
                model_path = model_best if model_best.exists() else (model_last if model_last.exists() else None)
                if not model_path:
                    continue

                if sub.name.startswith(f"{lang}_hf_"):
                    slug = sub.name[len(f"{lang}_hf_"):]
                    metrics_base = metrics_dir/f"ud_{lang}_custom_{slug}.json"
                elif sub.name.startswith(f"{lang}_spacy_"):
                    slug = sub.name[len(f"{lang}_spacy_"):]
                    metrics_base = metrics_dir/f"ud_{lang}_spacy_{slug}.json"
                else:
                    slug = slugify(sub.name)
                    metrics_base = metrics_dir/f"ud_{lang}_{slug}.json"

                # Resume for benchmarks: skip if metrics already exist (CPU and GPU if requested)
                if args.resume and metrics_exist(metrics_base, require_gpu=(gpu_id is not None)):
                    print(f"⏭  Resume: metrics already exist for {sub.name}, skipping benchmark")
                    continue

                print(f"→ Benchmark {sub.name} ({'model-best' if model_best.exists() else 'model-last'})")
                evaluate_cpu_gpu(model_path, test_spacy, metrics_base, gpu_id)
        print("Done. Metrics saved in:", metrics_dir)
        return

    # 2) download spaCy baselines if missing
    for lang in langs:
        for m in SPACY_BASELINES.get(lang, []):
            try:
                if not has_model(m):
                    run([sys.executable, "-m", "spacy", "download", m])
            except subprocess.CalledProcessError:
                print(f"Warning: could not download {m}")

    # ===== Training + Eval (with optional resume) =====
    for lang in langs:
        ud_dir = data_dir/f"ud_{lang}"
        train_spacy = ud_dir/"train.spacy"
        dev_spacy = ud_dir/"dev.spacy"
        test_spacy = ud_dir/"test.spacy"

        encoders = hf_cfg.get(lang, [])
        baselines = SPACY_BASELINES.get(lang, [])

        # If resume requested: skip entire language if ALL expected models already trained
        if args.resume and all_models_trained_for_lang(lang, encoders, baselines, models_dir):
            print(f"⏭  Resume: all models for '{lang}' already trained. Skipping language.")
            continue

        # 3a) train HF encoders (transformer, 2 epochs)
        if encoders:
            for enc in encoders:
                name = enc["id"]
                slug = enc.get("slug") or slugify(name)
                cfg_path = cfg_dir/f"{lang}_hf_{slug}.cfg"
                out_dir = models_dir/f"{lang}_hf_{slug}"

                if args.resume and model_best_exists(out_dir):
                    print(f"⏭  Resume: {out_dir.name} already has model-best/, skipping training")
                else:
                    init_config_transformer_spacy_tf(lang, name, cfg_path)
                    train(cfg_path, train_spacy, dev_spacy, out_dir, gpu_id)

                metrics_base = metrics_dir/f"ud_{lang}_custom_{slug}.json"
                if args.resume and metrics_exist(metrics_base, require_gpu=(gpu_id is not None)):
                    print(f"⏭  Resume: metrics exist for {out_dir.name}, skipping evaluation")
                else:
                    evaluate_cpu_gpu(out_dir/"model-best", test_spacy, metrics_base, gpu_id)

        else:
            print(f"No HF encoders listed for {lang} in {args.hf_json}. Skipping custom HF for this lang.")

        # 3b) train spaCy baselines (finetune)
        for m in baselines:
            slug = slugify(m)
            out_dir = models_dir/f"{lang}_spacy_{slug}"
            cfg_path = cfg_dir/f"{lang}_spacy_{slug}.cfg"

            if m.endswith("_trf"):
                encoder = DEFAULT_TRF_ENCODER.get(lang, "xlm-roberta-base")
                if args.resume and model_best_exists(out_dir):
                    print(f"⏭  Resume: {out_dir.name} already has model-best/, skipping training")
                else:
                    if USE_CURATED_FOR_TRF_BASELINES:
                        try:
                            print(f"→ Finetune spaCy TRF baseline '{m}' with **curated** encoder: {encoder}")
                            init_config_transformer_curated(lang, encoder, cfg_path)
                            train(cfg_path, train_spacy, dev_spacy, out_dir, gpu_id)
                        except subprocess.CalledProcessError as e:
                            print(f"⚠ Curated setup failed for '{m}' ({e}). Falling back to spacy-transformers.")
                            init_config_transformer_spacy_tf(lang, encoder, cfg_path)
                            train(cfg_path, train_spacy, dev_spacy, out_dir, gpu_id)
                    else:
                        print(f"→ Finetune spaCy TRF baseline '{m}' with spacy-transformers encoder: {encoder}")
                        init_config_transformer_spacy_tf(lang, encoder, cfg_path)
                        train(cfg_path, train_spacy, dev_spacy, out_dir, gpu_id)
            else:
                if args.resume and model_best_exists(out_dir):
                    print(f"⏭  Resume: {out_dir.name} already has model-best/, skipping training")
                else:
                    print(f"→ Finetune spaCy tok2vec baseline '{m}' (init from package tok2vec if available)")
                    init_config_tok2vec(lang, cfg_path, init_from_package=m)
                    train(cfg_path, train_spacy, dev_spacy, out_dir, gpu_id)

            metrics_base = metrics_dir/f"ud_{lang}_spacy_{slug}.json"
            if args.resume and metrics_exist(metrics_base, require_gpu=(gpu_id is not None)):
                print(f"⏭  Resume: metrics exist for {out_dir.name}, skipping evaluation")
            else:
                evaluate_cpu_gpu(out_dir/"model-best", test_spacy, metrics_base, gpu_id)

    print("Done. Metrics saved in:", metrics_dir)

if __name__ == "__main__":
    main()
