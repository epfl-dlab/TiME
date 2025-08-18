#find-best-checkpoint/find-best-checkpint.py
import os
import re
import json
import glob
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel
from datasets import load_dataset

# --- Local Module Imports ---
# Ensure these modules are in the Python path.
try:
    from minilmv2_fast import MiniLMv2Fast
    from minilmv2_ltg_fast import MiniLMLtgAdapted

    print("Successfully imported custom MiniLM modules.")
except ImportError as e:
    raise ImportError(
        "Could not import MiniLMv2Fast or MiniLMLtgAdapted. "
        "Ensure the containing modules are in your PYTHONPATH."
    ) from e

# --- Configuration & Constants ---
LANGUAGES = ["en", "fr", "de", "da", "ur", "ga", "hu"]
BASE_MODEL_DIR = Path("./models")


# --- Helper Functions ---

def read_model_details(model_dir: Path) -> Dict[str, Any]:
    """
    Reads a 'model_details.txt' file from a model directory and parses hyperparameters.

    Args:
        model_dir: The directory containing the model_details.txt file.

    Returns:
        A dictionary with parsed hyperparameters.
    """
    details_file = model_dir / "model_details.txt"
    if not details_file.exists():
        return {}

    details = {}
    with open(details_file, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = [x.strip() for x in line.split(':', 1)]

            if "Teacher_Model" in key:
                details["teacher_model_name"] = val
            elif "Teacher_Distillation_Layer" in key:
                details["teacher_layer_idx"] = int(val)
            elif "Student_Hidden_Size" in key:
                details["student_hidden_size"] = int(val)
            elif "Student_Num_Layers" in key:
                details["student_num_layers"] = int(val)
            elif "Num_Relation_Heads" in key or "Student_Attention_Heads" in key:
                details["attention_heads"] = int(val)
            elif "Minilm_Relations" in key:
                try:
                    details["relations"] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    pass  # Ignore malformed relation strings
    return details


def derive_heuristic_hparams(teacher_name: str, student_num_layers: Optional[int]) -> Tuple[int, int, int]:
    """
    Provides fallback hyperparameters if model_details.txt is incomplete.
    - Teacher Layer: Total layers - 5
    - Student Layers: Uses provided value or defaults to 6
    - Attention Heads: Defaults to 64
    """
    try:
        config = AutoConfig.from_pretrained(teacher_name, trust_remote_code=True)
        teacher_layer_idx = max(1, config.num_hidden_layers - 5)
        student_layer_idx = student_num_layers or 6
        attention_heads = 64  # A common default for base/large models
        return teacher_layer_idx, student_layer_idx, attention_heads
    except Exception as e:
        print(f"Warning: Could not derive heuristic hparams for {teacher_name}. Using safe defaults. Error: {e}")
        return 12, 6, 64  # Safe defaults


def get_model_variant_paths_for_lang(lang: str) -> List[Path]:
    """Finds all model variant directories for a given language."""
    lang_folder = BASE_MODEL_DIR / f"uonlp_CulturaX_{lang}"
    if not lang_folder.exists():
        print(f"Warning: Language folder not found: {lang_folder}")
        return []
    return [d for d in lang_folder.iterdir() if d.is_dir() and d.name.startswith("minilm")]


def get_eval_dataset(lang: str) -> List[str]:
    """
    Loads the appropriate evaluation dataset for a given language.
    Uses 'google/wmt24pp' for most languages and 'facebook/flores' for Irish (ga).
    """
    print(f"  Loading evaluation data for '{lang}'...")
    try:
        if lang == "ga":
            dataset = load_dataset("facebook/flores", "eng_Latn-gle_Latn", split='dev')
            eval_texts = [item['sentence_gle_Latn'] for item in dataset]
        else:
            wmt_configs = {"en": "en-de_DE", "de": "en-de_DE", "fr": "en-fr_FR", "da": "en-da_DK", "ur": "en-ur_PK",
                           "hu": "en-hu_HU"}
            config_name = wmt_configs.get(lang)
            if not config_name:
                print(f"  Skipping '{lang}': No WMT24++ configuration found.")
                return []

            dataset = load_dataset("google/wmt24pp", name=config_name, split="train")
            column = 'source' if lang == 'en' else 'target'
            eval_texts = [item[column] for item in dataset]

        print(f"  Successfully loaded {len(eval_texts):,} samples for '{lang}'.")
        return eval_texts
    except Exception as e:
        print(f"  Error loading evaluation data for '{lang}': {e}")
        return []


def get_base_model(model: nn.Module) -> nn.Module:
    """Extracts the core transformer model (e.g., BertModel) from a larger model."""
    if hasattr(model, 'base_model'):
        return model.base_model
    # Add other common attribute names for base models
    for attr in ['bert', 'roberta', 'electra']:
        if hasattr(model, attr):
            return getattr(model, attr)
    return model  # Assume the passed model is already the base model


# --- Core Evaluation Logic ---

def calculate_distillation_loss(
        student_checkpoint_path: str,
        teacher_model_name: str,
        hparams: Dict[str, Any],
        eval_texts: List[str],
        device: str
) -> float:
    """
    Calculates the MiniLMv2 distillation loss for a given student checkpoint.

    This function handles model loading, MiniLM module instantiation, and batch
    processing to compute the average loss over the evaluation data.
    """
    if not eval_texts:
        print("    No evaluation texts provided. Returning infinity loss.")
        return float('inf')

    # The tokenizer is expected to be in the parent directory of the 'student' folder.
    tokenizer_path = str(Path(student_checkpoint_path).parent.parent)

    teacher_model, student_model, distillation_module = None, None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Load models
        teacher_model = AutoModel.from_pretrained(teacher_model_name, trust_remote_code=True).to(device).eval()
        student_model = AutoModel.from_pretrained(student_checkpoint_path, trust_remote_code=True).to(device).eval()

        # Instantiate the correct distillation module
        DistillationClass = MiniLMLtgAdapted if "hplt" in teacher_model_name.lower() else MiniLMv2Fast

        distillation_module = DistillationClass(
            teacher=get_base_model(teacher_model),
            student=get_base_model(student_model),
            L=hparams["teacher_layer_idx"],
            M=hparams["student_num_layers"],
            relations=hparams["relations"],
            A_r=hparams["attention_heads"]
        ).to(device).eval()

    except Exception as e:
        print(f"    Error during model setup for {student_checkpoint_path}: {e}")
        return float('inf')
    finally:
        # Ensure initial models are cleaned up even if distillation module fails
        if 'teacher_model' in locals() and teacher_model and not distillation_module: del teacher_model
        if 'student_model' in locals() and student_model and not distillation_module: del student_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    total_loss, num_batches = 0.0, 0
    batch_size = 32  # Keep this fixed for comparable loss values
    max_length = min(tokenizer.model_max_length or 512, 512)

    for i in tqdm(range(0, len(eval_texts), batch_size), desc="    Calculating Loss", leave=False, dynamic_ncols=True):
        batch_texts = eval_texts[i:i + batch_size]
        try:
            encodings = tokenizer(
                batch_texts, return_tensors='pt', padding='max_length',
                truncation=True, max_length=max_length
            ).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                loss, *_ = distillation_module(**encodings)

            total_loss += loss.item()
            num_batches += 1
        except Exception as e:
            print(f"    Error in batch {i // batch_size}. Skipping. Error: {e}")
            continue

    # --- Cleanup ---
    if distillation_module:
        for handle in getattr(distillation_module, '_hook_handles', []):
            handle.remove()
        del distillation_module
    del teacher_model
    del student_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / num_batches if num_batches > 0 else float('inf')


# --- Main Execution Block ---

def main():
    """
    Main function to run the checkpoint evaluation grid search.
    Iterates through all languages, model variants, and checkpoints,
    calculating distillation loss and saving results to a CSV file.
    Supports resuming from a previously generated CSV.
    """
    parser = argparse.ArgumentParser(description="Find the best model checkpoint based on distillation loss.")
    parser.add_argument(
        "--results_csv",
        type=str,
        default="/dlabscratch1/schulmei/minilmv2/evaluation/find-best-model-checkpoint/best_model_results.csv",
        help="Path to the CSV file for saving and resuming results."
    )
    args = parser.parse_args()

    # Load existing results to support resuming
    if os.path.exists(args.results_csv):
        results_df = pd.read_csv(args.results_csv)
        processed_checkpoints = set(zip(
            results_df.lang,
            results_df.model_variant,
            results_df.checkpoint_step
        ))
    else:
        results_df = pd.DataFrame()
        processed_checkpoints = set()

    print(f"Found {len(processed_checkpoints)} previously processed checkpoints.")

    new_results = []

    # Grid search over languages, models, and checkpoints
    for lang in LANGUAGES:
        eval_texts = get_eval_dataset(lang)
        if not eval_texts:
            continue

        model_variant_paths = get_model_variant_paths_for_lang(lang)
        for variant_path in tqdm(model_variant_paths, desc=f"Variants for {lang}", leave=False):
            variant_name = variant_path.name

            # Load metadata for the variant
            hparams = read_model_details(variant_path)

            checkpoint_dir = variant_path / "student"
            if not checkpoint_dir.exists():
                continue

            checkpoints = sorted(
                checkpoint_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split('-')[-1])
            )

            for ckpt_path in checkpoints:
                step = int(ckpt_path.name.split('-')[-1])

                if (lang, variant_name, step) in processed_checkpoints:
                    continue

                print(f"\nProcessing: [{lang}] {variant_name} - Step {step}")

                # Finalize hyperparameters, using heuristics as a fallback
                if any(k not in hparams for k in ["teacher_layer_idx", "student_num_layers", "attention_heads"]):
                    t_l, s_m, a_h = derive_heuristic_hparams(hparams.get("teacher_model_name"),
                                                             hparams.get("student_num_layers"))
                    hparams.setdefault("teacher_layer_idx", t_l)
                    hparams.setdefault("student_num_layers", s_m)
                    hparams.setdefault("attention_heads", a_h)

                hparams.setdefault("relations", {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0})  # Default relations
                hparams.setdefault("teacher_model_name", "FacebookAI/xlm-roberta-large")  # Ultimate fallback

                loss = calculate_distillation_loss(
                    student_checkpoint_path=str(ckpt_path),
                    teacher_model_name=hparams["teacher_model_name"],
                    hparams=hparams,
                    eval_texts=eval_texts,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                print(f"  > Result: Loss = {loss:.4f}")

                new_results.append({
                    "lang": lang,
                    "model_variant": variant_name,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_step": step,
                    "teacher_model": hparams["teacher_model_name"],
                    "student_hidden_size": hparams.get("student_hidden_size"),
                    "student_num_layers": hparams["student_num_layers"],
                    "attention_heads": hparams["attention_heads"],
                    "loss": loss,
                })
                processed_checkpoints.add((lang, variant_name, step))

    # Save new results to the CSV file
    if new_results:
        new_df = pd.DataFrame(new_results)
        final_df = pd.concat([results_df, new_df], ignore_index=True)
        final_df.to_csv(args.results_csv, index=False)
        print(f"\nAppended {len(new_results)} new results to '{args.results_csv}'.")

    print("\n✓ Evaluation run complete.")


if __name__ == "__main__":
    main()