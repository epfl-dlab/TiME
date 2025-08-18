# trqin/data_pipeline.py
import logging
import os
import glob
from dotenv import load_dotenv
from typing import Callable, Tuple, Optional

from datasets import (
    load_dataset,
    Dataset,
    IterableDataset,
    DatasetDict,
    IterableDatasetDict,
    Features,
    Value,
    disable_caching,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- Setup ---
load_dotenv()
logger = logging.getLogger(__name__)

# Configure Hugging Face Hub access token
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    logger.info("Using HF_TOKEN from environment for Hub dataset access.")
else:
    logger.warning("HF_TOKEN not set. Will attempt cached login or public access for Hub datasets.")


# --- Private Helper Functions ---

def _get_tokenizer(tokenizer_name_or_path: str, max_seq_len: int) -> PreTrainedTokenizer:
    """Initializes and returns a PreTrainedTokenizer."""
    logger.info(f"Loading tokenizer: {tokenizer_name_or_path}")
    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        max_length=max_seq_len
    )


def _create_tokenize_function(tokenizer: PreTrainedTokenizer, text_column: str, max_seq_len: int) -> Callable:
    """Creates a batched tokenization function for use with .map()."""

    def tokenize_function(examples: dict) -> dict:
        """Applies tokenization to a batch of examples."""
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,  # Data collator will handle padding.
            max_length=max_seq_len,
            return_token_type_ids=True,
            add_special_tokens=True
        )

    return tokenize_function


def _load_local_arrow_dataset(
        file_patterns: list[str],
        stream_local: bool,
        text_column: str
) -> Optional[Tuple[Dataset | IterableDataset, bool]]:
    """Loads a dataset from local Arrow files specified by glob patterns."""
    logger.info(f"Local Arrow data source configured. Attempting to load patterns: {file_patterns}")

    concrete_files = []
    for pattern in file_patterns:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            logger.warning(f"No files matched local glob pattern: {pattern}")
        concrete_files.extend(expanded)

    if not concrete_files:
        logger.error(f"No local Arrow files found for patterns: {file_patterns}.")
        return None, False

    logger.info(f"Found {len(concrete_files)} local Arrow files. First 3: {concrete_files[:3]}")

    try:
        arrow_data_files = {"train": concrete_files}
        # Define expected features to ensure consistency.
        expected_features = Features({text_column: Value('string')})

        loaded_data = load_dataset(
            "arrow",
            data_files=arrow_data_files,
            streaming=stream_local,
            features=expected_features
        )

        raw_train_dataset = loaded_data.get("train")
        if raw_train_dataset is None:
            raise ValueError("Could not extract 'train' split from local Arrow load.")

        dataset_source_info = f"Local Arrow ({'Streamed' if stream_local else 'Map-Style'})"
        logger.info(f"Successfully loaded dataset from {dataset_source_info}")
        if not stream_local:
            logger.info(f"Local map-style dataset has {len(raw_train_dataset):,} rows.")

        return raw_train_dataset, stream_local

    except Exception as e:
        logger.error(f"Failed to load local Arrow dataset from {file_patterns}: {e}", exc_info=True)
        return None, False


def _preprocess_streamed_dataset(
        dataset: IterableDataset,
        shuffle_buffer: int,
        take_size: int,
        seed: Optional[int]
) -> IterableDataset:
    """Applies shuffling and take operations to a streamed dataset."""
    if take_size > 0:
        logger.info(f"Taking first {take_size} samples from the streamed dataset.")
        dataset = dataset.take(take_size)
    if shuffle_buffer > 0:
        logger.info(f"Shuffling streamed dataset with buffer size: {shuffle_buffer}")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return dataset


# --- Public API Functions (Behavior Preserved) ---

def get_tokenized_datasets(data_args, tokenizer_config, training_args=None):
    """
    Loads, tokenizes, and prepares datasets for training and evaluation.

    This is the primary data loading function. It handles loading from local Arrow files
    or the Hugging Face Hub, supports streaming and map-style processing, and
    applies tokenization. The exact behavior is controlled by the `data_args` object.

    Args:
        data_args: A configuration object containing data-related arguments
                   (e.g., dataset_name, is_local_arrow_config, max_seq_len).
        tokenizer_config: A configuration object for the tokenizer
                          (e.g., tokenizer_name_or_path).
        training_args: (Optional) A configuration object for training arguments,
                       used here to get the random seed for shuffling.

    Returns:
        A tuple containing:
        - train_dataset (Dataset or IterableDataset): The tokenized training dataset.
        - eval_dataset (Dataset, IterableDataset, or None): The tokenized eval dataset.
        - tokenizer (PreTrainedTokenizer): The tokenizer instance.
    """
    tokenizer = _get_tokenizer(
        tokenizer_config.tokenizer_name_or_path,
        data_args.max_seq_len
    )
    text_column = data_args.text_column_name or 'text'
    logger.info(f"Using text column: '{text_column}'")

    # 1. Load Raw Datasets
    raw_train_dataset, raw_eval_dataset = None, None
    is_streaming = False
    dataset_source_info = "N/A"

    if data_args.is_local_arrow_config and data_args.local_arrow_files_config:
        raw_train_dataset, is_streaming = _load_local_arrow_dataset(
            data_args.local_arrow_files_config,
            data_args.stream_local_files,
            text_column
        )
        dataset_source_info = f"Local Arrow files (streaming={is_streaming})"

    if raw_train_dataset is None:
        if not data_args.dataset_name:
            raise ValueError("No local data loaded and no Hub `dataset_name` provided.")

        is_streaming = data_args.streaming
        dataset_source_info = f"Hub: {data_args.dataset_name} (config: {data_args.dataset_config_name}, streaming={is_streaming})"
        logger.info(f"Loading dataset from {dataset_source_info}")

        raw_train_dataset = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name,
            split="train", streaming=is_streaming, token=HF_TOKEN
        )
        if data_args.do_eval:
            try:
                raw_eval_dataset = load_dataset(
                    data_args.dataset_name, data_args.dataset_config_name,
                    split="validation", streaming=is_streaming, token=HF_TOKEN
                )
                logger.info("Loaded 'validation' split from Hub.")
            except Exception as e:
                logger.warning(f"Could not load 'validation' split: {e}")

    if raw_train_dataset is None:
        raise ValueError("Failed to load raw training dataset. Check configurations.")

    # 2. Pre-process Streamed Datasets (Shuffle/Take)
    seed = training_args.seed if training_args else None
    if isinstance(raw_train_dataset, IterableDataset):
        raw_train_dataset = _preprocess_streamed_dataset(
            raw_train_dataset,
            data_args.shuffle_buffer_size,
            data_args.stream_take_size,
            seed
        )
    if raw_eval_dataset and isinstance(raw_eval_dataset, IterableDataset):
        raw_eval_dataset = _preprocess_streamed_dataset(
            raw_eval_dataset, 0, data_args.stream_take_size_eval, seed  # No shuffle for eval
        )

    # 3. Tokenize Datasets
    tokenize_fn = _create_tokenize_function(tokenizer, text_column, data_args.max_seq_len)

    # The .map() function with `remove_columns` will automatically drop all original columns
    # and keep only the new columns generated by `tokenize_fn`.
    columns_to_remove = raw_train_dataset.column_names

    map_kwargs = {
        "function": tokenize_fn,
        "batched": True,
        "batch_size": data_args.map_batch_size,
        "remove_columns": columns_to_remove
    }

    logger.info(f"Tokenizing datasets from {dataset_source_info}...")

    if is_streaming:
        train_dataset = raw_train_dataset.map(**map_kwargs)
        eval_dataset = raw_eval_dataset.map(**map_kwargs) if raw_eval_dataset else None
    else:
        num_proc = data_args.preprocessing_num_workers or max(1, os.cpu_count() // 2)
        map_kwargs.update({
            "num_proc": num_proc,
            "load_from_cache_file": not data_args.overwrite_cache
        })
        train_dataset = raw_train_dataset.map(desc="Tokenizing training set", **map_kwargs)
        eval_dataset = raw_eval_dataset.map(desc="Tokenizing validation set",
                                            **map_kwargs) if raw_eval_dataset else None

    logger.info("Dataset tokenization complete.")
    return train_dataset, eval_dataset, tokenizer


# --- Legacy Functions (Refactored for consistency, behavior preserved) ---

def _get_stream_data_legacy(ds_name, ds_config_name, tokenizer, max_seq_len):
    """
    Legacy function to load a dataset from the Hub with specific hardcoded logic.
    Refactored to use modern helpers but preserves original filtering and shuffling.
    """
    disable_caching()
    dataset = load_dataset(ds_name, ds_config_name, split="train", streaming=True)

    # Legacy filtering logic
    def is_not_empty_or_whitespace(example):
        text_content = example.get("text")
        return isinstance(text_content, str) and text_content.strip()

    dataset_filtered = dataset.filter(is_not_empty_or_whitespace)

    # Legacy tokenization logic
    tokenize_fn = _create_tokenize_function(tokenizer, 'text', max_seq_len)

    tokenized_ds = dataset_filtered.map(
        tokenize_fn,
        batched=True,
        remove_columns=['text', 'timestamp', 'url', 'source']
    )

    # Legacy shuffling logic
    BUFFER_SIZE = 10_000
    shuffled_ds = tokenized_ds.shuffle(buffer_size=BUFFER_SIZE, seed=42)
    return shuffled_ds.with_format("torch")


def get_data(data_args, tokenizer_config, training_args=None):
    """
    A legacy data loading function with specific, hardcoded logic for certain
    dataset configs. Its use is discouraged in favor of `get_tokenized_datasets`.
    The original behavior is preserved.
    """
    max_seq_len = data_args.max_seq_len or 512
    tokenizer = _get_tokenizer(tokenizer_config.tokenizer_name_or_path, max_seq_len)

    dataset_name = data_args.dataset_name
    dataset_config_name = data_args.dataset_config_name

    # This branching is brittle and specific to the original script's logic.
    if "en" in dataset_config_name or "fr" in dataset_config_name:
        dataset = _get_stream_data_legacy(dataset_name, dataset_config_name, tokenizer, max_seq_len)
        return dataset, None, tokenizer
    else:
        dataset = load_dataset(dataset_name, dataset_config_name, split="train")
        tokenize_fn = _create_tokenize_function(tokenizer, 'text', max_seq_len)
        # with_transform applies tokenization on-the-fly, which is efficient.
        tokenized_dataset = dataset.with_transform(tokenize_fn)
        return tokenized_dataset, None, tokenizer


def get_data2(data_args, tokenizer_config, training_args=None):
    """
    Another legacy/alternative data loading function.
    Its use is discouraged in favor of `get_tokenized_datasets`.
    The original behavior is preserved.
    """
    max_seq_len = data_args.max_seq_len or 512
    text_column = data_args.text_column_name or 'text'
    tokenizer = _get_tokenizer(tokenizer_config.tokenizer_name_or_path, max_seq_len)

    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split="train"
    )

    tokenize_fn = _create_tokenize_function(tokenizer, text_column, max_seq_len)
    columns_to_remove = dataset.column_names

    processed_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=data_args.map_batch_size,
        remove_columns=columns_to_remove
    )

    return processed_dataset, None, tokenizer