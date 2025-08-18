# train/distillation.py
import argparse
import ast
import json
import logging
import sys
import torch
import transformers
import datetime
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


from data_pipeline import  get_data
from minilmv2_ltg_fast import MiniLMLtgAdapted
from minilmv2_fast import MiniLMv2Fast as MiniLM
from minilmv2_debertav3_hybrid import DeBERTaToBERTDistiller

import os
from transformers import TrainerCallback, TrainerControl, TrainerState

def get_data_parser():

    parser = argparse.ArgumentParser(
        epilog="Parameters for specifying data paths and configurations"
    )
    parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="The name of the dataset to use (via the datasets library) if not loading local files."
    )

    parser.add_argument(
        "--stream_local_files",
        action="store_true",
        default=True,
        help="If loading local Arrow files (via train_config or local_arrow_glob_pattern), try to stream them."
    )
    parser.add_argument(
        "--pretokenized_dataset_path",
        type=str,
        default=None,
        help="Path to the pre-tokenized dataset. This should be a directory containing the pre-tokenized data."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=False,
        help="Configuration file for training dataset. Must be an array of {dir : directory, format: (txt|json|csv), columns : Comma separated list of column names (for csv) or field names (for json) containing the text. Content of the columns will be concatenated with a new line separator.} ",
    )
    parser.add_argument(
        "--val_config",
        type=str,
        required=False,
        help="Configuration file for validation dataset. Must be an array of {dir : directory, format: (txt|json|csv), columns : Comma separated list of column names (for csv) or field names (for json) containing the text. Content of the columns will be concatenated with a new line separator.} ",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Max Sequence Length considered for an input",
    )
    return parser


def get_model_parser():

    parser = argparse.ArgumentParser(
        epilog="Parameters for specifying data paths and configurations"
    )
    parser.add_argument(
        "--input_model_dir",
        type=str,
        required=True,
        help="The directory from which the pre-trained model is loaded.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=False,
        help="Checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=False,
        help="The directory from which the pre-trained model's tokenizer is loaded. If not provided, 'input_model_dir' will be used.",
    )
    parser.add_argument(
        "--student_hidden_size",
        type=int,
        required=True,
        help="Student model hidden size.",
    )
    parser.add_argument(
        "--student_num_layers",
        type=int,
        required=True,
        help="Student model hidden layer number.",
    )
    parser.add_argument(
        "--student_attention_heads",
        type=int,
        required=True,
        help="Student model attention head number.",
    )
    parser.add_argument(
        "--L", type=int, required=True, help="Teacher's layer to distill from."
    )
    parser.add_argument(
        "--num_relation_heads",
        type=int,
        required=True,
        help="Number of relation heads in MiniLM.",
    )
    parser.add_argument(
        "--minilm_relations",
        type=str,
        required=False,
        default="{(1, 1): 1, (2, 2): 1, (3, 3): 1}",
        help="Relations to use and their weights. The format is {(relation id1, relation id2): weight}. Relation ids are denoted as 1: Query, 2: Key, 3: Value",
    )
    return parser


def split_args_by_parser(args, parsers):

    parser_names = parsers.keys()
    parser_name_locs = sorted(
        map(
            lambda parser_name: (
                parser_name,
                args.index(parser_name) if parser_name in args else -1,
            ),
            parser_names,
        ),
        key=lambda x: x[1],
    )
    assert all(
        filter(lambda x: x[1] != -1, parser_name_locs)
    ), f"Required keys {parser_names} must be present"
    parsed_args = {}
    for idx, parser_name_loc in enumerate(parser_name_locs):
        parser_name, loc = parser_name_loc
        end_loc = (
            len(args)
            if idx == len(parser_name_locs) - 1
            else parser_name_locs[idx + 1][1]
        )
        parsed_args[parser_name] = parsers[parser_name].parse_args(
            args[loc + 1 : end_loc]
        )
    return parsed_args


logger = logging.getLogger(__name__)
# os.environ["WANDB_MODE"] = "offline" # Disable WandB online mode
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(console_formatter)
if not logger.hasHandlers():  # Avoid duplicate handlers if script is re-run in same session
    logger.addHandler(console_handler)




class SaveStudentCallback(TrainerCallback):
    """Every time the Trainer does a checkpoint, also dump out
       distiller.student.save_pretrained(...) so you get config + .bin"""

    def on_save(self,
                args,  # TrainingArguments
                state: TrainerState,
                control: TrainerControl,
                **kwargs
                ):
        # the Trainer has already made the folder “checkpoint-<step>”
        # make the student checkpoint dir
        student_dir = os.path.join(args.output_dir, "student")
        os.makedirs(student_dir, exist_ok=True)
        ckpt_dir = os.path.join(student_dir, f"checkpoint-{state.global_step}")
        # kwargs["model"] is your distiller → grab .student
        student = kwargs["model"].student
        student.save_pretrained(ckpt_dir, safe_serialization=False)  # False to avoid using safetensors
        # save the model config
        # (this is the same as student.config.save_pretrained(ckpt_dir))
        # but we can’t use the Trainer’s save_model() because it
        # doesn’t know about the student
        student.config.save_pretrained(ckpt_dir)
        # also save the tokenizer if you like:
        if kwargs.get("tokenizer") is not None:
            kwargs["tokenizer"].save_pretrained(ckpt_dir)
        else:
            try:
                kwargs["model"].tokenizer.save_pretrained(ckpt_dir)
            except Exception as e:
                logger.warning(f"Could not save tokenizer: {e}")
        # save the training arguments
        # save teacher tokenizer in the student checkpoint dir
        try:
            kwargs["model"].teacher.tokenizer.save_pretrained(ckpt_dir)
        except Exception as e:
            logger.warning(f"Could not save teacher tokenizer: {e}")

        try:
            main_output_dir_rng_state_file = os.path.join(args.output_dir, "rng_state.pth")
            specific_ckpt_dir_rng_state_file = os.path.join(args.output_dir, f"checkpoint-{state.global_step}",
                                                            "rng_state.pth")

            if os.path.exists(main_output_dir_rng_state_file):
                logger.info(f"Deleting problematic main RNG state file: {main_output_dir_rng_state_file}")
                try:
                    os.remove(main_output_dir_rng_state_file)
                    logger.info(f"Successfully deleted {main_output_dir_rng_state_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete main RNG state file {main_output_dir_rng_state_file}: {e}",
                                   exc_info=True)

            if os.path.exists(specific_ckpt_dir_rng_state_file):
                logger.info(
                    f"Deleting problematic checkpoint-specific RNG state file: {specific_ckpt_dir_rng_state_file}")
                try:
                    os.remove(specific_ckpt_dir_rng_state_file)
                    logger.info(f"Successfully deleted {specific_ckpt_dir_rng_state_file}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete checkpoint-specific RNG state file {specific_ckpt_dir_rng_state_file}: {e}",
                        exc_info=True)
        except Exception as e:
            logger.warning(f"Failed to delete RNG state file: {e}")

        return control  # Ensure the callback returns control


def _get_args():
    data_parser = get_data_parser()
    model_parser = get_model_parser()
    training_parser = HfArgumentParser((TrainingArguments))

    parsers = {
        "data_params": data_parser,
        "model_params": model_parser,
        "training_params": training_parser,
    }
    raw_cli_args = sys.argv[1:]
    params = split_args_by_parser(raw_cli_args, parsers)

    if hasattr(params["training_params"], "label_names") and \
            params["training_params"].label_names == ["start_positions", "end_positions"]:
        logger.info("Removing default QA-style label_names for MiniLM distillation.")
        delattr(params["training_params"], "label_names")


    params["training_params"].dataloader_drop_last = True

    if "accelerator_config" in params["training_params"] and isinstance(params["training_params"].accelerator_config,
                                                                        dict):
        logger.info(f"Keeping accelerator_config: {params['training_params'].accelerator_config}")
    elif "accelerator_config" in params["training_params"]:
        delattr(params["training_params"], "accelerator_config")

    hf_training_args_dict = vars(params["training_params"])


    if 'local_rank' not in hf_training_args_dict or hf_training_args_dict['local_rank'] == -1:
        hf_training_args_dict['local_rank'] = _get_rank() if _is_distributed() else -1
        logger.info(
            f"Explicitly setting local_rank for TrainingArguments constructor to: {hf_training_args_dict['local_rank']}")

    hf_training_args = TrainingArguments(**hf_training_args_dict)
    data_args = params["data_params"]
    model_args = params["model_params"]

    if hasattr(model_args, "minilm_relations"):
        if model_args.minilm_relations and isinstance(model_args.minilm_relations, str):
            try:
                model_args.minilm_relations = ast.literal_eval(model_args.minilm_relations)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse minilm_relations string: '{model_args.minilm_relations}'. Error: {e}")
    return data_args, model_args, hf_training_args


def _is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank():

    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None:
        return int(local_rank_env)

    rank_env = os.environ.get("RANK", "0")
    return int(rank_env)


def main():
    try:
        data_args, model_args, hf_training_args = _get_args()

        logger.info(f"Python sys.version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"Transformers path: {transformers.__file__}")

        logger.info(f"--- Effective TrainingArguments ---")
        logger.info(f"  Process Index (Global Rank): {hf_training_args.process_index}")
        logger.info(f"  Local Process Index (Local Rank on Node): {hf_training_args.local_process_index}")
        logger.info(f"  Device: {hf_training_args.device}")
        logger.info(f"  World Size: {hf_training_args.world_size}")
        logger.info(f"  N_GPU (total GPUs for this process): {hf_training_args.n_gpu}")
        logger.info(f"  TrainingArguments.local_rank (often device index): {hf_training_args.local_rank}")
        logger.info(f"  Output Directory: {hf_training_args.output_dir}")
        logger.info(f"-----------------------------------")

        # --- START: Data Configuration Logic ---
        path_to_train_config_json = getattr(data_args, 'train_config', None)

        data_args.is_local_arrow_config = False
        data_args.local_arrow_files_config = None

        logger.debug(f"Initial data_args.train_config path: {path_to_train_config_json}")

        if path_to_train_config_json and os.path.exists(path_to_train_config_json):
            try:
                with open(path_to_train_config_json, 'r', encoding='utf-8') as f:
                    loaded_train_config = json.load(f)
                logger.info(f"Successfully loaded train_config from: {path_to_train_config_json}")

                data_args.text_column_name = loaded_train_config.get("text_column_name",
                                                                     getattr(data_args, "text_column_name", "text"))

                if loaded_train_config.get("format") == "arrow" and loaded_train_config.get("urls"):
                    data_args.is_local_arrow_config = True
                    data_args.local_arrow_files_config = loaded_train_config["urls"]
                    logger.info(
                        f"Configured for local Arrow files: {data_args.local_arrow_files_config}. Stream if --stream_local_files.")
                    data_args.dataset_name = None
                    data_args.dataset_config_name = None
                else:
                    data_args.is_local_arrow_config = False
                    data_args.dataset_name = loaded_train_config.get("dataset_name", data_args.dataset_name)
                    data_args.dataset_config_name = loaded_train_config.get("dataset_config_name",
                                                                            data_args.dataset_config_name)
                    logger.info(
                        f"train_config specified non-Arrow or no URLs. Using Hub params: name='{data_args.dataset_name}', config='{data_args.dataset_config_name}'")

            except Exception as e:
                logger.error(
                    f"Failed to load/parse train_config '{path_to_train_config_json}': {e}. Using CLI/defaults.",
                    exc_info=True)
                data_args.is_local_arrow_config = False
        else:
            logger.info(f"train_config '{path_to_train_config_json}' not found/provided. Using CLI/defaults.")
            data_args.is_local_arrow_config = False

        data_args.text_column_name = getattr(data_args, 'text_column_name', "text")
        data_args.shuffle_buffer_size = getattr(data_args, 'shuffle_buffer_size', 10000)
        data_args.stream_take_size = getattr(data_args, 'stream_take_size', 0)
        data_args.map_batch_size = getattr(data_args, 'map_batch_size', 1000)

        if getattr(data_args, 'streaming', False) and \
                not getattr(data_args, 'is_local_arrow_config', False) and \
                getattr(data_args, 'dataset_name', None):
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "True"
            logger.info("HF_HUB_ENABLE_HF_TRANSFER=True for Hub streaming.")

        path_to_val_config_json = getattr(data_args, 'val_config', None)
        if path_to_val_config_json and os.path.exists(path_to_val_config_json):
            try:
                with open(path_to_val_config_json, 'r', encoding='utf-8') as f:
                    data_args.val_config_content = json.load(f)
            except Exception as e:
                logger.error(f"Could not load/parse val_config '{path_to_val_config_json}': {e}")
                data_args.val_config_content = None
        else:
            data_args.val_config_content = None


        set_seed(hf_training_args.seed)

        if hf_training_args.process_index == 0:
            logger.info(f"Process index {hf_training_args.process_index}: Creating output directory and log file.")
            if not os.path.exists(hf_training_args.output_dir):
                os.makedirs(hf_training_args.output_dir)
            log_file_path = os.path.join(hf_training_args.output_dir, "training.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            if file_handler not in logger.handlers:
                logger.addHandler(file_handler)
        else:
            logger.info(f"Process index {hf_training_args.process_index}: Not creating output directory or log file.")

        input_model_dir = model_args.input_model_dir
        checkpoint_to_resume_from = hf_training_args.resume_from_checkpoint

        if hasattr(model_args, "student_architecture") and model_args.student_architecture:
            logger.info(f"Student architecture specified: {model_args.student_architecture}")
            student_config = AutoConfig.from_pretrained(model_args.student_architecture, trust_remote_code=True)
        else:
            logger.info("No student architecture specified. Using default Bert Style Architecture (bert-base-uncased).")
            student_config = AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)

        tokenizer_dir = model_args.tokenizer_dir if model_args.tokenizer_dir else input_model_dir

        logger.info(f"Loading Teacher from: {input_model_dir}")
        teacher = AutoModel.from_pretrained(input_model_dir, trust_remote_code=True)
        logger.info("Teacher loaded.")

        logger.info("Initializing student model...")
        temp_tokenizer_for_student_config = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        # student_config = AutoConfig.from_pretrained(input_model_dir)
        student_config.hidden_size = model_args.student_hidden_size
        student_config.num_hidden_layers = model_args.student_num_layers
        student_config.num_attention_heads = model_args.student_attention_heads
        student_config.intermediate_size = model_args.student_hidden_size * 4
        student_config.vocab_size = max(temp_tokenizer_for_student_config.vocab_size,
                                        len(temp_tokenizer_for_student_config))
        if student_config.vocab_size > teacher.config.vocab_size:
            logger.warning(
                f"Student vocab size ({student_config.vocab_size}) is larger than teacher vocab size ({teacher.config.vocab_size}). Resizing teacher embeddings.")
            teacher.resize_token_embeddings(student_config.vocab_size)
        del temp_tokenizer_for_student_config

        logger.info(f"Student Configuration: {student_config}")
        student = AutoModel.from_config(student_config, trust_remote_code=True)
        logger.info("Student model initialized.")

        logger.info("Initializing MiniLMv2 distiller...")
        minilm_relations_val = model_args.minilm_relations
        if not isinstance(minilm_relations_val, dict):
            logger.warning(f"minilm_relations type is {type(minilm_relations_val)}, not dict. Attempting eval.")
            minilm_relations_val = ast.literal_eval(str(minilm_relations_val))
        logger.info(f"MiniLM relations for distiller: {minilm_relations_val}")

        if 'deberta-v3' in input_model_dir.lower():
            logger.info("Teacher is a DeBERTa-v3 model.")
            distiller = DeBERTaToBERTDistiller(
                teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
                relations=minilm_relations_val, A_r=model_args.num_relation_heads
            )
        elif 'ltg' in str(teacher.config.architectures[0]).lower():
            logger.info("Teacher is a LTG-BERT model.")
            distiller = MiniLMLtgAdapted(
                teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
                relations=minilm_relations_val, A_r=model_args.num_relation_heads
            )
        elif 'bert' in str(teacher.config.architectures[0]).lower():
            logger.info("Teacher is a BERT model.")
            distiller = MiniLM(
                teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
                relations=minilm_relations_val, A_r=model_args.num_relation_heads
            )
        elif 'roberta' in str(teacher.config.architectures[0]).lower():
            logger.info("Teacher is a RoBERTa model.")
            distiller = MiniLM(
                teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
                relations=minilm_relations_val, A_r=model_args.num_relation_heads
            )
        elif 'electra' in str(teacher.config.architectures[0]).lower():
            logger.info("Teacher is an ELECTRA model.")
            distiller = MiniLM(
                teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
                relations=minilm_relations_val, A_r=model_args.num_relation_heads
            )
        else:
            logger.warning(
                f"Teacher model architecture '{teacher.config.architectures[0]}' is not recognized. Proceeding with caution. Standard BERT model is assumed.")
            distiller = MiniLM(
                teacher=teacher, student=student, L=model_args.L, M=model_args.student_num_layers,
                relations=minilm_relations_val, A_r=model_args.num_relation_heads
            )
        logger.info(
            f"Teacher model architecture: {teacher.config.architectures[0] if hasattr(teacher.config, 'architectures') and teacher.config.architectures else 'Unknown or deberta-v3'}")

        logger.info("Loading and tokenizing datasets via data_utils1...")

        class TokenizerPathConfig:
            def __init__(self, path): self.tokenizer_name_or_path = path

        tokenizer_cfg_obj = TokenizerPathConfig(tokenizer_dir)


        logger.info("Loading and tokenizing datasets on-the-fly...")
        ##train_dataset, val_dataset, tokenizer_from_data_util = get_tokenized_datasets( # Deine alte Funktion
        #  data_args, tokenizer_cfg_obj, hf_training_args
        # )
        train_dataset, val_dataset, tokenizer_from_data_util = get_data(
            data_args, tokenizer_cfg_obj, hf_training_args
        )
        if _is_distributed() and os.environ.get("TOKENIZERS_PARALLELISM", "true").lower() == "true":
            logger.info("Disabling TOKENIZERS_PARALLELISM for distributed training.")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if hf_training_args.process_index == 0:
            transformers.utils.logging.set_verbosity_info()

        logger.info(f"Final Data Arguments: {data_args}")
        logger.info(f"Final Model Arguments: {model_args}")
        logger.info(
            f"Final Training Arguments (condensed): output_dir={hf_training_args.output_dir}, per_device_train_batch_size={hf_training_args.per_device_train_batch_size}, max_steps={hf_training_args.max_steps}, seed={hf_training_args.seed}")
        logger.info(f"Per-GPU Batch Size: {hf_training_args.per_device_train_batch_size}")

        logger.info("Initializing Trainer...")
        hf_training_args.remove_unused_columns = False
        trainer = Trainer(
            model=distiller,
            args=hf_training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer_from_data_util,
            data_collator=transformers.DataCollatorWithPadding(
                tokenizer_from_data_util, padding="longest"
            ),
            callbacks=[SaveStudentCallback()],
        )

        logger.info(
            f"Starting training. Resume from checkpoint: {checkpoint_to_resume_from if checkpoint_to_resume_from else 'No specific path (Trainer will check output_dir if True)'}")
        trainer.train(resume_from_checkpoint=checkpoint_to_resume_from)

        logger.info("---- TRAINING COMPLETED -----")

        if hf_training_args.save_strategy != "no":
            if hf_training_args.process_index == 0:
                logger.info(
                    f"Process index {hf_training_args.process_index}: Saving final model to {hf_training_args.output_dir} (save_strategy is '{hf_training_args.save_strategy}').")
                trainer.save_model()
                trainer.save_state()
            else:
                logger.info(f"Process index {hf_training_args.process_index}: Not saving model (not main process).")
        else:
            logger.info(f"Not saving model as save_strategy is '{hf_training_args.save_strategy}'.")


    except Exception as e:
        errors_dir = "errors"
        if not os.path.exists(errors_dir): os.makedirs(errors_dir)
        import traceback
        filename = os.path.join(errors_dir,
                                "error_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-") + ".txt")
        with open(filename, "w") as f:
            f.write(f"Error in {__file__} (PID: {os.getpid()}):\n{str(e)}\n\n")
            f.write(traceback.format_exc())
        logger.error(f"An error occurred: {e}. Full traceback saved to {filename}", exc_info=True)
        raise


if __name__ == "__main__":
    main()