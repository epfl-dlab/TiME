#!/bin/bash
# run_ud_eval.sh
# This script runs the Universal Dependencies (UD) evaluation pipeline for a fine-tuned model.
# It trains parsers, taggers, and lemmatizers on a specified UD treebank and evaluates them.
#
# Usage:
#   ./run_evaluation.sh [LANGUAGE_CODE] [PATH_TO_MODEL]
#
# Example:
#   ./run_evaluation.sh hunL "HPLT/hplt_bert_base_2_0_hun-Latn"
#

# --- Configuration ---

# --- System & Dependencies ---
# Ensure the script exits on any error and treats unset variables as an error.
set -eu
export TOKENIZERS_PARALLELISM=false
PYTHON_EXECUTABLE="/opt/conda/envs/default/bin/python"

# --- Model & Language ---
# Default values, can be overridden by command-line arguments.
DEFAULT_LANGUAGE="hunL"
DEFAULT_MODEL_PATH="HPLT/hplt_bert_base_2_0_hun-Latn"

# --- Paths & Directories ---
# Path to the root directory of the UD treebanks.
UD_TREEBANK_ROOT_DIR="/dlabscratch1/schulmei/minilmv2/evaluation/WikiAnn/hplt_eval/ud/ud_data/ud-treebanks-v2.15"
UD_MAPPING_VERSION="2_0" # Corresponds to the language_treebank_mapping_X_Y.json file.
BASE_OUTPUT_DIR="/dlabscratch1/schulmei/minilmv2/evaluation/WikiAnn/hplt_eval/ud/results3"

# --- Training Hyperparameters ---
EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=2e-5
DROPOUT=0.3
MIN_WORD_COUNT=3 # For vocabulary creation in the downstream task.

# --- End of Configuration ---


# --- Argument Parsing ---
# Use command-line arguments if provided, otherwise use defaults.
LANGUAGE=${1:-$DEFAULT_LANGUAGE}
MODEL_PATH=${2:-$DEFAULT_MODEL_PATH}


# --- Setup & Pre-flight Checks ---
echo "--- Starting Evaluation Pipeline ---"
echo "Python Interpreter:  $PYTHON_EXECUTABLE"
echo "Language Code:       $LANGUAGE"
echo "Model Path:          $MODEL_PATH"
echo "UD Treebank Path:    $UD_TREEBANK_ROOT_DIR"
echo "UD Mapping Version:  $UD_MAPPING_VERSION"

# Construct the full output directory path for this specific run.
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/ud_evaluation_results_v2_15_${LANGUAGE}"
echo "Output Directory:      $RUN_OUTPUT_DIR"
mkdir -p "${RUN_OUTPUT_DIR}/tmp" # For temporary CoNLL-U files
mkdir -p "${RUN_OUTPUT_DIR}/checkpoints"


# --- Dependency Installation ---
# This combines and deduplicates the original install commands.
# For production use, it's highly recommended to use a requirements.txt file.
echo "Installing/updating required Python packages..."
"$PYTHON_EXECUTABLE" -m pip install --upgrade \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    transformers \
    accelerate>=0.28.0 \
    datasets \
    evaluate \
    seqeval \
    conllu \
    tqdm \
    smart_open \
    scikit-learn \
    python-dotenv \
    numpyencoder \
    "ufal.chu_liu_edmonds>=0.9.1"

echo "Package installation complete."


# --- Execution ---
# Navigate to the script's working directory.
WORKING_DIR="/dlabscratch1/schulmei/minilmv2/evaluation/WikiAnn/hplt_eval/ud"
cd "$WORKING_DIR"

echo "Executing training and evaluation script..."
"$PYTHON_EXECUTABLE" train0.py \
    --language "${LANGUAGE}" \
    --custom_model_path "${MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --treebank_path "${UD_TREEBANK_ROOT_DIR}" \
    --version "${UD_MAPPING_VERSION}" \
    --results_path "${RUN_OUTPUT_DIR}/" \
    --checkpoints_path "${RUN_OUTPUT_DIR}/checkpoints/" \
    --min_count ${MIN_WORD_COUNT} \
    --lr "${LEARNING_RATE}" \
    --dropout ${DROPOUT}


# --- Completion ---
echo "--- Evaluation Finished ---"
echo "Results, logs, and model checkpoints are saved in: ${RUN_OUTPUT_DIR}"
echo "Key metrics (LAS, MLAS, BLEX) can be found in the .jsonl files in the results directory and in the console output from the script."