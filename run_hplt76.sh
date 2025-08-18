#!/bin/bash
export TOKENIZERS_PARALLELISM=false
# --- Configurations ---

SEED=21
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPU_COUNT: $GPU_COUNT"
FULL_BATCH_SIZE=256 # Desired total batch size across all GPUs minilmv2 uses 256

# Ensure batch size is evenly divisible by GPU_COUNT if GPU_COUNT > 0
if [ "$GPU_COUNT" -gt 0 ] && (( FULL_BATCH_SIZE % GPU_COUNT != 0 )); then
    echo "Warning: Adjusting FULL_BATCH_SIZE for even division by GPU_COUNT ($GPU_COUNT)"
    # Calculate batch size per GPU, then recalculate full batch size
    TEMP_PER_GPU_BATCH_SIZE=$((FULL_BATCH_SIZE / GPU_COUNT)) # Integer division
    FULL_BATCH_SIZE=$((GPU_COUNT * TEMP_PER_GPU_BATCH_SIZE))
    echo "Adjusted FULL_BATCH_SIZE to $FULL_BATCH_SIZE"
fi

# Calculate per_gpu_batch_size, ensure it's at least 1 if GPU_COUNT > 0
if [ "$GPU_COUNT" -gt 0 ]; then
    PER_GPU_BATCH_SIZE=$((FULL_BATCH_SIZE / GPU_COUNT))
    if [ "$PER_GPU_BATCH_SIZE" -lt 1 ]; then
        echo "Error: Calculated PER_GPU_BATCH_SIZE is less than 1. FULL_BATCH_SIZE ($FULL_BATCH_SIZE) might be too small for GPU_COUNT ($GPU_COUNT)."
        exit 1
    fi
else
    # Fallback for CPU or if GPU_COUNT is 0 for some reason
    PER_GPU_BATCH_SIZE=$FULL_BATCH_SIZE
    echo "Warning: GPU_COUNT is 0 or not detected. Running with PER_GPU_BATCH_SIZE = FULL_BATCH_SIZE = $PER_GPU_BATCH_SIZE (assuming CPU)."
fi


# Python Interpreter
PYBIN="/opt/conda/envs/default/bin/python" # Make sure this path is correct
echo "Using Python interpreter: $PYBIN"

# Dependencies (optional: run only once or check if already installed)
echo "Installing/updating packages..."

$PYBIN -m pip install --upgrade transformers "accelerate>=0.28.0" evaluate scikit-learn python-dotenv "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "evaluate[quality]" datasets

# --- Student Architecture ---
STUDENT_HIDDEN_SIZE=768 #768
STUDENT_NUM_LAYERS=6 #6
STUDENT_ATTENTION_HEADS=12

# --- Teacher-specific Parameters ---
TEACHER_MODEL_NAME="HPLT/hplt_bert_base_2_0_eng-Latn"
TEACHER_DISTILL_LAYER=12 # Layer of the teacher to distill from (For 

# --- Relation Heads ---
NUM_RELATION_HEADS=48 #64 for large models, 48 for base models

# --- Checkpoint and Output Directories ---

DATASET_NAME="uonlp/CulturaX"
DATASET_SUBSET="en"
model_folder=${DATASET_NAME//\//_}_${DATASET_SUBSET}
CLEANED_TEACHER_MODEL_NAME_FOR_PATH="${TEACHER_MODEL_NAME//\//_}" # Replaces all '/' with '_'
OUTPUT_BASE_DIR="./models/${model_folder}" # Base directory for models
CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/minilm-H${STUDENT_HIDDEN_SIZE}-L${STUDENT_NUM_LAYERS}-${DATASET_SUBSET}-${CLEANED_TEACHER_MODEL_NAME_FOR_PATH}"

# Create the main output/checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Logic for resuming from checkpoint
# This argument will be passed to TrainingArguments' resume_from_checkpoint
RESUME_FROM_CHECKPOINT_ARG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Found existing top-level output directory: $CHECKPOINT_DIR"

    LATEST_HF_CHECKPOINT=$(ls -td $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | head -n1)
    if [ ! -z "$LATEST_HF_CHECKPOINT" ] && [ -d "$LATEST_HF_CHECKPOINT" ]; then
        echo "Found latest Hugging Face Trainer checkpoint: $LATEST_HF_CHECKPOINT"
        RESUME_FROM_CHECKPOINT_ARG="--resume_from_checkpoint $LATEST_HF_CHECKPOINT"
    else
        echo "No specific Hugging Face Trainer checkpoints (checkpoint-xxxxx) found in $CHECKPOINT_DIR."
        # If output_dir exists, Trainer might try to resume if resume_from_checkpoint=True is passed.
        # To start fresh if no specific checkpoint is found, even if output_dir exists,
        # you might need to ensure resume_from_checkpoint is not True or a path.
        # Or, if you want to resume from the base output_dir if it has model files:
        if [ -f "$CHECKPOINT_DIR/pytorch_model.bin" ] || [ -f "$CHECKPOINT_DIR/model.safetensors" ]; then
             echo "Found model files in $CHECKPOINT_DIR. Consider resuming by passing this directory.
             RESUME_FROM_CHECKPOINT_ARG="--resume_from_checkpoint True" # Option 2: let Trainer try to find the latest in output_dir
        else
             echo "No model files for resume found in $CHECKPOINT_DIR. Starting fresh."
        fi
    fi
else
    echo "No top-level output directory found ($CHECKPOINT_DIR), starting fresh training."
fi



ARGS="data_params \
      --max_seq_len 512 \
      --stream_local_files \
      --dataset_name ${DATASET_NAME} \
      --dataset_config_name ${DATASET_SUBSET} \
    training_params \
      --per_device_train_batch_size ${PER_GPU_BATCH_SIZE} \
      --learning_rate 6e-4 \
      --adam_epsilon 1e-6 \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --weight_decay 0.01 \
      --max_steps 200000 \
      --save_strategy steps \
      --save_steps 10000 \
      --logging_strategy steps \
      --logging_steps 100 \
      --warmup_steps 4000 \
      --gradient_accumulation_steps 1 \
      --bf16 true \
      --dataloader_drop_last true \
      --max_grad_norm 1.0 \
      --ddp_find_unused_parameters true \
      --output_dir ${CHECKPOINT_DIR} \
      --seed=${SEED} \
      --torch_compile False \
      --dataloader_num_workers 16 \
      ${RESUME_FROM_CHECKPOINT_ARG} \
    model_params \
     --input_model_dir ${TEACHER_MODEL_NAME} \
     --student_hidden_size ${STUDENT_HIDDEN_SIZE} \
     --student_num_layers ${STUDENT_NUM_LAYERS} \
     --student_attention_heads ${STUDENT_ATTENTION_HEADS} \
     --L ${TEACHER_DISTILL_LAYER} \
     --num_relation_heads ${NUM_RELATION_HEADS} \
     --minilm_relations \"{(1,1):1,(2,2):1,(3,3):1}\" \
"


if [[ "$GPU_COUNT" -gt 1 ]]; then

  CMD="torchrun --nproc_per_node=${GPU_COUNT} -m train.distillation -- "
else
  CMD="$PYBIN -m train.distillation -- " # For single GPU or CPU
fi

echo "Executing command: $CMD $ARGS"
$CMD $ARGS

# This will only execute if the Python script finishes successfully (due to `set -e`)
echo "Saving model details to $CHECKPOINT_DIR/model_details.txt"
echo "Teacher_Model: $TEACHER_MODEL_NAME" > "$CHECKPOINT_DIR/model_details.txt"
echo "Student_Hidden_Size: $STUDENT_HIDDEN_SIZE" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Student_Num_Layers: $STUDENT_NUM_LAYERS" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Student_Attention_Heads: $STUDENT_ATTENTION_HEADS" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Teacher_Distillation_Layer (L): $TEACHER_DISTILL_LAYER" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Num_Relation_Heads: $NUM_RELATION_HEADS" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Minilm_Relations: {(1,1):1,(2,2):1,(3,3):1}" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Output Directory (contains checkpoints): $CHECKPOINT_DIR" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Seed: $SEED" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Full Batch Size (across all GPUs): $FULL_BATCH_SIZE" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Per GPU Batch Size: $PER_GPU_BATCH_SIZE" >> "$CHECKPOINT_DIR/model_details.txt"
echo "GPU Count: $GPU_COUNT" >> "$CHECKPOINT_DIR/model_details.txt"
echo "Final Command Args passed to Python script: $ARGS" >> "$CHECKPOINT_DIR/model_details.txt"
