#!/usr/bin/env bash
#ner_eval.sh
set -euo pipefail


PYBIN="/opt/conda/envs/default/bin/python"



$PYBIN -m pip install --upgrade transformers "accelerate>=0.28.0" evaluate scikit-learn python-dotenv "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" datasets seqeval numpyencoder


working_dir="./"

cd "$working_dir"
python_script="ner_bert.py"

model_path="FacebookAI/roberta-base"
dataset_name="wikiann/de"
model_path_sanitized=${model_path//\//-} # Replaces all '/' with '-'
output_dir="./results/german_ner_results_wikiann_de/${model_path_sanitized}"

first_arg=$1
second_arg=$2
third_arg=$3
if [ -n "$first_arg" ]; then
    dataset_name=$first_arg
fi
if [ -n "$second_arg" ]; then
    model_path=$second_arg
fi
if [ -n "$third_arg" ]; then
    output_dir=$third_arg
fi


cd "$working_dir"

$PYBIN -u "$python_script" \
    --model_name_or_path "$model_path" \
    --dataset_name "$dataset_name" \
    --output_dir "$output_dir" \
    --trust_remote_code True \


exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Success!"
else
    echo "Error with Exit-Code $exit_code" >&2
    exit $exit_code
fi
