# Tiny Language Models for NLP Pipelines

This repository contains the official code and resources for the paper **"Tiny Language Models for NLP Pipelines"**. Our work demonstrates how to train small, efficient, and high-performing monolingual language models for common NLP tasks using knowledge distillation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
  - [1. Distillation (Training)](#1-distillation-training)
  - [2. Finding the Best Checkpoint](#2-finding-the-best-checkpoint)
  - [3. Downstream Task Evaluation (UD & NER)](#3-downstream-task-evaluation-ud--ner)
- [Citation](#citation)

## Overview

Large language models, while powerful, are often too slow and computationally expensive for real-time applications or large-scale data processing. This project addresses this gap by providing a complete pipeline to distill large teacher models (like XLM-RoBERTa and HPLT) into compact, purpose-built "tiny" student models.

The key components of this repository are:
- **Distillation Training:** Scripts to perform knowledge distillation using the MiniLMv2 methodology.
- **Checkpoint Selection:** A robust script to evaluate all saved checkpoints against a validation set to find the optimal one.
- **Downstream Evaluation:** A comprehensive evaluation suite for core NLP tasks, including Part-of-Speech (POS) tagging, lemmatization, dependency parsing (LAS), and Named Entity Recognition (NER).

## Project Structure

The repository is organized into two main parts: `train/` for model distillation and `evaluation/` for assessing model performance.

```
tiny-language-models/
├── .venv/                   # Python virtual environment
├── evaluation/
│   ├── find-best-checkpoint/
│   │   └── find-best-checkpoint.py  # Script to find the best model checkpoint
│   └── NLP-eval/
│       ├── ner/                     # Scripts for NER evaluation
│       │   ├── ner_eval.sh
│       │   └── ...
│       └── ud/                      # Scripts for UD tasks (POS, Lemma, LAS)
│           ├── run_ud_eval.sh
│           └── ...
├── train/
│   ├── data_pipeline.py             # Data loading and preprocessing utilities
│   ├── distillation.py            # Main script to run the distillation process
│   ├── minilmv2_fast.py           # Core MiniLMv2 distillation logic
│   ├── minilmv2_ltg_fast.py       # Adapter for HPLT (LTG-BERT) teachers
│   └── ...
├── requirements.txt         # Project dependencies
├── run_hplt76.sh            # Example script to train a model with an HPLT teacher
├── run_xlmr76.sh            # Example script to train a model with an XLM-R teacher
└── README.md                # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/tiny-language-models.git
    cd tiny-language-models
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The project's dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Environment Variables:**
    If you need to access private models from the Hugging Face Hub, create a `.env` file in the root directory and add your token:
    ```
    HF_TOKEN=your_hugging_face_token_here
    ```

## How to Run

The project workflow consists of three main stages: training the student models, finding the best checkpoint from the training run, and finally, evaluating that checkpoint on downstream NLP tasks.

### 1. Distillation (Training)

The model distillation process is orchestrated by shell scripts that call `train/distillation.py`. We provide two example scripts: `run_hplt76.sh` and `run_xlmr76.sh`.

-   **To train a model using an HPLT model as the teacher:**
    -   Customize the parameters (e.g., `TEACHER_MODEL_NAME`, `DATASET_SUBSET`) at the top of `run_hplt76.sh`.
    -   Execute the script:
        ```bash
        bash run_hplt76.sh
        ```

-   **To train a model using XLM-RoBERTa as the teacher:**
    -   Customize `run_xlmr76.sh` as needed.
    -   Execute the script:
        ```bash
        bash run_xlmr76.sh
        ```
The scripts are configured to automatically handle multi-GPU training and will save model checkpoints and logs into a `models/` directory.

### 2. Finding the Best Checkpoint

After training, multiple checkpoints are saved. The `find-best-checkpoint.py` script evaluates each of these checkpoints against an unseen validation set to identify the one with the lowest distillation loss, indicating the best generalization.

-   **Run the script:**
    -   Verify the `BASE_MODEL_DIR` path inside the script points to your training output directory.
    -   Execute the script:
        ```bash
        python evaluation/find-best-checkpoint/find-best-checkpoint.py --results_csv evaluation/find-best-checkpoint/my_results.csv
        ```
This will produce a CSV file (`my_results.csv`) ranking all checkpoints. You can then use the `checkpoint_path` of the top-ranked model for the final evaluation.

### 3. Downstream Task Evaluation (UD & NER)

We provide evaluation pipelines for Universal Dependencies tasks (POS, Lemma, LAS) and Named Entity Recognition (NER).

#### Universal Dependencies (UD) Evaluation

The `run_ud_eval.sh` script handles the fine-tuning and evaluation on UD tasks.

-   **Prerequisites:**
    -   Download the required UD treebanks (e.g., v2.15) and place them in the `evaluation/NLP-eval/ud/ud_data/` directory.

-   **Run the evaluation:**
    The script accepts the language code and the path to your best model checkpoint as arguments.
    ```bash
    # Usage: ./run_ud_eval.sh [LANGUAGE_CODE] [PATH_TO_BEST_CHECKPOINT]
    bash evaluation/NLP-eval/ud/run_ud_eval.sh hunL "/path/to/your/best/student/checkpoint-XXXXX"
    ```
    Results will be saved as JSONL files in the specified output directory.

#### Named Entity Recognition (NER) Evaluation

The `ner_eval.sh` script handles evaluation on the WikiAnn dataset for NER.

-   **Run the evaluation:**
    The script takes the dataset name (e.g., `wikiann/de`) and the model path as arguments.
    ```bash
    # Usage: ./ner_eval.sh [DATASET_NAME] [MODEL_PATH] [OUTPUT_DIR]
    bash evaluation/NLP-eval/ner/ner_eval.sh "wikiann/de" "/path/to/your/best/student/checkpoint-XXXXX" "./results/ner/my_model_de"
    ```
