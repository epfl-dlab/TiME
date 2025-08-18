# NLP Downstream Task Evaluation

This directory contains the scripts and utilities for evaluating the fine-tuned performance of our distilled language models on core NLP tasks. The evaluation pipeline is divided into two main components:

1.  **Universal Dependencies (UD):** For Part-of-Speech (POS) tagging, lemmatization, and dependency parsing (LAS).
2.  **Named Entity Recognition (NER):** For evaluating token classification on the WikiAnn dataset.

## Acknowledgement

The evaluation framework in this directory, particularly the Universal Dependencies pipeline (`ud/`), is heavily based on and adapted from the excellent evaluation suite developed for the **HPLT project**. We extend our sincere gratitude to the HPLT authors for making their robust and comprehensive evaluation code publicly available, which provided a strong foundation for our benchmarking.

Original HPLT evaluation resources can be found with their model releases. (https://github.com/hplt-project/HPLT-WP4)
## Directory Structure

-   **`/ner`**: Contains all scripts and utilities for running NER evaluation.
    -   `ner_eval.sh`: The main bash script to orchestrate the NER fine-tuning and evaluation process.
    -   `ner_bert.py`: The core Python script that handles model loading, training, and prediction using the `transformers` library.
    -   `ner_eval.py`: A helper script for calculating entity-level metrics.
    -   `tsa_utils.py`: Contains dataclasses and helper functions for the NER pipeline.
    -   `constants.py`: Holds language code mappings.

-   **`/ud`**: Contains all scripts and utilities for running the UD evaluation (POS, Lemmatization, Parsing).
    -   `run_ud_eval.sh`: The main bash script to run the complete UD pipeline for a given language and model.
    -   `train0.py`: The primary Python script that trains the multi-task UD model.
    -   `model.py`: Defines the neural network architecture for the UD tasks.
    -   `lemma_rule.py`: Implements the rule-based lemmatization logic.
    -   `/ud_data/`: A placeholder directory where the user must place the downloaded Universal Dependencies treebank files (e.g., from `v2.15`).

## How to Use

Instructions for running these evaluation scripts are provided in the main `README.md` at the root of the repository. Please refer to the **"How to Run" -> "Downstream Task Evaluation"** section for detailed commands.

### Prerequisites

Before running the UD evaluation, you must download the required Universal Dependencies treebanks and place them into the `ud/ud_data/` directory.

```
evaluation/NLP-eval/ud/
└── ud_data/
    ├── ud-treebanks-v2.15/
    │   ├── UD_English-EWT/
    │   ├── UD_German-GSD/
    │   └── ...
    └── place ud data here  <-- (You can delete this placeholder file)
```