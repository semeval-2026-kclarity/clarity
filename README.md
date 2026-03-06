# KCLarity at SemEval-2026 Task 6: Encoder and Zero-Shot Approaches to Political Evasion Detection

This repository contains the code for the KCLarity team's participation in the [CLARITY shared task](https://konstantinosftw.github.io/CLARITY-SemEval-2026/) at SemEval 2026. We investigate fine-tuned encoder models (RoBERTa, DeBERTa-v3) and zero-shot decoder models (GPT-5.2, Llama 3, Qwen, Gemma 3) for classifying response clarity and evasion strategies in political discourse.

**Authors:** [Archie Sage](mailto:archie.sage@kcl.ac.uk)\* and [Salvatore Greco](mailto:salvatore.greco@kcl.ac.uk)\* (King's College London)
*\*Equal contribution.*

**Paper:** *KCLarity at SemEval-2026 Task 6: Encoder and Zero-Shot Approaches to Political Evasion Detection*

## Repository Structure

```
├── bash/
│   ├── task_a/                  # Bash scripts to run Task 1 experiments on HPC
│   ├── task_b/                  # Bash scripts to run Task 2 experiments on HPC
│   ├── base_runtime.sh          # Environment validation (checks .env, CUDA, conda)
│   └── test_runtime.sh          # Quick check that environment vars are set correctly
├── configs/
│   ├── data/                    # Dataset configurations
│   ├── experiments/
│   │   ├── task_a/              # Experiment configs for clarity-level classification (Task 1)
│   │   └── task_b/              # Experiment configs for evasion-level classification (Task 2)
│   ├── models/                  # Model architecture configs
│   └── tasks/                   # Task-level configs
├── scripts/
│   ├── evaluation/
│   │   ├── aggregate_seeds_local.py   # Aggregate encoder results across seeds
│   │   ├── test_zero_shot_gpt.py      # Zero-shot evaluation via OpenAI API
│   │   └── test_zero_shot.py          # Zero-shot evaluation via HuggingFace Inference API
│   ├── submissions/                   # Submission file generation for CodaBench
│   ├── visualisation/
│   │   └── render_confusion_matrix.py # Generate confusion matrices from evaluation JSON
│   ├── evaluate.py                    # Evaluation during training (reports to W&B)
│   └── train_single_task.py           # Main entry point for encoder training
├── src/
│   ├── data/                    # Dataset loading, splitting, tokenisation
│   ├── models/                  # Classifier architectures (RoBERTa, DeBERTa-v3)
│   ├── trainers/                # Training loop logic (single-task)
│   └── utils/                   # Config checks, masking, metrics, callbacks, W&B integration
├── .env.example                 # Template for all required environment variables
├── requirements.txt             # Python dependencies
└── README.md
```

## Terminology Mapping

The paper and codebase use slightly different names in a couple of places. These mappings are helpful when navigating configs and scripts.

**Task naming:**

| Paper | Codebase | Description |
|-------|----------|-------------|
| Task 1 | `task_a` | Clarity-level classification |
| Task 2 | `task_b` | Evasion-level classification |

**Input representations** (Section 3.3 in paper):

| Paper | Codebase | Format |
|-------|----------|--------|
| Segmented | `pair` | `[CLS] answer [SEP] question [SEP]` |
| Marked | `qa_marked` | `[QUESTION] question [ANSWER] answer` |


## Reproducing Results

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (all encoder experiments were trained on a single NVIDIA A100-SXM4-40GB)
- Conda (we used Miniforge)
- [Weights & Biases](https://wandb.ai) account (for experiment tracking)

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n semeval-env python=3.10
conda activate semeval-env

# Install dependencies
pip install -r requirements.txt
```

Some experiments require additional model downloads (e.g. spaCy models for the person-name masking ablation):

```bash
python -m spacy download en_core_web_lg
```

### 2. Configure Environment Variables

Copy the example file and fill in all values:

```bash
cp .env.example .env
```

You will need to set paths for scratch storage, model caches, and output directories, as well as API keys for W&B, HuggingFace, and OpenAI. See `.env.example` for the full list with descriptions.

Ensure the directories referenced in your `.env` exist before running experiments:

```bash
mkdir -p $MODEL_OUTPUTS_DIR $MASKING_CACHE_DIR $MODEL_CACHE_DIR
```

You can verify your environment is configured correctly:

```bash
bash bash/test_runtime.sh
```

### 3. Encoder Experiments

The main training entry point is `scripts/train_single_task.py`, which reads a YAML experiment config:

```bash
python scripts/train_single_task.py --config configs/experiments/task_a/fine_tuning/roberta/large/qa_marked/sqrt_weighted_stratified.yaml
```

Pre-configured bash scripts for all experiments reported in the paper are available in `bash/task_a/` and `bash/task_b/`. These are intended for HPC clusters with Slurm but can be adapted for other environments.

Once training is complete across all seeds, aggregate results:

```bash
python scripts/evaluation/aggregate_seeds_local.py
```

This produces the JSON files used by the visualisation scripts and outputs evaluation metrics as reported in the paper.

### 4. Zero-Shot Experiments

Zero-shot evaluation is self-contained in two scripts.

**OpenAI models (GPT-5.2):**

```bash
python scripts/evaluation/test_zero_shot_gpt.py --model gpt-5.2-2025-12-11
```

**HuggingFace Inference API models (Llama, Qwen, Gemma):**

```bash
python scripts/evaluation/test_zero_shot.py --model meta-llama/Llama-3.3-70B-Instruct google/gemma-3-27b-it
```

Both scripts include caching (predictions are saved incrementally), retry logic, and output metrics in a consistent JSON format. API keys must be set in your `.env` or passed via command-line arguments.

## Extending the Codebase

The pipeline is designed to be modular:

- **New model architectures:** Add a classifier in `src/models/` following the pattern of `roberta_classifier.py` or `deberta_classifier.py`.
- **New training regimes:** Add a trainer in `src/trainers/` (e.g., for multi-task learning).
- **New experiments:** Create a YAML config in `configs/experiments/` referencing your model and data configuration.

## Notes

- This repository does not include the complete code for the exploratory experiments described in Appendix G of the paper (cross-domain transfer and cognitive distortion features).
- The `scripts/submissions/` directory contains the scripts used to generate our CodaBench submission files during the shared task. These are included for transparency but are not needed for reproducing the results reported in the paper.

## Citation

```bibtex
@inproceedings{sage-greco-2026-kclarity,
    title     = "{KCL}arity at {S}em{E}val-2026 Task 6: Encoder and Zero-Shot Approaches to Political Evasion Detection",
    author    = "Sage, Archie and Greco, Salvatore",
    booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)",
    year      = "2026",
    publisher = "Association for Computational Linguistics"
}
```

## Contact

For questions or issues, please contact:

- **Archie Sage** — [archie.sage@kcl.ac.uk](mailto:archie.sage@kcl.ac.uk)
- **Salvatore Greco** — [salvatore.greco@kcl.ac.uk](mailto:salvatore.greco@kcl.ac.uk)