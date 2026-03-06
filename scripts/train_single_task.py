"""
Main training script for CLARITY SemEval-2026 Competition
Usage: python scripts/train.py --config configs/experiments/fine_tuning/task_a/bert_base.yaml

This script:
- Verifies GPU availability before training
- Trains the model
- Saves ONLY the best checkpoint (based on validation F1)
- Automatically calls evaluate.py for detailed metrics after training
- Logs training metrics to W&B
- Works for both Task A (3 classes) and Task B (multiple classes)

"""

import argparse
import os
import sys
import torch
import random
import numpy as np
import wandb
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config, validate_config, create_output_dirs
from src.utils.metrics import format_metrics_for_trainer, get_classification_report
# from src.utils.tokenizer_specials import add_clarity_special_tokens
from src.models import create_model
from src.data.clarity_dataset import create_datasets
from src.trainers import create_trainer
from src.utils.wandb import build_wandb_group_name, build_wandb_run_name
from transformers import set_seed as hf_set_seed
from transformers import AutoTokenizer

# Hard-disable SDPA kernels
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_gpu_setup():
    """
    Verify GPU is available and being used.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\n" + "="*60)
    print("GPU Setup Verification")
    print("="*60)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"✅ CUDA is available")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")

        # Check GPU memory
        print(f"\n   GPU Memory:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
            print(f"     GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"       Total: {total_memory:.2f} GB")
            print(f"       Allocated: {allocated_memory:.2f} GB")
            print(f"       Reserved: {reserved_memory:.2f} GB")

        return True
    else:
        print("❌ CUDA is NOT available - training will use CPU!")

        # Print environment info
        print(f"\n   Environment variables:")
        print(f"     CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"     CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

        return False


def compute_metrics(eval_pred, label_names=None):
    """
    Compute metrics for evaluation using unified metrics module.

    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer
        label_names: List of label names for per-class metrics

    Returns:
        Dictionary of metrics for Trainer logging
    """
    predictions, labels = eval_pred

    # Use unified metrics function
    metrics = format_metrics_for_trainer(predictions, labels, label_names)

    # Print classification report for visibility
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(get_classification_report(predictions, labels, label_names))

    return metrics


def setup_wandb(configs, seed):
    """
    Initialize W&B with enhanced organization and visualization.

    Args:
        configs: Dictionary with all configs (experiment, model, task, data)
    """
    experiment_config = configs['experiment']
    model_config = configs['model']
    task_config = configs['task']
    data_config = configs['data']

    if experiment_config['tracking']['use_wandb']:
        print("\nInitializing W&B...")

        # Extract key info for tagging and grouping
        task_name = task_config.get('task_name', 'unknown')
        model_type = model_config.get('model_type', 'unknown')
        experiment_type = experiment_config.get('experiment_type', 'unknown')

        # Create descriptive tags
        tags = [
            experiment_type,
            task_name,
            model_type,
            # NOTE - Commented out due to char-limit on tags (this breaks run)
            # NOTE - If you uncomment below, ensure exper_name is < 64 chars
            # experiment_config['experiment_name']
        ]

        # Add masking mode if present
        masking_mode = experiment_config.get('preprocessing', {}).get('masking_mode')
        if masking_mode:
            tags.append(masking_mode)

        hier = experiment_config.get("hierarchical", {}) or {}
        if hier.get("enabled", False):
            tags.append(f"hier_stage_{hier.get('stage')}")

        # Add model size
        model_name = model_config['model_name'].lower()
        if 'large' in model_name:
            tags.append('large')
        elif 'base' in model_name:
            tags.append('base')
        elif 'small' in model_name:
            tags.append('small')

        # Add GPU info to tags
        if torch.cuda.is_available():
            tags.append('gpu')
        else:
            tags.append('cpu')

        group_name = build_wandb_group_name(
            experiment_config,
            task_config,
            model_config,
            data_config,
        )

        wandb_run_name = build_wandb_run_name(
            experiment_config,
            seed=seed
        )

        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ["WANDB_ENTITY"],
            name=wandb_run_name,
            tags=tags,
            group=group_name,
            job_type="training",
            config={
                'experiment': experiment_config,
                'model': model_config,
                'task': task_config,
                'data': configs['data'],
                'seed': seed,
                'seed_mode': experiment_config.get('seed_control', {}).get('mode', 'single'),
                'gpu_info': {
                    'cuda_available': torch.cuda.is_available(),
                    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
                }
            },
            notes=experiment_config.get('description', '')
        )

        print(f"W&B logging enabled: {wandb.run.url}")
        print(f"  Tags: {tags}")
        print(f"  Group: {group_name}")
    else:
        print("W&B logging disabled")


def call_evaluate_script(checkpoint_path, config_path, split='test'):
    """
    Call the evaluate.py script as a subprocess.

    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the experiment config
        split: Dataset split to evaluate on ('validation' or 'test')
    """
    print(f"\n{'='*60}")
    print(f"Running detailed evaluation on {split} set...")
    print(f"{'='*60}")

    evaluate_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'evaluate.py'
    )

    cmd = [
        sys.executable,
        evaluate_script,
        '--checkpoint', checkpoint_path,
        '--config', config_path,
        '--split', split
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Evaluation on {split} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Evaluation on {split} failed with error: {e}")
        return False


def _unwrap_hf_model(model):
    # handles DDP/DataParallel and wrapper classes
    if hasattr(model, "module"):
        model = model.module
    return getattr(model, "model", model)

def run_single_seed(seed, configs, args, gpu_available):
    """
    Run one complete training + evaluation cycle for a single random seed.

    This function:
    - Sets all relevant random seeds (python, numpy, torch, HF)
    - Creates deterministic train / val / test splits (data seed is fixed elsewhere)
    - Trains the model using HuggingFace Trainer
    - Selects the best checkpoint based on validation macro F1
    - Saves the best model for this seed
    - Optionally evaluates on validation and test sets
    - Logs metrics and artefacts to W&B (if enabled)

    It is designed to be called multiple times in multi-seed mode so that
    performance can be averaged across different random initialisations
    while keeping the data split fixed.

    Args:
        seed (int):
            Random seed controlling model initialisation, dropout,
            dataloader shuffling, and optimiser behaviour.

        configs (dict):
            Loaded experiment, model, task, and data configurations.

        args (argparse.Namespace):
            Command-line arguments (e.g. resume checkpoint, skip eval).

        gpu_available (bool):
            Whether a CUDA device is available for training.

    Returns:
        dict:
            Summary information for this seed run, including:
            - seed used
            - output directory
            - best validation macro F1
    """

    print(f"Starting training with config: {args.config}")

    import copy
    configs = copy.deepcopy(configs)

    print("\n" + "-" * 80)
    print(f"INITIALISING SEED {seed}")
    print("-" * 80)

    set_seed(seed)
    hf_set_seed(seed)

    print(
        f"Seed {seed} set for: "
        f"python.random, numpy, torch, torch.cuda, hf_set_seed"
    )

    experiment_config = configs['experiment']
    model_config = configs['model']
    task_config = configs['task']
    data_config = configs['data']
    base_output_dir = experiment_config['output_dir']

    # --- HIERARCHICAL LABEL OVERRIDE ---
    hier = experiment_config.get("hierarchical", {}) or {}
    if hier.get("enabled", False):
        stage = int(hier.get("stage", 0) or 0)
        if stage not in (1, 2):
            raise ValueError(f"hierarchical.stage must be 1 or 2, got: {stage}")

        if stage == 1:
            # Ambivalent vs Not-Ambivalent
            new_labels = {
                "num_labels": 2,
                "label_names": ["NOT_AMBIVALENT", "AMBIVALENT"],
                "label2id": {"NOT_AMBIVALENT": 0, "AMBIVALENT": 1},
                "id2label": {0: "NOT_AMBIVALENT", 1: "AMBIVALENT"},
            }
        else:
            # Clear Reply vs Clear Non-Reply (Ambivalent filtered out in dataset)
            new_labels = {
                "num_labels": 2,
                "label_names": ["CLEAR_REPLY", "CLEAR_NON_REPLY"],
                "label2id": {"CLEAR_REPLY": 0, "CLEAR_NON_REPLY": 1},
                "id2label": {0: "CLEAR_REPLY", 1: "CLEAR_NON_REPLY"},
            }

        # Replace labels in task_config for this run only (configs is already deepcopy'd)
        task_config = dict(task_config)
        task_config["labels"] = new_labels
        configs["task"] = task_config
    # --- END HIERARCHICAL LABEL OVERRIDE ---

    # Modify output dir and run name
    experiment_config['output_dir'] = os.path.join(
        base_output_dir, f"seed_{seed}"
    )
    experiment_config['tracking']['wandb_run_name'] = (
        f"{experiment_config['experiment_name']}-seed{seed}"
    )
    
    # Masking
    preproc_cfg = experiment_config.get('preprocessing', {})
    masking_mode = preproc_cfg.get('masking_mode', 'none')

    data_config = dict(data_config)  # shallow copy to avoid mutation
    data_config['preprocessing'] = experiment_config.get('preprocessing', {})   

    print("\n" + "=" * 80)
    print(f"Text preprocessing:")
    print(f"  Masking mode: {masking_mode.upper()}")
    print("=" * 80)

    # Put input format on data config object
    input_format_config = experiment_config.get("input_format", "pair")
    data_config['input_format'] = input_format_config

    print("\n" + "=" * 80)
    print(f"Input format:")
    print(f"  Mode: {input_format_config.upper()}")
    print("=" * 80)

    data_config['cd'] = bool(experiment_config.get('cd', False))

    # Hierarchical task behaviour is data-time logic (filtering, label remap)
    if experiment_config.get("hierarchical", {}).get("enabled", False):
        data_config["hierarchical"] = experiment_config["hierarchical"]

    print(f"\nExperiment: {experiment_config['experiment_name']}")
    print(f"Description: {experiment_config['description']}")
    print(f"Model: {model_config['model_type']} ({model_config['model_name']})")
    print(f"Task: {task_config.get('task_name', 'unknown').upper()}")
    print(f"Device: {'GPU' if gpu_available else 'CPU'}")
    print(f"Seed: {seed}")

    # Create output directories
    create_output_dirs(experiment_config)

    # Setup W&B logging
    setup_wandb(configs, seed)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        data_config, task_config, model_config
    )

    # Create model
    print(f"\nCreating {model_config['model_type']} model...")
    model = create_model(model_config, task_config)

    # Tokeniser already has CLARITY special tokens (added in create_datasets).
    # We MUST still resize embeddings to match the tokeniser length.
    hf_model = _unwrap_hf_model(model)
    hf_model.resize_token_embeddings(len(tokenizer))
    print("HF model class:", type(hf_model))
    print("config.model_type:", getattr(hf_model.config, "model_type", None))
    print("config.attention_type:", getattr(hf_model.config, "attention_type", None))
    print("max_position_embeddings:", getattr(hf_model.config, "max_position_embeddings", None))
    print(f"Tokenizer size: {len(tokenizer)}")
    print(f"Embedding size: {hf_model.get_input_embeddings().num_embeddings}")
    assert hf_model.get_input_embeddings().num_embeddings == len(tokenizer), "Tokenizer/model vocab mismatch"

    # If we are running a hierarchical stage, persist label mappings in the HF config.
    # This avoids confusion later when loading checkpoints.
    if experiment_config.get("hierarchical", {}).get("enabled", False):
        label2id = task_config["labels"]["label2id"]
        id2label = task_config["labels"]["id2label"]
        hf_model.config.label2id = {str(k): int(v) for k, v in label2id.items()}
        hf_model.config.id2label = {int(k): str(v) for k, v in id2label.items()}

    if torch.cuda.is_available() and not args.force_cpu:
        model = model.to("cuda")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Verify model is on correct device
    if gpu_available:
        device = next(model.parameters()).device
        print(f"Model device: {device}")
        if device.type == 'cpu':
            # TODO - Silence warning when running on GPU (it's glitchy)
            print("⚠️  WARNING: Model is on CPU despite GPU being available!")

    # Create tokenizer for trainer
    print("\nSetting up trainer...")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_config['model_name'],
    #     cache_dir=model_config['loading'].get('cache_dir')
    # )

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set smart defaults for checkpoint management
    if 'training' not in experiment_config:
        experiment_config['training'] = {}

    training_config = experiment_config['training']

    if 'load_best_model_at_end' not in training_config:
        training_config['load_best_model_at_end'] = True

    if 'metric_for_best_model' not in training_config:
        training_config['metric_for_best_model'] = 'eval_f1_macro'

    if 'greater_is_better' not in training_config:
        training_config['greater_is_better'] = True

    # Print checkpoint strategy
    save_total_limit = training_config.get('save_total_limit', 'all')
    save_strategy = training_config.get('save_strategy', 'epoch')
    metric_for_best = training_config.get('metric_for_best_model', 'eval_f1_macro')

    print(f"\nCheckpoint strategy:")
    print(f"  - Save strategy: {save_strategy}")
    print(f"  - Save total limit: {save_total_limit} checkpoint(s)")
    print(f"  - Best model metric: {metric_for_best}")
    print(f"  - Load best at end: {training_config.get('load_best_model_at_end', True)}")
    print(f"  - Checkpoint location: {experiment_config['output_dir']}")

    # Create trainer
    trainer = create_trainer(
        configs=configs,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        seed=seed,
    )

    # Get label names for metrics computation
    id2label = task_config['labels']['id2label']
    label_names = [id2label.get(str(i), id2label.get(i, f"class_{i}"))
                   for i in range(len(id2label))]

    # Add compute_metrics to trainer with label names
    trainer.compute_metrics = lambda eval_pred: compute_metrics(eval_pred, label_names)

    # Print dataset distribution info
    print(f"\nDataset information:")
    print(f"  Train size: {len(train_dataset)} - Distribution: {train_dataset.get_label_distribution()}")
    print(f"  Validation size: {len(val_dataset)} - Distribution: {val_dataset.get_label_distribution()}")
    print(f"  Test size: {len(test_dataset)} - Distribution: {test_dataset.get_label_distribution()}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")

    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume)

        # Print training results
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"Final train loss: {train_result.training_loss:.4f}")

        # Get best metric from training
        if hasattr(train_result, 'metrics'):
            best_metric = train_result.metrics.get('eval_f1_macro', 'N/A')
            print(f"Best validation F1 (macro): {best_metric}")

        # Save final model
        best_model_path = os.path.join(experiment_config['output_dir'], 'best_model')
        print(f"\nSaving best model to {best_model_path}")

        best_ckpt = trainer.state.best_model_checkpoint
        print("Best checkpoint:", best_ckpt)

        if best_ckpt is None:
            print("WARNING: trainer.state.best_model_checkpoint is None")
        

        trainer.save_model(best_model_path)
        # Explicitly save HF config
        trainer.model.model.config.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)

        # Evaluate and log test metrics to W&B
        print("\n" + "="*60)
        print("Evaluating on test set and logging to W&B...")
        print("="*60)

        task_name = task_config.get('task_name', 'task')

        # Log final training loss to W&B
        if experiment_config['tracking']['use_wandb']:
            wandb.log({
                f"{task_name}/train/final_loss": train_result.training_loss,
            })
            wandb.run.summary[f"{task_name}_train_loss"] = train_result.training_loss

        # Call evaluate.py for detailed metrics
        if not args.skip_eval:
            print("\n" + "="*60)
            print("Running detailed evaluation...")
            print("="*60)

            call_evaluate_script(best_model_path, args.config, split='validation')
            call_evaluate_script(best_model_path, args.config, split='test')
        else:
            print("\n⚠️  Skipping automatic evaluation (--skip-eval flag set)")
            print(f"   To evaluate manually, run:")
            print(f"   python scripts/evaluate.py --checkpoint {best_model_path} --config {args.config} --split test")

        print("\n" + "="*60)
        print("✅ Training pipeline completed successfully!")
        print("="*60)
        print(f"📁 Best model saved to: {best_model_path}")
        print(f"📊 Outputs directory: {experiment_config['output_dir']}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        if experiment_config['tracking']['use_wandb']:
            wandb.finish(exit_code=1)
        raise

    # Clean up W&B
    if experiment_config['tracking']['use_wandb']:
        wandb.finish()

    return {
        "seed": seed,
        "output_dir": experiment_config['output_dir'],
        "best_val_f1": best_metric if 'best_metric' in locals() else None
    }

def main():
    parser = argparse.ArgumentParser(description='Train model for CLARITY competition')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file (configs/experiments/*.yaml)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip automatic evaluation after training'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    args = parser.parse_args()


    # Verify GPU setup
    gpu_available = verify_gpu_setup()

    #if not gpu_available and not args.force_cpu:
    #    print("\n" + "="*60)
    #    print("⚠️  WARNING: No GPU detected!")
    #    print("="*60)
    #    print("Training transformer models on CPU is extremely slow.")
    #    print("This job may take days or weeks to complete.")
    #    print("\nOptions:")
    #    print("  1. Check your SLURM configuration (--gres=gpu:1)")
    #    print("  2. Verify CUDA installation (module load cuda/...)")
    #    print("  3. Use --force-cpu flag to continue anyway")
    #    print("="*60)
    #    sys.exit(1)

    #if args.force_cpu:
    #    print("\n⚠️  CPU mode forced by --force-cpu flag")

    # Load and validate configurations
    configs = load_experiment_config(args.config)
    validate_config(configs)

    experiment_config = configs['experiment']
    model_config = configs['model']
    task_config = configs['task']
    data_config = configs['data']

    # ------------------------------------------------------------------
    # Seed handling
    # ------------------------------------------------------------------

    seed_cfg = experiment_config.get("seed_control", {})

    seed_mode = seed_cfg.get("mode", "single")

    if seed_mode == "multi":
        seeds = seed_cfg.get("seeds")
        if not seeds:
            raise ValueError(
                "seed_control.mode is 'multi' but no seeds were provided"
            )
    else:
        seed = seed_cfg.get("seed")
        if seed is None:
            raise ValueError(
                "seed_control.mode is 'single' but no seed was provided"
            )
        seeds = [seed]

    print("\n" + "=" * 80)
    print("SEED CONFIGURATION")
    print("=" * 80)
    print(f"Seed mode: {seed_mode}")
    print(f"Seeds to run: {seeds}")
    print(f"Number of runs: {len(seeds)}")
    print("=" * 80)

    all_run_summaries = []

    for seed in seeds:
        summary = run_single_seed(seed, configs, args, gpu_available)
        all_run_summaries.append(summary)
    


if __name__ == "__main__":
    main()