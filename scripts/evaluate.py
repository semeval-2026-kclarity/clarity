"""
Evaluation script for CLARITY SemEval-2026 Competition
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import wandb
from datetime import datetime

# Try to import plotly for confusion matrix visualization
try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Warning: plotly not installed. Some visualizations will be disabled.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config, validate_config
from src.utils.metrics import (
    compute_all_metrics,
    get_classification_report,
    compute_multi_annotator_metrics
)
from src.utils.tokenizer_specials import add_clarity_special_tokens
from src.models import create_model
from src.data.clarity_dataset import create_datasets
from src.utils.wandb import build_wandb_group_name, build_wandb_run_name

def extract_seed_from_checkpoint(checkpoint_path):
    """
    Extract seed from checkpoint path assuming structure .../seed_<N>/best_model
    """
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    for part in parts:
        if part.startswith("seed_"):
            try:
                return int(part.replace("seed_", ""))
            except ValueError:
                pass
    return None


def load_model_from_checkpoint(checkpoint_path, model_config, task_config, tokenizer=None):
    """Load a trained model from checkpoint."""
    print(f"\nLoading model from checkpoint: {checkpoint_path}")

    # Create model architecture
    model = create_model(model_config, task_config)

    # If tokenizer is provided, make sure embeddings match training
    if tokenizer is not None:
        # Tokeniser should already contain special tokens (loaded from checkpoint),
        # but resizing the embeddings is still mandatory.
        if hasattr(model, "module"):
            base = model.module
        else:
            base = model

        hf_model = getattr(base, "model", base)
        hf_model.resize_token_embeddings(len(tokenizer))

    # Load weights
    model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_path, 'model.safetensors')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")

    print(f"Loading weights from: {model_path}")

    # Load state dict
    if model_path.endswith('.bin'):
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        from safetensors.torch import load_file
        state_dict = load_file(model_path)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠️  WARNING: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  WARNING: Unexpected keys: {unexpected_keys}")

    print(f"✅ Model weights loaded successfully")
    model.eval()
    return model


def detect_multi_annotator_format(dataset, data_config, task_config):
    """
    Detect if dataset uses multi-annotator format (for Task B test set).

    Returns:
        Tuple (is_multi_annotator: bool, annotator_columns: list or None)
    """
    # Check if this is Task B
    if task_config['task_name'] != 'task_b':
        return False, None

    # Check if dataset has the raw dataframe
    if not hasattr(dataset, 'data'):
        return False, None

    df = dataset.data

    # Check for annotator columns
    annotator_cols = ['annotator1', 'annotator2', 'annotator3']
    has_annotators = all(col in df.columns for col in annotator_cols)

    # Check if evasion_label is null
    label_col = task_config['data']['label_column']
    has_null_labels = label_col in df.columns and df[label_col].isna().all()

    if has_annotators and has_null_labels:
        print(f"\n✅ Multi-annotator format detected!")
        print(f"   Using lenient evaluation: prediction correct if it matches ANY of the 3 annotators")
        return True, annotator_cols

    return False, None


def extract_annotator_labels(dataset, annotator_columns, task_config):
    """Extract labels from multiple annotator columns."""
    df = dataset.data
    label2id = task_config['labels']['label2id']

    annotator_labels = []
    for col in annotator_columns:
        # Convert string labels to IDs
        labels = df[col].map(label2id).values
        annotator_labels.append(labels)

        # Print distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n   {col} distribution:")
        id2label = task_config['labels']['id2label']
        for label_id, count in zip(unique, counts):
            label_name = id2label.get(str(int(label_id)), id2label.get(int(label_id), f"class_{label_id}"))
            print(f"      {label_name}: {count}")

    return annotator_labels


def compute_detailed_metrics(predictions, labels, task_config, multi_annotator=False):
    """Compute comprehensive metrics."""
    # Get label names
    id2label = task_config['labels']['id2label']
    num_classes = len(id2label)
    label_names = [id2label.get(str(i), id2label.get(i, f"class_{i}"))
                   for i in range(num_classes)]

    # Compute all metrics
    all_metrics = compute_all_metrics(
        predictions=predictions,
        labels=labels,
        label_names=label_names,
        include_confusion_matrix=True,
        multi_annotator=multi_annotator
    )

    # Convert format for backward compatibility
    results = {
        'overall': {
            'accuracy': all_metrics['overall']['accuracy'],
            'macro_avg': {
                'f1': all_metrics['overall']['f1_macro'],
                'precision': all_metrics['overall']['precision_macro'],
                'recall': all_metrics['overall']['recall_macro']
            },
            'weighted_avg': {
                'f1': all_metrics['overall']['f1_weighted'],
                'precision': all_metrics['overall']['precision_weighted'],
                'recall': all_metrics['overall']['recall_weighted']
            }
        },
        'per_class': {},
        'confusion_matrix': all_metrics['confusion_matrix'].tolist() if isinstance(all_metrics['confusion_matrix'], np.ndarray) else all_metrics['confusion_matrix']
    }

    # Add multi-annotator specific metrics
    if multi_annotator:
        results['overall']['accuracy_lenient'] = all_metrics['overall']['accuracy_lenient']
        results['overall']['accuracy_majority'] = all_metrics['overall']['accuracy_majority']
        results['overall']['mean_annotator_agreement'] = all_metrics['overall']['mean_annotator_agreement']
        results['annotator_stats'] = all_metrics['annotator_stats']

    # Convert per-class format
    for label_name, metrics in all_metrics['per_class'].items():
        results['per_class'][label_name] = {
            'class_id': label_names.index(label_name),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'support': metrics['support']
        }

    # Get predicted labels
    pred_labels = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions

    # Print classification report
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)

    if multi_annotator:
        # Use majority vote labels for report
        from collections import Counter
        majority_labels = []
        for i in range(len(labels[0])):
            labels_for_sample = [annotator[i] for annotator in labels]
            label_counts = Counter(labels_for_sample)
            majority_label = label_counts.most_common(1)[0][0]
            majority_labels.append(majority_label)
        majority_labels = np.array(majority_labels)

        print("(Using majority vote from 3 annotators for this report)")
        print("(Primary metric uses lenient evaluation - matches ANY annotator)")
        report_labels = majority_labels
    else:
        report_labels = labels

    print(get_classification_report(predictions, report_labels, label_names))

    return results, pred_labels


def print_detailed_results(results, split_name, multi_annotator=False):
    """Print detailed evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} SET EVALUATION RESULTS")
    if multi_annotator:
        print("(Multi-Annotator Evaluation)")
    print(f"{'=' * 60}")

    print(f"\n📊 Overall Metrics:")

    if multi_annotator:
        print(f"  PRIMARY METRIC:")
        print(f"    Accuracy (Lenient): {results['overall']['accuracy_lenient']:.4f}")
        print(f"      → Prediction correct if it matches ANY annotator\n")
        print(f"  SECONDARY METRICS:")
        print(f"    Accuracy (Majority Vote): {results['overall']['accuracy_majority']:.4f}")
        print(f"      → Using majority label from 3 annotators")
        print(f"    Mean Annotator Agreement: {results['overall']['mean_annotator_agreement']:.4f}\n")

        stats = results['annotator_stats']
        print(f"  Annotator Agreement:")
        print(f"    All 3 agree:  {stats['full_agreement_rate']:.1%}")
        print(f"    2 out of 3:   {stats['partial_agreement_rate']:.1%}")
        print(f"    All differ:   {stats['no_agreement_rate']:.1%}")
    else:
        print(f"  Accuracy: {results['overall']['accuracy']:.4f}")

    print(f"\n  Macro Average:")
    print(f"    Precision: {results['overall']['macro_avg']['precision']:.4f}")
    print(f"    Recall:    {results['overall']['macro_avg']['recall']:.4f}")
    print(f"    F1-Score:  {results['overall']['macro_avg']['f1']:.4f}")

    print(f"\n  Weighted Average:")
    print(f"    Precision: {results['overall']['weighted_avg']['precision']:.4f}")
    print(f"    Recall:    {results['overall']['weighted_avg']['recall']:.4f}")
    print(f"    F1-Score:  {results['overall']['weighted_avg']['f1']:.4f}")

    print(f"\n📈 Per-Class Metrics:")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-' * 75}")
    for class_name, metrics in results['per_class'].items():
        print(f"  {class_name:<25} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1_score']:>10.4f} "
              f"{metrics['support']:>10}")


def predict_on_dataset(model, dataset, device='cpu', batch_size=32):
    """Generate predictions for an entire dataset."""
    from torch.utils.data import DataLoader
    from transformers import DefaultDataCollator

    model.to(device)
    model.eval()

    data_collator = DefaultDataCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    all_predictions = []
    all_labels = []

    print(f"\nGenerating predictions...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total samples: {len(dataset)}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels')

            forward_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

            if 'token_type_ids' in batch:
                forward_kwargs['token_type_ids'] = batch['token_type_ids'].to(device)

            outputs = model(**forward_kwargs)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs

            all_predictions.append(logits.cpu().numpy())
            if labels is not None:
                all_labels.append(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples")

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None

    print(f"  ✅ Predictions generated")
    print(f"     Predictions shape: {predictions.shape}")
    print(f"     First 5 predicted classes: {np.argmax(predictions[:5], axis=1)}")

    return predictions, labels


def create_visualizations(results, pred_labels, true_labels, task_config, split_name, output_dir):
    """Create and save visualization plots."""
    visualizations = {}

    if not PLOTLY_AVAILABLE:
        print("  ⚠️  Plotly not available, skipping visualizations")
        return visualizations

    # Only create visualizations for single-label case
    if isinstance(true_labels, list):
        print("  ℹ️  Skipping visualizations for multi-annotator evaluation")
        return visualizations

    # Confusion matrix heatmap
    try:
        cm = np.array(results['confusion_matrix'])
        id2label = task_config['labels']['id2label']
        num_classes = len(id2label)
        label_names = [id2label.get(str(i), id2label.get(i, f"class_{i}"))
                      for i in range(num_classes)]

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=label_names,
            y=label_names,
            colorscale='Blues',
            showscale=True
        )

        fig.update_layout(
            title=f'Confusion Matrix - {split_name.capitalize()} Set',
            xaxis_title='Predicted',
            yaxis_title='True',
            height=600,
            width=800
        )

        cm_file = os.path.join(output_dir, f'{split_name}_confusion_matrix.html')
        fig.write_html(cm_file)
        visualizations['confusion_matrix'] = cm_file
        print(f"  ✅ Saved confusion matrix to: {cm_file}")

    except Exception as e:
        print(f"  ⚠️  Could not create confusion matrix: {e}")

    return visualizations


def log_to_wandb(results, visualizations, split_name, task_config, experiment_config, data_config, model_config, seed):
    """Log results to Weights & Biases."""

    if experiment_config['tracking'].get('use_wandb', False) is False:
        return


    group_name = build_wandb_group_name(
        experiment_config,
        task_config,
        model_config,
        data_config,
    )

    run_name = build_wandb_run_name(
        experiment_config,
        seed=seed,
        suffix=f"eval-{split_name}"
    )

    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        name=run_name,
        group=group_name,
        job_type="evaluation",
        tags=["evaluation", split_name],
        config={
            'experiment_name': experiment_config['experiment_name'],
            'task': task_config['task_name'],
            'split': split_name,
            'seed': seed,
        },
        reinit=True,
    )

    # Log overall metrics
    wandb.log({
        f'{split_name}/accuracy': results['overall']['accuracy'],
        f'{split_name}/f1_macro': results['overall']['macro_avg']['f1'],
        f'{split_name}/precision_macro': results['overall']['macro_avg']['precision'],
        f'{split_name}/recall_macro': results['overall']['macro_avg']['recall'],
    })

    # Log multi-annotator metrics if present
    if 'accuracy_lenient' in results['overall']:
        wandb.log({
            f'{split_name}/accuracy_lenient': results['overall']['accuracy_lenient'],
            f'{split_name}/accuracy_majority': results['overall']['accuracy_majority'],
            f'{split_name}/mean_annotator_agreement': results['overall']['mean_annotator_agreement'],
        })

    # Log per-class metrics
    for class_name, metrics in results['per_class'].items():
        wandb.log({
            f'{split_name}/{class_name}/f1': metrics['f1_score'],
            f'{split_name}/{class_name}/precision': metrics['precision'],
            f'{split_name}/{class_name}/recall': metrics['recall'],
        })

    # Log visualizations
    if 'confusion_matrix' in visualizations:
        wandb.log({f'{split_name}/confusion_matrix': wandb.Html(visualizations['confusion_matrix'])})

    print(f"  ✅ Logged to W&B")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on CLARITY dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    parser.add_argument('--split', type=str, choices=['validation', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"CLARITY EVALUATION SCRIPT")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")

    # Load configurations
    configs = load_experiment_config(args.config)
    validate_config(configs)

    experiment_config = configs['experiment']
    model_config = configs['model']
    task_config = configs['task']
    data_config = configs['data']

    # Keep eval consistent with hierarchical training behaviour (filtering, label remap)
    if experiment_config.get("hierarchical", {}).get("enabled", False):
        data_config = dict(data_config)  # shallow copy
        data_config["hierarchical"] = experiment_config["hierarchical"]

    # --- HIERARCHICAL LABEL OVERRIDE (must match scripts/train.py) ---
    hier = experiment_config.get("hierarchical", {}) or {}
    if hier.get("enabled", False):
        stage = int(hier.get("stage", 0) or 0)
        if stage not in (1, 2):
            raise ValueError(f"hierarchical.stage must be 1 or 2, got: {stage}")

        if stage == 1:
            new_labels = {
                "num_labels": 2,
                "label_names": ["NOT_AMBIVALENT", "AMBIVALENT"],
                "label2id": {"NOT_AMBIVALENT": 0, "AMBIVALENT": 1},
                "id2label": {0: "NOT_AMBIVALENT", 1: "AMBIVALENT"},
            }
        else:
            new_labels = {
                "num_labels": 2,
                "label_names": ["CLEAR_REPLY", "CLEAR_NON_REPLY"],
                "label2id": {"CLEAR_REPLY": 0, "CLEAR_NON_REPLY": 1},
                "id2label": {0: "CLEAR_REPLY", 1: "CLEAR_NON_REPLY"},
            }

        task_config = dict(task_config)
        task_config["labels"] = new_labels
    # --- END HIERARCHICAL LABEL OVERRIDE ---

    # Keep eval consistent with training (masking + input format)
    data_config = dict(data_config)  # shallow copy, stops accidental mutation
    data_config['preprocessing'] = experiment_config.get('preprocessing', data_config.get('preprocessing', {}))
    data_config['input_format'] = experiment_config.get('input_format', data_config.get('input_format', 'pair'))
    data_config['cd'] = bool(experiment_config.get('cd', False))

    print(f"Masking mode (eval): {data_config.get('preprocessing', {}).get('masking_mode', 'none')}")
    print(f"Input format (eval): {data_config.get('input_format')}")

    if args.no_wandb:
        experiment_config['tracking']['use_wandb'] = False

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint, 'evaluation')
    os.makedirs(args.output_dir, exist_ok=True)

    seed = extract_seed_from_checkpoint(args.checkpoint)

    # Seeds
    if seed is None:
        print("⚠️  Warning: Could not infer seed from checkpoint path")
    else:
        print(f"Detected seed from checkpoint: {seed}")

    print(f"Output directory: {args.output_dir}")

    # Load dataset (and the exact tokenizer used for tokenisation)
    print(f"\nLoading {args.split} dataset...")
    from transformers import AutoTokenizer

    # Always load the checkpoint tokenizer so special tokens match training
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
        print("✓ Loaded tokenizer from checkpoint (fast)")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
        print("✓ Loaded tokenizer from checkpoint (slow)")

    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(
        data_config, task_config, model_config, tokenizer=tokenizer
    )

    dataset = val_dataset if args.split == 'validation' else test_dataset

    hier = experiment_config.get("hierarchical", {}) or {}
    if hier.get("enabled", False) and int(hier.get("stage", 0) or 0) == 2:
        dist = dataset.get_label_distribution()
        if "AMBIVALENT" in dist:
            print("⚠️  Stage 2 eval dataset still contains AMBIVALENT. Metrics will be invalid.")
            print(f"   Distribution: {dist}")

    print(f"  Dataset size: {len(dataset)}")

    # Load model (resize embeddings using the SAME tokenizer)
    model = load_model_from_checkpoint(
        args.checkpoint, model_config, task_config, tokenizer=tokenizer
    )

    # Detect multi-annotator format
    is_multi_annotator, annotator_columns = detect_multi_annotator_format(
        dataset, data_config, task_config
    )

    # Generate predictions
    predictions, labels = predict_on_dataset(
        model, dataset, device=args.device, batch_size=args.batch_size
    )

    # For multi-annotator case, extract annotator labels
    if is_multi_annotator:
        print(f"\nExtracting labels from {len(annotator_columns)} annotators...")
        annotator_labels = extract_annotator_labels(dataset, annotator_columns, task_config)
        labels = annotator_labels
    else:
        print(f"  Label distribution: {dataset.get_label_distribution()}")

    # Compute metrics
    print(f"\nComputing metrics...")
    results, pred_labels = compute_detailed_metrics(
        predictions, labels, task_config, multi_annotator=is_multi_annotator
    )

    # Print results
    print_detailed_results(results, args.split, multi_annotator=is_multi_annotator)

    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    if not is_multi_annotator:
        visualizations = create_visualizations(
            results, pred_labels, labels, task_config, args.split, args.output_dir
        )
    else:
        visualizations = {}

    # Save results to JSON
    results_file = os.path.join(args.output_dir, f'{args.split}_results.json')
    results_with_metadata = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'timestamp': datetime.now().isoformat(),
        'task': task_config.get('task_name', 'unknown'),
        'model': model_config['model_type'],
        'multi_annotator': is_multi_annotator,
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    print(f"\n  ✅ Saved results to: {results_file}")

    # Log to W&B
    if experiment_config['tracking'].get('use_wandb', False):
        print(f"\n📤 Logging to W&B...")
        try:
            log_to_wandb(results, visualizations, args.split, task_config, experiment_config, data_config, model_config, seed)
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to log to W&B: {e}")

    print(f"\n{'=' * 60}")
    print(f"✅ EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"📁 Results saved to: {args.output_dir}")

    if is_multi_annotator:
        print(f"📊 {args.split.capitalize()} Accuracy (Lenient - matches ANY): {results['overall']['accuracy_lenient']:.4f}")
        print(f"📊 {args.split.capitalize()} F1 (macro): {results['overall']['macro_avg']['f1']:.4f}")
    else:
        print(f"📊 {args.split.capitalize()} Accuracy: {results['overall']['accuracy']:.4f}")
        print(f"📊 {args.split.capitalize()} F1 (macro): {results['overall']['macro_avg']['f1']:.4f}")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()