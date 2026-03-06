# src/utils/metrics.py
"""
Unified metrics computation for CLARITY SemEval-2026 Competition

Single source of truth for all evaluation metrics.
Used by both train.py (basic metrics) and evaluate.py (detailed metrics).

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def compute_multi_annotator_accuracy(predictions: np.ndarray,
                                     annotator_labels: List[np.ndarray]) -> float:
    """
    Compute accuracy where prediction is correct if it matches ANY annotator.

    For Task B test set: each sample has 3 annotators, prediction is correct
    if it matches at least one of them.

    Args:
        predictions: Predicted class IDs [n_samples]
        annotator_labels: List of label arrays, one per annotator [n_annotators x n_samples]

    Returns:
        Accuracy (fraction of samples where prediction matches at least one annotator)
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    n_samples = len(predictions)
    n_correct = 0

    for i in range(n_samples):
        pred = predictions[i]
        # Check if prediction matches any annotator
        matches_any = any(pred == annotator[i] for annotator in annotator_labels)
        if matches_any:
            n_correct += 1

    return n_correct / n_samples


def compute_multi_annotator_metrics(
        predictions: np.ndarray,
        annotator_labels: List[np.ndarray],
        label_names: List[str]
) -> Dict[str, Any]:
    """
    Compute metrics for multi-annotator evaluation.

    Uses a simple majority voting approach:
    - For each sample, determine the majority label from the 3 annotators
    - Compute standard metrics using these majority labels
    - Additionally report lenient accuracy (matches ANY annotator)

    Args:
        predictions: Predicted class IDs [n_samples]
        annotator_labels: List of 3 label arrays [3 x n_samples]
        label_names: List of label names

    Returns:
        Dictionary with standard metrics plus lenient accuracy
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    n_samples = len(predictions)
    n_annotators = len(annotator_labels)

    # 1. Compute lenient accuracy (matches ANY annotator)
    lenient_accuracy = compute_multi_annotator_accuracy(predictions, annotator_labels)

    # 2. Create majority vote labels for standard metrics
    # For each sample, use the most common label from the 3 annotators
    # If there's a tie, use the first annotator's label
    majority_labels = []
    for i in range(n_samples):
        labels_for_sample = [annotator[i] for annotator in annotator_labels]

        # Count occurrences
        from collections import Counter
        label_counts = Counter(labels_for_sample)

        # Get majority label (most common)
        majority_label = label_counts.most_common(1)[0][0]
        majority_labels.append(majority_label)

    majority_labels = np.array(majority_labels)

    # 3. Compute standard metrics using majority labels
    # These are the "reference" metrics for comparison
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        majority_labels, predictions, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        majority_labels, predictions, average='weighted', zero_division=0
    )

    # Accuracy using majority labels
    majority_accuracy = accuracy_score(majority_labels, predictions)

    # 4. Compute per-class metrics using majority labels
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        majority_labels, predictions, average=None, zero_division=0
    )

    per_class = {}
    for i, label_name in enumerate(label_names):
        if i < len(precision_per_class):
            per_class[label_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }

    # 5. Compute confusion matrix using majority labels
    cm = confusion_matrix(majority_labels, predictions, labels=range(len(label_names)))

    # 6. Compute annotator agreement statistics
    agreement_scores = []
    for i in range(n_samples):
        labels_for_sample = [annotator[i] for annotator in annotator_labels]
        # All 3 annotators agree
        if len(set(labels_for_sample)) == 1:
            agreement_scores.append(1.0)
        # 2 out of 3 agree
        elif len(set(labels_for_sample)) == 2:
            agreement_scores.append(2 / 3)
        # All different
        else:
            agreement_scores.append(0.0)

    mean_agreement = np.mean(agreement_scores)

    # 7. Build results dictionary
    results = {
        'overall': {
            'accuracy': lenient_accuracy,  # PRIMARY METRIC: matches ANY annotator
            'accuracy_lenient': lenient_accuracy,  # Explicit name
            'accuracy_majority': majority_accuracy,  # Secondary: using majority vote
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'mean_annotator_agreement': mean_agreement
        },
        'per_class': per_class,
        'confusion_matrix': cm,
        'annotator_stats': {
            'n_annotators': n_annotators,
            'mean_agreement': mean_agreement,
            'full_agreement_rate': np.mean([s == 1.0 for s in agreement_scores]),
            'partial_agreement_rate': np.mean([s == 2 / 3 for s in agreement_scores]),
            'no_agreement_rate': np.mean([s == 0.0 for s in agreement_scores])
        }
    }

    return results


def compute_basic_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute basic metrics for validation during training.

    Used by train.py for quick validation metrics.

    Args:
        predictions: Predicted class IDs (already argmax-ed) or logits
        labels: True class IDs

    Returns:
        Dictionary with accuracy, f1_macro, precision_macro, recall_macro
    """
    # Handle logits (if predictions are 2D)
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    # Compute metrics with zero_division handling
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }


def compute_per_class_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        label_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1, and support.

    Args:
        predictions: Predicted class IDs
        labels: True class IDs
        label_names: List of label names in order

    Returns:
        Dictionary mapping label_name -> {precision, recall, f1_score, support}
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,  # Get per-class metrics
        zero_division=0
    )

    # Build dictionary
    per_class = {}
    for i, label_name in enumerate(label_names):
        per_class[label_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }

    return per_class


def compute_weighted_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute weighted average metrics (weighted by support).

    Args:
        predictions: Predicted class IDs
        labels: True class IDs

    Returns:
        Dictionary with weighted precision, recall, f1
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    return {
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1)
    }


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class IDs
        labels: True class IDs

    Returns:
        Confusion matrix as numpy array
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    return confusion_matrix(labels, predictions)


def get_classification_report(
        predictions: np.ndarray,
        labels: np.ndarray,
        label_names: List[str],
        digits: int = 4
) -> str:
    """
    Get sklearn classification report as string.

    Args:
        predictions: Predicted class IDs
        labels: True class IDs
        label_names: List of label names
        digits: Number of decimal places

    Returns:
        Classification report string
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    return classification_report(
        labels,
        predictions,
        target_names=label_names,
        digits=digits,
        zero_division=0
    )


def compute_all_metrics(
        predictions: np.ndarray,
        labels: Union[np.ndarray, List[np.ndarray]],
        label_names: List[str],
        include_confusion_matrix: bool = True,
        multi_annotator: bool = False
) -> Dict[str, Any]:
    """
    Compute ALL metrics in one go.

    This is the main function that combines everything.
    Used by evaluate.py for comprehensive evaluation.

    Args:
        predictions: Predicted class IDs or logits
        labels: True class IDs OR list of annotator label arrays (for multi-annotator)
        label_names: List of label names in order
        include_confusion_matrix: Whether to compute confusion matrix
        multi_annotator: If True, labels is a list of annotator arrays

    Returns:
        Dictionary with all metrics
    """
    if multi_annotator:
        # Use multi-annotator evaluation
        if not isinstance(labels, list):
            raise ValueError("For multi_annotator=True, labels must be a list of annotator arrays")

        return compute_multi_annotator_metrics(predictions, labels, label_names)

    else:
        # Standard single-label evaluation
        # Handle logits
        if len(predictions.shape) > 1:
            predictions_argmax = np.argmax(predictions, axis=1)
        else:
            predictions_argmax = predictions

        # Compute all metrics
        basic = compute_basic_metrics(predictions_argmax, labels)
        weighted = compute_weighted_metrics(predictions_argmax, labels)
        per_class = compute_per_class_metrics(predictions_argmax, labels, label_names)

        result = {
            'overall': {
                **basic,
                **weighted
            },
            'per_class': per_class
        }

        if include_confusion_matrix:
            result['confusion_matrix'] = compute_confusion_matrix(predictions_argmax, labels)

        return result


def format_metrics_for_trainer(
        predictions: np.ndarray,
        labels: np.ndarray,
        label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Format metrics for HuggingFace Trainer's compute_metrics callback.

    This is what train.py should use in its compute_metrics function.
    Returns flat dictionary that Trainer can log.

    Args:
        predictions: Predicted class IDs or logits
        labels: True class IDs
        label_names: Optional list of label names for per-class metrics

    Returns:
        Flat dictionary suitable for Trainer logging
    """
    # Handle logits
    if len(predictions.shape) > 1:
        predictions_argmax = np.argmax(predictions, axis=1)
    else:
        predictions_argmax = predictions

    # Start with basic metrics
    metrics = compute_basic_metrics(predictions_argmax, labels)

    # Add weighted metrics
    weighted = compute_weighted_metrics(predictions_argmax, labels)
    metrics.update(weighted)

    # Add per-class metrics if label names provided
    if label_names:
        per_class = compute_per_class_metrics(predictions_argmax, labels, label_names)

        # Flatten per-class metrics for Trainer logging
        for label_name, class_metrics in per_class.items():
            # Convert to snake_case for logging
            label_key = label_name.lower().replace(' ', '_')

            for metric_name, metric_value in class_metrics.items():
                metrics[f'{label_key}/{metric_name}'] = metric_value

    return metrics


def compute_metrics_for_classification(
        predictions: np.ndarray,
        labels: np.ndarray,
        label_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics for evaluation after inference.

    Used by zero_shot.py to evaluate predictions after ICL inference.

    Args:
        predictions: Predicted label IDs (numpy array)
        labels: True label IDs (numpy array)
        label_names: Optional list of label names for per-class metrics

    Returns:
        Dictionary of metrics
    """
    # Handle logits if provided
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    # Compute overall metrics
    accuracy = accuracy_score(labels, predictions)

    # Compute precision, recall, F1 for macro and weighted averaging
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # Build metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
    }

    # Add per-class metrics if label names provided
    if label_names:
        # Compute per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        for i, label_name in enumerate(label_names):
            if i < len(precision_per_class):
                # Use label name as key
                label_key = label_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')

                metrics[f'{label_key}/precision'] = float(precision_per_class[i])
                metrics[f'{label_key}/recall'] = float(recall_per_class[i])
                metrics[f'{label_key}/f1_score'] = float(f1_per_class[i])
                metrics[f'{label_key}/support'] = int(support_per_class[i])

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


# Export all functions
__all__ = [
    'compute_basic_metrics',
    'compute_per_class_metrics',
    'compute_weighted_metrics',
    'compute_confusion_matrix',
    'get_classification_report',
    'compute_all_metrics',
    'format_metrics_for_trainer',
    'compute_metrics_for_classification',
    'compute_multi_annotator_accuracy',
    'compute_multi_annotator_metrics'
]