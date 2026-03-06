"""
Data loading utilities for CLARITY SemEval-2026 Competition
Updated for model-agnostic architecture
"""

from typing import Dict, Any
from .clarity_dataset import ClarityDataset, TokenizedDataset, create_datasets


def get_class_weights(train_dataset, method: str = "inverse_frequency") -> Dict[str, Any]:
    """
    Get class weights and related information.

    Args:
        train_dataset: Training dataset (ClarityDataset or TokenizedDataset)
        method: Weighting method ("inverse_frequency" or "balanced")

    Returns:
        Dictionary with weights and distribution info
    """
    weights = train_dataset.get_class_weights(method=method)
    distribution = train_dataset.get_label_distribution()

    return {
        'weights': weights,
        'method': method,
        'distribution': distribution,
        'weights_list': weights.tolist()
    }


# Export main functions and classes
__all__ = [
    'ClarityDataset',
    'TokenizedDataset',
    'create_datasets',
    'get_class_weights'
]