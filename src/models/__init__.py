"""
Models for CLARITY SemEval-2026 Competition
"""

from typing import Dict, Any
from .base_classifier import BaseClassifier
from .roberta_classifier import RobertaClassifier
from .deberta_classifier import DebertaClassifier

# Available models for the competition
AVAILABLE_MODELS = {
    'bert': BertClassifier,
    'roberta': RobertaClassifier,
    'distilbert': DistilBertClassifier,
    'deberta': DebertaClassifier,
    'electra': ElectraClassifier,
    't5': T5Classifier,
    # 'xlm-roberta': XlmRobertaClassifier,
    # 'albert': AlbertClassifier,
}



def create_model(model_config: Dict[str, Any], task_config: Dict[str, Any]) -> BaseClassifier:
    """
    Factory function to create models based on configuration.

    Args:
        model_config: Model configuration dictionary
        task_config: Task configuration dictionary

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_config.get('model_type', '').lower()

    if model_type not in AVAILABLE_MODELS:
        available_models = list(AVAILABLE_MODELS.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {available_models}"
        )

    model_class = AVAILABLE_MODELS[model_type]
    return model_class(model_config, task_config)


# Export main classes and functions
__all__ = [
    'BaseClassifier',
    'RobertaClassifier',
    'DebertaClassifier',
    'create_model',
    'AVAILABLE_MODELS'
]