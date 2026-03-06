"""
Base classifier interface for CLARITY SemEval-2026 Competition

Design principle: Models only handle forward pass and embedding extraction.
Loss computation is delegated to trainers for maximum flexibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseClassifier(ABC, nn.Module):
    """Abstract base class for all classification models."""

    def __init__(self, model_config: Dict[str, Any], task_config: Dict[str, Any]):
        """
        Initialize base classifier.

        Args:
            model_config: Model configuration dictionary
            task_config: Task configuration dictionary
        """
        super().__init__()
        self.model_config = model_config
        self.task_config = task_config
        self.num_labels = task_config['labels']['num_labels']

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for models that use them)
            labels: Ground truth labels (passed for compatibility, loss handled by trainers)

        Returns:
            Dictionary containing logits and optionally loss
        """
        pass

    @abstractmethod
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get model embeddings (typically [CLS] token representation).

        Used for:
        - Contrastive learning
        - Multi-task learning
        - Feature extraction
        - Similarity computations

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for models that use them)

        Returns:
            Embeddings tensor: [batch_size, hidden_size]
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'num_labels': self.num_labels,
            'model_name': self.model_config.get('model_name', 'unknown'),
            'model_type': self.model_config.get('model_type', 'unknown')
        }