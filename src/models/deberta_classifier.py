"""
DeBERTa Classifier implementation for CLARITY SemEval-2026 Competition

Design principle: Model only handles forward pass.
Loss computation is handled by trainers for flexibility:
- Standard loss → BaseClarityTrainer
- Weighted loss → ClarityWeightedSingleTaskTrainer
- Multi-task loss → MultiTaskTrainer
- Contrastive loss → ContrastiveLearningTrainer
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from typing import Dict, Any, Optional


class DebertaClassifier(nn.Module):
    """Simple DeBERTa classifier using HuggingFace AutoModel."""

    def __init__(self, model_config: Dict[str, Any], task_config: Dict[str, Any]):
        """
        Initialize DeBERTa classifier.

        Args:
            model_config: Model configuration from deberta_v3_base.yaml or deberta_v3_large.yaml
            task_config: Task configuration from task_a.yaml
        """
        super().__init__()
        self.model_config = model_config
        self.task_config = task_config
        self.num_labels = task_config['labels']['num_labels']

        # DeBERTa doesn't support classifier_dropout parameter
        # It only uses hidden_dropout_prob and attention_probs_dropout_prob
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_config['model_name'],
            num_labels=self.num_labels,
            hidden_dropout_prob=model_config['model_params']['hidden_dropout_prob'],
            attention_probs_dropout_prob=model_config['model_params']['attention_probs_dropout_prob'],
            cache_dir=model_config['loading'].get('cache_dir')
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DeBERTa.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels (passed for compatibility but loss ignored by custom trainers)

        Returns:
            Model outputs with logits (and optionally loss, though custom trainers ignore it)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get DeBERTa embeddings (useful for contrastive learning, multi-task, etc.)

        Returns:
            [CLS] token embeddings: [batch_size, hidden_size]
        """
        outputs = self.model.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Return [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]