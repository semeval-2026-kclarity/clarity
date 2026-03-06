"""
RoBERTa Classifier implementation for CLARITY SemEval-2026 Competition

Design principle: Model only handles forward pass.
Loss computation is handled by trainers for flexibility:
- Standard loss → BaseClarityTrainer
- Weighted loss → ClarityWeightedSingleTaskTrainer
- Multi-task loss → MultiTaskTrainer
- Contrastive loss → ContrastiveLearningTrainer
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from typing import Dict, Any, Optional


class RobertaClassifier(nn.Module):
    """Simple RoBERTa classifier using HuggingFace AutoModel."""

    def __init__(self, model_config: Dict[str, Any], task_config: Dict[str, Any]):
        """
        Initialize RoBERTa classifier.

        Args:
            model_config: Model configuration from roberta_base.yaml
            task_config: Task configuration from task_a.yaml
        """
        super().__init__()
        self.model_config = model_config
        self.task_config = task_config
        self.num_labels = task_config['labels']['num_labels']

        # Optional intermediate-task initialisation
        init_ckpt = model_config.get("init_from_checkpoint")

        if init_ckpt is not None and not os.path.exists(init_ckpt):
            raise ValueError(f"init_from_checkpoint path does not exist: {init_ckpt}")

        # Decide where to load weights from:
        # - init_from_checkpoint if provided (intermediate-task transfer)
        # - otherwise base pretrained model
        model_source = init_ckpt if init_ckpt is not None else model_config['model_name']

        print(f"🧠 Initialising RoBERTa from: {model_source}")

        ignore_mismatched_sizes = init_ckpt is not None

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=self.num_labels,
            hidden_dropout_prob=model_config['model_params']['hidden_dropout_prob'],
            attention_probs_dropout_prob=model_config['model_params']['attention_probs_dropout_prob'],
            classifier_dropout=model_config['model_params']['classifier_dropout'],
            cache_dir=model_config['loading'].get('cache_dir'),
            ignore_mismatched_sizes=ignore_mismatched_sizes,  # IMPORTANT
        )

        if init_ckpt is not None:
            print("🔁 Resetting classifier head for Task A")

            for module in self.model.classifier.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass through RoBERTa.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels (passed for compatibility but loss ignored by custom trainers)

        Returns:
            Model outputs with logits (and optionally loss, though custom trainers ignore it)
        """
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # BigBird/bert-like models sometimes use this, RoBERTa ignores it
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        return self.model(**kwargs)

    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get encoder embeddings (useful for contrastive learning, multi-task, etc.)

        Returns:
            [CLS] token embeddings: [batch_size, hidden_size]
        """
        # works for roberta, bigbird, deberta, etc
        base_attr = getattr(self.model, "base_model_prefix", None)

        if base_attr is None:
            raise RuntimeError("Model does not expose base_model_prefix, can't grab encoder outputs cleanly")

        encoder = getattr(self.model, base_attr)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = encoder(**kwargs)

        return outputs.last_hidden_state[:, 0, :]

    def gradient_checkpointing_enable(self, **kwargs):
    # HF Trainer expects this method to exist on the top-level model
        self.model.gradient_checkpointing_enable(**kwargs)
        self.model.config.use_cache = False  # required for checkpointing