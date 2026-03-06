# trainers/encoder_single_task_trainer.py
"""
Single-task trainer for encoder models (BERT, RoBERTa, DistilBERT, DeBERTa, ELECTRA).
Supports both weighted and standard cross-entropy loss.

This trainer works for Task A and Task B separately

Configurable loss types:
- Standard: use_weighted_loss=False
- Weighted: use_weighted_loss=True (handles class imbalance)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_trainer import BaseClarityTrainer


class EncoderSingleTaskTrainer(BaseClarityTrainer):
    """
    Single-task trainer for encoder models (BERT-style architectures).

    Design:
    - Works with any encoder model (BERT, RoBERTa, DistilBERT, etc.)
    - Supports both Task A and Task B (task-agnostic)
    - Handles weighted or standard loss via configuration
    - Model does forward pass, trainer computes loss

    Usage:
        # Standard training
        trainer = EncoderSingleTaskTrainer(
            class_weights_config={'use_weighted_loss': False},
            train_dataset=dataset,
            ...
        )

        # Weighted training (for class imbalance)
        trainer = EncoderSingleTaskTrainer(
            class_weights_config={
                'use_weighted_loss': True,
                'method': 'balanced'  # or 'inverse_frequency' or 'manual'
            },
            train_dataset=dataset,
            ...
        )
    """

    def __init__(self, class_weights_config: Dict[str, Any], train_dataset, **kwargs):
        """
        Initialize encoder single-task trainer.

        Args:
            class_weights_config: Configuration for class weighting
                - use_weighted_loss: bool (default: False)
                - method: 'inverse_frequency', 'balanced', 'sqrt', or 'manual' (if weighted)
                - manual_weights: list (if method='manual')
            train_dataset: Training dataset with get_class_weights() method
            **kwargs: Additional arguments passed to HuggingFace Trainer
        """
        super().__init__(train_dataset=train_dataset, **kwargs)

        self.class_weights_config = class_weights_config
        self.use_weighted_loss = class_weights_config.get("use_weighted_loss", False)
        self.class_weights = None

        self.warmup_epochs = int(class_weights_config.get("warmup_epochs", 0) or 0)
        if self.use_weighted_loss and self.warmup_epochs > 0:
            print(f"🔥 Class-weight warmup enabled for {self.warmup_epochs} epoch(s)")

        if self.use_weighted_loss:
            self._setup_class_weights(train_dataset)
            print(f"🔧 Using weighted loss with method: {class_weights_config['method']}")
            print(f"⚖️  Class weights: {self.class_weights.tolist()}")
        else:
            print("Using standard CrossEntropyLoss (no class weighting)")

    def train(self, *args, **kwargs):
        self.freeze_epochs = self.args.freeze_encoder_epochs or 0

        if self.freeze_epochs > 0 and hasattr(self.model, "freeze_encoder"):
            print(f"🔒 Freezing encoder for {self.freeze_epochs} epoch(s)")
            self.model.freeze_encoder()

        return super().train(*args, **kwargs)

    def on_epoch_begin(self):
        if (
            hasattr(self, "freeze_epochs")
            and self.freeze_epochs > 0
            and self.state.epoch == self.freeze_epochs
        ):
            if hasattr(self.model, "unfreeze_encoder"):
                print("🔓 Unfreezing encoder")
                self.model.unfreeze_encoder()

    def _get_weight_alpha(self) -> float:
        """
        Compute interpolation factor for class-weight warmup.

        Returns:
            alpha in [0, 1]
            0   -> uniform loss
            1   -> fully weighted loss
        """
        if self.warmup_epochs <= 0:
            return 1.0

        # HF Trainer stores current epoch as float (can be None early)
        epoch = self.state.epoch
        if epoch is None:
            return 0.0

        return min(1.0, (int(epoch) + 1) / self.warmup_epochs)

    def _setup_class_weights(self, train_dataset):
        """
        Compute class weights based on the selected method.

        Methods:
        - 'inverse_frequency': weight = 1 / frequency
          → More aggressive rebalancing, can be extreme

        - 'balanced': weight = n_samples / (n_classes * class_count)
          → Moderate rebalancing, recommended (scikit-learn default)

        - 'manual': use provided weights
          → Full control, specify exact weights per class
        """
        method = self.class_weights_config["method"]

        if method == "manual":
            manual_weights = self.class_weights_config.get("manual_weights")
            if not manual_weights:
                raise ValueError("Manual weights specified but not provided.")
            self.class_weights = torch.tensor(manual_weights, dtype=torch.float32)

        elif method in ["inverse_frequency", "balanced", "sqrt"]:
            self.class_weights = train_dataset.get_class_weights(method=method)
            distribution = train_dataset.get_label_distribution()
            print(f"📊 Training data distribution: {distribution}")

        else:
            raise ValueError(
                f"Unknown class weighting method: {method}. "
                f"Available: 'inverse_frequency', 'balanced', 'sqrt', 'manual'"
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss for training step.

        This method is called by HuggingFace Trainer during training.
        Workflow:
        1. Extract labels from inputs
        2. Run forward pass WITHOUT labels (prevents model's internal loss)
        3. Compute custom loss (weighted or standard based on config)

        Why we extract labels:
        - Encoder models (AutoModelForSequenceClassification) compute loss
          internally if labels are provided
        - We want to use our custom weighted loss, not the model's
        - So we pass inputs WITHOUT labels to model.forward()

        Args:
            model: The encoder model being trained
            inputs: Dictionary with input_ids, attention_mask, token_type_ids, labels
            return_outputs: Whether to return model outputs alongside loss

        Returns:
            loss (scalar) if return_outputs=False
            (loss, outputs) tuple if return_outputs=True
        """
        # Extract labels - create clean inputs without labels
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # Forward pass WITHOUT labels
        # Model returns logits only (no internal loss computation)
        outputs = model(**model_inputs)

        # Extract logits from model outputs
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = outputs.logits

        # Compute custom loss based on configuration
        if self.use_weighted_loss:
            loss = self._compute_weighted_loss(logits, labels)
        else:
            loss = self._compute_standard_loss(logits, labels)

        if self.use_weighted_loss and self.warmup_epochs > 0:
            if self.state.is_world_process_zero and self.state.global_step % self.args.logging_steps == 0:
                self.log({"class_weight_alpha": self._get_weight_alpha()})

        # Return in format expected by Trainer
        return (loss, outputs) if return_outputs else loss

    def _compute_weighted_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional class-weight warmup.

        Behaviour:
        - When warmup is disabled:
            Uses fixed per-class weights to handle class imbalance.

        - When warmup is enabled:
            Linearly blends between:
            * uniform (unweighted) loss
            * fully weighted loss
            over the first N epochs.

        Intuition:
        Early training is often unstable with aggressive class weighting.
        Warmup allows the model to learn a reasonable representation first,
        then gradually increases the penalty for minority classes.

        Example:
            Full class weights (sqrt method):
                [0.6, 1.0, 1.4]

            Epoch 0 (alpha = 0.5):
                effective weights ≈ [0.8, 1.0, 1.2]

            Epoch >= warmup_epochs:
                effective weights = [0.6, 1.0, 1.4]

        Args:
            logits:
                Model predictions of shape [batch_size, num_classes].
            labels:
                Ground-truth class indices of shape [batch_size].

        Returns:
            Scalar cross-entropy loss with warmup-adjusted class weights.
        """

        device = logits.device
        full_weights = self.class_weights.to(device)

        # Uniform weights = no class reweighting
        uniform_weights = torch.ones_like(full_weights)

        alpha = self._get_weight_alpha()

        # Linear interpolation
        effective_weights = (
            (1.0 - alpha) * uniform_weights
            + alpha * full_weights
        )

        loss_fct = nn.CrossEntropyLoss(weight=effective_weights)
        return loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

    def _compute_standard_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute standard cross-entropy loss (no weighting).

        All classes are treated equally regardless of their frequency.
        Used when use_weighted_loss=False.

        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: True labels [batch_size]

        Returns:
            Standard cross-entropy loss (scalar)
        """
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


# Backward compatibility alias
ClarityWeightedSingleTaskTrainer = EncoderSingleTaskTrainer