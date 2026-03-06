# trainers/base_trainer.py
"""
Base trainer for CLARITY SemEval-2026 Competition.
Provides shared utilities, device logging, and helper loss methods.
"""

import torch
import torch.nn as nn
from transformers import Trainer


class BaseClarityTrainer(Trainer):
    """Base class for all CLARITY custom trainers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_device_info()

    def _log_device_info(self):
        """Log basic device info."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU available: {device_name}")
            print(f"Device count: {device_count}, Current device: {current_device}")
            print(f"GPU memory: {memory_gb:.1f} GB")
        else:
            print("Using CPU - no GPU available")

    def compute_standard_loss(self, logits, labels):
        """Standard CrossEntropyLoss helper."""
        loss_fct = nn.CrossEntropyLoss()
        num_labels = logits.size(-1)
        return loss_fct(logits.view(-1, num_labels), labels.view(-1))
