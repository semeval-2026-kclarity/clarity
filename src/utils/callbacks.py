# src/utils/callbacks.py
"""
Training callbacks for CLARITY SemEval-2026 Competition
"""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Dict, Any


class CompactSummaryCallback(TrainerCallback):
    """
    Ultra-compact: one line per epoch with key metrics.
    Used by default in all trainers.
    """

    def __init__(self):
        self.best_f1 = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, **kwargs):
        """Print header."""
        print(f"\n{'='*110}")
        print(f"Epoch │   Loss │    Acc │  F1-M │ Clear-R │  Ambiv │ Clear-NR │ Status")
        print(f"{'-'*110}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, metrics=None, **kwargs):
        """Print one line."""
        if metrics is None:
            return

        epoch = metrics.get('epoch', 0)
        loss = metrics.get('eval_loss', 0)
        acc = metrics.get('eval_accuracy', 0)
        f1 = metrics.get('eval_f1_macro', 0)
        cr_f1 = metrics.get('eval_clear_reply/f1_score', 0)
        amb_f1 = metrics.get('eval_ambivalent/f1_score', 0)
        cnr_f1 = metrics.get('eval_clear_non_reply/f1_score', 0)

        is_best = self.best_f1 is None or f1 > self.best_f1
        if is_best:
            self.best_f1 = f1

        status = "✅ BEST" if is_best else ""

        print(f" {epoch:>4.1f} │ {loss:>6.4f} │ {acc:>6.4f} │ {f1:>5.4f} │ "
              f" {cr_f1:>6.4f} │ {amb_f1:>6.4f} │   {cnr_f1:>7.4f} │ {status}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Print summary."""
        print(f"{'-'*110}")
        print(f"✅ Done! Best F1: {self.best_f1:.4f}\n")


__all__ = ['CompactSummaryCallback']