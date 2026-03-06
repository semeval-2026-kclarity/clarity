# trainers/__init__.py
"""
Factory for creating CLARITY trainers.
Supports encoder models (BERT, RoBERTa, etc.) and encoder-decoder models (T5).

Trainer hierarchy:
- EncoderSingleTaskTrainer: For BERT, RoBERTa, etc. (supports weighted/standard)
- T5SingleTaskTrainer: For T5, Flan-T5 (text generation)
- Future: MultiTaskTrainer, ContrastiveTrainer, etc.
"""

from transformers import TrainingArguments, EarlyStoppingCallback
from .base_trainer import BaseClarityTrainer
from .encoder_single_task_trainer import EncoderSingleTaskTrainer
from .t5_single_task_trainer import T5SingleTaskTrainer
from src.utils.callbacks import CompactSummaryCallback

__all__ = [
    "BaseClarityTrainer",
    "EncoderSingleTaskTrainer",
    "T5SingleTaskTrainer",
    "create_trainer",
]


def create_trainer(configs, model, train_dataset, eval_dataset, tokenizer, seed):
    """
    Factory method to create the appropriate Trainer class.

    Trainer selection logic:
    1. If model is T5 → T5SingleTaskTrainer
    2. If model is encoder (BERT, RoBERTa, etc.) → EncoderSingleTaskTrainer
       - EncoderSingleTaskTrainer handles both weighted and standard loss
       - Configured via class_weights_config
    """
    experiment_config = configs["experiment"]
    model_config = configs["model"]
    task_config = configs["task"]
    class_weights_config = experiment_config.get("class_weights", {})
    training_config = experiment_config["training"].copy()

    training_config = experiment_config["training"].copy()

    # Remove non-TrainingArguments keys
    training_config.pop("init_from_checkpoint", None)
    freeze_epochs = training_config.pop("freeze_encoder_epochs", 0)

    # Determine model architecture type
    model_type = model_config.get("model_type", "").lower()

    # Select trainer based on model architecture
    if model_type == "t5":
        trainer_type = "t5"
        print("📝 Model architecture: Encoder-Decoder (T5)")
    else:
        trainer_type = "encoder"
        print("📝 Model architecture: Encoder-only (BERT-style)")

    # Build TrainingArguments
    if isinstance(training_config.get("learning_rate"), str):
        training_config["learning_rate"] = float(training_config["learning_rate"])

    # For T5 models, disable safetensors to avoid shared memory issues
    if model_type == "t5":
        training_config["save_safetensors"] = False
        print("📝 Disabled safetensors for T5 model (using PyTorch format)")

    # Extract early stopping config (not a TrainingArguments parameter)
    early_stopping_patience = training_config.pop('early_stopping_patience', None)
    early_stopping_threshold = training_config.pop('early_stopping_threshold', 0.0)

    training_args = TrainingArguments(
        output_dir=experiment_config["output_dir"],
        logging_dir=experiment_config["logging_dir"],
        seed=seed,
        data_seed=seed,
        **training_config,
    )

    # Attach custom attribute (HF allows this)
    training_args.freeze_encoder_epochs = experiment_config["training"].get(
        "freeze_encoder_epochs", 0
    )


    # Trainer selection map
    trainer_map = {
        "encoder": EncoderSingleTaskTrainer,
        "t5": T5SingleTaskTrainer,
    }

    TrainerClass = trainer_map.get(trainer_type)
    if TrainerClass is None:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available: {list(trainer_map.keys())}"
        )

    print(f"Creating {trainer_type} trainer...")

    # Prepare callbacks list
    callbacks = []

    try:
        callbacks.append(CompactSummaryCallback())
        print(f"✅ Compact table visualization enabled")
    except ImportError:
        print(f"⚠️  Warning: Could not import CompactSummaryCallback from src.utils.callbacks")

    # Add EarlyStoppingCallback if early_stopping_patience is specified
    if early_stopping_patience is not None:
        print(f"✅ Early stopping enabled with patience: {early_stopping_patience} epochs")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        )

    # Prepare extra kwargs based on trainer type
    extra_kwargs = {}

    if trainer_type == "encoder":
        # EncoderSingleTaskTrainer handles both weighted and standard loss
        extra_kwargs["class_weights_config"] = class_weights_config

        # Print loss configuration
        use_weighted = class_weights_config.get("use_weighted_loss", False)
        if use_weighted:
            method = class_weights_config.get("method", "unknown")
            print(f"✅ Loss: Weighted CrossEntropy (method: {method})")
        else:
            print(f"✅ Loss: Standard CrossEntropy")

    elif trainer_type == "t5":
        extra_kwargs["label2id"] = task_config["labels"]["label2id"]
        extra_kwargs["id2label"] = task_config["labels"]["id2label"]
        print(f"✅ Loss: Text generation loss with constrained decoding")

    # Add callbacks to all trainers
    if callbacks:
        extra_kwargs["callbacks"] = callbacks
        print(f"✅ Trainer created with {len(callbacks)} callback(s)")

    return TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        **extra_kwargs,
    )