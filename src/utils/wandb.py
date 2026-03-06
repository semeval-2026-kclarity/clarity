"""
Utilities for consistent Weights & Biases (W&B) setup.

The goal of this file is to define experiment *identity* exactly once.
Both training and evaluation should use the same group naming logic so
runs end up grouped correctly in the W&B UI.

If you change grouping logic, do it here - nowhere else.
"""

def build_wandb_group_name(
    experiment_config,
    task_config,
    model_config,
    data_config,
):
    """
    Build a stable W&B group name that represents an experiment setup,
    independent of seed.

    A group corresponds to:
    - same task
    - same model type
    - same data split method
    - same preprocessing / masking
    - same class weighting strategy
    - same seed mode (single vs multi)

    Seeds should NEVER be part of the group name.
    """

    experiment_type = experiment_config.get("experiment_type", "unknown")
    task_name = task_config.get("task_name", "unknown")
    model_name = model_config.get("model_name", "unknown")

    init_from_ckpt = model_config.get("init_from_checkpoint")
    init_suffix = "_init-aux" if init_from_ckpt else ""

    split_method = data_config.get("splitting", {}).get("method", "unknown")

    input_format = experiment_config.get("input_format", "pair")

    preprocessing = experiment_config.get("preprocessing", {})
    masking_mode = preprocessing.get("masking_mode", "none")

    class_weights_cfg = experiment_config.get("class_weights", {})
    use_weights = class_weights_cfg.get("use_weighted_loss", False)
    weight_method = class_weights_cfg.get("method", "unweighted")

    warmup_epochs = class_weights_cfg.get("warmup_epochs", 0)

    seed_mode = experiment_config.get("seed_control", {}).get("mode", "single")

    cd = experiment_config.get("cd", False)

    # Hierarchical Task A - keep stage runs grouped separately
    hier_cfg = experiment_config.get("hierarchical", {}) or {}
    if hier_cfg.get("enabled", False):
        hier_stage = hier_cfg.get("stage", "unknown")
        hier_suffix = f"_hier-stage{hier_stage}"
    else:
        hier_suffix = ""

    masking_str = f"{masking_mode}_" if masking_mode != "none" else ""

    if not use_weights:
        weight_str = "unweighted"
    elif warmup_epochs and warmup_epochs > 0:
        weight_str = f"{weight_method}_warmup{warmup_epochs}"
    else:
        weight_str = weight_method

    group_name = (
        f"{experiment_type}_"
        f"{task_name}_"
        f"{task_name}{hier_suffix}_"
        f"{model_name}"
        f"{init_suffix}_"
        f"{split_method}_"
        f"fmt-{input_format}_"
        f"{masking_str}"
        f"{weight_str}_"
        f"{seed_mode}"
    )

    if cd:
        group_name += "_cd"

    return group_name



def build_wandb_run_name(experiment_config, seed=None, suffix=None):
    """
    Build a human-readable W&B run name.

    This is allowed to include seed and other run-specific details.
    """

    base_name = experiment_config.get("experiment_name", "experiment")

    if seed is not None:
        base_name = f"{base_name}-seed{seed}"

    if suffix:
        base_name = f"{base_name}-{suffix}"

    return base_name
