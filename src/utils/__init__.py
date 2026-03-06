"""
Utility functions for CLARITY SemEval-2026 Competition
"""

from .config import (
    load_config,
    load_experiment_config,
    validate_config,
    create_output_dirs,
    setup_all_directories,
    substitute_variables,
    get_effective_paths
)

from .metrics import (
    compute_basic_metrics,
    compute_per_class_metrics,
    compute_weighted_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_all_metrics,
    format_metrics_for_trainer
)

from .callbacks import CompactSummaryCallback

from .constrained_generation import (
    ConstrainedLabelsLogitsProcessor,
    MultiTokenConstrainedProcessor,
    get_constrained_generation_config
)

from .masking import (
    mask_person_entities,
    mask_person_entities_pair
)

from .wandb import (
    build_wandb_group_name,
    build_wandb_run_name
)

from .tokenizer_specials import (
    build_clarity_special_tokens,
    add_clarity_special_tokens
)


__all__ = [
    # Config utilities
    'load_config',
    'load_experiment_config',
    'validate_config',
    'create_output_dirs',
    'setup_all_directories',
    'substitute_variables',
    'get_effective_paths',
    # Metrics
    'compute_basic_metrics',
    'compute_per_class_metrics',
    'compute_weighted_metrics',
    'compute_confusion_matrix',
    'get_classification_report',
    'compute_all_metrics',
    'format_metrics_for_trainer',
    # Callbacks
    'CompactSummaryCallback',
    # Masking
    'mask_person_entities',
    'mask_person_entities_pair',
    # wandb
    'build_wandb_group_name',
    'build_wandb_run_name'
    #
    'build_clarity_special_tokens',
    'add_clarity_special_tokens'
]