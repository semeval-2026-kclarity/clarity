"""
Configuration loading utilities for CLARITY SemEval-2026 Competition

Features:
- Variable substitution in paths (e.g., {experiment_name})
- Automatic directory creation for outputs, logs, and caches
- Configuration validation
- Numeric field type conversion
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        raw = file.read()

    # Expand ${VARS} from environment (.env, SLURM, shell, etc)
    expanded = os.path.expandvars(raw)

    config = yaml.safe_load(expanded)
    return config


def substitute_variables(config: Dict[str, Any], variables: Dict[str, str]) -> Dict[str, Any]:
    """
    Substitute variables in configuration strings.
    
    Supports patterns like:
    - {experiment_name}
    - {experiment_type}
    - {task_name}
    
    Args:
        config: Configuration dictionary
        variables: Dictionary of variable name -> value mappings
        
    Returns:
        Configuration with substituted values
    """
    def substitute_value(value):
        """Recursively substitute variables in values."""
        if isinstance(value, str):
            # Substitute all variables
            try:
                return value.format(**variables)
            except KeyError:
                # If variable not found, return original
                return value
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value
    
    return substitute_value(config)


def load_experiment_config(experiment_config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load experiment configuration and all referenced configs.
    Supports variable substitution in paths.
    
    Args:
        experiment_config_path: Path to experiment config file
        
    Returns:
        Dictionary with keys: 'experiment', 'model', 'task', 'data'
    """
    # Load main experiment config
    experiment_config = load_config(experiment_config_path)

    # Load referenced configs
    model_config = load_config(experiment_config['model_config'])
    task_config = load_config(experiment_config['task_config'])
    data_config = load_config(experiment_config['data_config'])

    # Create variable substitution dictionary
    variables = {
        'experiment_name': experiment_config.get('experiment_name', 'default_experiment'),
        'experiment_type': experiment_config.get('experiment_type', 'unknown'),
        'task_name': task_config.get('task_name', 'unknown'),
    }
    
    # Substitute variables in experiment config (for paths)
    experiment_config = substitute_variables(experiment_config, variables)
    
    # Also substitute in model config (for cache_dir)
    model_config = substitute_variables(model_config, variables)
    
    # And in data config (for cache_dir)
    data_config = substitute_variables(data_config, variables)

    # Fix numeric string values in training config (e.g., "2e-5" → 0.00002)
    if 'training' in experiment_config:
        training = experiment_config['training']
        numeric_fields = ['learning_rate', 'warmup_ratio', 'weight_decay']
        for field in numeric_fields:
            if field in training and isinstance(training[field], str):
                try:
                    training[field] = float(training[field])
                    print(f"Converting {field} from string '{training[field]}' to float")
                except ValueError:
                    pass

    return {
        'experiment': experiment_config,
        'model': model_config,
        'task': task_config,
        'data': data_config
    }


def validate_config(configs: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate loaded configurations for consistency.

    Args:
        configs: Dictionary of loaded configs

    Returns:
        True if valid, raises ValueError if invalid
    """
    experiment = configs['experiment']
    model = configs['model']
    task = configs['task']
    data = configs['data']

    # Check if model type is supported
    supported_models = ['bert', 'roberta', 'distilbert', 'deberta', 'electra', 't5'] 
    if model['model_type'] not in supported_models:
        raise ValueError(f"Unsupported model type: {model['model_type']}")

    # Check if task type is supported
    if task['task_type'] != 'classification':
        raise ValueError(f"Unsupported task type: {task['task_type']}")

    # Validate class weighting config
    if experiment.get('class_weights', {}).get('use_weighted_loss', False):
        method = experiment['class_weights']['method']
        if method not in ['inverse_frequency', 'balanced', 'sqrt', 'manual']:
            raise ValueError(f"Invalid class weighting method: {method}")

        if method == 'manual' and not experiment['class_weights']['manual_weights']:
            raise ValueError("Manual class weights not provided")

    return True


def create_output_dirs(experiment_config: Dict[str, Any]) -> None:
    """
    Create output directories for experiment.
    Creates all parent directories automatically.

    Args:
        experiment_config: Experiment configuration
    """
    output_dir = experiment_config['output_dir']
    logging_dir = experiment_config['logging_dir']

    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logging directory
    os.makedirs(logging_dir, exist_ok=True)
    
    # Create checkpoints directory (may not be needed with new checkpoint strategy)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    print(f"\n📁 Output directories created:")
    print(f"  ✅ Main: {output_dir}")
    print(f"  ✅ Logs: {logging_dir}")


def setup_all_directories(configs: Dict[str, Dict[str, Any]]) -> None:
    """
    Setup all required directories for the experiment.
    Creates directories if they don't exist, including:
    - Output directory
    - Logging directory
    - Model cache directory
    - Data cache directory (if specified)
    
    Args:
        configs: Dictionary with all configs (experiment, model, task, data)
    """
    directories_created = []
    
    # 1. Create output directory and subdirectories
    output_dir = configs['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    directories_created.append(('Output', output_dir))
    
    # 2. Create logging directory
    if 'logging_dir' in configs['experiment']:
        logging_dir = configs['experiment']['logging_dir']
        os.makedirs(logging_dir, exist_ok=True)
        directories_created.append(('Logging', logging_dir))
    
    # 3. Create evaluation directory
    eval_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    directories_created.append(('Evaluation', eval_dir))
    
    # 4. Create model cache directory
    if 'loading' in configs['model'] and 'cache_dir' in configs['model']['loading']:
        cache_dir = configs['model']['loading']['cache_dir']
        os.makedirs(cache_dir, exist_ok=True)
        directories_created.append(('Model cache', cache_dir))
    
    # 5. Create data cache directory (if specified)
    if 'cache_dir' in configs['data']:
        data_cache_dir = configs['data']['cache_dir']
        os.makedirs(data_cache_dir, exist_ok=True)
        directories_created.append(('Data cache', data_cache_dir))
    
    # Print summary
    print(f"\n📁 Directory setup complete:")
    for name, path in directories_created:
        print(f"  ✅ {name:15s}: {path}")


def get_effective_paths(configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Get all effective paths after variable substitution.
    Useful for debugging and logging.
    
    Args:
        configs: Dictionary with all configs
        
    Returns:
        Dictionary of path name -> resolved path
    """
    paths = {
        'output_dir': configs['experiment']['output_dir'],
        'logging_dir': configs['experiment'].get('logging_dir', 'N/A'),
    }
    
    if 'loading' in configs['model'] and 'cache_dir' in configs['model']['loading']:
        paths['model_cache'] = configs['model']['loading']['cache_dir']
    
    if 'cache_dir' in configs['data']:
        paths['data_cache'] = configs['data']['cache_dir']
    
    return paths


# Export functions
__all__ = [
    'load_config',
    'load_experiment_config',
    'validate_config',
    'create_output_dirs',
    'setup_all_directories',
    'substitute_variables',
    'get_effective_paths'
]
