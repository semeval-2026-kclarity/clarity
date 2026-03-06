"""
CLARITY Dataset implementation for SemEval-2026 Competition - Model Agnostic
Supports multiple splitting strategies: random, stratified, president_based
"""

import os
from datasets import load_from_disk
import pandas as pd
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from collections import Counter
from itertools import combinations
from typing import Dict, Any, Tuple, List
import torch
from torch.utils.data import Dataset as TorchDataset
import hashlib
import json
from sklearn.model_selection import train_test_split
from src.utils.masking import (
    mask_person_entities,
    mask_person_entities_pair
)


class ClarityDataset(TorchDataset):
    """Model-agnostic dataset class for CLARITY competition with multiple splitting strategies."""

    # Class-level cache for splits (shared across all instances)
    _splits_cache = {}

    def __init__(self, data_config: Dict[str, Any], task_config: Dict[str, Any], split: str = "train"):
        """
        Initialize CLARITY dataset.

        Args:
            data_config: Data configuration from configs/data/*.yaml
            task_config: Task configuration from configs/tasks/*.yaml
            split: Dataset split ("train", "validation", "test")
        """
        self.data_config = data_config
        self.task_config = task_config
        self.split = split

        # Masking flags
        self.preprocessing = self.data_config.get('preprocessing', {})
        self.masking_mode = self.preprocessing.get('masking_mode', 'none')

        # How should inputs be passed into the model?
        self.input_format = self.data_config.get("input_format", "pair")

        # CD prefix (hardcoded behaviour, controlled by experiment_config['cd'])
        # We assume the annotated CSV has: cd_bucket in {"CD_LOW", "CD_HIGH"}
        self.cd_enabled = bool(self.data_config.get("cd", False))
        self._cd_col = "cd_bucket"
        self._cd_default = "CD_LOW"

        # Hierarchical Task A config (optional)
        self.hier_cfg = (self.data_config.get("hierarchical", {}) or {})
        self.hier_enabled = bool(self.hier_cfg.get("enabled", False))
        self.hier_stage = int(self.hier_cfg.get("stage", 0) or 0)

        self.hier_amb_label = self.hier_cfg.get("ambivalent_label", "Ambivalent")
        self.hier_clear_label = self.hier_cfg.get("clear_label", "Clear Reply")
        self.hier_nonreply_label = self.hier_cfg.get("nonreply_label", "Clear Non-Reply")

        # Store raw data for multi-annotator detection (Task B)
        self.data = None  # Will be set in _load_and_split_dataset if needed

        # Load and process dataset
        self.dataset = self._load_and_split_dataset()

        # Get label mappings
        self.label2id = task_config['labels']['label2id']
        self.id2label = task_config['labels']['id2label']

        # Check if this is multi-annotator format (Task B)
        self.is_multi_annotator = self._check_multi_annotator_format()

        # Keep original mapping around (useful for debugging)
        self.orig_label2id = task_config['labels']['label2id']
        self.orig_id2label = task_config['labels']['id2label']

        # When hierarchical is enabled, the train script overrides task_config labels to binary.
        # So at this point task_config['labels'] is already stage-specific and safe.
        self.label2id = task_config['labels']['label2id']
        self.id2label = task_config['labels']['id2label']

    def _create_cache_key(self) -> str:
        """Create a hashable cache key from config dictionaries."""
        key_data = {
            'dataset_name': self.data_config['dataset']['name'],
            'split_method': self.data_config['splitting'].get('method', 'random'),
            'val_ratio': self.data_config['splitting']['val_ratio'],
            'label_column': self.task_config['data']['label_column'],
            'random_seed': self.data_config['splitting'].get('random_seed', 42),
            'dual_stratification': self.data_config['splitting'].get('dual_stratification', False),
            'masking_mode': self.preprocessing.get('masking_mode', 'none'),
            'input_format': self.data_config.get("input_format", "pair"),
            'hier_enabled': self.hier_enabled,
            'hier_stage': self.hier_stage,
            'hier_amb_label': self.hier_amb_label,
            'hier_clear_label': self.hier_clear_label,
            'hier_nonreply_label': self.hier_nonreply_label,
            'cd_enabled': self.cd_enabled,
        }
        # Create a stable hash from the dictionary
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_mask_cache_path(self, split: str) -> str:
        """
        Get cache path for masked dataset.
        """
        base_dir = os.environ["MASKING_CACHE_DIR"]

        dataset_name = self.data_config['dataset']['name'].replace('/', '_')
        task_name = self.task_config.get("task_name", "unknown")
        split_method = self.data_config['splitting'].get('method', 'random')
        seed = self.data_config['splitting'].get('random_seed', 42)
        masking_mode = self.masking_mode

        hier_tag = "hier_off" if not self.hier_enabled else f"hier_stage{self.hier_stage}"

        text_col = self.task_config['data']['text_column']
        text_pair_col = self.task_config['data']['text_pair_column']

        return os.path.join(
            base_dir,
            dataset_name,
            task_name,
            split_method,
            f"seed_{seed}",
            f"{text_col}__{text_pair_col}",
            masking_mode,
            hier_tag,
            split
        )

    def _load_and_split_dataset(self) -> Dataset:
        """Load dataset and create splits based on configured method."""
        from datasets import load_dataset, Dataset as HFDataset

        dataset_cfg = self.data_config["dataset"]
        source = (dataset_cfg.get("source") or "").lower().strip()
        data_files = dataset_cfg.get("data_files")  # only meaningful for CSV/local

        # ------------------------------------------------------------------
        # Load raw dataset (HF or CSV)
        # ------------------------------------------------------------------
        if source in ["huggingface", "hf"] and not data_files:
            # HuggingFace Hub dataset (e.g. ailsntua/QEvasion)
            raw_dataset = load_dataset(
                dataset_cfg["name"],
                cache_dir=dataset_cfg.get("cache_dir"),
                trust_remote_code=bool(dataset_cfg.get("trust_remote_code", False)),
            )

            # Some HF datasets don't have a 'validation' split
            # you create custom splits from raw_dataset['train'] anyway.
            if "train" not in raw_dataset:
                raise ValueError(
                    f"HuggingFace dataset '{dataset_cfg['name']}' did not return a 'train' split. "
                    f"Splits available: {list(raw_dataset.keys())}"
                )

        else:
            # CSV / local files 
            import pandas as pd
            from datasets import Features, Value

            if not data_files:
                raise ValueError(
                    "CSV dataset config requires dataset.data_files (or dataset.data_files.train). "
                    f"Got: {data_files}"
                )

            # Use train header as the canonical column list
            train_path = data_files["train"] if isinstance(data_files, dict) else data_files
            if not train_path:
                raise ValueError(f"dataset.data_files resolved to an invalid train path: {train_path}")

            cols = pd.read_csv(train_path, nrows=0).columns.tolist()

            # Force everything to string to avoid Arrow cross-split casting errors
            features = Features({c: Value("string") for c in cols})

            raw_dataset = load_dataset(
                dataset_cfg.get("name", "csv"),
                data_files=data_files,
                cache_dir=dataset_cfg.get("cache_dir"),
                features=features,
            )

            if "train" not in raw_dataset:
                raise ValueError(
                    f"CSV dataset did not return a 'train' split. Splits available: {list(raw_dataset.keys())}"
                )

        if self.data_config["splitting"]["create_custom_splits"]:
            split_method = self.data_config["splitting"].get("method", "random")
            cache_key = self._create_cache_key()

            if cache_key not in ClarityDataset._splits_cache:
                print(f"\n{'='*80}")
                print(f"Computing {split_method.upper()} splits (first time)...")
                print(f"{'='*80}")

                if split_method == "random":
                    ClarityDataset._splits_cache[cache_key] = self._create_random_splits(raw_dataset["train"])
                elif split_method == "stratified":
                    ClarityDataset._splits_cache[cache_key] = self._create_stratified_splits(raw_dataset["train"])
                elif split_method == "president_based":
                    ClarityDataset._splits_cache[cache_key] = self._create_president_based_splits(raw_dataset["train"])
                else:
                    raise ValueError(f"Unknown split method: {split_method}")
            else:
                print(f"✓ Using cached {split_method} splits")

            splits = ClarityDataset._splits_cache[cache_key]

            if self.split == "train":
                train_dataset = self._apply_masking(splits["train"])
                self.data = train_dataset.to_pandas()
                return train_dataset
            elif self.split == "validation":
                val_dataset = self._apply_masking(splits["validation"])
                self.data = val_dataset.to_pandas()
                return val_dataset
            else:  # test
                if "test" in raw_dataset:
                    test_dataset = raw_dataset["test"]

                    if self.hier_enabled and self.hier_stage == 2:
                        label_col = self.task_config["data"]["label_column"]
                        test_dataset = test_dataset.filter(lambda x: x[label_col] != self.hier_amb_label)

                    test_dataset = self._apply_masking(test_dataset)
                    self.data = test_dataset.to_pandas()
                    return test_dataset
                else:
                    print("⚠ No test split found - using validation split as test")
                    test_dataset = splits["validation"]
                    test_dataset = self._apply_masking(test_dataset)
                    self.data = test_dataset.to_pandas()
                    return test_dataset

        else:
            dataset = raw_dataset[self.split]
            dataset = self._apply_masking(dataset)
            self.data = dataset.to_pandas()
            return dataset

    def _print_split_statistics(self, train_data: Dataset, train_indices: List[int],
                               val_indices: List[int], split_method: str):
        """Print comprehensive statistics for train/validation splits."""
        label_col = self.task_config['data']['label_column']
        is_task_b = self.task_config.get('task_name') == 'task_b'

        print(f"\n{'='*80}")
        print(f"SPLIT STATISTICS - {split_method.upper()} METHOD")
        print(f"{'='*80}")

        # 1. Sample counts
        print(f"\n📊 Sample Counts:")
        print(f"  Total samples: {len(train_data)}")
        print(f"  Training samples: {len(train_indices)}")
        print(f"  Validation samples: {len(val_indices)}")
        print(f"  Validation ratio: {len(val_indices)/len(train_data):.2%}")

        # 2. President distribution
        if "president" in train_data.column_names:
            print(f"\n👥 President Distribution:")
            train_presidents = [train_data[i]['president'] for i in train_indices]
            val_presidents = [train_data[i]['president'] for i in val_indices]

            train_pres_counts = Counter(train_presidents)
            val_pres_counts = Counter(val_presidents)

            all_presidents = sorted(set(train_presidents) | set(val_presidents))

            print(f"  Training presidents ({len(train_pres_counts)} distinct):")
            for pres in sorted(train_pres_counts.keys()):
                print(f"    {pres}: {train_pres_counts[pres]} samples")

            print(f"  Validation presidents ({len(val_pres_counts)} distinct):")
            for pres in sorted(val_pres_counts.keys()):
                print(f"    {pres}: {val_pres_counts[pres]} samples")

            # Check for overlap
            overlap = set(train_pres_counts.keys()) & set(val_pres_counts.keys())
            if overlap:
                print(f"  ⚠️  President overlap: {sorted(overlap)}")
            else:
                print(f"  ✅ No president overlap (strict generalization test)")

        # 3. Label distributions
        if is_task_b:
            # For Task B: show both evasion and clarity labels
            evasion_to_clarity = self.task_config['labels']['evasion_to_clarity']

            # Evasion labels (9 classes)
            train_evasion = [train_data[i][label_col] for i in train_indices]
            val_evasion = [train_data[i][label_col] for i in val_indices]

            train_evasion_counts = {k: int(v) for k, v in Counter(train_evasion).items()}
            val_evasion_counts = {k: int(v) for k, v in Counter(val_evasion).items()}

            print(f"\n🏷️  Evasion Label Distribution (9 classes):")
            all_evasion_labels = sorted(set(train_evasion) | set(val_evasion))

            print(f"  {'Label':<25} {'Train':<15} {'Val':<15} {'Total':<10}")
            print(f"  {'-'*65}")
            for label in all_evasion_labels:
                train_count = train_evasion_counts.get(label, 0)
                val_count = val_evasion_counts.get(label, 0)
                total_count = train_count + val_count
                train_pct = f"{train_count} ({train_count/len(train_indices)*100:.1f}%)"
                val_pct = f"{val_count} ({val_count/len(val_indices)*100:.1f}%)"
                print(f"  {label:<25} {train_pct:<15} {val_pct:<15} {total_count:<10}")

            # Clarity labels (3 classes)
            train_clarity = [evasion_to_clarity[train_data[i][label_col]] for i in train_indices]
            val_clarity = [evasion_to_clarity[train_data[i][label_col]] for i in val_indices]

            train_clarity_counts = {k: int(v) for k, v in Counter(train_clarity).items()}
            val_clarity_counts = {k: int(v) for k, v in Counter(val_clarity).items()}

            print(f"\n🏷️  Clarity Label Distribution (3 classes):")
            all_clarity_labels = sorted(set(train_clarity) | set(val_clarity))

            print(f"  {'Label':<25} {'Train':<15} {'Val':<15} {'Total':<10}")
            print(f"  {'-'*65}")
            for label in all_clarity_labels:
                train_count = train_clarity_counts.get(label, 0)
                val_count = val_clarity_counts.get(label, 0)
                total_count = train_count + val_count
                train_pct = f"{train_count} ({train_count/len(train_indices)*100:.1f}%)"
                val_pct = f"{val_count} ({val_count/len(val_indices)*100:.1f}%)"
                print(f"  {label:<25} {train_pct:<15} {val_pct:<15} {total_count:<10}")
        else:
            # For Task A: show only clarity labels
            train_labels = [train_data[i][label_col] for i in train_indices]
            val_labels = [train_data[i][label_col] for i in val_indices]

            train_label_counts = {k: int(v) for k, v in Counter(train_labels).items()}
            val_label_counts = {k: int(v) for k, v in Counter(val_labels).items()}

            print(f"\n🏷️  Label Distribution:")
            all_labels = sorted(set(train_labels) | set(val_labels))

            print(f"  {'Label':<25} {'Train':<15} {'Val':<15} {'Total':<10}")
            print(f"  {'-'*65}")
            for label in all_labels:
                train_count = train_label_counts.get(label, 0)
                val_count = val_label_counts.get(label, 0)
                total_count = train_count + val_count
                train_pct = f"{train_count} ({train_count/len(train_indices)*100:.1f}%)"
                val_pct = f"{val_count} ({val_count/len(val_indices)*100:.1f}%)"
                print(f"  {label:<25} {train_pct:<15} {val_pct:<15} {total_count:<10}")

        print(f"\n{'='*80}\n")

    def _create_random_splits(self, train_data: Dataset) -> Dict[str, Dataset]:
        label_col = self.task_config['data']['label_column']

        # stage 2 must not contain Ambivalent
        if self.hier_enabled and self.hier_stage == 2:
            train_data = train_data.filter(lambda x: x[label_col] != self.hier_amb_label)
            
        """Create random train/validation splits."""
        val_ratio = self.data_config['splitting']['val_ratio']
        random_seed = self.data_config['splitting'].get('random_seed', 42)

        print(f"🎲 Random split configuration:")
        print(f"  Validation ratio: {val_ratio:.2%}")
        print(f"  Random seed: {random_seed}")

        # Get total size
        total_size = len(train_data)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        # Create random indices with seed
        np.random.seed(random_seed)
        indices = np.random.permutation(total_size)

        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()

        # Create splits
        train_split = train_data.select(train_indices)
        val_split = train_data.select(val_indices)

        # Print comprehensive statistics
        self._print_split_statistics(train_data, train_indices, val_indices, "random")

        return {'train': train_split, 'validation': val_split}

    def _create_stratified_splits(self, train_data: Dataset) -> Dict[str, Dataset]:
        """Create stratified train/validation splits maintaining label distribution."""
        val_ratio = self.data_config['splitting']['val_ratio']
        random_seed = self.data_config['splitting'].get('random_seed', 42)
        # Dual stratification ensures train/val splits preserve the joint distribution
        # of fine-grained evasion labels (9 classes) and their mapped clarity labels (3 classes).
        dual_strat = self.data_config['splitting'].get('dual_stratification', False)
        label_col = self.task_config['data']['label_column']

        print(f"📊 Stratified split configuration:")
        print(f"  Validation ratio: {val_ratio:.2%}")
        print(f"  Random seed: {random_seed}")
        print(f"  Dual stratification: {dual_strat}")

        # Drop any rows where the label is missing or empty.
        # HuggingFace datasets don't use pandas, so we filter at the dataset level
        # instead of calling something like .notnull().
        train_data = train_data.filter(
            lambda x: x[label_col] is not None and x[label_col] != ""
        )

        # If hierarchical stage 2, remove Ambivalent BEFORE splitting
        # so stratification happens on the actual stage-2 label space.
        if self.hier_enabled and self.hier_stage == 2:
            train_data = train_data.filter(lambda x: x[label_col] != self.hier_amb_label)

        # Convert to pandas
        raw_labels = train_data[label_col]
        if self.hier_enabled and self.hier_stage in (1, 2):
            mapped = [self._map_label_text(x) for x in raw_labels]
            df = pd.DataFrame({'label': mapped, 'index': range(len(train_data))})
        else:
            df = pd.DataFrame({'label': raw_labels, 'index': range(len(train_data))})

        # Determine stratification strategy
        if dual_strat and self.task_config.get('task_name') == 'task_b':
            # For Task B with dual stratification: create composite labels
            evasion_to_clarity = self.task_config['labels']['evasion_to_clarity']
            df['clarity_label'] = df['label'].map(evasion_to_clarity)
            # Create composite label for stratification (evasion + clarity)
            df['strat_label'] = df['label'] + '_' + df['clarity_label']
            stratify_col = 'strat_label'
            print(f"  Stratifying on: evasion + clarity labels (composite)")
        else:
            # Single-level stratification
            stratify_col = 'label'
            print(f"  Stratifying on: task-specific labels only")

        # Perform stratified split with seed
        train_indices, val_indices = train_test_split(
            df['index'].values,
            test_size=val_ratio,
            stratify=df[stratify_col].values,
            random_state=random_seed
        )

        # Convert numpy arrays to Python int lists (CRITICAL for HuggingFace datasets)
        train_indices = [int(i) for i in train_indices]
        val_indices = [int(i) for i in val_indices]

        # Create splits
        train_split = train_data.select(train_indices)
        val_split = train_data.select(val_indices)

        # Print comprehensive statistics
        self._print_split_statistics(train_data, train_indices, val_indices, "stratified")

        return {'train': train_split, 'validation': val_split}

    def _create_president_based_splits(self, train_data: Dataset) -> Dict[str, Dataset]:
        """Create train/validation splits ensuring no president overlap."""
        label_col = self.task_config['data']['label_column']
        dual_strat = self.data_config['splitting'].get('dual_stratification', False)
        is_task_b = self.task_config.get('task_name') == 'task_b'
        random_seed = self.data_config['splitting'].get('random_seed', 42)

        # (nice to do this for all split methods so behaviour matches)
        train_data = train_data.filter(lambda x: x[label_col] is not None and x[label_col] != "")

        # stage 2 must not contain Ambivalent, or mapping will crash later
        if self.hier_enabled and self.hier_stage == 2:
            train_data = train_data.filter(lambda x: x[label_col] != self.hier_amb_label)

        print(f"👥 President-based split configuration:")
        print(f"  Validation ratio target: {self.data_config['splitting']['val_ratio']:.2%}")
        print(f"  Random seed: {random_seed}")
        print(f"  Dual stratification: {dual_strat}")
        print(f"  Max presidents in validation: {self.data_config['splitting'].get('max_presidents_in_val', 5)}")

        np.random.seed(random_seed)

        # Build df ONCE (dont rebuild it later, easy mistake)
        df = pd.DataFrame({
            'president': train_data['president'],
            'label': train_data[label_col],
        })

        # Only map labels for Task A hierarchical stuff (Task B labels are different)
        if (not is_task_b) and self.hier_enabled and self.hier_stage in (1, 2):
            df['label'] = df['label'].apply(self._map_label_text)

        # Dual strat is only meaningful for Task B
        if dual_strat and is_task_b:
            evasion_to_clarity = self.task_config['labels']['evasion_to_clarity']
            df['clarity_label'] = df['label'].map(evasion_to_clarity)
            print("  Optimizing for: evasion labels + clarity labels")
        else:
            df['clarity_label'] = df['label']
            print("  Optimizing for: task-specific labels")

        label_dist = df['label'].value_counts(normalize=True).sort_index()
        clarity_dist = df['clarity_label'].value_counts(normalize=True).sort_index()

        best_split = self._find_optimal_president_split(
            df, label_dist, clarity_dist, use_dual_strat=(dual_strat and is_task_b)
        )
        if best_split is None:
            raise ValueError("Could not find a valid president-based split")

        val_presidents_set = set(best_split['val_presidents'])

        train_indices, val_indices = [], []
        for i, president in enumerate(train_data['president']):
            if president in val_presidents_set:
                val_indices.append(i)
            else:
                train_indices.append(i)

        train_split = train_data.select(train_indices)
        val_split = train_data.select(val_indices)

        self._print_split_statistics(train_data, train_indices, val_indices, "president_based")
        return {'train': train_split, 'validation': val_split}

    def _find_optimal_president_split(self, df: pd.DataFrame,
                                      label_dist: pd.Series,
                                      clarity_dist: pd.Series,
                                      use_dual_strat: bool) -> Dict[str, Any]:
        """
        Find optimal president split that maintains label distribution.

        Args:
            df: DataFrame with president, label, and clarity_label columns
            label_dist: Target distribution for primary labels (evasion for Task B, clarity for Task A)
            clarity_dist: Target distribution for clarity labels
            use_dual_strat: Whether to use dual stratification (True for Task B)
        """
        unique_presidents = df['president'].unique()
        best_split = None
        best_score = float('inf')

        target_val_ratio = self.data_config['splitting']['val_ratio']
        max_presidents = self.data_config['splitting'].get('max_presidents_in_val', 5)

        # Get weights from config
        weights = self.data_config['splitting'].get('stratification_weights', {})
        clarity_weight = weights.get('clarity', 1.0)
        evasion_weight = weights.get('evasion', 2.0)
        size_weight = weights.get('size', 5.0)

        print(f"\n🔍 Searching for optimal president split...")
        print(f"  Trying combinations of 1-{max_presidents} presidents in validation")
        print(f"  Scoring weights: evasion={evasion_weight}, clarity={clarity_weight}, size={size_weight}")

        # Try different numbers of presidents in validation set
        for r in range(1, min(len(unique_presidents), max_presidents + 1)):
            for val_presidents in combinations(unique_presidents, r):
                val_mask = df['president'].isin(val_presidents)

                train_df = df[~val_mask]
                val_df = df[val_mask]

                if len(train_df) == 0 or len(val_df) == 0:
                    continue

                # Calculate distributions
                train_label_dist = train_df['label'].value_counts(normalize=True).sort_index()
                val_label_dist = val_df['label'].value_counts(normalize=True).sort_index()

                train_clarity_dist = train_df['clarity_label'].value_counts(normalize=True).sort_index()
                val_clarity_dist = val_df['clarity_label'].value_counts(normalize=True).sort_index()

                train_presidents = train_df['president'].unique()

                # Calculate stratification scores
                # 1. Primary label stratification (evasion for Task B, clarity for Task A)
                label_stratification_score = 0
                for label in label_dist.index:
                    target_ratio = label_dist[label]
                    train_ratio = train_label_dist.get(label, 0)
                    val_ratio = val_label_dist.get(label, 0)
                    label_stratification_score += abs(train_ratio - target_ratio) + abs(val_ratio - target_ratio)

                # 2. Clarity label stratification (for dual stratification)
                clarity_stratification_score = 0
                for label in clarity_dist.index:
                    target_ratio = clarity_dist[label]
                    train_ratio = train_clarity_dist.get(label, 0)
                    val_ratio = val_clarity_dist.get(label, 0)
                    clarity_stratification_score += abs(train_ratio - target_ratio) + abs(val_ratio - target_ratio)

                # 3. Size score
                val_ratio = len(val_df) / len(df)
                size_score = abs(val_ratio - target_val_ratio)

                # Combined score with weights
                if use_dual_strat:
                    # For Task B with dual stratification: balance both levels
                    combined_score = (
                        label_stratification_score * evasion_weight +      # Evasion labels (fine-grained)
                        clarity_stratification_score * clarity_weight +    # Clarity labels (coarse-grained)
                        size_score * size_weight                           # Split size
                    )
                else:
                    # For Task A or single stratification: only primary label matters
                    combined_score = (
                        label_stratification_score * (evasion_weight + clarity_weight) +  # Primary labels
                        size_score * size_weight                                           # Split size
                    )

                if combined_score < best_score:
                    best_score = combined_score
                    best_split = {
                        'train_presidents': train_presidents,
                        'val_presidents': val_presidents,
                        'train_size': len(train_df),
                        'val_size': len(val_df),
                        'val_ratio': val_ratio,
                        'label_stratification_score': label_stratification_score,
                        'clarity_stratification_score': clarity_stratification_score if use_dual_strat else None,
                        'size_score': size_score,
                        'combined_score': combined_score,
                    }

        if best_split:
            print(f"\n✅ Found optimal split:")
            print(f"  Combined score: {best_split['combined_score']:.4f}")
            print(f"  Label stratification score: {best_split['label_stratification_score']:.4f}")
            if use_dual_strat:
                print(f"  Clarity stratification score: {best_split['clarity_stratification_score']:.4f}")
            print(f"  Size score: {best_split['size_score']:.4f}")
            print(f"  Validation presidents: {sorted(best_split['val_presidents'])}")

        return best_split

    def _check_multi_annotator_format(self) -> bool:
        """Check if this dataset uses multi-annotator format (Task B test set)."""
        if self.data is None:
            return False

        if self.task_config.get('task_name') != 'task_b':
            return False

        has_annotators = all(
            col in self.data.columns
            for col in ['annotator1', 'annotator2', 'annotator3']
        )

        if not has_annotators:
            return False

        label_col = self.task_config['data']['label_column']
        if label_col not in self.data.columns:
            return False

        label_is_null = self.data[label_col].isna().all()

        if has_annotators and label_is_null:
            print("📌 Multi-annotator test set detected")
            print("   Using annotator1 labels for dataset loading")
            print("   Evaluation will use all 3 annotators")
            return True

        return False

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)

    def _map_label_text(self, label_text: str) -> str:
        """
        Map original Task A label text into the stage-specific label space.
        When hierarchical is disabled, returns label_text unchanged.
        """
        if not self.hier_enabled:
            return label_text

        if self.hier_stage == 1:
            # Ambivalent vs Not-Ambivalent
            if label_text == self.hier_amb_label:
                return "AMBIVALENT"
            return "NOT_AMBIVALENT"

        if self.hier_stage == 2:
            # Ambivalent should never appear in stage 2 data. If it does, something upstream is wrong.
            if label_text == self.hier_amb_label:
                raise ValueError(
                    "Stage 2 received an Ambivalent label. "
                    "Stage 2 datasets must filter Ambivalent examples before evaluation/training."
                )

            if label_text == self.hier_clear_label:
                return "CLEAR_REPLY"
            if label_text == self.hier_nonreply_label:
                return "CLEAR_NON_REPLY"

            # If the dataset has unexpected label strings, fail loudly
            raise ValueError(f"Unknown Task A label for stage 2: {label_text}")

        # If misconfigured, fall back to original
        return label_text

    def _get_cd_token(self, example: Dict[str, Any]) -> str | None:
        if not self.cd_enabled:
            return None

        tok = example.get(self._cd_col)

        if tok is None:
            return self._cd_default

        if isinstance(tok, float) and np.isnan(tok):
            return self._cd_default

        tok = str(tok).strip()
        if tok == "" or tok.lower() == "nan":
            return self._cd_default

        # if something odd is in the column, fall back
        if tok not in ("CD_LOW", "CD_HIGH"):
            return self._cd_default

        return tok
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset - returns raw text."""
        example = self.dataset[idx]

        # Get label - handle multi-annotator case
        if self.is_multi_annotator:
            label_text = example.get('annotator1')
            if label_text is None or pd.isna(label_text) or label_text == '':
                label_text = list(self.label2id.keys())[0]
                print(f"⚠️ Warning: annotator1 is empty at index {idx}, using fallback: {label_text}")
        else:
            label_text = example.get(self.task_config['data']['label_column'])
            if label_text is None or pd.isna(label_text) or label_text == '':
                label_text = example.get('annotator1')
                if label_text is None or pd.isna(label_text) or label_text == '':
                    label_text = list(self.label2id.keys())[0]
                    print(f"⚠️ Warning: Both evasion_label and annotator1 empty at index {idx}, using fallback: {label_text}")

        answer = example[self.task_config['data']['text_column']]
        question = example[self.task_config['data']['text_pair_column']]

        cd_tok = self._get_cd_token(example)

        label_text = self._map_label_text(label_text)

        item = {
            'label_text': label_text,
            'label_id': self.label2id[label_text],
        }

        if self.input_format == "pair":
            # prepend to answer only (CD is derived from the answer)
            if cd_tok:
                answer = f"[{cd_tok}] {answer}"
            item['answer'] = answer
            item['question'] = question

        elif self.input_format == "qa_marked":
            prefix = f"[{cd_tok}] " if cd_tok else ""
            item['text'] = f"{prefix}[QUESTION] {question} [ANSWER] {answer}"
        else:
            raise ValueError(f"Unknown input_format: {self.input_format}")

        return item

    def _apply_masking(self, dataset: Dataset) -> Dataset:
        """Apply PERSON masking once to the entire dataset split, with caching."""
        masking_mode = self.masking_mode

        if masking_mode == 'none':
            return dataset

        cache_path = self._get_mask_cache_path(self.split)

        # Try loading from cache
        if os.path.exists(cache_path):
            print(f"⚡ Loading cached masked dataset: {cache_path}")
            return load_from_disk(cache_path)

        print(f"🔒 Applying PERSON masking ({masking_mode}) to split: {self.split}")

        answer_col = self.task_config['data']['text_column']
        question_col = self.task_config['data']['text_pair_column']

        if masking_mode == 'naive':
            def mask_example(example):
                new_example = dict(example)  # shallow copy
                new_example[answer_col] = mask_person_entities(example[answer_col])
                new_example[question_col] = mask_person_entities(example[question_col])
                return new_example

        elif masking_mode == 'entity_aware':
            def mask_example(example):
                new_example = dict(example)  # copy

                q, a = mask_person_entities_pair(
                    example[question_col],
                    example[answer_col]
                )

                new_example[question_col] = q
                new_example[answer_col] = a
                return new_example
        else:
            raise ValueError(f"Unknown masking_mode: {masking_mode}")

        masked_dataset = dataset.map(
            mask_example,
            desc=f"Masking PERSON entities ({masking_mode}, {self.split})"
        )

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        masked_dataset.save_to_disk(cache_path)
        print(f"💾 Saved masked dataset to cache: {cache_path}")

        return masked_dataset

    def get_class_weights(
        self,
        method: str = "inverse_frequency",
        *,
        max_weight: float = 2.5,
        eps: float = 1e-6,
        normalise_mean: bool = True,
    ) -> torch.Tensor:
        """Compute class weights for handling imbalanced dataset.

        Args:
            method:
                - "inverse_frequency": classic 1/freq (aggressive, can be unstable)
                - "balanced": sklearn-ish 1/freq with a constant factor (still aggressive)
                - "sqrt": sqrt-smoothed inverse frequency (much more stable)
            max_weight:
                Caps the biggest class weight. This stops rare classes nuking optimisation.
            eps:
                Tiny constant to avoid divide-by-zero and other silliness.
            normalise_mean:
                If True, scales weights so the mean weight is 1. Helps keep LR behaviour sane.
        """

        # Defensive warning - weights should come from train split
        if self.split != "train":
            print(
                f"⚠️ Warning: Computing class weights from '{self.split}' split. "
                f"This is usually unintended."
            )
            
        # Collect labels from the split
        if self.is_multi_annotator:
            labels = []
            for example in self.dataset:
                label_text = example.get('annotator1')
                if label_text and not pd.isna(label_text):
                    labels.append(self.label2id[label_text])
        else:
            label_col = self.task_config['data']['label_column']
            labels = []
            for example in self.dataset:
                raw = example.get(label_col)
                if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                    # Default safely: treat missing as NOT_AMBIVALENT or CLEAR_REPLY depending on stage
                    raw = self.hier_clear_label if (self.hier_enabled and self.hier_stage == 2) else self.hier_amb_label

                mapped = self._map_label_text(raw)
                labels.append(self.label2id[mapped])

        labels = np.asarray(labels, dtype=np.int64)

        # Count classes (note: make sure we include all classes even if one is missing in a split)
        num_classes = len(self.label2id)
        class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

        # Basic safety check (rare, but if a class is missing, raw inverse weighting explodes)
        # We'll just treat missing as eps count, but also warn once.
        if np.any(class_counts == 0):
            missing = np.where(class_counts == 0)[0].tolist()
            print(f"⚠️ Warning: Missing classes in this split: {missing} - weights will be dampened")
            class_counts = np.maximum(class_counts, eps)

        total_samples = float(labels.shape[0])

        # Convert counts -> frequencies
        freqs = class_counts / max(total_samples, 1.0)

        method = (method or "").lower().strip()

        if method in ["inverse_frequency", "inverse_freq", "inv_freq"]:
            # Aggressive: 1/freq up to a constant factor
            weights = 1.0 / (freqs + eps)

        elif method == "balanced":
            # Similar to many "balanced" recipes: basically 1/freq with a different constant
            weights = total_samples / (class_counts * num_classes)

        elif method == "sqrt":
            # Still favours minority classes, but doesn't go crazy.
            weights = 1.0 / np.sqrt(freqs + eps)

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Clip to stop catastrophic seeds.
        weights = np.clip(weights, 1.0, float(max_weight))

         # Normalise so avg weight is 1 (keeps your effective LR roughly comparable)
        if normalise_mean:
            weights = weights / np.mean(weights)

        # (handy when you're staring at W&B wondering why something died)
        try:
            ratio = float(weights.max() / max(weights.min(), eps))
        except Exception:
            ratio = -1.0

        print(f"🔧 Using weighted loss with method: {method}")
        print(f"⚖️  Class counts: {class_counts.astype(int).tolist()}")
        print(f"⚖️  Class freqs: {[round(float(x), 4) for x in freqs.tolist()]}")
        print(f"⚖️  Class weights: {[round(float(w), 4) for w in weights.tolist()]} (max/min={ratio:.2f})")

        return torch.tensor(weights, dtype=torch.float32)


    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in current split."""
        if self.is_multi_annotator:
            labels = []
            for example in self.dataset:
                label_text = example.get('annotator1')
                if label_text and not pd.isna(label_text):
                    labels.append(label_text)
            dist = dict(Counter(labels))
            print("   (Distribution based on annotator1 - evaluation uses all 3)")
            return dist
        else:
            label_col = self.task_config['data']['label_column']
            labels = []
            for example in self.dataset:
                raw = example.get(label_col)
                if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                    continue
                labels.append(self._map_label_text(raw))
            return dict(Counter(labels))

    



class TokenizedDataset(TorchDataset):
    """Wrapper that adds model-specific tokenization."""

    def __init__(
        self,
        clarity_dataset: ClarityDataset,
        tokenizer,
        model_type: str = "bert",
        tokenization_cfg: Dict[str, Any] | None = None,
    ):
        self.clarity_dataset = clarity_dataset
        self.tokenizer = tokenizer
        self.model_type = (model_type or "bert").lower()

        tok = tokenization_cfg or {}

        # pull these from yaml so BigBird can actually go past 512
        self.max_length = int(tok.get("max_length", 512))
        self.padding = tok.get("padding", "max_length")
        self.truncation = tok.get("truncation", True)
        self.return_special_tokens_mask = bool(tok.get("return_special_tokens_mask", False))

    def _pair_truncation(self):
        # HF likes 'longest_first' for pair inputs - bool also works but this is usually nicer
        if not self.truncation:
            return False
        if isinstance(self.truncation, str):
            return self.truncation
        return "longest_first"

    def __len__(self):
        return len(self.clarity_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_item = self.clarity_dataset[idx]

        if self._use_single_text() and self.model_type not in ["roberta", "deberta", "bigbird", "big_bird"]:
            raise ValueError(
                f"input_format='qa_marked' (single text) is only supported for model_type='roberta' right now. "
                f"Got model_type='{self.model_type}'. Either switch input_format to 'pair' "
                f"or implement single-text tokenisation for {self.model_type}."
            )

        if self.model_type == "bert":
            return self._tokenize_bert(raw_item)
        elif self.model_type == "t5":
            return self._tokenize_t5(raw_item)
        elif self.model_type in ["roberta", "bigbird", "big_bird"]:
            return self._tokenize_roberta(raw_item)
        elif self.model_type == "distilbert":
            return self._tokenize_distilbert(raw_item)
        elif self.model_type == "deberta":
            return self._tokenize_deberta(raw_item)
        elif self.model_type == "electra":
            return self._tokenize_electra(raw_item)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _use_single_text(self) -> bool:
        return getattr(self.clarity_dataset, "input_format", "pair") == "qa_marked"

    def _tokenize_bert(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """BERT-style tokenization. Tokenized as [CLS] answer [SEP] question [SEP]."""
            
        encoding = self.tokenizer(
            item['answer'],  # answer as first argument
            item['question'],  # question as second argument
            max_length=self.max_length,
            padding=self.padding,
            truncation=self._pair_truncation(),
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'labels': torch.tensor(item['label_id'], dtype=torch.long)
        }

    def _tokenize_roberta(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """RoBERTa-style tokenization (no token_type_ids). Tokenized as <s> answer </s></s> question </s>."""
        if self._use_single_text():
            encoding = self.tokenizer(
                item['text'],
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_special_tokens_mask=self.return_special_tokens_mask,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer(
                item['answer'],
                item['question'],
                max_length=self.max_length,
                padding=self.padding,
                truncation=self._pair_truncation(),
                return_special_tokens_mask=self.return_special_tokens_mask,
                return_tensors='pt'
            )

        out = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label_id'], dtype=torch.long)
        }

        # BigBird often has token_type_ids (like BERT). RoBERTa won't, so this is safe.
        if 'token_type_ids' in encoding:
            out['token_type_ids'] = encoding['token_type_ids'].squeeze()

        return out

    def _tokenize_distilbert(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """DistilBERT-style tokenization (no token_type_ids). Tokenized as [CLS] answer [SEP] question [SEP]."""
        encoding = self.tokenizer(
            item['answer'],
            item['question'],
            max_length=512,
            padding='max_length',
            truncation='longest_first',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label_id'], dtype=torch.long)
        }

    def _tokenize_deberta(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """DeBERTa-style tokenization (no token_type_ids). Tokenized as [CLS] answer [SEP] question [SEP]."""
        if self._use_single_text():
            encoding = self.tokenizer(
                item['text'],
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer(
                item['answer'],
                item['question'],
                max_length=512,
                padding='max_length',
                truncation='longest_first',
                return_tensors='pt'
            )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label_id'], dtype=torch.long)
        }

    def _tokenize_electra(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """ELECTRA-style tokenization (no token_type_ids). Tokenized as [CLS] answer [SEP] question [SEP]."""
        encoding = self.tokenizer(
            item['answer'],
            item['question'],
            max_length=512,
            padding='max_length',
            truncation='longest_first',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label_id'], dtype=torch.long)
        }

    def _tokenize_t5(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """T5-style tokenization (text-to-text format)."""
        # T5 expects text-to-text format
        input_text = f"question: {item['question']} answer: {item['answer']}"
        target_text = item['label_text']  # e.g., "clear" or "unclear"

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

    # Forward methods from underlying dataset
    def get_class_weights(self, method: str = "inverse_frequency", **kwargs) -> torch.Tensor:
        return self.clarity_dataset.get_class_weights(method=method, **kwargs)

    def get_label_distribution(self) -> Dict[str, int]:
        return self.clarity_dataset.get_label_distribution()

    # Expose data attribute for multi-annotator detection in evaluate.py (Task B)
    @property
    def data(self):
        """Expose raw data for multi-annotator detection."""
        return self.clarity_dataset.data


def create_datasets(data_config: Dict[str, Any], task_config: Dict[str, Any],
                    model_config: Dict[str, Any], tokenizer=None):
    """
    Create datasets with appropriate tokenization for the specified model.
    Automatically handles fast/slow tokenizer fallback for compatibility.

    UPDATED: Now supports multi-annotator format for Task B test set.

    Args:
        data_config: Data configuration
        task_config: Task configuration
        model_config: Model configuration

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from transformers import AutoTokenizer
    import warnings

    # Create raw datasets
    train_raw = ClarityDataset(data_config, task_config, split="train")
    val_raw = ClarityDataset(data_config, task_config, split="validation")
    test_raw = ClarityDataset(data_config, task_config, split="test")

    # Create tokenizer with special handling for DeBERTa-v3
    model_name = model_config['model_name']

    if tokenizer is None:
        # Special handling for DeBERTa-v3 models
        if 'deberta-v3' in model_name.lower():
            print(f"⚠ DeBERTa-v3 detected, loading slow tokenizer directly to avoid tiktoken issues...")
            try:
                from transformers import DebertaV2Tokenizer
                tokenizer = DebertaV2Tokenizer.from_pretrained(
                    model_name,
                    cache_dir=model_config['loading'].get('cache_dir')
                )
                print(f"✓ Successfully loaded slow DeBERTa-v3 tokenizer")
            except Exception as e:
                raise RuntimeError(f"Failed to load DeBERTa-v3 slow tokenizer: {e}")
        else:
            # For other models, try fast first, then slow
            for use_fast in [True, False]:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=use_fast,
                        cache_dir=model_config['loading'].get('cache_dir')
                    )
                    tokenizer_type = "fast" if use_fast else "slow"
                    print(f"✓ Successfully loaded {tokenizer_type} tokenizer for {model_name}")
                    break
                except Exception as e:
                    if use_fast:
                        print(f"⚠ Fast tokenizer failed, falling back to slow tokenizer...")
                        print(f"  Error: {str(e)[:100]}...")
                    else:
                        raise RuntimeError(
                            f"Both fast and slow tokenizers failed for {model_name}. "
                            f"Error: {e}"
                        )
    else:
        print("✓ Using provided tokenizer (external)")

    # Add pad token if missing (needed for some models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create tokenized datasets
    model_type = model_config['model_type']

    from src.utils.tokenizer_specials import add_clarity_special_tokens
    add_clarity_special_tokens(tokenizer, model=None, max_person_tags=32)

    tok_cfg = model_config.get("tokenization", {}) or {}

    train_dataset = TokenizedDataset(train_raw, tokenizer, model_type, tokenization_cfg=tok_cfg)
    val_dataset = TokenizedDataset(val_raw, tokenizer, model_type, tokenization_cfg=tok_cfg)
    test_dataset = TokenizedDataset(test_raw, tokenizer, model_type, tokenization_cfg=tok_cfg)

    print(f"Created {model_type} tokenized datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset, tokenizer