#!/usr/bin/env python3
"""
aggregate_seeds_local.py

Local multi-seed evaluation + aggregation for CLARITY encoder checkpoints.

- Looks for: <experiment-dir>/seed_*/best_model/
- Evaluates on the TEST split
  - If --config is provided and dataset.data_files exists -> uses the CSV test file
  - Otherwise falls back to HF dataset test split (default: ailsntua/QEvasion)
- Computes paper-style metrics:
  Task 1:
    - Clarity macro F1 / P / R
  Task 2:
    - Evasion ACC_match (match-any annotator)
    - Per-annotator macro F1 / P / R (A1/A2/A3)
    - Avg across annotators (macro F1/P/R)
    - Clarity macro F1 / P / R inferred from EVASION_TO_CLARITY mapping
- Adds per-label breakdowns (good for error analysis):
  - Per-label P/R/F1/support for clarity (Task A and Task B-derived clarity)
  - Per-label P/R/F1/support for evasion vs annotator1/2/3
  - Confusion matrices (as nested lists) with explicit label order
- CD handling (matches your dataset behaviour):
  - When CD enabled: reads cd_bucket (or --cd-col)
  - Accepts CD_HEIGH typo and normalises to CD_HIGH
  - pair mode: prefixes ANSWER only: "[CD_*] {answer}"
  - qa_marked mode: prefixes whole string: "[CD_*] [QUESTION] ... [ANSWER] ..."

Usage examples:


  /// ROBERTA BASE:


  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_a/roberta_base_task_a_stratified_pair \
  --config configs/experiments/task_a/fine_tuning/roberta/base/pair/stratified.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_a/roberta_base_task_a_stratified_qa_marked \
  --config configs/experiments/task_a/fine_tuning/roberta/base/qa_marked/stratified.yaml \
  --cd off \
  --recompute


  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_base_task_b_stratified_pair \
  --config configs/experiments/task_b/fine_tuning/roberta/base/pair/stratified.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_base_task_b_stratified_qa_marked \
  --config configs/experiments/task_b/fine_tuning/roberta/base/qa_marked/stratified.yaml \
  --cd off \
  --recompute


  /// ROBERTA LARGE:

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_a/roberta_large_task_a_stratified_qa_marked \
  --config configs/experiments/task_a/fine_tuning/roberta/large/qa_marked/stratified.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_large_task_b_stratified_qa_marked \
  --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/stratified.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_large_task_b_stratified_balanced_weighted_qa_marked \
  --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/balanced_weighted_stratified.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_large_task_b_stratified_sqrt_weighted_qa_marked \
  --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/sqrt_weighted_stratified.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_large_task_b_president_disjoint_qa_marked \
  --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/president_disjoint.yaml \
  --cd off \
  --recompute

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/roberta_large_task_b_stratified_qa_marked_cd_prepend \
  --config configs/experiments/task_b/fine_tuning/roberta/large/qa_marked/stratified_cd_prepend.yaml \
  --cd on \
  --recompute


  /// DEBERTA V3 BASE:

  python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/deberta_v3_base_task_b_stratified_qa_marked \
  --config configs/experiments/task_b/fine_tuning/deberta_v3/base/qa_marked/stratified.yaml \
  --cd off \
  --recompute

    /// DEBERTA V3 LARGE:

    python scripts/evaluation/aggregate_seeds_local.py \
  --experiment-dir ../model_outputs/task_b/deberta_v3_large_task_b_stratified_qa_marked \
  --config configs/experiments/task_b/fine_tuning/deberta_v3/large/qa_marked/stratified.yaml \
  --cd off \
  --recompute

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------
# Canonical labels
# ---------------------------------------------------------------------

EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

EVASION_TO_CLARITY = {
    "Explicit":             "Clear Reply",
    "Implicit":             "Ambivalent",
    "Dodging":              "Ambivalent",
    "General":              "Ambivalent",
    "Deflection":           "Ambivalent",
    "Partial/half-answer":  "Ambivalent",
    "Declining to answer":  "Clear Non-Reply",
    "Claims ignorance":     "Clear Non-Reply",
    "Clarification":        "Clear Non-Reply",
}

CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _macro_prf_str(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {"f1": float(f1), "p": float(p), "r": float(r)}


def _per_label_prf_str(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Per-label P/R/F1/support.
    """
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    out: Dict[str, Dict[str, float]] = {}
    for i, lab in enumerate(labels):
        out[lab] = {
            "p": float(p[i]),
            "r": float(r[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
    return out


def _confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.astype(int).tolist()


def _pick_device(device: str) -> str:
    d = (device or "auto").lower().strip()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _normalise_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _normalise_label(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _canonise_by_list(x: str, allowed: List[str], fallback: Optional[str] = None) -> str:
    """
    Best-effort mapping to canonical label spellings.
    """
    xx = _normalise_label(x)
    for a in allowed:
        if xx.lower() == a.lower():
            return a
    # special case
    if xx.lower() == "clear non reply":
        return "Clear Non-Reply"
    if fallback is not None:
        return fallback
    return xx


def _normalise_cd_bucket(x: Any, default: str = "CD_LOW") -> str:
    if x is None:
        return default
    try:
        if isinstance(x, float) and np.isnan(x):
            return default
    except Exception:
        pass

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return default

    s = s.upper().replace(" ", "_")

    # common typos / variants (yep, people do this...)
    if s in ("CD_HEIGH", "CD_HIEGH", "CD_HGH", "CD_HI"):
        s = "CD_HIGH"
    if s in ("HIGH", "H"):
        s = "CD_HIGH"
    if s in ("LOW", "L"):
        s = "CD_LOW"

    if s not in ("CD_LOW", "CD_HIGH"):
        return default

    return s


def _infer_input_format(experiment_dir: Path, choice: str) -> str:
    choice = (choice or "auto").lower().strip()
    if choice in ("pair", "qa_marked"):
        return choice
    name = experiment_dir.name.lower()
    if "qa_marked" in name or "qa-marked" in name or "qamarked" in name:
        return "qa_marked"
    return "pair"


def _infer_task_from_checkpoint(best_model_dir: Path) -> str:
    cfg_path = best_model_dir / "config.json"
    if not cfg_path.exists():
        return "unknown"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"

    id2label = cfg.get("id2label") or {}
    labels = []
    for k, v in id2label.items():
        try:
            _ = int(k)
        except Exception:
            continue
        labels.append(_normalise_label(v))

    s = set(labels)
    if len(labels) == 9 or s.issuperset(set(EVASION_LABELS)):
        return "task_b"
    if len(labels) == 3 or s.issuperset(set(CLARITY_LABELS)):
        return "task_a"
    return "unknown"


def _get_cols_from_config(configs: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (question_col, answer_col, label_col).
    For pipeline:
      answer_col = task.data.text_column
      question_col = task.data.text_pair_column
      label_col = task.data.label_column
    """
    if not configs:
        return None, None, None
    try:
        tcfg = configs["task"]
        d = tcfg.get("data", {}) or {}
        qcol = d.get("text_pair_column")
        acol = d.get("text_column")
        lcol = d.get("label_column")
        return qcol, acol, lcol
    except Exception:
        return None, None, None


def _guess_q_a_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = set(df.columns)
    qcol = "question" if "question" in cols else "interview_question"
    for cand in ["interview_answer", "answer", "response"]:
        if cand in cols:
            return qcol, cand
    raise ValueError(f"Could not find answer column in df. Columns: {list(df.columns)}")


def _load_experiment_config_if_available(cfg_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not cfg_path:
        return None

    try:
        root = Path(__file__).resolve().parents[1]
        sys.path.append(str(root))
        from src.utils.config import load_experiment_config, validate_config  # type: ignore
        configs = load_experiment_config(cfg_path)
        validate_config(configs)
        return configs
    except Exception as e:
        print(f"⚠️ Could not load config via src.utils.config ({e}). Will continue with fallbacks.")
        return None


def _resolve_test_source(
    *,
    configs: Optional[Dict[str, Any]],
    dataset_name: str,
    explicit_test_csv: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (source_kind, info) where source_kind in {"csv","hf"}.
    info includes e.g. {"test_csv": "..."} or {"dataset_name": "..."}.
    """
    if explicit_test_csv:
        return "csv", {"test_csv": explicit_test_csv}

    if configs:
        dcfg = configs.get("data", {}) or {}
        ds_cfg = dcfg.get("dataset", {}) or {}
        data_files = ds_cfg.get("data_files")

        # if data_files exists - this is the "csv/local" path
        if data_files:
            if isinstance(data_files, dict):
                if data_files.get("test"):
                    return "csv", {"test_csv": data_files["test"]}
                if data_files.get("validation"):
                    # matches your dataset fallback behaviour
                    return "csv", {"test_csv": data_files["validation"], "note": "used validation as test (no test file)"}
                if data_files.get("train"):
                    return "csv", {"test_csv": data_files["train"], "note": "used train as test (no test/val file?)"}
            elif isinstance(data_files, str):
                return "csv", {"test_csv": data_files}

        # otherwise, assume HF
        ds_name = ds_cfg.get("name") or dataset_name
        return "hf", {"dataset_name": ds_name}

    # no config - default HF
    return "hf", {"dataset_name": dataset_name}


def _should_apply_cd(*, cd_mode: str, configs: Optional[Dict[str, Any]], df: pd.DataFrame, tokenizer) -> bool:
    cd_mode = (cd_mode or "auto").lower().strip()
    if cd_mode == "on":
        return True
    if cd_mode == "off":
        return False

    # auto mode
    # config flag wins if present
    if configs:
        exp = configs.get("experiment", {}) or {}
        if bool(exp.get("cd", False)):
            return True

    # if we have the column...
    if "cd_bucket" in df.columns:
        return True

    try:
        specials = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if "[CD_LOW]" in specials or "[CD_HIGH]" in specials:
            return True
    except Exception:
        pass

    return False


@dataclass
class SeedRun:
    seed: int
    best_model_dir: Path


def find_seed_runs(experiment_dir: Path) -> List[SeedRun]:
    out: List[SeedRun] = []
    for p in sorted(experiment_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"seed_(\d+)$", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        best = p / "best_model"
        if best.exists() and best.is_dir():
            out.append(SeedRun(seed=seed, best_model_dir=best))
    return out


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_test_df(source_kind: str, info: Dict[str, Any], limit: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta = dict(info)

    if source_kind == "csv":
        test_csv = meta["test_csv"]
        df = pd.read_csv(test_csv)
        if limit is not None:
            df = df.iloc[:limit].reset_index(drop=True)
        meta["n"] = int(len(df))
        return df, meta

    # HF
    ds_name = meta["dataset_name"]
    ds = load_dataset(ds_name)
    test = ds["test"]
    if limit is not None:
        test = test.select(range(min(limit, len(test))))
    df = test.to_pandas()
    meta["n"] = int(len(df))
    return df, meta


# ---------------------------------------------------------------------
# HF weight loading (fixes 'model.' / 'module.' prefixes etc)
# ---------------------------------------------------------------------

def _find_weight_file(model_dir: Path) -> Optional[Path]:
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_state_dict_any(weight_path: Path) -> Dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as e:
            raise RuntimeError("Need safetensors installed to read .safetensors weights") from e
        return load_file(str(weight_path))
    return torch.load(str(weight_path), map_location="cpu")


def _best_prefix_strip(state_keys: List[str], target_keys: set) -> str:
    prefixes = set(["", "module.", "model.", "base_model.", "base_model.model."])

    # get a few plausible prefixes from the checkpoint itself
    for k in state_keys[:2000]:
        parts = k.split(".")
        for n in (1, 2, 3):
            if len(parts) > n:
                prefixes.add(".".join(parts[:n]) + ".")

    best_p = ""
    best_overlap = -1

    for p in prefixes:
        overlap = 0
        for k in state_keys:
            kk = k[len(p):] if p and k.startswith(p) else k
            if kk in target_keys:
                overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_p = p

    return best_p


def _filter_ignorable_missing(missing_keys: List[str]) -> List[str]:
    ignore_suffixes = ("position_ids", "token_type_ids")
    keep = []
    for k in missing_keys:
        if k.endswith(ignore_suffixes):
            continue
        keep.append(k)
    return keep


def robust_load_model(model_dir: Path, *, task: str, device: str):
    """
    Loads a SequenceClassification model even if the saved state_dict has prefixes
    like 'model.' or 'module.' (DDP etc).
    """
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    # enforce sane label space if config is missing / generic (LABEL_0 etc)
    if task == "task_a":
        cfg.num_labels = 3
        if not getattr(cfg, "id2label", None) or any(str(v).startswith("LABEL_") for v in cfg.id2label.values()):
            cfg.id2label = {i: CLARITY_LABELS[i] for i in range(3)}
            cfg.label2id = {CLARITY_LABELS[i]: i for i in range(3)}
    elif task == "task_b":
        cfg.num_labels = 9
        if not getattr(cfg, "id2label", None) or any(str(v).startswith("LABEL_") for v in cfg.id2label.values()):
            cfg.id2label = {i: EVASION_LABELS[i] for i in range(9)}
            cfg.label2id = {EVASION_LABELS[i]: i for i in range(9)}

    model = AutoModelForSequenceClassification.from_config(cfg)

    weight_path = _find_weight_file(model_dir)
    if weight_path is None:
        raise RuntimeError(
            f"No weight file found in {model_dir} (expected model.safetensors / pytorch_model.*)"
        )

    sd = _load_state_dict_any(weight_path)

    target_keys = set(model.state_dict().keys())
    strip_p = _best_prefix_strip(list(sd.keys()), target_keys)

    remapped = {}
    for k, v in sd.items():
        kk = k[len(strip_p):] if strip_p and k.startswith(strip_p) else k
        remapped[kk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing2 = _filter_ignorable_missing(list(missing))

    total_keys = len(model.state_dict())
    bad_missing = len(missing2) > max(25, int(0.10 * total_keys))
    bad_unexpected = len(unexpected) > max(25, int(0.10 * total_keys))

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint keys still don't line up after prefix stripping.\n"
            f"model_dir={model_dir}\n"
            f"weights={weight_path.name}\n"
            f"prefix_stripped='{strip_p}'\n"
            f"missing={len(missing2)} unexpected={len(unexpected)}\n"
            f"missing_examples={missing2[:10]}\n"
            f"unexpected_examples={list(unexpected)[:10]}"
        )

    model.to(torch.device(device))
    model.eval()
    return model


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def batched_predict(
    *,
    model,
    tokenizer,
    questions: List[str],
    answers: List[str],
    input_format: str,
    device: str,
    batch_size: int,
    max_length: int,
    cd_vals: Optional[List[str]],
) -> np.ndarray:
    model.to(device)
    model.eval()

    preds: List[int] = []
    n = len(questions)

    # clamp to tokenizer limit
    try:
        tmax = int(getattr(tokenizer, "model_max_length", 0) or 0)
        if 0 < tmax < 100000:
            max_length = min(max_length, tmax)
    except Exception:
        pass

    for start in range(0, n, batch_size):
        q = questions[start:start + batch_size]
        a = answers[start:start + batch_size]

        if input_format == "qa_marked":
            if cd_vals is not None:
                c = cd_vals[start:start + batch_size]
                texts = [f"[{cc}] [QUESTION] {qq} [ANSWER] {aa}" for cc, qq, aa in zip(c, q, a)]
            else:
                texts = [f"[QUESTION] {qq} [ANSWER] {aa}" for qq, aa in zip(q, a)]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            # pair mode - training used (answer, question)
            enc = tokenizer(
                a,
                q,
                padding=True,
                truncation="longest_first",
                max_length=max_length,
                return_tensors="pt",
            )

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out[0]
            batch_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            preds.extend(batch_pred)

    return np.asarray(preds, dtype=np.int64)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _load_id2label(best_model_dir: Path) -> Dict[str, str]:
    cfg_path = best_model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg.get("id2label") or {}


def eval_task_a_from_df(
    pred_ids: np.ndarray,
    id2label: Dict[str, str],
    df: pd.DataFrame,
    label_col: str
) -> Dict[str, Any]:
    # truth
    y_true_raw = df[label_col].tolist()
    y_true = [_canonise_by_list(x, CLARITY_LABELS) for x in y_true_raw]

    # preds
    y_pred = []
    for i in pred_ids.tolist():
        lab = id2label.get(str(i), id2label.get(int(i), ""))
        lab = _canonise_by_list(lab, CLARITY_LABELS)
        y_pred.append(lab)

    macro = _macro_prf_str(y_true, y_pred, labels=CLARITY_LABELS)
    per_label = _per_label_prf_str(y_true, y_pred, labels=CLARITY_LABELS)
    cm = _confusion(y_true, y_pred, labels=CLARITY_LABELS)

    return {
        "clarity": macro,
        "clarity_per_label": per_label,
        "clarity_confusion_matrix": {
            "labels": CLARITY_LABELS,
            "matrix": cm,
        }
    }


def eval_task_b_from_df(
    pred_ids: np.ndarray,
    id2label: Dict[str, str],
    df: pd.DataFrame
) -> Dict[str, Any]:
    # gold
    ann1 = [_canonise_by_list(x, EVASION_LABELS) for x in df["annotator1"].tolist()]
    ann2 = [_canonise_by_list(x, EVASION_LABELS) for x in df["annotator2"].tolist()]
    ann3 = [_canonise_by_list(x, EVASION_LABELS) for x in df["annotator3"].tolist()]
    clarity_true = [_canonise_by_list(x, CLARITY_LABELS) for x in df["clarity_label"].tolist()]

    # predicted evasion label strings
    ev_pred = []
    for i in pred_ids.tolist():
        lab = id2label.get(str(i), id2label.get(int(i), ""))
        lab = _canonise_by_list(lab, EVASION_LABELS, fallback="General")
        # keep to known set
        if lab not in EVASION_LABELS:
            lab = "General"
        ev_pred.append(lab)

    # match-any accuracy
    acc_match = float(np.mean([
        (p == a1) or (p == a2) or (p == a3)
        for p, a1, a2, a3 in zip(ev_pred, ann1, ann2, ann3)
    ]))

    # per-annotator macro PRF (table uses macro)
    m1 = _macro_prf_str(ann1, ev_pred, labels=EVASION_LABELS)
    m2 = _macro_prf_str(ann2, ev_pred, labels=EVASION_LABELS)
    m3 = _macro_prf_str(ann3, ev_pred, labels=EVASION_LABELS)

    avg = {
        "f1": float((m1["f1"] + m2["f1"] + m3["f1"]) / 3.0),
        "p":  float((m1["p"] + m2["p"] + m3["p"]) / 3.0),
        "r":  float((m1["r"] + m2["r"] + m3["r"]) / 3.0),
    }

    # per-label breakdowns (per annotator)
    per1 = _per_label_prf_str(ann1, ev_pred, labels=EVASION_LABELS)
    per2 = _per_label_prf_str(ann2, ev_pred, labels=EVASION_LABELS)
    per3 = _per_label_prf_str(ann3, ev_pred, labels=EVASION_LABELS)

    cm1 = _confusion(ann1, ev_pred, labels=EVASION_LABELS)
    cm2 = _confusion(ann2, ev_pred, labels=EVASION_LABELS)
    cm3 = _confusion(ann3, ev_pred, labels=EVASION_LABELS)

    # derived clarity from evasion preds
    cl_pred = [EVASION_TO_CLARITY.get(e, "Ambivalent") for e in ev_pred]
    cl_macro = _macro_prf_str(clarity_true, cl_pred, labels=CLARITY_LABELS)
    cl_per = _per_label_prf_str(clarity_true, cl_pred, labels=CLARITY_LABELS)
    cl_cm = _confusion(clarity_true, cl_pred, labels=CLARITY_LABELS)

    return {
        "clarity": cl_macro,
        "clarity_per_label": cl_per,
        "clarity_confusion_matrix": {
            "labels": CLARITY_LABELS,
            "matrix": cl_cm,
        },
        "evasion": {
            "acc_match": acc_match,
            "annotator1": m1,
            "annotator2": m2,
            "annotator3": m3,
            "avg": avg,
            "per_label": {
                "annotator1": per1,
                "annotator2": per2,
                "annotator3": per3,
            },
            "confusion_matrices": {
                "labels": EVASION_LABELS,
                "annotator1": cm1,
                "annotator2": cm2,
                "annotator3": cm3,
            }
        }
    }


# ---------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------

def evaluate_seed(
    sr: SeedRun,
    *,
    df: pd.DataFrame,
    task: str,
    input_format: str,
    label_col_task_a: str,
    cd_mode: str,
    cd_col: str,
    cd_default: str,
    device: str,
    batch_size: int,
    max_length: int,
    save_predictions: bool,
    out_dir: Path,
    configs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    best = sr.best_model_dir

    # tokeniser + model
    tokenizer = AutoTokenizer.from_pretrained(best, use_fast=True, local_files_only=True)
    model = robust_load_model(best, task=task, device=device)

    # columns for q/a
    qcol, acol = _guess_q_a_cols(df)  # fallback
    qcol_cfg, acol_cfg, _ = _get_cols_from_config(configs)
    if qcol_cfg and acol_cfg and (qcol_cfg in df.columns) and (acol_cfg in df.columns):
        qcol, acol = qcol_cfg, acol_cfg

    questions = [_normalise_text(x) for x in df[qcol].tolist()]
    answers = [_normalise_text(x) for x in df[acol].tolist()]

    # CD behaviour
    apply_cd = _should_apply_cd(cd_mode=cd_mode, configs=configs, df=df, tokenizer=tokenizer)

    cd_vals: Optional[List[str]] = None
    if apply_cd:
        if cd_col in df.columns:
            cd_vals = [_normalise_cd_bucket(v, default=cd_default) for v in df[cd_col].tolist()]
        else:
            cd_vals = [cd_default] * len(df)

        if input_format == "pair":
            # prepend only to answer
            answers = [f"[{c}] {a}" for c, a in zip(cd_vals, answers)]
        else:
            # qa_marked gets cd_vals passed down and prefixed in text builder
            pass

        # not fatal but nice to know
        if ("[CD_LOW]" not in tokenizer.get_vocab()) or ("[CD_HIGH]" not in tokenizer.get_vocab()):
            print(f"  ⚠️ Seed {sr.seed}: CD tags not in vocab as single tokens - they'll be split (still might match training)")

    # predict
    pred_ids = batched_predict(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        answers=answers,
        input_format=input_format,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        cd_vals=(cd_vals if input_format == "qa_marked" else None),
    )

    id2label = getattr(getattr(model, "config", None), "id2label", None) or _load_id2label(best)

    # metrics
    if task == "task_a":
        if label_col_task_a not in df.columns:
            raise RuntimeError(f"Task A label column '{label_col_task_a}' not found in df columns: {list(df.columns)}")
        metrics = eval_task_a_from_df(pred_ids, id2label, df, label_col=label_col_task_a)
    elif task == "task_b":
        for c in ["annotator1", "annotator2", "annotator3", "clarity_label"]:
            if c not in df.columns:
                raise RuntimeError(f"Task B needs column '{c}' in test data. Columns: {list(df.columns)}")
        metrics = eval_task_b_from_df(pred_ids, id2label, df)
    else:
        raise ValueError(f"Unknown task: {task}")

    if save_predictions:
        out = df.copy()
        out["pred_label_id"] = pred_ids.tolist()
        out["pred_label"] = [id2label.get(str(i), id2label.get(int(i), "")) for i in pred_ids.tolist()]

        if task == "task_b":
            ev = [_canonise_by_list(x, EVASION_LABELS, fallback="General") for x in out["pred_label"].tolist()]
            ev = [x if x in EVASION_LABELS else "General" for x in ev]
            out["evasion_pred"] = ev
            out["clarity_pred"] = [EVASION_TO_CLARITY.get(x, "Ambivalent") for x in ev]

        pred_csv = out_dir / f"seed_{sr.seed}_predictions.csv"
        out.to_csv(pred_csv, index=False)

    return {
        "seed": sr.seed,
        "best_model_dir": str(best),
        "task": task,
        "input_format": input_format,
        "cd_applied": bool(apply_cd),
        "n": int(len(df)),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def _aggregate_per_label_across_seeds(
    per_seed: List[Dict[str, Any]],
    *,
    metric_key: str,   # e.g. "clarity_per_label"
    labels: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Produces:
      label -> metric -> {mean,std}
    metrics are p/r/f1 (support is kept as int from the first seed)
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for lab in labels:
        vals_p, vals_r, vals_f1 = [], [], []
        support = None

        for s in per_seed:
            block = s["metrics"].get(metric_key) or {}
            if lab not in block:
                continue
            vals_p.append(block[lab]["p"])
            vals_r.append(block[lab]["r"])
            vals_f1.append(block[lab]["f1"])
            if support is None:
                support = int(block[lab].get("support", 0))

        mp, sp = _mean_std(vals_p)
        mr, sr = _mean_std(vals_r)
        mf, sf = _mean_std(vals_f1)

        out[lab] = {
            "p": {"mean": mp, "std": sp},
            "r": {"mean": mr, "std": sr},
            "f1": {"mean": mf, "std": sf},
            "support": int(support if support is not None else 0),
        }

    return out


def _aggregate_taskb_evasion_per_label(
    per_seed: List[Dict[str, Any]],
    labels: List[str],
) -> Dict[str, Any]:
    """
    Aggregates evasion per-label metrics across seeds, separately for each annotator.

    Returns:
      {
        "annotator1": {label -> {p/r/f1 mean/std + support}},
        "annotator2": ...
        "annotator3": ...
      }
    """
    out = {}
    for ann in ["annotator1", "annotator2", "annotator3"]:
        ann_out: Dict[str, Any] = {}
        for lab in labels:
            vp, vr, vf = [], [], []
            support = None
            for s in per_seed:
                ev = s["metrics"].get("evasion", {}) or {}
                per = (ev.get("per_label", {}) or {}).get(ann, {}) or {}
                if lab not in per:
                    continue
                vp.append(per[lab]["p"])
                vr.append(per[lab]["r"])
                vf.append(per[lab]["f1"])
                if support is None:
                    support = int(per[lab].get("support", 0))

            mp, sp = _mean_std(vp)
            mr, sr = _mean_std(vr)
            mf, sf = _mean_std(vf)

            ann_out[lab] = {
                "p": {"mean": mp, "std": sp},
                "r": {"mean": mr, "std": sr},
                "f1": {"mean": mf, "std": sf},
                "support": int(support if support is not None else 0),
            }
        out[ann] = ann_out
    return out


def aggregate(per_seed: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    cl_f1 = [x["metrics"]["clarity"]["f1"] for x in per_seed]
    cl_p  = [x["metrics"]["clarity"]["p"] for x in per_seed]
    cl_r  = [x["metrics"]["clarity"]["r"] for x in per_seed]

    agg = {
        "clarity": {
            "f1": {"mean": _mean_std(cl_f1)[0], "std": _mean_std(cl_f1)[1]},
            "p":  {"mean": _mean_std(cl_p)[0],  "std": _mean_std(cl_p)[1]},
            "r":  {"mean": _mean_std(cl_r)[0],  "std": _mean_std(cl_r)[1]},
        },
        "clarity_per_label": _aggregate_per_label_across_seeds(
            per_seed,
            metric_key="clarity_per_label",
            labels=CLARITY_LABELS,
        ),
        "evasion": None,
        "evasion_per_label": None,
        "table_row_means": {
            "f1": _mean_std(cl_f1)[0],
            "p":  _mean_std(cl_p)[0],
            "r":  _mean_std(cl_r)[0],
        }
    }

    if task == "task_b":
        acc = [x["metrics"]["evasion"]["acc_match"] for x in per_seed]
        f1a1 = [x["metrics"]["evasion"]["annotator1"]["f1"] for x in per_seed]
        f1a2 = [x["metrics"]["evasion"]["annotator2"]["f1"] for x in per_seed]
        f1a3 = [x["metrics"]["evasion"]["annotator3"]["f1"] for x in per_seed]
        favg = [x["metrics"]["evasion"]["avg"]["f1"] for x in per_seed]

        agg["evasion"] = {
            "acc_match": {"mean": _mean_std(acc)[0], "std": _mean_std(acc)[1]},
            "f1_a1": {"mean": _mean_std(f1a1)[0], "std": _mean_std(f1a1)[1]},
            "f1_a2": {"mean": _mean_std(f1a2)[0], "std": _mean_std(f1a2)[1]},
            "f1_a3": {"mean": _mean_std(f1a3)[0], "std": _mean_std(f1a3)[1]},
            "f1_avg": {"mean": _mean_std(favg)[0], "std": _mean_std(favg)[1]},
            "p_avg": {"mean": _mean_std([x["metrics"]["evasion"]["avg"]["p"] for x in per_seed])[0],
                      "std":  _mean_std([x["metrics"]["evasion"]["avg"]["p"] for x in per_seed])[1]},
            "r_avg": {"mean": _mean_std([x["metrics"]["evasion"]["avg"]["r"] for x in per_seed])[0],
                      "std":  _mean_std([x["metrics"]["evasion"]["avg"]["r"] for x in per_seed])[1]},
        }

        agg["evasion_per_label"] = _aggregate_taskb_evasion_per_label(per_seed, labels=EVASION_LABELS)

        agg["table_row_means"].update({
            "acc_match": _mean_std(acc)[0],
            "f1_a1": _mean_std(f1a1)[0],
            "f1_a2": _mean_std(f1a2)[0],
            "f1_a3": _mean_std(f1a3)[0],
            "f1_avg": _mean_std(favg)[0],
        })

    return agg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Local multi-seed aggregation for CLARITY checkpoints (no wandb)")
    ap.add_argument("--experiment-dir", type=str, required=True, help="Folder containing seed_*/best_model")
    ap.add_argument("--out-dir", type=str, default=None, help="Output folder (default: <experiment-dir>/evaluation_tables_local)")
    ap.add_argument("--config", type=str, default=None, help="Experiment config yaml (recommended, esp for CD/CSV datasets)")
    ap.add_argument("--task", type=str, default="auto", choices=["auto", "task_a", "task_b"])
    ap.add_argument("--input-format", type=str, default="auto", choices=["auto", "pair", "qa_marked"])
    ap.add_argument("--dataset", type=str, default="ailsntua/QEvasion", help="HF dataset name if using HF source")
    ap.add_argument("--test-csv", type=str, default=None, help="Explicit test CSV path (overrides config/HF)")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Debug: only first N examples")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="Ignore cached per_seed JSON and rerun")
    ap.add_argument("--cd", type=str, default="auto", choices=["auto", "on", "off"], help="Apply CD prefixing")
    ap.add_argument("--cd-col", type=str, default="cd_bucket", help="Column name for CD bucket")
    ap.add_argument("--cd-default", type=str, default="CD_LOW", choices=["CD_LOW", "CD_HIGH"], help="Default CD bucket")
    ap.add_argument("--task-a-label-col", type=str, default="clarity_label", help="Truth label column for Task A")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (exp_dir / "evaluation_tables_local")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_experiment_config_if_available(args.config)

    device = _pick_device(args.device)
    input_format = _infer_input_format(exp_dir, args.input_format)

    seed_runs = find_seed_runs(exp_dir)
    if not seed_runs:
        raise RuntimeError(f"No seed runs found under {exp_dir} (expected seed_*/best_model)")

    # infer task
    task = args.task
    if task == "auto":
        guess = _infer_task_from_checkpoint(seed_runs[0].best_model_dir)
        task = guess if guess != "unknown" else "task_a"

    # choose test source (csv/hf)
    source_kind, source_info = _resolve_test_source(
        configs=configs,
        dataset_name=args.dataset,
        explicit_test_csv=args.test_csv,
    )
    df, meta = load_test_df(source_kind, source_info, limit=args.limit)

    # if config provides label col for Task 1, prefer it
    _, _, label_col_cfg = _get_cols_from_config(configs)
    label_col_task_a = label_col_cfg if (label_col_cfg and label_col_cfg in df.columns) else args.task_a_label_col

    print(f"Experiment: {exp_dir.name}")
    print(f"Seeds: {[s.seed for s in seed_runs]}")
    print(f"Task: {task}")
    print(f"Input format: {input_format}")
    print(f"Device: {device}")
    print(f"Test source: {source_kind} | info: {meta}")
    if task == "task_a":
        print(f"Task A label column: {label_col_task_a}")

    # cache
    per_seed_path = out_dir / "per_seed_test_metrics.json"
    per_seed: List[Dict[str, Any]] = []

    if per_seed_path.exists() and not args.recompute:
        try:
            per_seed = json.loads(per_seed_path.read_text(encoding="utf-8"))
            if not isinstance(per_seed, list):
                per_seed = []
        except Exception:
            per_seed = []

    if not per_seed:
        for sr in seed_runs:
            print(f"\n--- Evaluating seed {sr.seed} ---")
            res = evaluate_seed(
                sr,
                df=df,
                task=task,
                input_format=input_format,
                label_col_task_a=label_col_task_a,
                cd_mode=args.cd,
                cd_col=args.cd_col,
                cd_default=args.cd_default,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                save_predictions=args.save_predictions,
                out_dir=out_dir,
                configs=configs,
            )
            per_seed.append(res)

        per_seed_path.write_text(json.dumps(per_seed, indent=2), encoding="utf-8")
        print(f"\nSaved per-seed metrics: {per_seed_path}")
    else:
        print(f"\nLoaded cached per-seed metrics: {per_seed_path}")

    agg = aggregate(per_seed, task=task)

    payload = {
        "schema": "clarity_local_aggregation_v2",
        "experiment_dir": str(exp_dir),
        "experiment_name": exp_dir.name,
        "task": task,
        "split": "test",
        "input_format": input_format,
        "device": device,
        "test_source": {"kind": source_kind, **meta},
        "cd": {"mode": args.cd, "cd_col": args.cd_col, "cd_default": args.cd_default},
        "seeds": [x["seed"] for x in per_seed],
        "per_seed": per_seed,
        "aggregate": agg,
    }

    out_json = out_dir / "aggregate_test_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote aggregate JSON: {out_json}")

    row = payload["aggregate"]["table_row_means"]
    if task == "task_b":
        print("\nTable row means (Task B style):")
        print(f"  Clarity  F1/P/R : {row['f1']:.4f} / {row['p']:.4f} / {row['r']:.4f}")
        print(f"  ACC_match       : {row['acc_match']:.4f}")
        print(f"  F1_A1/A2/A3     : {row['f1_a1']:.4f} / {row['f1_a2']:.4f} / {row['f1_a3']:.4f}")
        print(f"  F1_avg          : {row['f1_avg']:.4f}")
        print("  (per-label breakdowns are in aggregate_test_metrics.json now)")
    else:
        print("\nTable row means (Task A style):")
        print(f"  Clarity  F1/P/R : {row['f1']:.4f} / {row['p']:.4f} / {row['r']:.4f}")
        print("  (per-label breakdowns are in aggregate_test_metrics.json now)")


if __name__ == "__main__":
    main()