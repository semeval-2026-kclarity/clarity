#!/usr/bin/env python3
"""
Ensemble inference + Codabench packaging for CLARITY Task 1.

Does NOT modify any saved model files.

Output:
- <outdir>/prediction  (NO extension)
- <outdir>/prediction.zip  (zip contains ONLY prediction, no folders)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


# Fixed mapping from your task config
IDX2LABEL = {
    0: "Clear Reply",
    1: "Ambivalent",
    2: "Clear Non-Reply",
}
LABELS = [IDX2LABEL[i] for i in range(3)]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def discover_best_models(models_root: Path) -> list[Path]:
    model_dirs = sorted([p for p in models_root.glob("seed_*/best_model") if p.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No best_model dirs found under: {models_root} (expected seed_*/best_model)")
    return model_dirs


def pick_text_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = set(df.columns)

    if "question" in cols:
        qcol = "question"
    elif "interview_question" in cols:
        qcol = "interview_question"
    else:
        raise ValueError(f"Couldn't find a question column. Columns: {sorted(cols)}")

    if "interview_answer" in cols:
        acol = "interview_answer"
    elif "answer" in cols:
        acol = "answer"
    elif "response" in cols:
        acol = "response"
    else:
        raise ValueError(f"Couldn't find an answer column. Columns: {sorted(cols)}")

    return qcol, acol


def build_qa_marked(question: str, answer: str) -> str:
    q = "" if question is None else str(question)
    a = "" if answer is None else str(answer)
    if q == "nan":
        q = ""
    if a == "nan":
        a = ""
    return f"[QUESTION] {q.strip()} [ANSWER] {a.strip()}"


def find_weight_file(model_dir: Path) -> Path | None:
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_state_dict_any(weight_path: Path) -> dict:
    if weight_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("Need safetensors installed to read .safetensors weights") from e
        return load_file(str(weight_path))
    return torch.load(str(weight_path), map_location="cpu")


def best_prefix_strip(state_keys: list[str], target_keys: set[str]) -> str:
    """
    Tries to find the best prefix to remove so checkpoint keys match model keys.
    """
    prefixes = set(["", "module.", "model.", "base_model.", "base_model.model."])

    # also add prefixes derived from the actual ckpt keys (first 1-3 segments)
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


def _filter_ignorable_missing(missing_keys: list[str]) -> list[str]:
    """
    Some models have buffers like position_ids that may not be in the checkpoint.
    """
    ignore_suffixes = ("position_ids", "token_type_ids")
    keep = []
    for k in missing_keys:
        if k.endswith(ignore_suffixes):
            continue
        keep.append(k)
    return keep


def robust_load_model(model_dir: Path, device: torch.device) -> AutoModelForSequenceClassification:
    """
    Always loads weights manually from the weight file, and strips prefixes in-memory.
    """
    logging.info(f"Loading model (manual weights): {model_dir}")

    # load config and force correct label mapping
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.num_labels = 3
    config.id2label = {0: IDX2LABEL[0], 1: IDX2LABEL[1], 2: IDX2LABEL[2]}
    config.label2id = {IDX2LABEL[0]: 0, IDX2LABEL[1]: 1, IDX2LABEL[2]: 2}

    model = AutoModelForSequenceClassification.from_config(config)

    weight_path = find_weight_file(model_dir)
    if weight_path is None:
        raise RuntimeError(f"No weight file found in {model_dir} (expected model.safetensors / pytorch_model.*)")

    size_mb = weight_path.stat().st_size / (1024 * 1024)
    logging.info(f"  Found weights: {weight_path.name} ({size_mb:.1f} MB)")

    sd = load_state_dict_any(weight_path)

    target_keys = set(model.state_dict().keys())
    strip_p = best_prefix_strip(list(sd.keys()), target_keys)
    logging.info(f"  Best prefix to strip: '{strip_p}'")

    remapped = {}
    for k, v in sd.items():
        kk = k[len(strip_p):] if strip_p and k.startswith(strip_p) else k
        remapped[kk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing2 = _filter_ignorable_missing(list(missing))

    logging.info(f"  Manual load - missing={len(missing2)} unexpected={len(unexpected)} (ignoring tiny buffer stuff)")

    # if most of the model didn't load, just fail fast
    total_keys = len(model.state_dict())
    bad_missing = len(missing2) > max(25, int(0.10 * total_keys))
    bad_unexpected = len(unexpected) > max(25, int(0.10 * total_keys))

    if bad_missing or bad_unexpected:
        logging.error(f"  Example missing keys: {missing2[:10]}")
        logging.error(f"  Example unexpected keys: {list(unexpected)[:10]}")
        raise RuntimeError(
            "Weights still don't line up after prefix stripping. "
            "This usually means the checkpoint is not a full HF model state_dict "
            "(or it's from a different architecture)."
        )

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_with_model(
    model_dir: Path,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = robust_load_model(model_dir, device)

    all_preds = []
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(torch.argmax(probs, dim=-1).detach().cpu().numpy())

        if (start // batch_size) % 20 == 0:
            logging.info(f"  Inference progress: {min(start + batch_size, len(texts))}/{len(texts)}")

    preds = np.concatenate(all_preds, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    return preds, probs


def majority_vote(pred_matrix: np.ndarray, prob_tensor: np.ndarray) -> np.ndarray:
    M, N = pred_matrix.shape
    C = prob_tensor.shape[-1]
    final = np.zeros((N,), dtype=np.int64)

    for i in range(N):
        votes = pred_matrix[:, i]
        counts = np.bincount(votes, minlength=C)
        top = counts.max()
        winners = np.where(counts == top)[0]

        if len(winners) == 1:
            final[i] = int(winners[0])
        else:
            # tie break - mean probs for tied classes
            mean_probs = prob_tensor[:, i, :].mean(axis=0)
            best = max(winners, key=lambda c: float(mean_probs[c]))
            final[i] = int(best)

    return final


def write_prediction_file(out_path: Path, label_lines: list[str]) -> None:
    out_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def zip_prediction_file(pred_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction")


def main():
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--expected-rows", type=int, default=237)
    ap.add_argument("--allow-wrong-rows", action="store_true", help="Don't crash if row count != expected")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    models_root = Path(args.models_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    logging.info(f"Using device: {device}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info(f"Loaded CSV: {csv_path} | rows={len(df)} cols={len(df.columns)}")

    if len(df) != args.expected_rows:
        msg = (
            f"Row count is {len(df)} but expected {args.expected_rows}. "
            "Codabench will reject the submission if this is wrong. "
            "Check you're using the official test CSV."
        )
        if args.allow_wrong_rows:
            logging.warning(msg)
        else:
            raise RuntimeError(msg)

    qcol, acol = pick_text_columns(df)
    logging.info(f"Using columns - question: '{qcol}', answer: '{acol}'")

    texts = [build_qa_marked(q, a) for q, a in zip(df[qcol].tolist(), df[acol].tolist())]
    logging.info(f"Built qa_marked texts: {len(texts)}")

    model_dirs = discover_best_models(models_root)
    logging.info(f"Found {len(model_dirs)} model(s):")
    for p in model_dirs:
        w = find_weight_file(p)
        if w is not None:
            size_mb = w.stat().st_size / (1024 * 1024)
            logging.info(f"  - {p} | weights={w.name} ({size_mb:.1f} MB)")
        else:
            logging.info(f"  - {p} | weights=NOT FOUND")

    per_model_preds = []
    per_model_probs = []

    for i, model_dir in enumerate(model_dirs, start=1):
        logging.info(f"[{i}/{len(model_dirs)}] Predicting with {model_dir.name} ...")

        preds, probs = predict_with_model(
            model_dir=model_dir,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        per_model_preds.append(preds)
        per_model_probs.append(probs)

        uniq, cnt = np.unique(preds, return_counts=True)
        dist = {IDX2LABEL[int(k)]: int(v) for k, v in zip(uniq, cnt)}
        logging.info(f"  Pred distribution: {dist}")

    pred_matrix = np.stack(per_model_preds, axis=0)
    prob_tensor = np.stack(per_model_probs, axis=0)

    logging.info("Ensembling with majority vote...")
    final_ids = majority_vote(pred_matrix, prob_tensor)
    final_labels = [IDX2LABEL[int(i)] for i in final_ids]

    pred_path = outdir / "prediction"
    write_prediction_file(pred_path, final_labels)
    logging.info(f"Wrote prediction file: {pred_path} (lines={len(final_labels)})")

    zip_path = outdir / "prediction.zip"
    zip_prediction_file(pred_path, zip_path)
    logging.info(f"Wrote zip file: {zip_path}")

    logging.info("Done. Upload prediction.zip (zip should contain only 'prediction').")


if __name__ == "__main__":
    main()
