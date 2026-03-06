#!/usr/bin/env python3
"""
Compute seed-averaged confusion matrices for clarity + evasion, then write:

  - CSVs (raw + optional row-normalised)
  - Heatmaps (raw + optional row-normalised)

Input JSON is expected to look like:
  data["per_seed"][i]["metrics"]["clarity_confusion_matrix"] -> {"labels": [...], "matrix": [...]}
  data["per_seed"][i]["metrics"]["evasion"]["confusion_matrices"] -> {"labels": [...], "annotator1": [...], ...}

Usage:
  python scripts/visualisation/render_confusion_matrix.py \
    ../model_outputs/task_b/roberta_large_task_b_stratified_qa_marked/evaluation_tables_local/aggregate_test_metrics.json \
    --outdir ./paper_artifacts/confusion_matrix \
    --normalise \
    --no_fig_titles
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _set_paperish_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False,
    })


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _as_np(mat) -> np.ndarray:
    arr = np.array(mat, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    return arr


def _check_labels(all_labels: List[List[str]], name: str) -> List[str]:
    base = all_labels[0]
    for i, labs in enumerate(all_labels[1:], start=1):
        if labs != base:
            raise ValueError(
                f"{name} labels mismatch at index {i}.\n"
                f"base: {base}\nthis: {labs}"
            )
    return base


def _mean_stack(mats: List[np.ndarray]) -> np.ndarray:
    if not mats:
        raise ValueError("No matrices to average")
    stacked = np.stack(mats, axis=0)  # (k, r, c)
    return stacked.mean(axis=0)


def _row_normalise(mat: np.ndarray) -> np.ndarray:
    # normalise each row to sum to 1 (if row is all zeros, leave it zeros)
    row_sums = mat.sum(axis=1, keepdims=True)

    out = np.zeros_like(mat, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(mat, row_sums, out=out, where=row_sums != 0)

    out[np.isnan(out)] = 0.0
    return out


def write_csv(path: str, labels: List[str], mat: np.ndarray) -> None:
    # CSV with header row + first column as true label
    with open(path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for lab, row in zip(labels, mat):
            # keep a few decimals if it's averaged, otherwise ints are fine
            row_str = ",".join(f"{v:.4f}" for v in row)
            f.write(f"{lab},{row_str}\n")


def plot_cm(
    path: str,
    labels: List[str],
    mat: np.ndarray,
    title: str,
    normalised: bool = False,
    show_title: bool = True,
) -> None:
    n = len(labels)

    if n <= 4:
        fig_w, fig_h = 6.2, 5.2
    elif n <= 7:
        fig_w, fig_h = 8.5, 7.2
    else:
        fig_w, fig_h = 9.5, 8.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, interpolation="nearest", cmap="Reds", aspect="equal")

    if show_title:
        ax.set_title(title, pad=18, fontweight="bold")
    ax.set_xlabel("Predicted", labelpad=14)
    ax.set_ylabel("Gold", labelpad=14)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    ax.tick_params(axis="x", which="major", pad=7)
    ax.tick_params(axis="y", which="major", pad=7)

    # subtle cell gridlines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fmt = ".2f" if normalised else ".1f"
    thresh = mat.max() * 0.6 if mat.size else 0.0

    base_fs = 8 if n > 6 else 10
    fs = base_fs - 1 if normalised else base_fs

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                fontsize=fs,
                color="white" if val > thresh else "black",
            )

    fig.tight_layout(pad=1.4)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def extract_clarity(per_seed: List[dict]) -> Tuple[List[str], List[np.ndarray]]:
    labels_all: List[List[str]] = []
    mats: List[np.ndarray] = []

    for entry in per_seed:
        cm = entry["metrics"]["clarity_confusion_matrix"]
        labels_all.append(cm["labels"])
        mats.append(_as_np(cm["matrix"]))

    labels = _check_labels(labels_all, "clarity")
    return labels, mats


def extract_evasion(per_seed: List[dict]) -> Tuple[List[str], Dict[str, List[np.ndarray]]]:
    labels_all: List[List[str]] = []
    by_annotator: Dict[str, List[np.ndarray]] = {}

    for entry in per_seed:
        cms = entry["metrics"]["evasion"]["confusion_matrices"]
        labels_all.append(cms["labels"])

        for k, v in cms.items():
            if k == "labels":
                continue
            # expect keys like annotator1/annotator2/annotator3
            by_annotator.setdefault(k, []).append(_as_np(v))

    labels = _check_labels(labels_all, "evasion")

    n_seeds = len(per_seed)
    for ann, mats in by_annotator.items():
        if len(mats) != n_seeds:
            raise ValueError(f"{ann} has {len(mats)} matrices, expected {n_seeds}")

    return labels, by_annotator


def main() -> None:
    _set_paperish_style()

    ap = argparse.ArgumentParser(
        description="Compute seed-averaged confusion matrices for clarity and evasion."
    )
    ap.add_argument("json_path", help="Path to the JSON file")
    ap.add_argument("--outdir", default="cm_out", help="Output directory")
    ap.add_argument(
        "--normalise",
        action="store_true",
        help="Also output row-normalised (per-gold-label) heatmaps/CSVs",
    )
    ap.add_argument(
        "--no_fig_titles",
        action="store_true",
        help="Do not draw titles on the figures (use LaTeX captions instead).",
    )
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_seed = data.get("per_seed", [])
    if not per_seed:
        raise ValueError("JSON has no 'per_seed' entries")

    show_title = not args.no_fig_titles

    # ---- A) Clarity (seed-averaged) ----
    clarity_labels, clarity_mats = extract_clarity(per_seed)
    clarity_mean = _mean_stack(clarity_mats)

    write_csv(os.path.join(args.outdir, "clarity_cm_seed_mean.csv"), clarity_labels, clarity_mean)
    plot_cm(
        os.path.join(args.outdir, "clarity_cm_seed_mean.png"),
        clarity_labels,
        clarity_mean,
        title="Evasion-based clarity confusion matrix",
        normalised=False,
        show_title=show_title,
    )

    if args.normalise:
        clarity_norm = _row_normalise(clarity_mean)
        write_csv(
            os.path.join(args.outdir, "clarity_cm_seed_mean_row_norm.csv"),
            clarity_labels,
            clarity_norm,
        )
        plot_cm(
            os.path.join(args.outdir, "clarity_cm_seed_mean_row_norm.png"),
            clarity_labels,
            clarity_norm,
            title="Evasion-based clarity confusion matrix (row-normalised)",
            normalised=True,
            show_title=show_title,
        )

    # ---- B) Evasion (seed-averaged) ----
    ev_labels, ev_by_ann = extract_evasion(per_seed)

    # per-annotator seed mean
    ev_ann_means: Dict[str, np.ndarray] = {}
    for ann, mats in sorted(ev_by_ann.items()):
        ev_ann_means[ann] = _mean_stack(mats)
        write_csv(
            os.path.join(args.outdir, f"evasion_cm_seed_mean_{ann}.csv"),
            ev_labels,
            ev_ann_means[ann],
        )
        plot_cm(
            os.path.join(args.outdir, f"evasion_cm_seed_mean_{ann}.png"),
            ev_labels,
            ev_ann_means[ann],
            title=f"Evasion confusion matrix ({ann})",
            normalised=False,
            show_title=show_title,
        )

        if args.normalise:
            ann_norm = _row_normalise(ev_ann_means[ann])
            write_csv(
                os.path.join(args.outdir, f"evasion_cm_seed_mean_{ann}_row_norm.csv"),
                ev_labels,
                ann_norm,
            )
            plot_cm(
                os.path.join(args.outdir, f"evasion_cm_seed_mean_{ann}_row_norm.png"),
                ev_labels,
                ann_norm,
                title=f"Evasion confusion matrix (seed-mean, {ann}, row-normalised)",
                normalised=True,
                show_title=show_title,
            )

    # overall average across annotators (and seeds)
    all_mats: List[np.ndarray] = []
    for mats in ev_by_ann.values():
        all_mats.extend(mats)
    ev_overall_mean = _mean_stack(all_mats)

    write_csv(os.path.join(args.outdir, "evasion_cm_seed_mean_overall.csv"), ev_labels, ev_overall_mean)
    plot_cm(
        os.path.join(args.outdir, "evasion_cm_seed_mean_overall.png"),
        ev_labels,
        ev_overall_mean,
        title="Evasion confusion matrix (mean over seeds+annotators)",
        normalised=False,
        show_title=show_title,
    )

    if args.normalise:
        overall_norm = _row_normalise(ev_overall_mean)
        write_csv(
            os.path.join(args.outdir, "evasion_cm_seed_mean_overall_row_norm.csv"),
            ev_labels,
            overall_norm,
        )
        plot_cm(
            os.path.join(args.outdir, "evasion_cm_seed_mean_overall_row_norm.png"),
            ev_labels,
            overall_norm,
            title="Evasion confusion matrix (row-normalised)",
            normalised=True,
            show_title=show_title,
        )

    print(f"Done. Wrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()