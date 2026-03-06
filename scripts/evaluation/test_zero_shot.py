#!/usr/bin/env python3
"""
CLARITY Task B (evasion level) + Task A (clarity level) evaluation
using HuggingFace Inference API.

- Predicts evasion label using the same taxonomy prompt from make_evasion_submission_gpt.py
- Infers clarity label from the evasion->clarity mapping
- Evaluates ONLY on the test split of ailsntua/QEvasion
- Evasion metrics:
    * Match-any accuracy: prediction is correct if it matches ANY of the 3 annotators
    * Per-annotator F1 / Precision / Recall (macro) for each of the 3 annotators
    * Average F1 / Precision / Recall across the 3 annotators (main comparison number)
- Clarity: prediction is evaluated against the single clarity_label
- Supports multiple models in a single run

Supported Gemma 3 models (via HF router):
  google/gemma-3-27b-it  -> routed via Scaleway (only Gemma 3 size available on HF)
  google/gemma-3-4b-it and google/gemma-3-12b-it are NOT on any HF provider

Usage:
    python test_zero_shot.py
    python test_zero_shot.py --model meta-llama/Llama-3.1-8B-Instruct
    python test_zero_shot.py --model meta-llama/Llama-3.1-8B-Instruct google/gemma-3-27b-it
    python test_zero_shot.py --no-cache --output-dir my_results/
    python test_zero_shot.py --batch-size 8 --model modelA modelB modelC
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import requests
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = "INSERT_YOUR_HF_TOKEN_HERE"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Models that need a specific provider on the HF router.
MODEL_PROVIDER_MAP = {
    "google/gemma-3-27b-it": "scaleway",
}

MODEL_BATCH_SLEEP = {
    "openai/gpt-oss-120b": 5.0,
    "openai/gpt-oss-20b":  3.0,
}

MODEL_MAX_TOKENS = {
    "openai/gpt-oss-120b": 4000,
    "openai/gpt-oss-20b":  3000,
}

MODEL_BATCH_SIZE = {
    "openai/gpt-oss-120b": 4,
    "openai/gpt-oss-20b":  4,
}

DISABLE_THINKING_MODELS: set[str] = set()

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


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_taxonomy_prompt() -> str:
    return (
        "You are an expert annotator for CLARITY Task B (evasion level).\n"
        "Input: a QUESTION and an ANSWER (political interview style).\n"
        "Output: EXACTLY ONE label for each item from this set:\n"
        f"{', '.join(EVASION_LABELS)}.\n\n"
        "Core principle: decide based on whether the ANSWER supplies the *requested commitment*.\n"
        "Requested commitment = the specific yes/no, person, time, place, number, policy stance, or concrete plan the QUESTION asks for.\n"
        "If that commitment is present (even indirectly), it is NOT Dodging/Deflection.\n\n"
        "Step 0 - normalise the question:\n"
        "- Treat multi-part questions as requiring ALL parts unless the question clearly foregrounds one part.\n"
        "- If the question contains a yes/no + 'why/how/what specifics', then a bare yes/no is incomplete.\n\n"
        "Decision ladder (apply in order; stop at the first that clearly applies):\n"
        "1) Clarification\n"
        "   - The answer primarily asks the interviewer to repeat/clarify/restate.\n"
        "   - If it both asks for clarification AND gives the requested commitment, choose based on the dominant function.\n\n"
        "2) Claims ignorance\n"
        "   - The answer asserts lack of knowledge/recall/awareness.\n"
        "   - IMPORTANT: If the answer later gives the requested commitment anyway, do NOT pick Claims ignorance.\n\n"
        "3) Declining to answer\n"
        "   - The answer refuses to provide the requested commitment now.\n"
        "   - If it refuses but then gives the requested commitment, label by what dominates.\n\n"
        "4) Explicit\n"
        "   - Directly states the requested commitment (even if brief), in the form requested.\n\n"
        "5) Implicit\n"
        "   - The requested commitment is not stated verbatim, but is clearly recoverable via a straightforward inference.\n"
        "   - Test: a reasonable listener could paraphrase the commitment in one sentence without guessing.\n\n"
        "6) Partial/half-answer\n"
        "   - Answers ONE required part but omits other required parts.\n\n"
        "7) General vs Deflection vs Dodging (non-answer family)\n"
        "   7a) General - On-topic, but avoids the requested commitment by staying vague or non-committal.\n"
        "   7b) Deflection - Acknowledges the question then shifts to a different frame. No requested commitment.\n"
        "   7c) Dodging - Does NOT engage the requested commitment AND does NOT meaningfully address the topic.\n\n"
        "Tie-breakers:\n"
        "- Concrete list of steps in response to 'what will you do' -> prefer Explicit (or Partial if incomplete).\n"
        "- Gives constraints but never the asked commitment -> Declining to answer.\n"
        "- Explains the topic but not the specific asked stance -> General or Deflection, not Dodging.\n"
        "- 'I already answered that' without the commitment -> Declining to answer.\n\n"
        "Mini-examples:\n"
        "A) Dodging vs Deflection\n"
        "Q: Did you meet the leader yesterday?\n"
        "A (Dodging): Our country is doing very well economically.\n"
        "A (Deflection): I understand the question - what matters is focusing on jobs for families.\n\n"
        "B) General vs Implicit\n"
        "Q: Do you support the bill?\n"
        "A (General): We're considering all options and working with colleagues.\n"
        "A (Implicit): I'll be voting yes when it reaches the floor.\n\n"
        "C) Declining vs General\n"
        "Q: What concessions do you want them to make?\n"
        "A (Declining): I'll tell you after the meeting.\n"
        "A (General): We want cooperation and stability.\n\n"
        "D) Claims ignorance vs General\n"
        "Q: Have you seen evidence of acceleration?\n"
        "A (Claims ignorance): I haven't seen that report.\n"
        "A (General): We're concerned and monitoring closely.\n\n"
        "Output rules:\n"
        "- Choose exactly one label per item.\n"
        "- Keep the item order exactly.\n"
        "- Judge using only the provided QUESTION and ANSWER.\n"
        "\nRespond ONLY with a JSON object in this exact format: {\"labels\": [\"label1\", \"label2\", ...]}\n"
        "No explanation, no markdown fences, only the JSON.\n"
    )


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def extract_evasion_label_from_text(text: str) -> str:
    if not text:
        return "General"
    tl = text.lower()
    patterns = {
        "Partial/half-answer": ["partial/half-answer", "partial/half answer", "half-answer", "half answer", "partial answer", "partial"],
        "Claims ignorance":    ["claims ignorance", "claim ignorance", "claiming ignorance"],
        "Declining to answer": ["declining to answer", "decline to answer", "refuses to answer", "declining"],
        "Clarification":       ["clarification"],
        "Deflection":          ["deflection", "deflecting"],
        "Dodging":             ["dodging", "dodge"],
        "Explicit":            ["explicit"],
        "Implicit":            ["implicit"],
        "General":             ["general"],
    }
    for label, pats in patterns.items():
        for p in pats:
            if p in tl:
                return label
    return "General"


def parse_labels_from_response(raw: str, expected_n: int) -> list[str]:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "labels" in data:
            labels = data["labels"]
            if isinstance(labels, list):
                validated = [l if l in EVASION_LABELS else extract_evasion_label_from_text(l) for l in labels]
                while len(validated) < expected_n:
                    validated.append("General")
                return validated[:expected_n]
    except Exception:
        pass
    try:
        arr_match = re.search(r'\[.*?\]', clean, re.DOTALL)
        if arr_match:
            labels = json.loads(arr_match.group())
            if isinstance(labels, list):
                validated = [l if l in EVASION_LABELS else extract_evasion_label_from_text(l) for l in labels]
                while len(validated) < expected_n:
                    validated.append("General")
                return validated[:expected_n]
    except Exception:
        pass
    lines = [l.strip() for l in clean.split('\n') if l.strip()]
    labels = []
    for line in lines:
        labels.append(extract_evasion_label_from_text(line))
        if len(labels) == expected_n:
            break
    while len(labels) < expected_n:
        labels.append("General")
    return labels


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def get_router_url(model: str) -> str:
    provider = MODEL_PROVIDER_MAP.get(model)
    if provider:
        return f"https://router.huggingface.co/{provider}/v1/chat/completions"
    return "https://router.huggingface.co/v1/chat/completions"


def call_hf_for_batch(
    hf_token: str,
    model: str,
    system_prompt: str,
    items: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_retries: int = 6,
    base_sleep_s: float = 2.0,
) -> list[str]:
    url = get_router_url(model)
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    user_payload = json.dumps({"items": items}, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Label each item.\nInput JSON:\n{user_payload}"},
        ],
        "max_tokens": MODEL_MAX_TOKENS.get(model, max_tokens),
        "temperature": temperature,
        "top_p": 0.9,
        **(({"enable_thinking": False}) if model in DISABLE_THINKING_MODELS else {}),
    }

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            raw = (msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
            if not raw:
                raise RuntimeError(f"Empty model output. Full message: {msg}")
            labels = parse_labels_from_response(raw, expected_n=len(items))
            logging.debug(f"Batch raw response: {raw[:300]}")
            return labels
        except Exception as e:
            last_err = e
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            print(f"\n{'!'*60}")
            print(f"  API CALL FAILED — attempt {attempt}/{max_retries}")
            print(f"  Model      : {model}")
            print(f"  URL        : {url}")
            print(f"  Error type : {type(e).__name__}")
            print(f"  Error msg  : {e}")
            print("  Full traceback:")
            traceback.print_exc()
            print(f"{'!'*60}\n")
            if attempt < max_retries:
                logging.info(f"Sleeping {sleep_s:.1f}s before retry...")
                time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _prf_macro(gold: list[str], pred: list[str]) -> dict:
    """Return macro precision, recall, F1 and weighted F1 for a single gold list."""
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(gold, pred, average="weighted", zero_division=0)
    return {
        "f1_macro":        float(f1),
        "precision_macro": float(p),
        "recall_macro":    float(r),
        "f1_weighted":     float(f1_w),
    }


def evaluate_evasion(
    y_pred: list[str],
    ann1:   list[str],
    ann2:   list[str],
    ann3:   list[str],
) -> dict:
    """
    Returns a dict with:
      - match_any_accuracy          : fraction of predictions matching at least one annotator
      - annotator1/2/3              : per-annotator macro F1 / P / R / weighted-F1
      - avg_annotators              : mean of the three annotators' macro F1 / P / R / weighted-F1
    """
    # --- match-any accuracy -------------------------------------------
    match_any = sum(
        p == a1 or p == a2 or p == a3
        for p, a1, a2, a3 in zip(y_pred, ann1, ann2, ann3)
    ) / len(y_pred)

    # --- per-annotator metrics ----------------------------------------
    per_ann = {
        "annotator1": _prf_macro(ann1, y_pred),
        "annotator2": _prf_macro(ann2, y_pred),
        "annotator3": _prf_macro(ann3, y_pred),
    }

    # --- average across the three annotators --------------------------
    metric_keys = ["f1_macro", "precision_macro", "recall_macro", "f1_weighted"]
    avg_ann = {
        k: sum(per_ann[ann][k] for ann in per_ann) / 3
        for k in metric_keys
    }

    return {
        "match_any_accuracy": float(match_any),
        "annotator1":         per_ann["annotator1"],
        "annotator2":         per_ann["annotator2"],
        "annotator3":         per_ann["annotator3"],
        "avg_annotators":     avg_ann,
    }


def evaluate_clarity(y_true: list[str], y_pred: list[str]) -> dict:
    return _prf_macro(y_true, y_pred)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_prf_block(label: str, metrics: dict) -> None:
    """Print a single P/R/F1 block."""
    print(f"    {label:<22}  F1={metrics['f1_macro']:.4f}  P={metrics['precision_macro']:.4f}  R={metrics['recall_macro']:.4f}  F1w={metrics['f1_weighted']:.4f}")


def print_evasion_metrics(model: str, metrics: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  EVASION (Task B) — {model}")
    print(f"{'='*65}")
    print(f"  Match-any accuracy       : {metrics['match_any_accuracy']:.4f}")
    print(f"  Per-annotator (macro):")
    _print_prf_block("annotator1 (strict)",  metrics["annotator1"])
    _print_prf_block("annotator2",           metrics["annotator2"])
    _print_prf_block("annotator3",           metrics["annotator3"])
    print(f"  Average across annotators:")
    _print_prf_block("avg",                  metrics["avg_annotators"])
    print(f"{'='*65}")


def print_clarity_metrics(model: str, metrics: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  CLARITY (Task A) — {model}")
    print(f"{'='*65}")
    _print_prf_block("clarity", metrics)
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        return {int(k): v for k, v in json.loads(path.read_text()).items() if v in EVASION_LABELS}
    except Exception:
        return {}


def save_cache(path: Path, cache: dict[int, str]) -> None:
    path.write_text(json.dumps({str(k): v for k, v in sorted(cache.items())}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    *,
    model: str,
    hf_token: str,
    system_prompt: str,
    questions: list[str],
    answers: list[str],
    clarity_true: list[str],
    ann1: list[str],
    ann2: list[str],
    ann3: list[str],
    test_df: pd.DataFrame,
    base_output_dir: Optional[str],
    no_cache: bool,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    base_sleep_s: float,
    n: int,
) -> dict:
    model_slug = model.split("/")[-1].lower().replace("-", "_")
    outdir = Path(base_output_dir) / model_slug if base_output_dir else Path(f"hf_results/{model_slug}")
    outdir.mkdir(parents=True, exist_ok=True)

    provider = MODEL_PROVIDER_MAP.get(model, "auto")
    logging.info(f"\n{'#'*60}\n  Model    : {model}\n  Provider : {provider}\n  Output   : {outdir}\n{'#'*60}")

    # --- load / warm up cache ----------------------------------------
    cache_path = outdir / "cache_evasion_labels.json"
    cache: dict[int, str] = {} if no_cache else load_cache(cache_path)
    cache = {k: v for k, v in cache.items() if k < n}
    if cache:
        logging.info(f"Loaded cache: {len(cache)}/{n} entries")

    final_labels: list[str] = [""] * n
    for i, lab in cache.items():
        final_labels[i] = lab

    # --- inference loop ----------------------------------------------
    missing = [i for i in range(n) if not final_labels[i]]
    logging.info(f"Need to predict {len(missing)}/{n} items")

    _batch_size = MODEL_BATCH_SIZE.get(model, batch_size)
    for start in range(0, len(missing), _batch_size):
        idxs = missing[start : start + _batch_size]
        batch_items = [{"id": int(i), "question": questions[i], "answer": answers[i]} for i in idxs]
        batch_num    = start // _batch_size + 1
        total_batches = (len(missing) + _batch_size - 1) // _batch_size
        logging.info(f"Batch {batch_num}/{total_batches} — items {start+1}..{min(start+_batch_size, len(missing))}/{len(missing)}")

        labels = call_hf_for_batch(
            hf_token=hf_token, model=model, system_prompt=system_prompt, items=batch_items,
            temperature=temperature, max_tokens=max_tokens, max_retries=max_retries, base_sleep_s=base_sleep_s,
        )
        for i, lab in zip(idxs, labels):
            final_labels[i] = lab
            cache[int(i)] = lab
        if not no_cache:
            save_cache(cache_path, cache)
        if start + _batch_size < len(missing):
            time.sleep(MODEL_BATCH_SLEEP.get(model, 1.5))

    # --- fill invalid predictions ------------------------------------
    bad = [i for i, l in enumerate(final_labels) if l not in EVASION_LABELS]
    if bad:
        logging.warning(f"Filling {len(bad)} invalid predictions with 'General'")
        for i in bad:
            final_labels[i] = "General"

    clarity_pred = [EVASION_TO_CLARITY[l] for l in final_labels]

    # --- save predictions --------------------------------------------
    results_df = test_df.copy()
    results_df["evasion_pred"] = final_labels
    results_df["clarity_pred"] = clarity_pred
    results_df.to_csv(outdir / "predictions.csv", index=False)
    logging.info(f"Saved predictions: {outdir / 'predictions.csv'}")

    # --- compute metrics ---------------------------------------------
    evasion_metrics = evaluate_evasion(final_labels, ann1, ann2, ann3)
    clarity_metrics = evaluate_clarity(clarity_true, clarity_pred)

    # --- print metrics -----------------------------------------------
    print_evasion_metrics(model, evasion_metrics)

    # Per-annotator classification reports
    for ann_name, ann_gold in [("annotator1", ann1), ("annotator2", ann2), ("annotator3", ann3)]:
        print(f"\n{'='*65}")
        print(f"  EVASION — Classification Report vs {ann_name} — {model.split('/')[-1]}")
        print(f"{'='*65}")
        print(classification_report(ann_gold, final_labels, labels=EVASION_LABELS, zero_division=0))

    print_clarity_metrics(model, clarity_metrics)
    print(f"\n{'='*65}")
    print(f"  CLARITY — Classification Report — {model.split('/')[-1]}")
    print(f"{'='*65}")
    print(classification_report(clarity_true, clarity_pred, labels=CLARITY_LABELS, zero_division=0))

    # --- save metrics ------------------------------------------------
    summary = {
        "model":            model,
        "test_set_size":    n,
        "evasion_metrics":  evasion_metrics,
        "clarity_metrics":  clarity_metrics,
    }
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2))
    logging.info(f"Saved metrics: {outdir / 'metrics.json'}")
    return summary


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_summaries: list[dict]) -> None:
    if len(all_summaries) < 2:
        return
    print(f"\n{'='*95}")
    print(f"  MULTI-MODEL COMPARISON")
    print(f"{'='*95}")
    print(f"{'Model':<40} {'Match-Any':>10} {'Ann1-F1':>9} {'Ann2-F1':>9} {'Ann3-F1':>9} {'Avg-F1':>9} {'Cl-F1':>9}")
    print("-" * 95)
    for s in all_summaries:
        em = s["evasion_metrics"]
        cm = s["clarity_metrics"]
        print(
            f"{s['model'].split('/')[-1][:39]:<40}"
            f" {em['match_any_accuracy']:>10.4f}"
            f" {em['annotator1']['f1_macro']:>9.4f}"
            f" {em['annotator2']['f1_macro']:>9.4f}"
            f" {em['annotator3']['f1_macro']:>9.4f}"
            f" {em['avg_annotators']['f1_macro']:>9.4f}"
            f" {cm['f1_macro']:>9.4f}"
        )
    print("=" * 95)





# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description="HF Inference API evasion + clarity evaluation (multi-model)")
    ap.add_argument("--model", nargs="+", default=[DEFAULT_MODEL], metavar="MODEL_ID",
                    help="One or more HuggingFace model IDs. google/gemma-3-27b-it is routed via Scaleway.")
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--temperature",  type=float, default=0.0)
    ap.add_argument("--max-tokens",   type=int,   default=600)
    ap.add_argument("--max-retries",  type=int,   default=6)
    ap.add_argument("--base-sleep-s", type=float, default=2.0)
    ap.add_argument("--output-dir",   default=None)
    ap.add_argument("--no-cache",     action="store_true")
    ap.add_argument("--hf-token",     default=HF_TOKEN)
    ap.add_argument("--debug",        action="store_true", help="Run on first 32 samples only")
    args = ap.parse_args()

    logging.info("Loading ailsntua/QEvasion test split...")
    ds = load_dataset("ailsntua/QEvasion")
    test_df = ds["test"].to_pandas()
    logging.info(f"Test set: {len(test_df)} rows | columns: {test_df.columns.tolist()}")

    qcol = "question" if "question" in test_df.columns else "interview_question"
    acol = next(c for c in ["interview_answer", "answer", "response"] if c in test_df.columns)
    logging.info(f"Using columns — question: '{qcol}', answer: '{acol}'")

    questions    = [str(x).strip() if str(x).strip().lower() != "nan" else "" for x in test_df[qcol]]
    answers      = [str(x).strip() if str(x).strip().lower() != "nan" else "" for x in test_df[acol]]
    clarity_true = test_df["clarity_label"].tolist()
    ann1 = test_df["annotator1"].tolist()
    ann2 = test_df["annotator2"].tolist()
    ann3 = test_df["annotator3"].tolist()
    n = len(test_df)

    if args.debug:
        logging.info("DEBUG MODE — restricting to first 32 samples")
        test_df      = test_df.iloc[:32].reset_index(drop=True)
        questions    = questions[:32]
        answers      = answers[:32]
        clarity_true = clarity_true[:32]
        ann1, ann2, ann3 = ann1[:32], ann2[:32], ann3[:32]
        n = 32

    system_prompt = build_taxonomy_prompt()
    logging.info(f"\nModels to evaluate ({len(args.model)}): {args.model}")

    all_summaries = []
    for model in args.model:
        summary = run_model(
            model=model, hf_token=args.hf_token, system_prompt=system_prompt,
            questions=questions, answers=answers, clarity_true=clarity_true,
            ann1=ann1, ann2=ann2, ann3=ann3, test_df=test_df,
            base_output_dir=args.output_dir, no_cache=args.no_cache,
            batch_size=args.batch_size, temperature=args.temperature,
            max_tokens=args.max_tokens, max_retries=args.max_retries,
            base_sleep_s=args.base_sleep_s, n=n,
        )
        all_summaries.append(summary)

    print_comparison_table(all_summaries)

    # --- final summary -----------------------------------------------
    print(f"\n{'='*65}")
    print(f"  FULL SUMMARY")
    print(f"{'='*65}")
    for s in all_summaries:
        em, cm = s["evasion_metrics"], s["clarity_metrics"]
        print(f"\n  Model              : {s['model']}")
        print(f"  Test samples       : {s['test_set_size']}")
        print(f"  Match-any accuracy : {em['match_any_accuracy']:.4f}")
        print(f"  Ann1  F1/P/R       : {em['annotator1']['f1_macro']:.4f} / {em['annotator1']['precision_macro']:.4f} / {em['annotator1']['recall_macro']:.4f}")
        print(f"  Ann2  F1/P/R       : {em['annotator2']['f1_macro']:.4f} / {em['annotator2']['precision_macro']:.4f} / {em['annotator2']['recall_macro']:.4f}")
        print(f"  Ann3  F1/P/R       : {em['annotator3']['f1_macro']:.4f} / {em['annotator3']['precision_macro']:.4f} / {em['annotator3']['recall_macro']:.4f}")
        print(f"  Avg   F1/P/R       : {em['avg_annotators']['f1_macro']:.4f} / {em['avg_annotators']['precision_macro']:.4f} / {em['avg_annotators']['recall_macro']:.4f}")
        print(f"  Clarity F1/P/R     : {cm['f1_macro']:.4f} / {cm['precision_macro']:.4f} / {cm['recall_macro']:.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()