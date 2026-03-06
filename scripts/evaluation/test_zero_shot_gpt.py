#!/usr/bin/env python3
"""
CLARITY Task B (evasion level) + Task A (clarity level) evaluation
using OpenAI API (Responses API + Structured Outputs).

Mirror of test_zero_shot.py but for OpenAI models.
Output structure is identical: <output_dir>/<model_slug>/metrics.json
so print_results_table.py can read results from both scripts together.

- Predicts evasion label using the same taxonomy prompt
- Infers clarity label from the evasion->clarity mapping
- Evaluates ONLY on the test split of ailsntua/QEvasion
- Evasion metrics:
    * Match-any accuracy: prediction is correct if it matches ANY of the 3 annotators
    * Per-annotator F1 / Precision / Recall (macro) for each of the 3 annotators
    * Average F1 / Precision / Recall across the 3 annotators (main comparison number)
- Clarity: prediction is evaluated against the single clarity_label
- Supports multiple models in a single run
- Reads OPENAI_API_KEY from environment (or pass --openai-api-key)

Usage:
    python test_zero_shot_gpt.py
    python test_zero_shot_gpt.py --model gpt-4o-2024-08-06
    python test_zero_shot_gpt.py --model gpt-4o-2024-08-06 gpt-5
    python test_zero_shot_gpt.py --no-cache --output-dir my_results/
    python test_zero_shot_gpt.py --batch-size 12 --model gpt-4o gpt-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5.2-2025-12-11"

# Per-model batch size overrides
MODEL_BATCH_SIZE: dict[str, int] = {}

# Per-model inter-batch sleep overrides (seconds)
MODEL_BATCH_SLEEP: dict[str, float] = {}

# Per-model max_output_tokens overrides
MODEL_MAX_TOKENS: dict[str, int] = {}

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
        "   - The answer primarily asks the interviewer to repeat/clarify/restate (eg 'Say it again', 'Which one?', 'What do you mean?').\n"
        "   - If it both asks for clarification AND gives the requested commitment, choose based on the dominant function.\n\n"
        "2) Claims ignorance\n"
        "   - The answer asserts lack of knowledge/recall/awareness (eg 'I don't know', 'I haven't seen that', 'I'm not aware').\n"
        "   - IMPORTANT: If the answer later gives the requested commitment anyway, do NOT pick Claims ignorance.\n\n"
        "3) Declining to answer\n"
        "   - The answer refuses to provide the requested commitment now (eg 'no comment', 'I won't get into that', 'I'll tell you when it's over').\n"
        "   - If it refuses but then gives the requested commitment, label by what dominates.\n\n"
        "4) Explicit\n"
        "   - Directly states the requested commitment (even if brief), in the form requested.\n"
        "   - If the answer gives the key fact/stance but also includes lots of extra talk, it can still be Explicit.\n\n"
        "5) Implicit\n"
        "   - The requested commitment is not stated verbatim, but is clearly recoverable via a straightforward inference.\n"
        "   - Test: a reasonable listener could paraphrase the commitment in one sentence without guessing.\n\n"
        "6) Partial/half-answer\n"
        "   - Answers ONE required part but omits other required parts.\n"
        "   - Common case: yes/no given but 'why/how/specifics' requested and missing.\n\n"
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
        "- Keep the item order exactly - do not sort, do not use the id for ordering.\n"
        "- Judge using only the provided QUESTION and ANSWER.\n"
    )


def build_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": EVASION_LABELS},
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _extract_response_text(resp) -> str:
    """Extract text from OpenAI Responses API response across SDK versions."""
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text
    try:
        out = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        out.append(getattr(c, "text", ""))
        return "\n".join(out).strip()
    except Exception:
        return ""


def call_gpt_for_batch(
    client: OpenAI,
    model: str,
    system_prompt: str,
    schema: dict,
    items: list[dict],
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 600,
    max_retries: int = 6,
    base_sleep_s: float = 1.0,
) -> list[str]:
    user_payload = json.dumps({"items": items}, ensure_ascii=False)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Label each item.\nInput JSON:\n{user_payload}",
                    },
                ],
                temperature=temperature,
                max_output_tokens=MODEL_MAX_TOKENS.get(model, max_output_tokens),
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "task_b_labels",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )

            raw = _extract_response_text(resp)
            if not raw:
                raise RuntimeError(f"Empty model output. Full response: {resp}")

            data = json.loads(raw)
            labels = data.get("labels", None)
            if not isinstance(labels, list):
                raise RuntimeError(f"Expected labels list, got: {type(labels)}")
            if len(labels) < len(items):
                raise RuntimeError(f"Too few labels: got {len(labels)}, expected {len(items)}")
            if len(labels) > len(items):
                logging.warning(f"Extra labels: got {len(labels)}, expected {len(items)} — truncating")
                labels = labels[:len(items)]
            for lab in labels:
                if lab not in EVASION_LABELS:
                    raise RuntimeError(f"Unknown label produced: {lab}")

            return labels

        except Exception as e:
            last_err = e
            sleep_s = base_sleep_s * (2 ** (attempt - 1))

            # Respect rate limit retry-after if present
            msg = str(e)
            if "Please try again in" in msg:
                try:
                    wait_s = float(msg.split("Please try again in")[1].split("s")[0].strip())
                    sleep_s = max(sleep_s, wait_s + 1.0)
                except Exception:
                    pass

            print(f"\n{'!'*60}")
            print(f"  API CALL FAILED — attempt {attempt}/{max_retries}")
            print(f"  Model      : {model}")
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
# Evaluation helpers  (identical to test_zero_shot.py)
# ---------------------------------------------------------------------------

def _prf_macro(gold: list[str], pred: list[str]) -> dict:
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
    match_any = sum(
        p == a1 or p == a2 or p == a3
        for p, a1, a2, a3 in zip(y_pred, ann1, ann2, ann3)
    ) / len(y_pred)

    per_ann = {
        "annotator1": _prf_macro(ann1, y_pred),
        "annotator2": _prf_macro(ann2, y_pred),
        "annotator3": _prf_macro(ann3, y_pred),
    }

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
# Pretty-print helpers  (identical to test_zero_shot.py)
# ---------------------------------------------------------------------------

def _print_prf_block(label: str, metrics: dict) -> None:
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
# Cache helpers  (identical to test_zero_shot.py)
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
    client: OpenAI,
    schema: dict,
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
    max_output_tokens: int,
    max_retries: int,
    base_sleep_s: float,
    n: int,
) -> dict:
    model_slug = model.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    outdir = Path(base_output_dir) / model_slug if base_output_dir else Path(f"gpt_results/{model_slug}")
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'#'*60}\n  Model  : {model}\n  Output : {outdir}\n{'#'*60}")

    # --- cache -------------------------------------------------------
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
        batch_num     = start // _batch_size + 1
        total_batches = (len(missing) + _batch_size - 1) // _batch_size
        logging.info(f"Batch {batch_num}/{total_batches} — items {start+1}..{min(start+_batch_size, len(missing))}/{len(missing)}")

        labels = call_gpt_for_batch(
            client=client, model=model, system_prompt=system_prompt, schema=schema,
            items=batch_items, temperature=temperature, max_output_tokens=max_output_tokens,
            max_retries=max_retries, base_sleep_s=base_sleep_s,
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

    for ann_name, ann_gold in [("annotator1", ann1), ("annotator2", ann2), ("annotator3", ann3)]:
        print(f"\n{'='*65}")
        print(f"  EVASION — Classification Report vs {ann_name} — {model}")
        print(f"{'='*65}")
        print(classification_report(ann_gold, final_labels, labels=EVASION_LABELS, zero_division=0))

    print_clarity_metrics(model, clarity_metrics)
    print(f"\n{'='*65}")
    print(f"  CLARITY — Classification Report — {model}")
    print(f"{'='*65}")
    print(classification_report(clarity_true, clarity_pred, labels=CLARITY_LABELS, zero_division=0))

    # --- save metrics ------------------------------------------------
    summary = {
        "model":           model,
        "test_set_size":   n,
        "evasion_metrics": evasion_metrics,
        "clarity_metrics": clarity_metrics,
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
            f"{s['model'][:39]:<40}"
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

    ap = argparse.ArgumentParser(description="OpenAI API evasion + clarity evaluation (multi-model)")
    ap.add_argument("--model", nargs="+", default=[DEFAULT_MODEL], metavar="MODEL_ID",
                    help="One or more OpenAI model IDs (e.g. gpt-4o-2024-08-06 gpt-5)")
    ap.add_argument("--batch-size",        type=int,   default=12)
    ap.add_argument("--temperature",       type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int,   default=600)
    ap.add_argument("--max-retries",       type=int,   default=6)
    ap.add_argument("--base-sleep-s",      type=float, default=1.0)
    ap.add_argument("--output-dir",        default=None)
    ap.add_argument("--no-cache",          action="store_true")
    ap.add_argument("--openai-api-key",    default=None,
                    help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    ap.add_argument("--debug",             action="store_true", help="Run on first 32 samples only")
    args = ap.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY or pass --openai-api-key.")

    client = OpenAI()
    schema = build_json_schema()

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
            model=model, client=client, schema=schema, system_prompt=system_prompt,
            questions=questions, answers=answers, clarity_true=clarity_true,
            ann1=ann1, ann2=ann2, ann3=ann3, test_df=test_df,
            base_output_dir=args.output_dir, no_cache=args.no_cache,
            batch_size=args.batch_size, temperature=args.temperature,
            max_output_tokens=args.max_output_tokens, max_retries=args.max_retries,
            base_sleep_s=args.base_sleep_s, n=n,
        )
        all_summaries.append(summary)

    print_comparison_table(all_summaries)

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