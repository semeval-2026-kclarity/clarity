#!/usr/bin/env python3
"""
Codabench packaging for CLARITY Task 2 (evasion level) using an OpenAI GPT model,
PLUS a mapped Task A (clarity level) file derived from Task 2 outputs.

Output structure:
- <outdir>/task_b/prediction        (NO extension)
- <outdir>/task_b/prediction.zip    (zip contains ONLY prediction, no folders)

- <outdir>/task_a_mapped/prediction
- <outdir>/task_a_mapped/prediction.zip

Example:
python scripts/submissions/make_evasion_submission_gpt.py \
  --csv data/raw/semeval_test/clarity_task_evaluation_dataset.csv \
  --outdir ../model_outputs/task_b/gpt5_2/submissions \
  --model gpt-5.2-2025-12-11 \
  --batch-size 12

Notes:
- Expects OPENAI_API_KEY to be available in env (eg via .env export).
- Uses the Responses API + Structured Outputs (JSON schema) for reliable parsing.
- If you rerun, it will resume from a cache file in the outdir unless you disable it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
import zipfile
from typing import Any

import pandas as pd
from openai import OpenAI

IDX2EVASION = {
    0: "Explicit",
    1: "Implicit",
    2: "Dodging",
    3: "General",
    4: "Deflection",
    5: "Partial/half-answer",
    6: "Declining to answer",
    7: "Claims ignorance",
    8: "Clarification",
}
EVASION_LABELS = [IDX2EVASION[i] for i in range(9)]

# Task B -> Task A mapping
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

CLARITY_SET = {"Clear Reply", "Ambivalent", "Clear Non-Reply"}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


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


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.strip().lower() == "nan":
        return ""
    return s.strip()

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
        "   - If it refuses but then gives the requested commitment, label by what dominates (refusal vs answer).\n\n"
        "4) Explicit\n"
        "   - Directly states the requested commitment (even if brief), in the form requested.\n"
        "   - If the answer gives the key fact/stance but also includes lots of extra talk, it can still be Explicit.\n\n"
        "5) Implicit\n"
        "   - The requested commitment is not stated verbatim, but is clearly recoverable via a straightforward inference that the speaker intends.\n"
        "   - Test: a reasonable listener could paraphrase the commitment in one sentence without guessing.\n\n"
        "6) Partial/half-answer\n"
        "   - Answers ONE required part but omits other required parts.\n"
        "   - Common case: yes/no given but 'why/how/specifics' requested and missing.\n\n"
        "7) General vs Deflection vs Dodging (non-answer family)\n"
        "   First decide if the answer is an *attempted answer* vs a *pivot* vs *unrelated*.\n\n"
        "   7a) General (attempted but underspecified)\n"
        "       - On-topic, but avoids the requested commitment by staying vague, broad, or non-committal.\n"
        "       - May list principles, concerns, or context, but no specific stance/fact requested.\n\n"
        "   7b) Deflection (acknowledge + pivot)\n"
        "       - Acknowledges the question or its topic, then shifts to a different frame (values, process, talking point, criticism).\n"
        "       - Often contains 'what matters is...', 'the real issue is...', 'let me be clear about...', 'I’m focused on...'.\n"
        "       - Still no requested commitment.\n\n"
        "   7c) Dodging (unrelated or answers a different question)\n"
        "       - Does NOT engage the requested commitment AND does NOT meaningfully address the question’s topic.\n"
        "       - Includes answering a different question, or generic talking unrelated to the asked topic.\n\n"

        "Tie-breakers (use only when unsure):\n"
        "- If the answer gives a concrete list of steps/measures in response to 'what steps / what will you do', prefer Explicit (or Partial if it misses asked parts).\n"
        "- If the answer gives constraints/justification but never provides the asked commitment (eg 'I won't negotiate in public'), it's Declining to answer.\n"
        "- If the answer explains lots about the topic but does not answer the specific asked strategy/yes-no, it is usually General or Deflection, not Dodging.\n"
        "- 'I already answered that' without giving the commitment now counts as Declining to answer.\n\n"

        "Mini-examples (short, to anchor the hardest boundaries):\n"
        "A) Dodging vs Deflection\n"
        "Q: Did you meet the leader yesterday?\n"
        "A (Dodging): Our country is doing very well economically.\n"
        "A (Deflection): I understand the question - what matters is focusing on jobs for families.\n\n"
        "B) General vs Implicit\n"
        "Q: Do you support the bill?\n"
        "A (General): We’re considering all options and working with colleagues.\n"
        "A (Implicit): I’ll be voting yes when it reaches the floor.\n\n"
        "C) Declining vs General\n"
        "Q: What concessions do you want them to make?\n"
        "A (Declining): I’ll tell you after the meeting.\n"
        "A (General): We want cooperation and stability.\n\n"
        "D) Claims ignorance vs General\n"
        "Q: Have you seen evidence of acceleration?\n"
        "A (Claims ignorance): I haven’t seen that report.\n"
        "A (General): We’re concerned and monitoring closely.\n\n"

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


def response_text(resp) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text

    # fallback across SDK versions
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
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    base_sleep_s: float,
) -> list[str]:
    """
    Returns list[str] labels, one per item (same order).
    """
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
                        "content": (
                            "Label each item.\n"
                            "Input JSON (items with question + answer):\n"
                            f"{user_payload}"
                        ),
                    },
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
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

            raw = response_text(resp)
            if not raw:
                raise RuntimeError("Empty model output (no output_text).")

            data = json.loads(raw)
            labels = data.get("labels", None)
            if not isinstance(labels, list):
                raise RuntimeError(f"Bad schema - expected labels list, got: {type(labels)}")

            if len(labels) < len(items):
                raise RuntimeError(f"Too few labels: got {len(labels)} expected {len(items)}")

            if len(labels) > len(items):
                logging.warning(
                    f"Extra labels produced: got {len(labels)} expected {len(items)} - truncating"
                )
                labels = labels[: len(items)]

            for lab in labels:
                if lab not in EVASION_LABELS:
                    raise RuntimeError(f"Model produced an unknown label: {lab}")

            return labels

        except Exception as e:
            last_err = e
            msg = str(e)

            sleep_s = base_sleep_s * (2 ** (attempt - 1))

            if "Please try again in" in msg:
                try:
                    wait_s = float(msg.split("Please try again in")[1].split("s")[0].strip())
                    sleep_s = max(sleep_s, wait_s + 1.0)
                except Exception:
                    pass

            sleep_s = sleep_s * (0.9 + 0.2 * ((hash((attempt, time.time())) % 1000) / 1000.0))

            logging.warning(f"API call failed (attempt {attempt}/{max_retries}): {e}")

            if attempt < max_retries:
                time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def write_prediction_file(out_path: Path, label_lines: list[str]) -> None:
    out_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def zip_prediction_file(pred_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction")


def map_evasion_to_clarity(evasion_labels: list[str]) -> list[str]:
    mapped = []
    for lab in evasion_labels:
        if lab not in EVASION_TO_CLARITY:
            raise ValueError(f"Unknown evasion label for mapping: {lab}")
        c = EVASION_TO_CLARITY[lab]
        if c not in CLARITY_SET:
            raise ValueError(f"Mapped to unknown clarity label: {c}")
        mapped.append(c)
    return mapped


def load_cache(cache_path: Path) -> dict[int, str]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        out: dict[int, str] = {}
        for k, v in data.items():
            try:
                i = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v in EVASION_LABELS:
                out[i] = v
        return out
    except Exception:
        return {}


def save_cache(cache_path: Path, cache: dict[int, str]) -> None:
    cache_path.write_text(
        json.dumps({str(k): v for k, v in sorted(cache.items())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Official test CSV (237 rows)")
    ap.add_argument("--outdir", required=True, help="Where to write outputs")

    ap.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06"),
        help="OpenAI model id (or set OPENAI_MODEL).",
    )
    ap.add_argument("--batch-size", type=int, default=12, help="How many QA pairs per API call")
    ap.add_argument("--expected-rows", type=int, default=237)
    ap.add_argument("--allow-wrong-rows", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=600)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--base-sleep-s", type=float, default=1.0)

    ap.add_argument("--no-cache", action="store_true", help="Disable resume cache")
    ap.add_argument("--fail-soft", action="store_true", help="On hard failure, fill remaining with 'General'")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    out_task_b = outdir / "task_b"
    out_task_a = outdir / "task_a_mapped"
    out_task_b.mkdir(parents=True, exist_ok=True)
    out_task_a.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info(f"Loaded CSV: {csv_path} | rows={len(df)} cols={len(df.columns)}")

    if len(df) != args.expected_rows:
        msg = (
            f"Row count is {len(df)} but expected {args.expected_rows}. "
            "Codabench will reject the submission if this is wrong."
        )
        if args.allow_wrong_rows:
            logging.warning(msg)
        else:
            raise RuntimeError(msg)

    qcol, acol = pick_text_columns(df)
    logging.info(f"Using columns - question: '{qcol}', answer: '{acol}'")

    questions = [_clean_text(x) for x in df[qcol].tolist()]
    answers = [_clean_text(x) for x in df[acol].tolist()]
    n = len(df)

    # Reads OPENAI_API_KEY from env
    client = OpenAI()

    system_prompt = build_taxonomy_prompt()
    schema = build_json_schema()

    cache_path = outdir / "cache_task_b_labels.json"
    cache: dict[int, str] = {} if args.no_cache else load_cache(cache_path)
    if cache:
        logging.info(f"Loaded cache: {cache_path} (entries={len(cache)}/{n})")

    final_labels: list[str] = [""] * n

    for i, lab in cache.items():
        if 0 <= i < n:
            final_labels[i] = lab

    missing = [i for i in range(n) if not final_labels[i]]
    logging.info(f"Need to predict {len(missing)}/{n} items using model={args.model}")

    try:
        for start in range(0, len(missing), args.batch_size):
            idxs = missing[start : start + args.batch_size]
            batch_items = [{"id": int(i), "question": questions[i], "answer": answers[i]} for i in idxs]

            logging.info(
                f"Calling GPT for batch {start // args.batch_size + 1} "
                f"({min(start + args.batch_size, len(missing))}/{len(missing)}) ..."
            )

            labels = call_gpt_for_batch(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                schema=schema,
                items=batch_items,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                base_sleep_s=args.base_sleep_s,
            )

            for i, lab in zip(idxs, labels):
                final_labels[i] = lab
                cache[int(i)] = lab

            if not args.no_cache:
                save_cache(cache_path, cache)

            time.sleep(1.5)

    except Exception as e:
        if args.fail_soft:
            logging.error(f"Hard failure - continuing due to --fail-soft: {e}")
            for i in range(n):
                if not final_labels[i]:
                    final_labels[i] = "General"
        else:
            raise

    bad = [i for i, lab in enumerate(final_labels) if lab not in EVASION_LABELS]
    if bad:
        raise RuntimeError(f"Some outputs are missing/invalid at indices: {bad[:10]} (count={len(bad)})")

    # Task B
    pred_path_b = out_task_b / "prediction"
    write_prediction_file(pred_path_b, final_labels)
    logging.info(f"Wrote Task B prediction file: {pred_path_b} (lines={len(final_labels)})")

    zip_path_b = out_task_b / "prediction.zip"
    zip_prediction_file(pred_path_b, zip_path_b)
    logging.info(f"Wrote Task B zip file: {zip_path_b}")

    # Task A mapped
    final_clarity = map_evasion_to_clarity(final_labels)

    pred_path_a = out_task_a / "prediction"
    write_prediction_file(pred_path_a, final_clarity)
    logging.info(f"Wrote mapped Task A prediction file: {pred_path_a} (lines={len(final_clarity)})")

    zip_path_a = out_task_a / "prediction.zip"
    zip_prediction_file(pred_path_a, zip_path_a)
    logging.info(f"Wrote mapped Task A zip file: {zip_path_a}")

    logging.info("Done.")
    logging.info("Submit ONE file at a time on Codabench:")
    logging.info(f"  - Task B zip: {zip_path_b}")
    logging.info(f"  - Task A (mapped) zip: {zip_path_a}")


if __name__ == "__main__":
    main()
