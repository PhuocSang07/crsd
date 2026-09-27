"""Canonical trace content {id, question, thinking, answer, gold, ...} from every source.

    --source s1k11    simplescaling/s1K-1.1, DeepSeek-R1 trajectory + attempt: the exact training data
                      of the SGL / P-ALIGN / SSFT baselines ("read" tracks)
    --source openr1   open-r1/OpenR1-Math-220k: correct, complete DeepSeek-R1 generations for problems
                      sharing no 13-gram with s1K or a test set -- held-out R1-written traces for D1/D2/D4
    --source jsonl    anything with {id, question, response} (generate_traces.py --stage select, or
                      evaluate.py --export-traces rollouts): the response is split into thinking/answer
"""

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from prompting import nfc, split_response

SOURCES = ("s1k11", "openr1", "jsonl")


def from_s1k11(dataset_name: str | None) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset(dataset_name or "simplescaling/s1K-1.1", split="train")
    out = []
    for index, row in enumerate(rows):
        out.append({
            "id": f"s1k11-{index}",
            "question": nfc(row.get("problem") or row["question"]),
            "thinking": nfc(row["deepseek_thinking_trajectory"]).strip(),
            "answer": nfc(row["deepseek_attempt"]).strip(),
            "gold": None,
            "source": row.get("source_type"),
            "closed": True,
        })
    return out


def from_openr1(dataset_name: str | None, limit: int, seed: int) -> list[dict]:
    """First `limit` usable rows of a seeded shuffle of the stream (deterministic, source-bound)."""
    from datasets import load_dataset

    from decontaminate import contamination_ngrams, ngrams

    banned = contamination_ngrams()
    stream = load_dataset(dataset_name or "open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=5000)
    out, scanned = [], 0
    progress = tqdm(total=limit, desc="openr1", unit="trace")
    for row in stream:
        scanned += 1
        if not row.get("answer") or ngrams(row["problem"]) & banned:
            continue
        verified = row.get("correctness_math_verify") or []
        complete = row.get("is_reasoning_complete") or []
        picks = [k for k, ok in enumerate(verified) if ok and (k >= len(complete) or complete[k])]
        if not picks:
            continue
        parts = split_response(row["generations"][picks[0]])
        if not parts["closed"] or not parts["answer"]:
            continue
        uuid = row.get("uuid")
        if not isinstance(uuid, str) or not uuid or uuid.lower() == "nan":  # a few rows have no uuid
            uuid = "h" + hashlib.sha1(row["problem"].encode("utf-8")).hexdigest()[:16]
        out.append({
            "id": f"openr1-{uuid}",
            "question": nfc(row["problem"]),
            "thinking": nfc(parts["thinking"]),
            "answer": nfc(parts["answer"]),
            "gold": row["answer"],
            "source": f"openr1/{row.get('source')}",
            "closed": True,
        })
        progress.update(1)
        if len(out) >= limit:
            break
    progress.close()
    print(f"openr1: kept {len(out)} of {scanned} scanned rows")
    return out


def from_jsonl(path: str) -> list[dict]:
    out = []
    for line in open(path):
        row = json.loads(line)
        parts = split_response(row["response"])
        if row.get("finish_reason") == "length":  # cut at max_tokens: never a finished answer (D3)
            parts = {"thinking": parts["thinking"] + ("\n\n\n" + parts["answer"] if parts["answer"] else ""),
                     "answer": "", "closed": False}
        record = {
            "id": row["id"],
            "question": nfc(row["question"]),
            "thinking": nfc(parts["thinking"]),
            "answer": nfc(parts["answer"]),
            "gold": row.get("gold"),
            "source": row.get("source"),
            "closed": parts["closed"],
        }
        for key in ("correct", "teacher_solve_rate"):
            if key in row:
                record[key] = row[key]
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--input", help="dataset dir / HF id (s1k11, openr1) or JSONL path (jsonl)")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, default=300, help="openr1: number of held-out traces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "s1k11":
        records = from_s1k11(args.input)
    elif args.source == "openr1":
        records = from_openr1(args.input, args.limit, args.seed)
    else:
        if not args.input:
            parser.error("--source jsonl needs --input")
        records = from_jsonl(args.input)
    records = [r for r in records if r["thinking"]]
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):  # ids key every downstream file (records, signal banks, results)
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(f"duplicate trace ids: {dup[:5]}")
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    closed = sum(r["closed"] for r in records)
    print(f"wrote {len(records)} canonical traces ({closed} closed) -> {output}")
    if args.source == "openr1":
        # HF streaming threads abort the interpreter at shutdown (exit 134); output is flushed, so exit now.
        import os
        import sys

        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
