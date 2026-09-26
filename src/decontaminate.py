"""13-gram overlap between training/dev questions and the test benchmarks (proposal Sec. 6.2).

    python src/decontaminate.py --data traces.jsonl [--data dev.jsonl ...] --output report.json [--drop-to clean.jsonl]

Word-level 13-grams after lowercasing and stripping punctuation/LaTeX braces. AIME25 postdates s1K
and is reported separately as the low-risk set.
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

N = 13
_TOKEN = re.compile(r"[a-z0-9]+")


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    words = _TOKEN.findall(text.lower())
    return {tuple(words[k : k + n]) for k in range(len(words) - n + 1)}


@lru_cache(maxsize=1)
def contamination_ngrams() -> frozenset:
    """13-grams of s1K (training questions) and the four test sets: what dev/held-out must avoid."""
    from datasets import load_dataset

    from benchmarks import BENCHMARKS, _dataset_path

    grams = set()
    for question in load_dataset(_dataset_path("simplescaling/s1K", "s1K"), split="train")["question"]:
        grams |= ngrams(question)
    for name in ("aime24", "aime25", "amc12", "math500"):
        for record in BENCHMARKS[name]():
            grams |= ngrams(record["question"])
    return frozenset(grams)


def main() -> None:
    from benchmarks import BENCHMARKS

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True, help="JSONL with a `question` field")
    parser.add_argument("--benchmarks", default="aime24,aime25,amc12,math500",
                        help="test sets; 'dev' may be added to also check training data against the dev set")
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-to", help="write the first --data file minus contaminated rows here")
    args = parser.parse_args()

    test_grams = {}
    for name in args.benchmarks.split(","):
        grams = set()
        for record in BENCHMARKS[name]():
            grams |= ngrams(record["question"])
        test_grams[name] = grams

    report, contaminated_ids = {}, set()
    for path in args.data:
        rows = [json.loads(line) for line in open(path)]
        hits = {name: [] for name in test_grams}
        for row in rows:
            grams = ngrams(row["question"])
            for name, test in test_grams.items():
                if grams & test:
                    hits[name].append(row["id"])
                    if path == args.data[0]:
                        contaminated_ids.add(row["id"])
        report[path] = {"rows": len(rows), **{f"overlap_{name}": len(ids) for name, ids in hits.items()},
                        "ids": hits}
        print(path, {name: len(ids) for name, ids in hits.items()})

    Path(args.output).write_text(json.dumps(report, indent=2))
    if args.drop_to:
        rows = [json.loads(line) for line in open(args.data[0])]
        with open(args.drop_to, "w") as handle:
            for row in rows:
                if row["id"] not in contaminated_ids:
                    handle.write(json.dumps(row) + "\n")
        print(f"dropped {len(contaminated_ids)} rows -> {args.drop_to}")


if __name__ == "__main__":
    main()
