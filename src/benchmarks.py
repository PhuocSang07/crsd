"""Benchmark loaders -> [{id, question, gold, task_type}] (proposal Sec. 6.4).

AIME24 (30), AIME25 (30), AMC12 (AI-MO/aimo-validation-amc, 83 problems as in P-ALIGN, Appendix E),
MATH500, and "dev": 200 MATH-train level 3-5 problems disjoint from every test set (tuning and D3).
"""

import os

from datasets import load_dataset

from answer_scoring import extract_boxed

BENCH_DATA_ROOT = os.environ.get("BENCH_DATA_ROOT")
DEV_SIZE = 200


def _dataset_path(repo_id: str, local_dir_name: str) -> str:
    """BENCH_DATA_ROOT/<local_dir_name> (offline mirror, see download.txt), or the HF id if BENCH_DATA_ROOT is empty.

    A missing mirror is an error, never a silent fall back to the Hub.
    """
    if not BENCH_DATA_ROOT:
        return repo_id
    local = f"{BENCH_DATA_ROOT}/{local_dir_name}"
    if not os.path.isdir(local):
        raise FileNotFoundError(f"missing local mirror {local} of {repo_id} (download.txt: --hf-dataset {repo_id} "
                                f"<BENCH_DATA_ROOT>/{local_dir_name})")
    return local


def _records(rows, question_key: str, answer_key: str) -> list[dict]:
    return [
        {"id": index, "question": row[question_key], "gold": str(row[answer_key]), "task_type": "math"}
        for index, row in enumerate(rows)
    ]


def load_aime24() -> list[dict]:
    """math-ai/aime24 stores the gold answer as a boxed expression in `solution`."""
    records = _records(load_dataset(_dataset_path("math-ai/aime24", "aime24"), split="test"), "problem", "solution")
    for record in records:
        record["gold"] = extract_boxed(record["gold"]) or record["gold"]
    return records


def load_aime25() -> list[dict]:
    return _records(load_dataset(_dataset_path("math-ai/aime25", "aime25"), split="test"), "problem", "answer")


def load_amc12() -> list[dict]:
    rows = load_dataset(_dataset_path("AI-MO/aimo-validation-amc", "aimo-validation-amc"), split="train")
    return _records(rows, "problem", "answer")


def load_math500() -> list[dict]:
    return _records(load_dataset(_dataset_path("HuggingFaceH4/MATH-500", "MATH-500"), split="test"), "problem", "answer")


def load_dev() -> list[dict]:
    from generate_traces import load_questions

    rows = load_questions("math-train", _dataset_path("DigitalLearningGmbH/MATH-lighteval", "MATH-lighteval"),
                          seed=42, skip=0, limit=DEV_SIZE)
    return [
        {"id": index, "question": row["question"], "gold": row["gold"], "task_type": "math"}
        for index, row in enumerate(rows)
        if row["gold"] is not None
    ]


BENCHMARKS = {
    "aime24": load_aime24,
    "aime25": load_aime25,
    "amc12": load_amc12,
    "math500": load_math500,
    "dev": load_dev,
}

# Samples per problem (Sec. 6.4): AIME/AMC need many -- one AIME problem is ~3.3 points.
DEFAULT_SAMPLES = {"aime24": 16, "aime25": 16, "amc12": 16, "math500": 4, "dev": 4}

# 4-shot plain-text prompt for the pre-distillation Base student (baseline B0, few-shot row).
FEW_SHOT_EXAMPLES = [
    ("What is the value of $2^{10} - 2^{8}$?",
     "We have $2^{10} = 1024$ and $2^8 = 256$, so the difference is $1024 - 256 = 768$.\n\nThe answer is \\boxed{768}."),
    ("How many positive divisors does $36$ have?",
     "Since $36 = 2^2 \\cdot 3^2$, the number of divisors is $(2+1)(2+1) = 9$.\n\nThe answer is \\boxed{9}."),
    ("Solve for $x$: $3x + 7 = 22$.",
     "Subtracting 7 gives $3x = 15$, so $x = 5$.\n\nThe answer is \\boxed{5}."),
    ("A rectangle has perimeter $30$ and length $9$. What is its area?",
     "The width is $30/2 - 9 = 6$, so the area is $9 \\cdot 6 = 54$.\n\nThe answer is \\boxed{54}."),
]


def few_shot_prompt(question: str) -> str:
    shots = "".join(f"Problem: {q}\nSolution: {a}\n\n" for q, a in FEW_SHOT_EXAMPLES)
    return f"{shots}Problem: {question.strip()}\nSolution:"
