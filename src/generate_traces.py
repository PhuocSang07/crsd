"""Build s1K-Q8B (proposal Sec. 6.2): the teacher re-writes every trace it will later be read on.

Two stages, so the expensive one never has to be redone to change a filtering rule:

    --stage generate   Qwen3-8B (thinking) samples n traces per question with vLLM
                       (T=0.6, top-p 0.95, top-k 20, up to 32,768 tokens) -> raw JSONL
    --stage select     keep traces that closed </think>, were not cut at max_tokens and are correct;
                       per question pick one correct trace at random (seeded), keep the other
                       the other seven as RSR candidates -> traces JSONL + retention stats (gate G0)

Correctness: math-verify against the gold answer when one can be read off (a \\boxed{} in the
reference, or a short bare reference); otherwise an LLM judge (--judge-model, vLLM) compares the
trace's final answer with the reference solution. Without a judge those questions are dropped and
counted.

Sources: "s1k" (simplescaling/s1K, the 1,000 training questions) and "math-train" (MATH train split,
levels 3-5, minus every problem sharing a 13-gram with s1K or the four test sets) for the dev set
and the held-out set.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from answer_scoring import extract_boxed, math_answers_equal
from prompting import THINK_CLOSE, render_prompt

SOURCES = ("s1k", "math-train")
_BARE_ANSWER_MAX_CHARS = 64

JUDGE_TEMPLATE = (
    "You are grading a solution against a reference.\n\n"
    "Problem:\n{question}\n\nReference solution:\n{reference}\n\n"
    "Candidate final answer (the text after the candidate's reasoning):\n{answer}\n\n"
    "Does the candidate's final answer agree with the reference (same final result or conclusion; "
    "ignore formatting and presentation)? Reply with exactly one word: YES or NO."
)


def reference_answer(solution: str) -> str | None:
    """A gold string math-verify can compare with, or None when only a free-form reference exists."""
    solution = (solution or "").strip()
    boxed = extract_boxed(solution)
    if boxed:
        return boxed
    if solution and "\n" not in solution and len(solution) <= _BARE_ANSWER_MAX_CHARS:
        return solution
    return None


def load_questions(source: str, dataset_name: str | None, seed: int, skip: int, limit: int | None) -> list[dict]:
    """Unified rows {id, question, reference, gold, source, cot_type}; order is seeded and stable."""
    if source == "s1k":
        rows = load_dataset(dataset_name or "simplescaling/s1K", split="train")
        items = [
            {"question": r["question"], "reference": r["solution"], "source": r["source_type"], "cot_type": r["cot_type"]}
            for r in rows
        ]
    elif source == "math-train":
        from decontaminate import contamination_ngrams, ngrams

        rows = load_dataset(dataset_name or "DigitalLearningGmbH/MATH-lighteval", split="train")
        items = [
            {"question": r["problem"], "reference": r["solution"], "source": f"MATH/{r['type']}/{r['level']}", "cot_type": "math"}
            for r in rows
            if r["level"] in ("Level 3", "Level 4", "Level 5")
        ]
        # Sec. 6.2: dev and held-out must not overlap s1K or any test set (13-gram rule). s1K draws on
        # MATH itself (qfq/openaimath), so without this a few dev/held-out problems are s1K problems.
        banned = contamination_ngrams()
        items = [item for item in items if not (ngrams(item["question"]) & banned)]
        random.Random(seed).shuffle(items)
    else:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    items = items[skip:]
    if limit is not None:
        items = items[:limit]
    for index, item in enumerate(items):
        item["id"] = f"{source}-{skip + index}"
        item["gold"] = reference_answer(item["reference"])
    return items


def generate(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    items = load_questions(args.source, args.dataset_name, args.seed, args.skip, args.limit)
    items = items[args.shard_index :: args.num_shards]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = [render_prompt(tokenizer, item["question"], enable_thinking=True) for item in items]
    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        seed=args.seed,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        n=args.n_per_question,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        # Batched so a stop partway through keeps every finished question.
        for start in tqdm(range(0, len(items), args.batch_size), desc="generating", unit="batch"):
            batch = items[start : start + args.batch_size]
            results = llm.generate(prompts[start : start + args.batch_size], sampling, use_tqdm=False)
            for item, prompt, result in zip(batch, prompts[start : start + args.batch_size], results):
                handle.write(json.dumps({
                    **item,
                    "prompt": prompt,
                    "generations": [
                        {"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)}
                        for c in result.outputs
                    ],
                }) + "\n")
    print(f"wrote {len(items)} questions x {args.n_per_question} -> {output}")


def final_answer_text(text: str) -> str:
    return text.split(THINK_CLOSE, 1)[1].strip() if THINK_CLOSE in text else ""


def judge_rows(rows: list[dict], judge_model: str, args) -> dict[tuple[str, int], bool]:
    """LLM verdicts for generations whose question has no machine-checkable gold answer."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    keys, prompts = [], []
    for row in rows:
        for index, generation in enumerate(row["generations"]):
            answer = final_answer_text(generation["text"])
            if not answer:
                continue
            content = JUDGE_TEMPLATE.format(
                question=row["question"], reference=row["reference"][-4000:], answer=answer[-3000:]
            )
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))
            keys.append((row["id"], index))
    if not prompts:
        return {}
    llm = LLM(model=judge_model, max_model_len=16384, dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size, disable_log_stats=True)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=True)
    return {key: out.outputs[0].text.strip().upper().startswith("YES") for key, out in zip(keys, outputs)}


def select(args) -> None:
    rows = [json.loads(line) for line in open(args.raw_path)]
    needs_judge = [row for row in rows if row["gold"] is None]
    verdicts = judge_rows(needs_judge, args.judge_model, args) if (args.judge_model and needs_judge) else {}

    rng = random.Random(args.seed)
    counts = Counter()
    kept = []
    for row in rows:
        counts["questions"] += 1
        correct, verdict = [], {}
        for index, generation in enumerate(row["generations"]):
            counts["generations"] += 1
            text = generation["text"]
            if generation["finish_reason"] != "stop" or THINK_CLOSE not in text:
                counts["truncated_or_unclosed"] += 1
                continue
            if row["gold"] is not None:
                ok = math_answers_equal(extract_boxed(final_answer_text(text)), row["gold"])
            elif (row["id"], index) in verdicts:
                ok = verdicts[(row["id"], index)]
            else:
                counts["no_gold_no_judge"] += 1
                continue
            counts["correct"] += int(ok)
            verdict[index] = bool(ok)
            if ok:
                correct.append(index)
        if not correct:
            continue
        chosen = rng.choice(correct)
        kept.append({
            "id": row["id"],
            "question": row["question"],
            "gold": row["gold"],
            "reference": row["reference"],
            "source": row["source"],
            "cot_type": row["cot_type"],
            "prompt": row["prompt"],
            "response": row["generations"][chosen]["text"],
            "n_tokens": row["generations"][chosen]["n_tokens"],
            "teacher_solve_rate": len(correct) / len(row["generations"]),
            # RSR (baseline B6) ranks the other seven traces of the same question (Sec. 6.2); `correct` is
            # None for a truncated/unclosed trace.
            "candidates": [
                {"text": g["text"], "correct": verdict.get(i), "finish_reason": g["finish_reason"], "n_tokens": g["n_tokens"]}
                for i, g in enumerate(row["generations"]) if i != chosen
            ],
        })
        if args.max_keep and len(kept) >= args.max_keep:
            break

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in kept:
            handle.write(json.dumps(record) + "\n")
    stats = {**counts, "kept_questions": len(kept), "retention": len(kept) / max(1, counts["questions"])}
    stats["G0_pass (>=600 kept)"] = len(kept) >= 600
    Path(str(output) + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "select"), required=True)
    parser.add_argument("--source", choices=SOURCES, default="s1k")
    parser.add_argument("--dataset-name", help="local mirror or HF id (default per source)")
    parser.add_argument("--model-name", help="teacher (generate)")
    parser.add_argument("--raw-path", help="raw generations (select input)")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--skip", type=int, default=0, help="drop the first N questions (dev/held-out split)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-question", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=34816)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--judge-model", help="LLM judge for free-form references (select)")
    parser.add_argument("--max-keep", type=int, help="stop after this many kept questions (held-out set)")
    args = parser.parse_args()
    if args.stage == "generate":
        if not args.model_name:
            parser.error("--stage generate needs --model-name")
        generate(args)
    else:
        if not args.raw_path:
            parser.error("--stage select needs --raw-path")
        select(args)


if __name__ == "__main__":
    main()
