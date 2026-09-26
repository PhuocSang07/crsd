"""Render canonical traces for one model and map the step nodes onto its tokens.

Input: canonical JSONL (build_canonical.py). Output: one record per trace with

    input_ids               prompt ids + response ids + the tokenizer's eos token (SGL convention)
    response_token_span     [start, end) of the response (supervised, together with the eos token)
    nodes                   [{kind, char_start, char_end, token_start, token_end, hash}], v0 = q ... v_{n+1} = a

--style sgl renders the student's training text exactly as SpectralGuidedLearning/data_prep.py does
(the baselines' format); --style thinking renders how a teacher reads a trace as its own reasoning.
Prompt and response are tokenized separately and concatenated, as the student is trained. Node hashes
are what matches a teacher's routing signals to any student's records.
"""

import argparse
import json
import statistics
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from prompting import STYLES, render
from step_nodes import MAX_STEPS, MIN_STEP_CHARS, SEGMENT_MODES, assign_token_spans, build_nodes


def encode_offsets(tokenizer, text: str, base: int) -> tuple[list[int], list[int]]:
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoding["input_ids"], [base + start for start, _ in encoding["offset_mapping"]]


def build_record(tokenizer, content: dict, args) -> tuple[dict | None, str]:
    """(record, "ok") or (None, reason)."""
    if not content.get("closed", True) and not args.allow_unclosed:
        return None, "unclosed"
    rendered = render(tokenizer, content, args.style)
    if rendered is None:
        return None, "content_not_verbatim"
    nodes = build_nodes(rendered, mode=args.segment_mode, min_chars=args.min_step_chars, max_steps=args.max_steps)
    prompt, response = rendered["prompt"], rendered["response"]

    prompt_ids, prompt_starts = encode_offsets(tokenizer, prompt, 0)
    response_ids, response_starts = encode_offsets(tokenizer, response, len(prompt))
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no eos token")
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(response_ids) > args.max_tokens:  # SGL's rule: the eos token is not counted
        return None, "too_long"
    # The stop token starts past the text so no node can claim it.
    spans = assign_token_spans(nodes, prompt_starts + response_starts + [len(prompt) + len(response)])
    if spans is None:
        return None, "empty_node"
    for node, (start, end) in zip(nodes, spans):
        node["token_start"], node["token_end"] = start, end
    record = {
        "id": content["id"],
        "question": content["question"],
        "gold": content.get("gold"),
        "prompt": prompt,
        "response": response,
        "input_ids": input_ids,
        "response_token_span": [len(prompt_ids), len(prompt_ids) + len(response_ids)],
        "nodes": nodes,
        "n_steps": sum(n["kind"] == "step" for n in nodes),
        "n_tokens": len(input_ids),
        "closed": nodes[-1]["kind"] == "answer",
        "style": args.style,
    }
    for key in ("correct", "teacher_solve_rate", "source"):
        if key in content:
            record[key] = content[key]
    return record, "ok"


def percentiles(values: list[float]) -> str:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return f"p10={pick(0.1)} median={pick(0.5)} p90={pick(0.9)} max={ordered[-1]}"


def log_stats(records: list[dict]) -> dict:
    """Sec. 6.2 statistics: trace length, n, and the (step, target) distance distribution."""
    if not records:
        print("no records")
        return {}
    tokens = [r["n_tokens"] for r in records]
    steps = [r["n_steps"] for r in records]
    step_tokens = [n["token_end"] - n["token_start"] for r in records for n in r["nodes"] if n["kind"] == "step"]
    print(f"tokens/trace : mean={statistics.mean(tokens):.0f} {percentiles(tokens)}")
    print(f"steps/trace  : mean={statistics.mean(steps):.1f} {percentiles(steps)}")
    print(f"tokens/step  : mean={statistics.mean(step_tokens):.1f} {percentiles(step_tokens)}")
    bins = {"[4,8)": 0, "[8,16)": 0, "[16,32)": 0, "[32,64)": 0, "[64,inf)": 0}
    for n in steps:
        for i in range(1, n + 2):
            for j in range(1, i - 3):
                d = i - j
                key = "[4,8)" if d < 8 else "[8,16)" if d < 16 else "[16,32)" if d < 32 else "[32,64)" if d < 64 else "[64,inf)"
                bins[key] += 1
    print("far pairs by distance (d_min = 4): " + " ".join(f"{k}={v}" for k, v in bins.items()))
    return {"tokens": percentiles(tokens), "steps": percentiles(steps), "pairs_by_distance": bins}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True, help="canonical JSONL from build_canonical.py")
    parser.add_argument("--tokenizer", required=True, help="the model these records are for")
    parser.add_argument("--style", choices=STYLES, required=True, help="sgl = student training text; thinking = teacher reading")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--segment-mode", choices=SEGMENT_MODES, default="paragraph")
    parser.add_argument("--min-step-chars", type=int, default=MIN_STEP_CHARS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--allow-unclosed", action="store_true",
                        help="keep truncated responses (no answer node); for D3 on student rollouts")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records, reasons = [], {}
    for line in tqdm(open(args.canonical), desc="nodes", unit="trace"):
        record, reason = build_record(tokenizer, json.loads(line), args)
        reasons[reason] = reasons.get(reason, 0) + 1
        if record is not None:
            records.append(record)
            if args.limit and len(records) >= args.limit:
                break

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records ({args.style}) -> {output}  ({reasons})")
    stats = log_stats(records)
    Path(str(output) + ".stats.json").write_text(json.dumps({"counts": reasons, **stats}, indent=2))


if __name__ == "__main__":
    main()
