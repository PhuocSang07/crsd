"""Tokenize traces for one model and map the step nodes onto its tokens.

Input: traces JSONL from generate_traces.py --stage select ({id, question, prompt, response, ...}),
or s1K-1.1's DeepSeek-R1 traces (--source s1k11-r1, variant A15: the teacher only re-reads them).
Output: one record per trace with

    input_ids               prompt ids + response ids + <|im_end|>
    response_token_span     [start, end) of the response (the supervised part; the stop token after it too)
    nodes                   [{kind, char_start, char_end, token_start, token_end}], v0 = q ... v_{n+1} = a

Prompt and response are tokenized separately and concatenated, exactly as the student is trained,
so the same record serves training and teacher-forced extraction. Run once per tokenizer; the node
character spans are identical across models by construction (and asserted by extract_routing.py).
"""

import argparse
import json
import statistics
import unicodedata
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from prompting import END_OF_TURN, THINK_CLOSE, THINK_OPEN, question_char_span, render_prompt
from step_nodes import MAX_STEPS, MIN_STEP_CHARS, SEGMENT_MODES, assign_token_spans, build_nodes

SOURCES = ("traces", "s1k11-r1")


def encode_offsets(tokenizer, text: str, base: int) -> tuple[list[int], list[int]]:
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoding["input_ids"], [base + start for start, _ in encoding["offset_mapping"]]


def build_record(tokenizer, trace: dict, args) -> tuple[dict | None, str]:
    """(record, "ok") or (None, reason)."""
    prompt = trace["prompt"]
    response = unicodedata.normalize("NFC", trace["response"])
    q_span = question_char_span(prompt, trace["question"])
    nodes = build_nodes(prompt, response, q_span, mode=args.segment_mode, min_chars=args.min_step_chars,
                        max_steps=args.max_steps, allow_unclosed=getattr(args, "allow_unclosed", False))
    if nodes is None:
        return None, "no_closed_think_or_empty_answer"

    prompt_ids, prompt_starts = encode_offsets(tokenizer, prompt, 0)
    response_ids, response_starts = encode_offsets(tokenizer, response, len(prompt))
    stop_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if stop_id is None or stop_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no {END_OF_TURN} token")
    input_ids = prompt_ids + response_ids + [stop_id]
    if len(input_ids) > args.max_tokens:
        return None, "too_long"

    # The stop token starts past the text so no node can claim it.
    token_starts = prompt_starts + response_starts + [len(prompt) + len(response)]
    spans = assign_token_spans(nodes, token_starts)
    if spans is None:
        return None, "empty_node"
    for node, (start, end) in zip(nodes, spans):
        node["token_start"], node["token_end"] = start, end
    return {
        "id": trace["id"],
        "question": trace["question"],
        "gold": trace.get("gold"),
        "prompt": prompt,
        "response": response,
        "input_ids": input_ids,
        "response_token_span": [len(prompt_ids), len(prompt_ids) + len(response_ids)],
        "nodes": nodes,
        "n_steps": sum(n["kind"] == "step" for n in nodes),
        "closed": nodes[-1]["kind"] == "answer",
        "n_tokens": len(input_ids),
        "teacher_solve_rate": trace.get("teacher_solve_rate"),
    }, "ok"


def iter_traces(args, template_tokenizer):
    if args.source == "traces":
        with open(args.traces_path) as handle:
            for line in handle:
                yield json.loads(line)
        return
    # A15: s1K-1.1's DeepSeek-R1 traces; the prompt is still the teacher's chat template.
    from datasets import load_dataset

    rows = load_dataset(args.traces_path or "simplescaling/s1K-1.1", split="train")
    for index, row in enumerate(rows):
        response = (
            f"{THINK_OPEN}\n{row['deepseek_thinking_trajectory'].strip()}\n{THINK_CLOSE}\n\n"
            f"{row['deepseek_attempt'].strip()}"
        )
        yield {
            "id": f"s1k11-{index}",
            "question": row["question"],
            "gold": None,
            "prompt": render_prompt(template_tokenizer, row["question"], enable_thinking=True),
            "response": response,
        }


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
    # Number of (row, far-target) pairs per distance bin at d_min = 4.
    bins = {"[4,8)": 0, "[8,16)": 0, "[16,32)": 0, "[32,64)": 0, "[64,inf)": 0}
    for n in steps:
        for i in range(1, n + 2):
            for j in range(1, i - 3):
                d = i - j
                key = "[4,8)" if d < 8 else "[8,16)" if d < 16 else "[16,32)" if d < 32 else "[32,64)" if d < 64 else "[64,inf)"
                bins[key] += 1
    print("far pairs by distance: " + " ".join(f"{k}={v}" for k, v in bins.items()))
    return {"tokens": percentiles(tokens), "steps": percentiles(steps), "pairs_by_distance": bins}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-path", help="traces JSONL (source=traces) or s1K-1.1 mirror (source=s1k11-r1)")
    parser.add_argument("--source", choices=SOURCES, default="traces")
    parser.add_argument("--tokenizer", required=True, help="the model these records are for")
    parser.add_argument("--template-tokenizer", help="chat template for s1k11-r1 prompts (the teacher)")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--segment-mode", choices=SEGMENT_MODES, default="paragraph")
    parser.add_argument("--min-step-chars", type=int, default=MIN_STEP_CHARS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-unclosed", action="store_true",
                        help="keep responses cut before </think> (no answer node); for D3 on student rollouts")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = AutoTokenizer.from_pretrained(args.template_tokenizer) if args.template_tokenizer else tokenizer
    records, reasons = [], {}
    for trace in tqdm(iter_traces(args, template), desc="nodes", unit="trace"):
        record, reason = build_record(tokenizer, trace, args)
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
    print(f"wrote {len(records)} records -> {output}  ({reasons})")
    stats = log_stats(records)
    Path(str(output) + ".stats.json").write_text(json.dumps({"counts": reasons, **stats}, indent=2))


if __name__ == "__main__":
    main()
