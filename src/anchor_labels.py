"""Functional step labels and anchor flags (proposal Appendix B), after Bogdan et al. (2025).

Categories: planning, fact_retrieval, active_computation, uncertainty_management, self_checking,
result_consolidation, final_answer. Anchors = planning | uncertainty_management | self_checking.

    --labeler heuristic  keyword rules on each step's opening (default, CPU); an approximation only
    --labeler llm        instruct model via vLLM with the definitions in the prompt (the proposal's protocol)
Adds `node_labels` (one per node; "question" for v0) and `anchor` (0/1 per node) to every record.
"""

import argparse
import json
import re
from pathlib import Path

LABELS = (
    "planning",
    "fact_retrieval",
    "active_computation",
    "uncertainty_management",
    "self_checking",
    "result_consolidation",
    "final_answer",
)
ANCHOR_LABELS = frozenset({"planning", "uncertainty_management", "self_checking"})

# Ordered: the first matching rule wins.
_RULES = [
    ("final_answer", re.compile(r"\\boxed|final answer|the answer is", re.I)),
    ("self_checking", re.compile(r"^\W*(let me (double[- ]?)?check|let's (double[- ]?)?check|let me verify|let's verify|"
                                 r"verify|double[- ]check|to confirm|check:|sanity check|let me confirm|let me re-?examine)", re.I)),
    ("uncertainty_management", re.compile(r"^\W*(wait|hmm+|but wait|hold on|actually|alternatively|however|maybe|perhaps|"
                                          r"i'?m not sure|is that (right|correct)|did i|oh|no,|but )", re.I)),
    ("planning", re.compile(r"^\W*(okay,? so|ok,? so|alright|first,?|let me (think|start|try|consider|figure)|let's (start|try|"
                            r"think|consider|see|figure)|i need to|we need to|the problem (asks|says|is)|to solve|my plan|"
                            r"strategy|approach|so,? i need|next,?|now,? (let|we|i))", re.I)),
    ("fact_retrieval", re.compile(r"\b(recall|formula for|theorem|by definition|is defined as|it is known|remember that|"
                                  r"we know that|identity)\b", re.I)),
    ("result_consolidation", re.compile(r"^\W*(so,? |therefore|thus|hence|putting (it|this) together|in summary|"
                                        r"that means|this means|which gives|so the)", re.I)),
]

LLM_TEMPLATE = (
    "Classify the function of one step of a model's reasoning trace. Categories:\n"
    "planning: decides what to do next or sets up an approach.\n"
    "fact_retrieval: recalls a fact, formula or definition without computing.\n"
    "active_computation: carries out algebra or arithmetic.\n"
    "uncertainty_management: expresses doubt, backtracks, or considers an alternative.\n"
    "self_checking: verifies a previous result.\n"
    "result_consolidation: summarizes or aggregates results obtained so far.\n"
    "final_answer: states the final answer.\n\n"
    "Step:\n\"\"\"\n{step}\n\"\"\"\n\nAnswer with the category name only."
)


def heuristic_label(text: str) -> str:
    head = text.strip()[:160]
    for label, pattern in _RULES:
        if pattern.search(head if label != "final_answer" else text):
            return label
    return "active_computation"


def parse_label(reply: str) -> str:
    reply = reply.strip().lower()
    for label in LABELS:
        if label in reply:
            return label
    return "active_computation"


def node_texts(record: dict) -> list[str]:
    text = record["prompt"] + record["response"]
    return [text[n["char_start"] : n["char_end"]] for n in record["nodes"]]


def label_records(records: list[dict], labeler: str, model: str | None, tp: int) -> None:
    steps = [(r_i, n_i, t) for r_i, r in enumerate(records) for n_i, t in enumerate(node_texts(r)) if n_i > 0]
    if labeler == "llm":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(model)
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": LLM_TEMPLATE.format(step=t[:1200])}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for _, _, t in steps
        ]
        llm = LLM(model=model, max_model_len=4096, dtype="bfloat16", tensor_parallel_size=tp, disable_log_stats=True)
        replies = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))]
        labels = [parse_label(reply) for reply in replies]
    else:
        labels = [heuristic_label(t) for _, _, t in steps]

    for record in records:
        record["node_labels"] = ["question"] + [None] * (len(record["nodes"]) - 1)
    for (r_i, n_i, _), label in zip(steps, labels):
        records[r_i]["node_labels"][n_i] = label
    for record in records:
        record["node_labels"][-1] = "final_answer"
        record["anchor"] = [int(label in ANCHOR_LABELS) for label in record["node_labels"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="records from data_prep.py")
    parser.add_argument("--output-path", help="default: overwrite --data-path")
    parser.add_argument("--labeler", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--model-name", help="labeler model for --labeler llm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.labeler == "llm" and not args.model_name:
        parser.error("--labeler llm needs --model-name")

    records = [json.loads(line) for line in open(args.data_path)]
    label_records(records, args.labeler, args.model_name, args.tensor_parallel_size)
    output = Path(args.output_path or args.data_path)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    counts = {}
    for record in records:
        for label in record["node_labels"][1:]:
            counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    print(f"labeled {total} nodes in {len(records)} records ({args.labeler}) -> {output}")
    print(" ".join(f"{k}={v / total:.1%}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
