"""Distance-controlled error injection (proposal Sec. 6.7, Appendix C).

    --stage build     from correct held-out traces, rendered as the *student* reads them (data_prep.py
                      --style sgl records, so the student continues its own training format): a number r
                      introduced in step j, absent from
                      steps j+1..j+2 and first reused at step i with i - j in [4,16), [16,64) or [64,inf)
                      (d in {4, 16, 64}), is corrupted in step j only (r +- 1, r x 10 or r / 10, sign flip);
                      the prefix is cut after step j + 2. Each case has a control twin (clean prefix).
    --stage generate  the student continues every prefix with vLLM (4 samples, T=0.6, up to 16k tokens,
                      capped by the context left after the prefix)
    --stage score     detection: the continuation recomputes and states the original value r (reported
                      net of the control's base rate of stating r); recovery: the final answer is correct.
                      By distance bucket and corruption kind; controls give the base re-check/recovery.
Detection is a numeric string match here; the proposal adds an LLM-judge rubric on top.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import palign_grader
from prompting import stop_token_ids

DISTANCES = (4, 16, 64)
KINDS = ("plus_minus_one", "times_ten", "sign_flip")
# a number may end a sentence ("... is 137."), but not continue into a longer token or decimal
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)")
_RECHECK = re.compile(r"\b(wait|let me (?:re)?check|double[- ]check|verify|that doesn't|mistake|error)\b", re.I)


def numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def corrupt(value: str, kind: str, rng: random.Random) -> str:
    number = float(value)
    is_int = re.fullmatch(r"-?\d+", value) is not None
    if kind == "plus_minus_one":
        out = number + rng.choice((-1, 1))
    elif kind == "times_ten":
        out = number * 10 if rng.random() < 0.5 or is_int else number / 10
    else:
        out = -number
    return str(int(out)) if is_int and float(out).is_integer() else f"{out:g}"


def distance_bucket(gap: int) -> int | None:
    """d of the bucket [d, next d) holding the first-reuse gap i - j; None below the smallest d."""
    for d, upper in zip(DISTANCES, DISTANCES[1:] + (10**9,)):
        if d <= gap < upper:
            return d
    return None


def build(records: list[dict], per_distance: int, seed: int) -> list[dict]:
    """Injected/control case pairs, `per_distance` pairs per distance bucket.

    A candidate value r is introduced in step j (absent from the question and every earlier step) and
    reused later; its bucket is the gap to its *first* reuse, so each case sits in exactly one of
    [4,16), [16,64), [64,inf). r must not occur in steps j+1..j+2: those stay in the prefix, and a
    detection metric that looks for r in the continuation could then be passed by copying.
    """
    rng = random.Random(seed)
    by_distance = defaultdict(list)
    for record in records:
        text = record["prompt"] + record["response"]
        nodes = record["nodes"]
        steps = [text[n["char_start"] : n["char_end"]] for n in nodes]
        last_step = len(steps) - 2 if nodes[-1]["kind"] == "answer" else len(steps) - 1
        seen = set(numbers(steps[0]))  # values already in the question are not "introduced"
        first_step = {}
        for j in range(1, last_step + 1):
            for value in numbers(steps[j]):
                if value not in seen and value not in first_step and (abs(float(value)) >= 10 or "." in value):
                    first_step[value] = j
            seen |= set(numbers(steps[j]))
        for value, j in first_step.items():
            if j + 2 > last_step:
                continue
            reuse = [i for i in range(j + 1, last_step + 1) if value in numbers(steps[i])]
            if not reuse or reuse[0] <= j + 2:  # no reuse, or r still visible in the kept prefix
                continue
            d = distance_bucket(reuse[0] - j)
            if d is None:
                continue
            kind = rng.choice(KINDS)
            wrong = corrupt(value, kind, rng)
            cut = nodes[j + 2]["char_end"]
            clean_prefix = text[len(record["prompt"]) : cut]
            bad_step = re.sub(rf"(?<![\w.]){re.escape(value)}(?!\w|\.\d)", wrong, steps[j])
            start = nodes[j]["char_start"] - len(record["prompt"])
            bad_prefix = clean_prefix[:start] + bad_step + clean_prefix[start + len(steps[j]) :]
            base = {"trace_id": record["id"], "distance": d, "first_reuse_gap": reuse[0] - j, "kind": kind,
                    "value": value, "wrong": wrong, "step": j, "gold": record["gold"], "prompt": record["prompt"]}
            by_distance[d].append([
                {**base, "id": f"{record['id']}-{value}", "prefix": bad_prefix, "control": False},
                {**base, "id": f"{record['id']}-{value}-ctrl", "prefix": clean_prefix, "control": True},
            ])
    kept = []
    for d in DISTANCES:
        pairs = by_distance.get(d, [])
        rng.shuffle(pairs)
        kept += [case for pair in pairs[:per_distance] for case in pair]
    return kept


def generate(cases: list[dict], args) -> list[dict]:
    """Continue every prefix; each request gets max_tokens = min(--max-tokens, context left after its
    prompt), and cases leaving less than --min-new-tokens are dropped (reported), since the student's
    context is 32,768 tokens and a prefix cut after step j + 2 can already be long."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora = args.lora_adapter
    base = args.base_model if lora else args.model
    tokenizer = AutoTokenizer.from_pretrained(base)
    prompts = [c["prompt"] + c["prefix"] for c in cases]
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    budget = [min(args.max_tokens, args.max_model_len - n) for n in lengths]
    keep = [k for k, b in enumerate(budget) if b >= args.min_new_tokens]
    print(f"{len(cases) - len(keep)} / {len(cases)} cases dropped: prefix leaves < {args.min_new_tokens} tokens of context")
    llm = LLM(model=base, enable_lora=bool(lora), max_lora_rank=args.lora_r,
              max_model_len=args.max_model_len, dtype="bfloat16", disable_log_stats=True, seed=args.seed)
    request = LoRARequest("adapter", 1, args.model) if lora else None
    params = [SamplingParams(n=args.n_samples, temperature=0.6, top_p=0.95, top_k=20, max_tokens=budget[k],
                             seed=args.seed, stop_token_ids=stop_token_ids(tokenizer)) for k in keep]
    outputs = llm.generate([prompts[k] for k in keep], params, lora_request=request)
    kept = []
    for k, output in zip(keep, outputs):
        cases[k]["continuations"] = [o.text for o in output.outputs]
        kept.append(cases[k])
    return kept


def score(cases: list[dict]) -> dict:
    """Rates per (group, distance) and per (injected, distance, kind), plus the net detection.

    "states_r": the continuation writes the original value r. On injected cases r appears nowhere in the
    prefix, so writing it means recomputing it (detection); on controls r *is* in the prefix, so the rate
    is the base rate of restating it. `detection_net` = injected minus control, per distance and overall.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for case in cases:
        group = "control" if case["control"] else "injected"
        for text in case.get("continuations", []):
            states = int(case["value"] in numbers(text))
            keys = [(group, case["distance"]), (group, "all")]
            if not case["control"]:
                keys.append((group, case["distance"], case["kind"]))
            for key in keys:
                groups[key]["states_r"].append(states)
                groups[key]["recheck"].append(int(bool(_RECHECK.search(text))))
                if case["gold"]:
                    groups[key]["recovery"].append(palign_grader.grade([case["prefix"] + text], case["gold"])[0])
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    report = {"/".join(map(str, key)): {metric: mean(values) for metric, values in metrics.items()}
              | {"n": len(metrics["states_r"])}
              for key, metrics in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))}
    for d in list(DISTANCES) + ["all"]:
        inj, ctl = report.get(f"injected/{d}"), report.get(f"control/{d}")
        if inj:
            inj["detection"] = inj["states_r"]
            if ctl:
                inj["detection_net"] = inj["states_r"] - ctl["states_r"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "generate", "score"), required=True)
    parser.add_argument("--records", help="held-out student records, --style sgl (build)")
    parser.add_argument("--cases", required=True, help="cases JSONL (build output / generate+score input)")
    parser.add_argument("--output", help="generate: cases with continuations; score: report JSON")
    parser.add_argument("--per-distance", type=int, default=34, help="cases per distance (3 x 34 ~ 100 in the pilot)")
    parser.add_argument("--model")
    parser.add_argument("--base-model")
    parser.add_argument("--lora-adapter", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=32768, help="student context (Qwen3-*-Base: 32768)")
    parser.add_argument("--min-new-tokens", type=int, default=4096, help="drop cases leaving less room than this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.stage == "build":
        records = [json.loads(line) for line in open(args.records)]
        cases = build(records, args.per_distance, args.seed)
        Path(args.cases).parent.mkdir(parents=True, exist_ok=True)
        with open(args.cases, "w") as handle:
            for case in cases:
                handle.write(json.dumps(case) + "\n")
        counts = defaultdict(int)
        for case in cases:
            counts[case["distance"]] += not case["control"]
        if min(counts.get(d, 0) for d in DISTANCES) < args.per_distance:
            print(f"WARNING: fewer than {args.per_distance} cases in some bucket (the >=64 bucket needs long traces)")
        print(f"{len(cases)} cases (half controls) -> {args.cases}: {dict(counts)}")
        return
    cases = [json.loads(line) for line in open(args.cases)]
    if args.stage == "generate":
        cases = generate(cases, args)
        with open(args.output, "w") as handle:
            for case in cases:
                handle.write(json.dumps(case) + "\n")
        return
    report = score(cases)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
