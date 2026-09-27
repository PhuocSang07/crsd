"""Generate with vLLM and score pass@1 / pass@3 with the unbiased estimator; one grader for every arm.

    python src/evaluate.py --protocol proposal --model <ckpt> --base-model <base> --tag <tag>
    python src/evaluate.py --protocol palign   --model <ckpt> --base-model <base> --tag <tag>
    python src/evaluate.py --rescore results-proposal/<tag>/raw/aime24.jsonl        # no GPU needed

Protocols (any setting can be overridden by a flag):
    proposal  Sec. 6.4: T=0.6, top-p 0.95, top-k 20, n = 16 (AIME24/25, AMC12) / 4 (MATH500),
              max_model_len 32768, max_tokens 31744
    palign    the baselines' own protocol (SGL / P-ALIGN eval scripts): T=0.6, top-p 0.9,
              repetition penalty 1.05, n = 3, max_model_len 4096, max_tokens 3584, eager
Grader: P-ALIGN's (math_verify OR oat_math_grader) for every arm. --prompt-style sgl (default, the
students' training format), thinking, or zeroshot / fewshot (Base student, B0). --export-traces writes
rollouts for D3.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import palign_grader
from answer_scoring import score_generation
from benchmarks import BENCHMARKS, DEFAULT_SAMPLES, few_shot_prompt
from pass_at_k import bootstrap_ci, mean_pass_at_k
from prompting import render_prompt, stop_token_ids, user_content

PROMPT_STYLES = ("sgl", "thinking", "zeroshot", "fewshot")
PROTOCOLS = {
    "proposal": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
                 "max_model_len": 32768, "max_tokens": 31744, "enforce_eager": False},
    "palign": {"temperature": 0.6, "top_p": 0.9, "top_k": -1, "repetition_penalty": 1.05,
               "max_model_len": 4096, "max_tokens": 3584, "enforce_eager": True, "n_samples": 3},
}


def build_prompts(records: list[dict], config: dict, tokenizer) -> list[str]:
    style = config["prompt_style"]
    if style == "fewshot":
        return [few_shot_prompt(r["question"]) for r in records]
    if style == "zeroshot":
        return [user_content(r["question"]) + "\n" for r in records]
    return [render_prompt(tokenizer, r["question"], style) for r in records]


def make_llm(model_path: str, config: dict):
    from vllm import LLM

    kwargs = dict(
        max_model_len=config["max_model_len"], gpu_memory_utilization=config["gpu_memory_utilization"],
        dtype="bfloat16", enforce_eager=config["enforce_eager"], disable_log_stats=True,
        tensor_parallel_size=config.get("tensor_parallel_size", 1), seed=config["seed"],
    )
    if config.get("lora_adapter"):
        from vllm.lora.request import LoRARequest

        llm = LLM(model=config["base_model"], enable_lora=True, max_lora_rank=config["lora_r"], **kwargs)
        return llm, LoRARequest("adapter", 1, model_path)
    return LLM(model=model_path, **kwargs), None


def generate(llm, lora_request, records: list[dict], prompts: list[str], n: int, config: dict, tokenizer):
    """Yield (records_slice, prompts_slice, completions) batch by batch so a stop keeps finished problems."""
    from vllm import SamplingParams

    sampling = SamplingParams(
        n=n, temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
        repetition_penalty=config["repetition_penalty"], max_tokens=config["max_tokens"], seed=config["seed"],
        stop=["\n\nProblem:"] if config["prompt_style"] == "fewshot" else None,
        stop_token_ids=stop_token_ids(tokenizer),
    )
    batch = config.get("batch_size") or len(records)
    for start in range(0, len(records), batch):
        outputs = llm.generate(prompts[start : start + batch], sampling, lora_request=lora_request, use_tqdm=False)
        yield records[start : start + batch], prompts[start : start + batch], [
            [{"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)} for c in o.outputs]
            for o in outputs
        ]


def label_generations(texts: list[str], gold: str, task_type: str, grader: str) -> list[int]:
    if grader == "palign" and task_type == "math":
        return palign_grader.grade(texts, gold)
    return [int(score_generation(t, gold, task_type)) for t in texts]


def score_file(path: Path, ks=(1, 3), grader: str = "palign") -> tuple[dict, list[list[int]]]:
    rows = [json.loads(line) for line in path.open()]
    labels, lengths, truncated, unanswered = [], [], [], []
    for row in rows:
        texts = [g["text"] for g in row["generations"]]
        labels.append(label_generations(texts, row["gold"], row["task_type"], grader))
        lengths += [g.get("n_tokens", 0) for g in row["generations"]]
        truncated += [int(g.get("finish_reason") == "length") for g in row["generations"]]
        unanswered += [int("\\boxed" not in t) for t in texts]
    n = min(len(l) for l in labels) if labels else 0
    graders = "+".join(palign_grader.active_graders()) if grader == "palign" else "builtin"
    summary = {"benchmark": path.stem, "n_problems": len(rows), "samples_per_problem": n, "grader": graders,
               "length": float(np.mean(lengths)) if lengths else 0.0,
               "truncation_rate": float(np.mean(truncated)) if truncated else 0.0,
               "no_answer_rate": float(np.mean(unanswered)) if unanswered else 0.0}
    for k in ks:
        if k <= n:
            summary[f"pass@{k}"] = mean_pass_at_k(labels, k)
            summary[f"pass@{k}_ci"] = bootstrap_ci(labels, k, resamples=500)
    return summary, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="proposal")
    parser.add_argument("--model", help="checkpoint (adapter dir or full model)")
    parser.add_argument("--tag")
    parser.add_argument("--benchmarks", default="aime24,aime25,amc12,math500")
    parser.add_argument("--rescore")
    parser.add_argument("--grader", choices=("palign", "builtin"), default="palign")
    parser.add_argument("--shard", help="i/n round-robin share of each benchmark")
    parser.add_argument("--n-samples", type=int, help="one n for every benchmark (overrides the protocol)")
    parser.add_argument("--n-samples-map", help="per-benchmark n, e.g. aime24=8,aime25=8,amc12=8 (pilot)")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", help="default: results-<protocol>")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES)
    parser.add_argument("--template-tokenizer", help="tokenizer whose chat template renders prompts (default: base model)")
    parser.add_argument("--base-model", help="base weights for a LoRA adapter")
    parser.add_argument("--lora-adapter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--export-traces", help="write rollouts as traces JSONL (+ .labels.jsonl) for D3")
    args = parser.parse_args()

    if args.rescore:
        summary, _ = score_file(Path(args.rescore), grader=args.grader)
        print(json.dumps(summary, indent=2))
        return

    config = dict(PROTOCOLS[args.protocol])
    config.update(yaml.safe_load(Path(args.config).read_text()) if args.config else {})
    config.update({k: v for k, v in vars(args).items() if v is not None and k not in ("config", "rescore")})
    for key, value in {"gpu_memory_utilization": 0.9, "seed": 42, "prompt_style": "sgl", "lora_r": 16,
                       "results_dir": f"results-{args.protocol}", "batch_size": 64}.items():
        config.setdefault(key, value)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("template_tokenizer") or config.get("base_model") or config["model"])
    tag = config.get("tag") or Path(config["model"]).name
    shard_index, shard_count = (int(x) for x in config["shard"].split("/")) if config.get("shard") else (0, 1)
    suffix = f"-shard{shard_index}of{shard_count}" if shard_count > 1 else ""

    run_dir = Path(config["results_dir"]) / tag
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    llm, lora_request = make_llm(config["model"], config)
    summaries, exported = [], []
    per_bench = dict(item.split("=") for item in config.get("n_samples_map", "").split(",") if item)
    for name in config["benchmarks"].split(","):
        records = BENCHMARKS[name]()[shard_index::shard_count]
        n = int(per_bench.get(name) or config.get("n_samples") or DEFAULT_SAMPLES[name])
        prompts = build_prompts(records, config, tokenizer)
        raw_path = run_dir / "raw" / f"{name}{suffix}.jsonl"
        print(f"[{name}] {len(records)} problems x {n} ({args.protocol})")
        with raw_path.open("w") as handle:
            for batch, batch_prompts, completions in generate(llm, lora_request, records, prompts, n, config, tokenizer):
                for record, prompt, gens in zip(batch, batch_prompts, completions):
                    handle.write(json.dumps({"id": record["id"], "gold": record["gold"], "task_type": record["task_type"],
                                             "generations": gens}) + "\n")
                    if config.get("export_traces"):
                        labels = label_generations([g["text"] for g in gens], record["gold"], "math", config["grader"])
                        for g_index, (g, correct) in enumerate(zip(gens, labels)):
                            exported.append({"id": f"{name}-{record['id']}-r{g_index}", "question": record["question"],
                                             "gold": record["gold"], "prompt": prompt, "response": g["text"],
                                             "correct": int(correct), "finish_reason": g["finish_reason"]})
        summary, _ = score_file(raw_path, grader=config["grader"])
        summary.update(model=tag, protocol=args.protocol, prompt_style=config["prompt_style"])
        summaries.append(summary)
        print(f"[{name}] " + "  ".join(f"{k}={summary[k]:.2%}" for k in ("pass@1", "pass@3") if k in summary)
              + f"  length={summary['length']:.0f}  truncated={summary['truncation_rate']:.2%}")

    (run_dir / f"summary{suffix}.json").write_text(json.dumps(summaries, indent=2))
    mean = lambda key: np.mean([s[key] for s in summaries if key in s])  # noqa: E731
    print(f"\n{tag} [{args.protocol}]: mean pass@1 = {mean('pass@1'):.2%}  mean pass@3 = {mean('pass@3'):.2%}")
    if config.get("export_traces"):
        path = Path(config["export_traces"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as traces, open(str(path) + ".labels.jsonl", "w") as labels_file:
            for row in exported:
                traces.write(json.dumps(row) + "\n")
                labels_file.write(json.dumps({"id": row["id"], "correct": row["correct"]}) + "\n")
        print(f"exported {len(exported)} rollouts -> {path}")


if __name__ == "__main__":
    main()
