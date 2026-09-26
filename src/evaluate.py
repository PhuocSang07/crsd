"""Generate with vLLM and score pass@1 / pass@3 with the unbiased estimator (proposal Sec. 6.4).

    python src/evaluate.py --model <ckpt> --base-model <Qwen3-1.7B-Base> --tag csrd-q8b-1.7b
    python src/evaluate.py --rescore results/<tag>/raw/aime24.jsonl        # no GPU needed

n samples per problem default to 16 for AIME24/AIME25/AMC12 and 4 for MATH500 (--n-samples
overrides all). Sampling: T=0.6, top-p 0.95, top-k 20, max_model_len 32,768 (Qwen3-*-Base's
max_position_embeddings; prompt + generation must fit, so max_tokens defaults to 31,744); answers from the last
\\boxed{} graded with math-verify. Also reports mean response length, the share cut at the token
cap and the share with no final answer, and a two-level (problem x sample) bootstrap CI.

Prompts: the chat template in thinking mode (prompting.render_prompt, identical to training) or,
for the pre-distillation Base student (B0), --prompt-style fewshot (4-shot plain text) / zeroshot.
--export-traces writes the rollouts as traces (for D3: teacher-forcing on student text).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from answer_scoring import score_generation
from benchmarks import BENCHMARKS, DEFAULT_SAMPLES, few_shot_prompt
from pass_at_k import bootstrap_ci, mean_pass_at_k
from prompting import THINK_CLOSE, render_prompt, stop_token_ids, user_content

PROMPT_STYLES = ("chat", "zeroshot", "fewshot")


def build_prompts(records: list[dict], config: dict) -> list[str]:
    style = config.get("prompt_style", "chat")
    if style == "fewshot":
        return [few_shot_prompt(r["question"]) for r in records]
    if style == "zeroshot":
        return [user_content(r["question"]) + "\n" for r in records]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["template_tokenizer"])
    return [render_prompt(tokenizer, r["question"], enable_thinking=True) for r in records]


def make_llm(model_path: str, config: dict):
    from vllm import LLM

    kwargs = dict(
        max_model_len=config["max_model_len"], gpu_memory_utilization=config["gpu_memory_utilization"],
        dtype="bfloat16", enforce_eager=config.get("enforce_eager", False), disable_log_stats=True,
        tensor_parallel_size=config.get("tensor_parallel_size", 1), seed=config["seed"],
    )
    if config.get("lora_adapter"):
        from vllm.lora.request import LoRARequest

        llm = LLM(model=config["base_model"], enable_lora=True, max_lora_rank=config["lora_r"], **kwargs)
        return llm, LoRARequest("adapter", 1, model_path)
    return LLM(model=model_path, **kwargs), None


def generate(llm, lora_request, records: list[dict], prompts: list[str], n: int, config: dict):
    """Yield (records_slice, completions) batch by batch so a stop keeps finished problems."""
    from vllm import SamplingParams

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("template_tokenizer") or config.get("base_model") or config["model"])
    stop = ["\n\nProblem:"] if config.get("prompt_style") == "fewshot" else None
    sampling = SamplingParams(
        n=n, temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
        max_tokens=config["max_tokens"], seed=config["seed"], stop=stop, stop_token_ids=stop_token_ids(tokenizer),
    )
    batch = config.get("batch_size") or len(records)
    for start in range(0, len(records), batch):
        outputs = llm.generate(prompts[start : start + batch], sampling, lora_request=lora_request, use_tqdm=False)
        yield records[start : start + batch], prompts[start : start + batch], [
            [{"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)} for c in o.outputs]
            for o in outputs
        ]


def score_file(path: Path, ks=(1, 3)) -> tuple[dict, list[list[int]]]:
    rows = [json.loads(line) for line in path.open()]
    labels, lengths, truncated, unanswered = [], [], [], []
    for row in rows:
        texts = [g["text"] for g in row["generations"]]
        labels.append([int(score_generation(t, row["gold"], row["task_type"])) for t in texts])
        lengths += [g.get("n_tokens", 0) for g in row["generations"]]
        truncated += [int(g.get("finish_reason") == "length") for g in row["generations"]]
        unanswered += [int("\\boxed" not in t) for t in texts]
    n = min(len(l) for l in labels) if labels else 0
    summary = {"benchmark": path.stem, "n_problems": len(rows), "samples_per_problem": n,
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
    parser.add_argument("--model", help="checkpoint (adapter dir or full model)")
    parser.add_argument("--tag")
    parser.add_argument("--benchmarks", default="aime24,aime25,amc12,math500")
    parser.add_argument("--rescore")
    parser.add_argument("--shard", help="i/n round-robin share of each benchmark")
    parser.add_argument("--n-samples", type=int, help="override the per-benchmark defaults")
    parser.add_argument("--n-samples-map", help="per-benchmark override, e.g. aime24=8,aime25=8,amc12=8 (pilot)")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES)
    parser.add_argument("--template-tokenizer", help="tokenizer whose chat template renders prompts (the teacher's)")
    parser.add_argument("--base-model", help="base weights for a LoRA adapter")
    parser.add_argument("--lora-adapter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--export-traces", help="write rollouts as traces JSONL (+ .labels.jsonl) for D3")
    args = parser.parse_args()

    if args.rescore:
        summary, _ = score_file(Path(args.rescore))
        print(json.dumps(summary, indent=2))
        return

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    config.update({k: v for k, v in vars(args).items() if v is not None and k not in ("config", "rescore")})
    for key, value in {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 31744, "max_model_len": 32768,
                       "gpu_memory_utilization": 0.9, "seed": 42, "prompt_style": "chat", "lora_r": 64,
                       "results_dir": "results", "batch_size": 64}.items():
        config.setdefault(key, value)
    if config["prompt_style"] == "chat":
        config.setdefault("template_tokenizer", config.get("base_model") or config["model"])
    tag = config.get("tag") or Path(config["model"]).name
    shard_index, shard_count = (int(x) for x in config["shard"].split("/")) if config.get("shard") else (0, 1)
    suffix = f"-shard{shard_index}of{shard_count}" if shard_count > 1 else ""

    run_dir = Path(config["results_dir"]) / tag
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    llm, lora_request = make_llm(config["model"], config)
    summaries, exported = [], []
    for name in config["benchmarks"].split(","):
        records = BENCHMARKS[name]()[shard_index::shard_count]
        per_bench = dict(item.split("=") for item in config.get("n_samples_map", "").split(",") if item)
        n = int(per_bench.get(name) or config.get("n_samples") or DEFAULT_SAMPLES[name])
        prompts = build_prompts(records, config)
        raw_path = run_dir / "raw" / f"{name}{suffix}.jsonl"
        print(f"[{name}] {len(records)} problems x {n}")
        with raw_path.open("w") as handle:
            for batch, batch_prompts, completions in generate(llm, lora_request, records, prompts, n, config):
                for record, prompt, gens in zip(batch, batch_prompts, completions):
                    handle.write(json.dumps({"id": record["id"], "gold": record["gold"], "task_type": record["task_type"],
                                             "generations": gens}) + "\n")
                    if config.get("export_traces"):
                        for g_index, g in enumerate(gens):
                            exported.append({"id": f"{name}-{record['id']}-r{g_index}", "question": record["question"],
                                             "gold": record["gold"], "prompt": prompt, "response": g["text"],
                                             "correct": int(score_generation(g["text"], record["gold"], "math")),
                                             "closed": THINK_CLOSE in g["text"]})
        summary, _ = score_file(raw_path)
        summary.update(model=tag, prompt_style=config["prompt_style"])
        summaries.append(summary)
        print(f"[{name}] " + "  ".join(f"{k}={summary[k]:.2%}" for k in ("pass@1", "pass@3") if k in summary)
              + f"  length={summary['length']:.0f}  truncated={summary['truncation_rate']:.2%}")

    (run_dir / f"summary{suffix}.json").write_text(json.dumps(summaries, indent=2))
    mean = lambda key: np.mean([s[key] for s in summaries if key in s])  # noqa: E731
    print(f"\n{tag}: mean pass@1 = {mean('pass@1'):.2%}  mean pass@3 = {mean('pass@3'):.2%}")
    if config.get("export_traces"):
        path = Path(config["export_traces"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as traces, open(str(path) + ".labels.jsonl", "w") as labels:
            for row in exported:
                traces.write(json.dumps(row) + "\n")
                labels.write(json.dumps({"id": row["id"], "correct": row["correct"]}) + "\n")
        print(f"exported {len(exported)} rollouts -> {path}")


if __name__ == "__main__":
    main()
