"""CPU smoke test of the whole CSRD pipeline on tiny random Qwen3 models (no GPU, no vLLM).

    python scripts/smoke_test_pipeline.py --tokenizer <dir with a Qwen3 tokenizer> [--workdir /tmp/crsd-smoke]

Builds a tiny "teacher" (6 layers) and "student" (4 layers) with the real Qwen3 tokenizer, writes
synthetic traces, then runs every CLI stage in order: data_prep (both tokenizers) -> anchor labels
-> teacher calibrate/select/targets -> causal targets -> SFT, CSRD, CSRD-PQ, CSRD-QK training ->
student extraction (+ QK-Restore) -> diagnostics. It checks plumbing, not results.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(*args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": ""}
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, env=env)


def tiny_model(tokenizer_dir: str, out: Path, layers: int, heads: int, kv: int, hidden: int, seed: int) -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv, head_dim=hidden // heads, max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(out)
    tokenizer.save_pretrained(out)


def synthetic_traces(tokenizer_dir: str, path: Path, n: int) -> None:
    from transformers import AutoTokenizer

    sys.path.insert(0, str(SRC))
    from prompting import render_prompt

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    openers = ["Let me compute", "Wait, check", "So we get", "First, set up", "Hmm, maybe", "Now multiply"]
    rng = random.Random(0)
    with path.open("w") as handle:
        for index in range(n):
            question = f"What is {index} times {index + 3} plus {2 * index}?"
            steps = [
                f"{rng.choice(openers)} the value {rng.randint(1, 99)} with {rng.randint(1, 99)} and carry it forward to the next line."
                for _ in range(rng.randint(14, 24))
            ]
            response = "<think>\n" + "\n\n".join(steps) + "\n</think>\n\nThe answer is \\boxed{" + str(index * (index + 3) + 2 * index) + "}."
            handle.write(json.dumps({
                "id": f"smoke-{index}", "question": question, "gold": str(index), "prompt": render_prompt(tokenizer, question),
                "response": response, "teacher_solve_rate": 1.0,
            }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--workdir", default="/tmp/crsd-smoke")
    args = parser.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    teacher, student = work / "teacher", work / "student"
    tiny_model(args.tokenizer, teacher, layers=6, heads=4, kv=2, hidden=64, seed=0)
    tiny_model(args.tokenizer, student, layers=4, heads=4, kv=2, hidden=32, seed=1)
    synthetic_traces(args.tokenizer, work / "traces.jsonl", 12)

    s = str(SRC)
    run(f"{s}/data_prep.py", "--traces-path", str(work / "traces.jsonl"), "--tokenizer", str(teacher),
        "--output-path", str(work / "teacher.jsonl"))
    run(f"{s}/data_prep.py", "--traces-path", str(work / "traces.jsonl"), "--tokenizer", str(student),
        "--output-path", str(work / "student.jsonl"))
    run(f"{s}/anchor_labels.py", "--data-path", str(work / "student.jsonl"))
    run(f"{s}/anchor_labels.py", "--data-path", str(work / "teacher.jsonl"))

    routing_t = work / "routing-teacher"
    run(f"{s}/extract_routing.py", "--stage", "calibrate", "--model-name", str(teacher), "--data-path",
        str(work / "teacher.jsonl"), "--output-dir", str(routing_t), "--n-traces", "8", "--query-block", "64")
    run(f"{s}/extract_routing.py", "--stage", "select", "--output-dir", str(routing_t), "--k-per-band", "2")
    targets = work / "targets"
    run(f"{s}/extract_routing.py", "--stage", "targets", "--model-name", str(teacher), "--data-path",
        str(work / "teacher.jsonl"), "--heads-json", str(routing_t / "heads-excess_bg.json"), "--output-dir", str(targets),
        "--save-per-head")
    causal = work / "causal"
    run(f"{s}/causal_targets.py", "--model-name", str(teacher), "--data-path", str(work / "teacher.jsonl"),
        "--targets-dir", str(targets), "--output-dir", str(causal), "--fraction", "0.5", "--top-j", "5", "--query-block", "64")

    common = ["--model-name", str(student), "--data-path", str(work / "student.jsonl"), "--max-steps", "6",
              "--gradient-accumulation-steps", "2", "--logging-steps", "1", "--save-strategy", "no",
              "--lora-r", "4", "--lora-alpha", "4", "--learning-rate", "1e-3", "--ce-chunk", "64"]
    csrd = ["--csrd-lambda", "1.0", "--targets-dir", str(targets), "--causal-dir", str(causal), "--csrd-k-student", "2",
            "--csrd-warmup-frac", "0.34", "--csrd-ramp-frac", "0.17", "--csrd-grad-log-interval", "1"]
    run(f"{s}/train_sft.py", *common, "--output-dir", str(work / "ckpt-sft"))
    run(f"{s}/train_sft.py", *common, *csrd, "--output-dir", str(work / "ckpt-csrd"), "--csrd-anchor-beta", "1.0",
        "--metrics-log", str(work / "csrd-metrics.json"))
    run(f"{s}/train_sft.py", *common, *csrd, "--output-dir", str(work / "ckpt-csrd-pq"), "--csrd-loss-form", "per_query")
    run(f"{s}/train_sft.py", *common, *csrd, "--output-dir", str(work / "ckpt-csrd-qk"), "--csrd-qk-rank", "2")

    for tag in ("sft", "csrd"):
        routing_s = work / f"routing-student-{tag}"
        base = ["--model-name", str(student), "--adapter", str(work / f"ckpt-{tag}"), "--data-path", str(work / "student.jsonl")]
        run(f"{s}/extract_routing.py", "--stage", "calibrate", *base, "--output-dir", str(routing_s), "--n-traces", "8")
        run(f"{s}/extract_routing.py", "--stage", "select", "--output-dir", str(routing_s), "--k-per-band", "2")
        run(f"{s}/extract_routing.py", "--stage", "targets", *base, "--heads-json", str(routing_s / "heads-excess_bg.json"),
            "--output-dir", str(work / f"student-targets-{tag}"))
        run(f"{s}/extract_routing.py", "--stage", "targets", *base, "--qk-restore", "--heads-json",
            str(routing_s / "heads-excess_bg.json"), "--output-dir", str(work / f"student-targets-{tag}-qkrestore"))
        run(f"{s}/diagnostics.py", "--teacher-targets", str(targets), "--student-targets", str(work / f"student-targets-{tag}"),
            "--records", str(work / "student.jsonl"), "--causal-dir", str(causal), "--adapter", str(work / f"ckpt-{tag}"),
            "--output-dir", str(work / f"diag-{tag}"), "--bootstrap", "200")
    run(f"{s}/qk_restore.py", "--adapter", str(work / "ckpt-csrd"), "--output-dir", str(work / "ckpt-csrd-qkrestore"))
    print("\nsmoke test OK ->", work)


if __name__ == "__main__":
    main()
