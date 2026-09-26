"""CPU smoke test of the whole CSRD pipeline on tiny random models (no GPU, no vLLM).

    python scripts/smoke_test_pipeline.py --teacher-tokenizer <R1-Distill or Qwen3 tokenizer dir> \\
        --student-tokenizer <Qwen3 tokenizer dir> [--workdir /tmp/crsd-smoke]

The teacher is a tiny Qwen2 model with the teacher's tokenizer (a DeepSeek-R1-Distill tokenizer makes this the
cross-tokenizer case of the read-d32b-q8b track), the student a tiny Qwen3 model. Every CLI stage runs in order:
canonical content -> teacher (thinking) and student (SGL) records -> anchors -> teacher calibrate/select/targets
-> causal targets -> signal bank -> SFT, CSRD, CSRD-PQ, CSRD-QK training from the bank -> student extraction
(+ QK-Restore) -> diagnostics. It checks plumbing, not results.
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


def tiny_model(tokenizer_dir: str, out: Path, arch: str, layers: int, heads: int, kv: int, hidden: int, seed: int) -> None:
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    config_cls, model_cls = (Qwen2Config, Qwen2ForCausalLM) if arch == "qwen2" else (Qwen3Config, Qwen3ForCausalLM)
    config = config_cls(
        vocab_size=len(tokenizer), hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv, head_dim=hidden // heads, max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model_cls(config).save_pretrained(out)
    tokenizer.save_pretrained(out)


def synthetic_canonical(path: Path, n: int) -> None:
    openers = ["Let me compute", "Wait, check", "So we get", "First, set up", "Hmm, maybe", "Now multiply"]
    rng = random.Random(0)
    with path.open("w") as handle:
        for index in range(n):
            steps = [
                f"{rng.choice(openers)} the value {rng.randint(10, 99)} with {rng.randint(10, 99)} and carry it forward to the next line."
                for _ in range(rng.randint(14, 24))
            ]
            handle.write(json.dumps({
                "id": f"smoke-{index}", "question": f"What is {index} times {index + 3} plus {2 * index}?",
                "thinking": "\n\n".join(steps), "answer": "The answer is \\boxed{" + str(index * (index + 3) + 2 * index) + "}.",
                "gold": str(index * (index + 3) + 2 * index), "closed": True,
            }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--workdir", default="/tmp/crsd-smoke")
    args = parser.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    teacher, student = work / "teacher", work / "student"
    tiny_model(args.teacher_tokenizer, teacher, "qwen2", layers=6, heads=4, kv=2, hidden=64, seed=0)
    tiny_model(args.student_tokenizer, student, "qwen3", layers=4, heads=4, kv=2, hidden=32, seed=1)
    synthetic_canonical(work / "canonical.jsonl", 12)

    s = str(SRC)
    t_rec, s_rec = str(work / "teacher.jsonl"), str(work / "student.jsonl")
    run(f"{s}/data_prep.py", "--canonical", str(work / "canonical.jsonl"), "--tokenizer", str(teacher),
        "--style", "thinking", "--output-path", t_rec)
    run(f"{s}/data_prep.py", "--canonical", str(work / "canonical.jsonl"), "--tokenizer", str(student),
        "--style", "sgl", "--output-path", s_rec)
    run(f"{s}/anchor_labels.py", "--data-path", s_rec)

    routing_t, targets, causal = work / "routing-teacher", work / "targets", work / "causal"
    run(f"{s}/extract_routing.py", "--stage", "calibrate", "--model-name", str(teacher), "--data-path", t_rec,
        "--output-dir", str(routing_t), "--n-traces", "8", "--query-block", "64")
    run(f"{s}/extract_routing.py", "--stage", "select", "--output-dir", str(routing_t), "--k-per-band", "2")
    run(f"{s}/extract_routing.py", "--stage", "targets", "--model-name", str(teacher), "--data-path", t_rec,
        "--heads-json", str(routing_t / "heads-excess_bg.json"), "--output-dir", str(targets), "--save-per-head",
        "--source-name", "smoke")
    run(f"{s}/causal_targets.py", "--model-name", str(teacher), "--data-path", t_rec, "--targets-dir", str(targets),
        "--output-dir", str(causal), "--fraction", "0.5", "--top-j", "5", "--query-block", "64")
    bank = work / "signals.safetensors"
    run(f"{s}/signal_bank.py", "pack", "--targets-dir", str(targets), "--causal-dir", str(causal), "--output", str(bank))
    run(f"{s}/signal_bank.py", "info", str(bank))

    common = ["--model-name", str(student), "--data-path", s_rec, "--max-steps", "6",
              "--gradient-accumulation-steps", "2", "--logging-steps", "1", "--save-strategy", "no",
              "--lora-r", "4", "--lora-alpha", "4", "--learning-rate", "1e-3", "--ce-chunk", "64"]
    csrd = ["--csrd-lambda", "1.0", "--signals", str(bank), "--csrd-k-student", "2",
            "--csrd-warmup-frac", "0.34", "--csrd-ramp-frac", "0.17", "--csrd-grad-log-interval", "1"]
    run(f"{s}/train_sft.py", *common, "--output-dir", str(work / "ckpt-sft"))
    run(f"{s}/train_sft.py", *common, *csrd, "--output-dir", str(work / "ckpt-csrd"), "--csrd-anchor-beta", "1.0",
        "--metrics-log", str(work / "csrd-metrics.json"))
    run(f"{s}/train_sft.py", *common, *csrd, "--output-dir", str(work / "ckpt-csrd-pq"), "--csrd-loss-form", "per_query")
    run(f"{s}/train_sft.py", *common, *csrd, "--output-dir", str(work / "ckpt-csrd-qk"), "--csrd-qk-rank", "2")

    for tag in ("sft", "csrd"):
        routing_s = work / f"routing-student-{tag}"
        base = ["--model-name", str(student), "--adapter", str(work / f"ckpt-{tag}")]
        run(f"{s}/extract_routing.py", "--stage", "calibrate", *base, "--data-path", s_rec, "--output-dir", str(routing_s),
            "--n-traces", "8")
        run(f"{s}/extract_routing.py", "--stage", "select", "--output-dir", str(routing_s), "--k-per-band", "2")
        heads = str(routing_s / "heads-excess_bg.json")
        run(f"{s}/extract_routing.py", "--stage", "targets", *base, "--data-path", s_rec, "--heads-json", heads,
            "--output-dir", str(work / f"student-targets-{tag}"))
        run(f"{s}/extract_routing.py", "--stage", "targets", *base, "--qk-restore", "--data-path", s_rec, "--heads-json", heads,
            "--output-dir", str(work / f"student-targets-{tag}-qkrestore"))
        run(f"{s}/diagnostics.py", "--teacher-targets", str(targets), "--student-targets", str(work / f"student-targets-{tag}"),
            "--records", s_rec, "--causal-dir", str(causal), "--adapter", str(work / f"ckpt-{tag}"),
            "--distance-after", str(routing_s / "mean-distance.npy"), "--output-dir", str(work / f"diag-{tag}"),
            "--bootstrap", "200")
    run(f"{s}/qk_restore.py", "--adapter", str(work / "ckpt-csrd"), "--output-dir", str(work / "ckpt-csrd-qkrestore"))
    print("\nsmoke test OK ->", work)


if __name__ == "__main__":
    main()
