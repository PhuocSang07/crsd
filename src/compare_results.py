"""Merge per-run eval summaries into the comparison table (mean +- std over seeds) and check G5.

    python src/compare_results.py --results-dir results-proposal --track read-d32b-q8b

Tags "<arm>-<track>[-s<seed>]" are pooled over seeds; every arm is compared with --baseline:
per-benchmark deltas, G5 accuracy (mean pass@1 >= +1.5 pp over SFT), and a problem-level paired
permutation test (all benchmarks pooled) with Holm correction over the CSRD arms (Sec. 6.4).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import score_file
from pass_at_k import holm_bonferroni, paired_permutation_test

BENCHMARK_ORDER = ["aime24", "aime25", "amc12", "math500"]
SEED_SUFFIX = re.compile(r"-s\d+$")


def load(results_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """{arm_tag (seed stripped): {benchmark: [summary per seed]}}"""
    runs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(results_dir.glob("*/summary.json")):
        for row in json.loads(path.read_text()):
            runs[SEED_SUFFIX.sub("", row["model"])][row["benchmark"]].append(row)
    return runs


def per_problem_pass1(results_dir: Path, arm: str, benchmarks: list[str]) -> dict[tuple[str, int], float]:
    """{(benchmark, problem index): pass@1 averaged over the arm's seeds} from the raw generations."""
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or SEED_SUFFIX.sub("", run_dir.name) != arm:
            continue
        for name in benchmarks:
            raw = run_dir / "raw" / f"{name}.jsonl"
            if raw.exists():
                _, labels = score_file(raw, ks=(1,))
                for index, row in enumerate(labels):
                    values[(name, index)].append(float(np.mean(row)))
    return {key: float(np.mean(v)) for key, v in values.items()}


def significance(results_dir: Path, runs, baseline: str, family: str) -> dict[str, dict]:
    """Paired tests of every arm vs the baseline; Holm correction only over arms matching `family`."""
    benchmarks = [b for b in BENCHMARK_ORDER if any(b in r for r in runs.values())]
    base = per_problem_pass1(results_dir, baseline, benchmarks)
    raw_p, deltas = {}, {}
    for arm in runs:
        if arm == baseline:
            continue
        other = per_problem_pass1(results_dir, arm, benchmarks)
        shared = sorted(set(base) & set(other))
        if not shared:
            continue
        a, b = [other[k] for k in shared], [base[k] for k in shared]
        raw_p[arm] = paired_permutation_test(a, b)
        deltas[arm] = (float(np.mean(a) - np.mean(b)) * 100, len(shared))
    in_family = {arm: p for arm, p in raw_p.items() if re.search(family, arm)}
    adjusted = holm_bonferroni(in_family) if in_family else {}
    return {arm: {"delta_pp_problem_mean": deltas[arm][0], "problems": deltas[arm][1],
                  "p": raw_p[arm], "p_holm": adjusted.get(arm, float("nan"))} for arm in raw_p}


def cell(rows: list[dict], metric: str) -> tuple[float, float] | None:
    values = [r[metric] for r in rows if metric in r]
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0) if values else None


def table(runs, metric: str, baseline: str | None) -> tuple[str, dict[str, float]]:
    benchmarks = [b for b in BENCHMARK_ORDER if any(b in r for r in runs.values())]
    lines = ["| Arm | seeds | " + " | ".join(benchmarks) + " | Avg |", "|" + "---|" * (len(benchmarks) + 3)]
    averages = {}
    for tag, by_bench in runs.items():
        cells = [cell(by_bench.get(b, []), metric) for b in benchmarks]
        present = [c[0] for c in cells if c]
        if not present:
            continue
        averages[tag] = float(np.mean(present))
        seeds = max(len(by_bench.get(b, [])) for b in benchmarks)
        text = [f"{c[0]:.2%} ± {c[1]:.2%}" if c else "-" for c in cells]
        lines.append(f"| {tag} | {seeds} | " + " | ".join(text) + f" | {averages[tag]:.2%} |")
    if baseline and baseline in runs:
        for tag in runs:
            if tag == baseline:
                continue
            deltas = []
            for b in benchmarks:
                new, base = cell(runs[tag].get(b, []), metric), cell(runs[baseline].get(b, []), metric)
                deltas.append(f"{(new[0] - base[0]) * 100:+.2f}" if new and base else "-")
            avg = (averages.get(tag, np.nan) - averages.get(baseline, np.nan)) * 100
            lines.append(f"| **Δ {tag} − {baseline} (pp)** | | " + " | ".join(deltas) + f" | {avg:+.2f} |")
    return "\n".join(lines), averages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results-proposal")
    parser.add_argument("--track", help="keep only runs of this track (tags '<arm>-<track>[-s<seed>]')")
    parser.add_argument("--baseline", help="arm tag (seed stripped) every other arm is compared with (default sft-<track>)")
    parser.add_argument("--family", default=r"^csrd-(?!qkrestore)", help="regex of the arms forming the Holm family")
    args = parser.parse_args()

    runs = load(Path(args.results_dir))
    if args.track:
        runs = {tag: rows for tag, rows in runs.items() if tag.endswith(f"-{args.track}")}
        args.baseline = args.baseline or f"sft-{args.track}"
    if not runs:
        raise SystemExit(f"no */summary.json under {args.results_dir}" + (f" for {args.track}" if args.track else ""))
    parts, gates = [], {}
    for metric in ("pass@1", "pass@3"):
        text, averages = table(runs, metric, args.baseline)
        parts.append(f"### {metric}\n\n{text}")
        if metric == "pass@1" and args.baseline in averages:
            gates = {tag: {"delta_pass@1_pp": (avg - averages[args.baseline]) * 100,
                           "G5_accuracy": (avg - averages[args.baseline]) * 100 >= 1.5}
                     for tag, avg in averages.items() if tag != args.baseline}
    output = "\n\n".join(parts)
    tests = significance(Path(args.results_dir), runs, args.baseline, args.family) if args.baseline in runs else {}
    if tests:
        output += (f"\n\n### Paired permutation vs {args.baseline} (problem level, Holm-corrected)\n\n"
                   "| Arm | Δ pass@1 (pp, problem mean) | problems | p | p (Holm, CSRD family) |\n|---|---|---|---|---|\n"
                   + "\n".join(f"| {arm} | {t['delta_pp_problem_mean']:+.2f} | {t['problems']} | {t['p']:.4f} | {t['p_holm']:.4f} |"
                                for arm, t in tests.items()))
    if gates:
        output += "\n\n### G5 (pass@1 criterion)\n\n" + "\n".join(
            f"- {tag}: {g['delta_pass@1_pp']:+.2f} pp -> {'PASS' if g['G5_accuracy'] else 'fail'}" for tag, g in gates.items())
    print(output)
    suffix = f"-{args.track}" if args.track else ""
    Path(args.results_dir, f"comparison-table{suffix}.md").write_text(output + "\n")
    Path(args.results_dir, f"gates-g5{suffix}.json").write_text(json.dumps(gates, indent=2))
    Path(args.results_dir, f"significance{suffix}.json").write_text(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
