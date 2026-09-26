"""Pilot gates G0-G6 in one place, and the decision-tree branch they select (proposal Sec. 7, Table 6).

    python src/pilot_report.py --track read-q8b-1.7b

Reads what the pilot stages wrote: the teacher-trace retention stats (G0; gen-* tracks only -- read-*
tracks train on the fixed s1K-1.1 set), the SFT student's diagnostics (G1, G2, G4) and D3 (G3), the
proposal-protocol comparison (G5 accuracy), the error-injection reports (G5 detection), and run
summaries (G6 throughput). Missing inputs are reported as "not run".
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--lam", default="0.3", help="lambda of the CSRD arm the gates are read for")
    parser.add_argument("--seed-tag", default="s42", help="seed whose SFT run carries the week-1 diagnostics")
    parser.add_argument("--eval-dir", default="results-proposal")
    parser.add_argument("--diag-dir", default="results")
    parser.add_argument("--checkpoints", default="checkpoints")
    args = parser.parse_args()
    args.baseline = f"sft-{args.track}"
    args.arm = f"csrd-l{args.lam}-{args.track}"
    results, evals = Path(args.diag_dir), Path(args.eval_dir)
    gates: dict[str, object] = {}

    if args.track.startswith("gen-"):
        teacher = args.track.split("-")[1]
        stats = read(Path("data/gen") / teacher / "s1k-traces.jsonl.stats.json")
        gates["G0"] = None if stats is None else stats["kept_questions"] >= 600
    else:
        gates["G0"] = True  # s1K-1.1 as used by the baselines: nothing is filtered
    diag = read(results / f"diag-{args.baseline}-{args.seed_tag}" / "diagnostics.json")
    for gate in ("G1", "G2", "G4"):
        gates[gate] = None if diag is None else diag["gates"].get(gate)
    d3 = read(results / f"diag-{args.baseline}-{args.seed_tag}-d3" / "diagnostics.json")
    gates["G3"] = None if d3 is None else d3["gates"].get("G3")

    g5 = read(evals / f"gates-g5-{args.track}.json") or {}
    accuracy = g5.get(args.arm, {}).get("G5_accuracy")
    delta = g5.get(args.arm, {}).get("delta_pass@1_pp")
    inj_base = read(results / f"inject-{args.baseline}-{args.seed_tag}" / "report.json")
    inj_arm = read(results / f"inject-{args.arm}-{args.seed_tag}" / "report.json")
    detection_gain = None
    if inj_base and inj_arm:
        det = lambda rep: rep.get("injected/all", {}).get("detection_net", np.nan)  # noqa: E731
        detection_gain = 100 * (det(inj_arm) - det(inj_base))
    detection = None if detection_gain is None or delta is None else (detection_gain >= 5 and delta >= 0)
    gates["G5"] = None if accuracy is None and detection is None else bool(accuracy or detection)

    runtimes = {}
    for summary in Path(args.checkpoints).glob("*/run-summary.json"):
        tag = re.sub(r"-s\d+$", "", summary.parent.name)
        runtimes.setdefault(tag, []).append(read(summary).get("train_runtime_s") or np.nan)
    overhead = None
    if args.baseline in runtimes and args.arm in runtimes:
        overhead = np.nanmean(runtimes[args.arm]) / np.nanmean(runtimes[args.baseline]) - 1
    gates["G6"] = None if overhead is None else bool(overhead <= 0.25)

    if gates["G1"] is False:
        decision = "stop CSRD -> fallback B (analysis paper): gradient locality does not become a routing gap"
    elif gates["G1"] and gates["G3"] and gates["G5"]:
        decision = "full programme"
    elif gates["G1"] and gates["G4"] is False:
        decision = "switch default to CSRD-C (causal heads + L_causal), rerun week 2"
    elif gates["G1"] and gates["G3"] is False and gates["G5"] is False:
        decision = "fallback A (uncertainty-aware distillation); keep diagnostics as analysis"
    else:
        decision = "undetermined: some gates have not run yet"

    lines = [f"# CSRD pilot gates -- {args.track}", ""]
    fmt = lambda v: "not run" if v is None else ("PASS" if v else "fail")  # noqa: E731
    lines += [f"- {gate}: {fmt(value)}" for gate, value in gates.items()]
    lines += ["", f"- pass@1 delta ({args.arm} - {args.baseline}): {delta if delta is not None else 'n/a'} pp",
              f"- error-detection gain: {detection_gain if detection_gain is not None else 'n/a'} pp",
              f"- throughput overhead: {f'{overhead:.1%}' if overhead is not None else 'n/a'}", "",
              f"**Decision:** {decision}"]
    text = "\n".join(lines)
    print(text)
    out = results / f"pilot-gates-{args.track}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
