"""Held-out mass calibration on fixed student heads (MC-CSRD v4 Sec. 7.4: hypothesis H1, gate G2).

    python src/mass_diagnostics.py --teacher T --student base=D0 --student b1=D1 --student b3=D3 --reference b1 \
        --output-dir OUT

T and every D are extract_routing.py --stage targets outputs (with the raw mass M) over the same held-out records;
all students were read with one head list (the fixed train heads), so arms differ only in their weights. Rows are
the valid rows of both sides; every row averages all tokens of its step. Per arm and band, pooled over all rows:

    E_Z               mean |Z_S - Z_T|                                    mass calibration (H1)
    dM[g], |dM|[g]    unconditional mass gap sum_{j in g} D_S - D_T per group g (question, distance bins)
    KL_raw            mean KL(D_T || D_S) over {REST} U F(i); = Ber + cond (chain rule, both reported)
    KL_route          mean KL(P || Q) of the far-normalized routing (v3 L_route); dmu(>=32) as in diagnostics.py D1
"avg" is the mean of the two bands. CIs resample traces; contrasts arm - reference reuse each resample for both arms.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from routing import (
    DISTANCE_BINS,
    far_target_mask,
    kl_rows,
    mass_gap_by_bin,
    mass_groups,
    mc_kl_rows,
    mc_parts,
    routing_gap_by_bin,
)
from signal_bank import SignalSource

SCALARS = ("ez", "zT", "zS", "kl_raw", "ber", "cond", "kl_route")


def trace_terms(teacher: dict, student: dict, d_min: int) -> list[dict]:
    """Per band: {metric: (sum, count)} of one trace, so traces pool exactly."""
    to = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float64))  # noqa: E731
    num_nodes = len(teacher["hash"])
    far = far_target_mask(num_nodes, d_min)
    rows = torch.from_numpy(np.asarray(teacher["rows"]).astype(bool) & np.asarray(student["rows"]).astype(bool))
    names, groups = mass_groups(far)
    out = []
    for b in range(np.asarray(teacher["P"]).shape[0]):
        P, Q, M_T, M_S = to(teacher["P"][b]), to(student["P"][b]), to(teacher["M"][b]), to(student["M"][b])
        Z_T, Z_S = M_T.sum(-1), M_S.sum(-1)
        valid = rows & (Z_T > 0)
        n = int(valid.sum())
        terms = {
            "ez": (float((Z_S - Z_T)[valid].abs().sum()), n),
            "zT": (float(Z_T[valid].sum()), n),
            "zS": (float(Z_S[valid].sum()), n),
            "kl_raw": (float(mc_kl_rows(M_T, M_S)[valid].sum()), n),
            "kl_route": (float(kl_rows(P, Q)[valid].sum()), n),
        }
        ber, cond = mc_parts(M_T, M_S, valid)  # row means -> sums
        terms["ber"], terms["cond"] = (float(ber) * n, n), (float(cond) * n, n)
        for gap in mass_gap_by_bin(M_T, M_S, valid, groups, names):
            terms[f"dM{gap['bin']}"] = (gap["dM_sum"], gap["count"])
            terms[f"absdM{gap['bin']}"] = (gap["absdM_sum"], gap["count"])
        # v3 D1 dmu(>=32): a row with a target at >= 64 also has one in [32, 64), whose row count is the denominator
        gaps = routing_gap_by_bin(P, Q, valid, DISTANCE_BINS, far)
        terms["dmu_ge32"] = (gaps[3]["dmu_sum"] + gaps[4]["dmu_sum"], gaps[3]["dmu_count"])
        out.append(terms)
    return out


def load_arm(teacher_dir: str, student_dir: str, d_min: int) -> dict[str, list[dict]]:
    teacher, student = SignalSource(teacher_dir), SignalSource(student_dir)
    out = {}
    for trace_id in sorted(set(teacher.ids()) & set(student.ids())):
        t, s = teacher.get(trace_id), student.get(trace_id)
        if not np.array_equal(t["hash"], s["hash"]):
            raise ValueError(f"{trace_id}: teacher and student nodes differ")
        if "M" not in t or "M" not in s:
            raise SystemExit(f"{trace_id}: no raw mass M (re-extract {teacher_dir if 'M' not in t else student_dir} "
                             "with the v4 extract_routing.py)")
        out[trace_id] = trace_terms(t, s, d_min)
    if not out:
        raise SystemExit(f"no trace shared by {teacher_dir} and {student_dir}")
    return out


def arrays(per_trace: list[list[dict]], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(sums, counts) [traces, bands, keys]."""
    shape = (len(per_trace), len(per_trace[0]), len(keys))
    sums, counts = np.zeros(shape), np.zeros(shape)
    for t, bands in enumerate(per_trace):
        for b, terms in enumerate(bands):
            for k, key in enumerate(keys):
                sums[t, b, k], counts[t, b, k] = terms.get(key, (0.0, 0))
    return sums, counts


def pooled(sums: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """weights [R, traces] (resample multiplicities) -> [R, bands + 1 ("avg"), keys]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        per_band = np.einsum("rt,tbk->rbk", weights, sums) / np.einsum("rt,tbk->rbk", weights, counts)
    return np.concatenate([per_band, per_band.mean(1, keepdims=True)], axis=1)


def ci(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values.size else [float("nan")] * 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, help="teacher held-out targets (dir or bank) with M")
    parser.add_argument("--student", action="append", required=True, help="TAG=DIR, repeatable")
    parser.add_argument("--reference", help="tag every other arm is contrasted with (e.g. the v3-objective arm b1)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-min", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arms = dict(item.split("=", 1) for item in args.student)
    loaded = {tag: load_arm(args.teacher, path, args.d_min) for tag, path in arms.items()}
    shared = sorted(set.intersection(*(set(v) for v in loaded.values())))
    per_trace = {tag: [data[i] for i in shared] for tag, data in loaded.items()}
    first = next(iter(per_trace.values()))
    group_keys = [key for key in first[0][0] if key.startswith(("dM", "absdM"))]
    keys = list(SCALARS) + ["dmu_ge32"] + group_keys
    bands = [f"b{b}" for b in range(len(first[0]))] + ["avg"]
    rng = np.random.default_rng(args.seed)
    weights = np.stack([np.bincount(rng.integers(0, len(shared), len(shared)), minlength=len(shared))
                        for _ in range(args.bootstrap)]).astype(float)
    point = {tag: pooled(*arrays(data, keys), np.ones((1, len(shared))))[0] for tag, data in per_trace.items()}
    boot = {tag: pooled(*arrays(data, keys), weights) for tag, data in per_trace.items()}

    def entry(value, samples):
        return {"value": float(value), "ci": ci(samples)}

    result = {"traces": len(shared), "d_min": args.d_min, "teacher": args.teacher, "students": arms, "arms": {}}
    for tag in per_trace:
        result["arms"][tag] = {band: {key: entry(point[tag][b, k], boot[tag][:, b, k]) for k, key in enumerate(keys)}
                               for b, band in enumerate(bands)}
    if args.reference:
        if args.reference not in per_trace:
            raise SystemExit(f"--reference {args.reference} is not one of {list(per_trace)}")
        ref = args.reference
        contrast_keys = [key for key in keys if key in ("ez", "kl_raw", "ber", "cond", "kl_route", "dmu_ge32")
                         or key.startswith("absdM")]
        result["contrasts"] = {
            f"{tag}-{ref}": {band: {key: entry(point[tag][b, keys.index(key)] - point[ref][b, keys.index(key)],
                                               boot[tag][:, b, keys.index(key)] - boot[ref][:, b, keys.index(key)])
                                    for key in contrast_keys} for b, band in enumerate(bands)}
            for tag in per_trace if tag != ref}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mass-diagnostics.json").write_text(json.dumps(result, indent=2))
    fmt = lambda v: f"{v['value']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]"  # noqa: E731
    lines = [f"# Held-out mass calibration ({len(shared)} traces, fixed student heads)", ""]
    for band in bands:
        lines += [f"## {band}", "", "| arm | Z_S (Z_T) | E_Z | KL_raw | Ber | cond | KL_route | dmu(>=32) |",
                  "|---|---|---|---|---|---|---|---|"]
        for tag in per_trace:
            m = result["arms"][tag][band]
            lines.append(f"| {tag} | {m['zS']['value']:.4f} ({m['zT']['value']:.4f}) | {fmt(m['ez'])} | "
                         f"{fmt(m['kl_raw'])} | {m['ber']['value']:.4f} | {m['cond']['value']:.4f} | "
                         f"{fmt(m['kl_route'])} | {fmt(m['dmu_ge32'])} |")
        groups = [k[2:] for k in group_keys if k.startswith("dM")]
        lines += ["", "Unconditional mass gap dM = S - T per group (mean over rows with a target in the group):", "",
                  "| arm | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for tag in per_trace:
            m = result["arms"][tag][band]
            lines.append(f"| {tag} | " + " | ".join(f"{m['dM' + g]['value']:+.4f}" for g in groups) + " |")
        lines.append("")
    if args.reference:
        lines += [f"## Paired contrasts vs {args.reference} (avg of bands; E_Z / KL: negative = closer to the teacher)", "",
                  "| contrast | E_Z | KL_raw | KL_route | dmu(>=32) |", "|---|---|---|---|---|"]
        for name, by_band in result["contrasts"].items():
            c = by_band["avg"]
            lines.append(f"| {name} | {fmt(c['ez'])} | {fmt(c['kl_raw'])} | {fmt(c['kl_route'])} | {fmt(c['dmu_ge32'])} |")
    (out / "mass-diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
