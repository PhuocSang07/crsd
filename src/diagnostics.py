"""Diagnostic protocol D1-D5 and pilot gates G1-G4 (proposal Sec. 5 and Table 6).

Everything is teacher-forced on one text: --teacher-targets and --student-targets are
extract_routing.py outputs over the *same* records (held-out teacher traces for D1/D2/D4, student
rollouts on the dev set for D3), so P_i and Q_i correspond row by row.

    D1  distance profile: RG(b), Delta-mu(b) per bin, slope over log distance, RG(>=64)/RG([4,8)),
        trace-level bootstrap CIs                                             -> G1, Hypothesis 1
    D2  anchor rows vs computation rows (paired sign-flip test over traces), target-side gap,
        nDCG@k of the student's vertical-score ranking against the teacher's  -> G2, Hypothesis 2
    D3  error prediction on student rollouts: logistic regression "wrong answer" on length, NLL and
        reflection-marker count, with and without the routing gap; CV AUC gain  -> G3, Hypothesis 3
    D4  attention-causal agreement: median Spearman(P_i, C~_i) per band (and per head if saved) -> G4
    D5  LoRA update norms of W_Q/W_K vs W_V/W_O per layer; mean attention distance of the heads
D6 (QK-Restore) is D1 again on targets extracted with --qk-restore, plus evaluate.py on the
qk_restore.py adapter.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from pass_at_k import paired_permutation_test
from receiver_heads import vertical_scores
from routing import DISTANCE_BINS, causal_target, far_target_mask, js_divergence, routing_gap_by_bin

REFLECTION_MARKERS = re.compile(r"\b(wait|hmm|alternatively|let me (?:check|verify|double[- ]check)|actually|but let me)\b", re.I)
BIN_NAMES = ["[4,8)", "[8,16)", "[16,32)", "[32,64)", "[64,inf)"]
# geometric bin centres for the slope over log distance; the open bin uses sqrt(64 * 128)
BIN_CENTRES = [math.sqrt(4 * 8), math.sqrt(8 * 16), math.sqrt(16 * 32), math.sqrt(32 * 64), math.sqrt(64 * 128)]


def load_pairs(teacher_dir: Path, student_dir: Path) -> list[dict]:
    """Matched (teacher, student) routing per trace; node character spans must agree."""
    pairs = []
    for t_path in sorted(teacher_dir.glob("*.npz")):
        s_path = student_dir / t_path.name
        if not s_path.exists():
            continue
        t, s = np.load(t_path), np.load(s_path)
        if not np.array_equal(t["char_spans"], s["char_spans"]):
            raise ValueError(f"{t_path.stem}: teacher and student nodes differ")
        pairs.append({
            "id": t_path.stem, "P": torch.from_numpy(t["P"]), "Q": torch.from_numpy(s["Q"] if "Q" in s else s["P"]),
            "rows": torch.from_numpy(t["rows"]) & torch.from_numpy(s["rows"]),
            "nll": float(s["nll"]) if "nll" in s else None,
        })
    if not pairs:
        raise SystemExit(f"no matching <id>.npz between {teacher_dir} and {student_dir}")
    return pairs


def band_views(pair: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per band plus 'avg' (mean over bands of the row distributions)."""
    views = {f"band{b}": (pair["P"][b], pair["Q"][b]) for b in range(pair["P"].size(0))}
    views["avg"] = (pair["P"].mean(0), pair["Q"].mean(0))
    return views


# ---------------------------------------------------------------- D1

def _pooled(terms: list[list[dict]], key: str) -> np.ndarray:
    sums = np.array([[t[f"{key}_sum"] for t in trace] for trace in terms])
    counts = np.array([[t[f"{key}_count"] for t in trace] for trace in terms])
    return sums, counts


def _profile(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    total = counts.sum(0)
    return np.where(total > 0, sums.sum(0) / np.maximum(total, 1), np.nan)


def _summaries(rg: np.ndarray, dmu_sums: np.ndarray, dmu_counts: np.ndarray) -> dict:
    valid = np.isfinite(rg)
    slope = np.polyfit(np.log(np.asarray(BIN_CENTRES)[valid]), rg[valid], 1)[0] if valid.sum() >= 2 else np.nan
    # Delta-mu(>=32) sums a row's mass deficit over both far bins; a row with a target at distance
    # >= 64 always has targets in [32, 64) too, so the [32, 64) row count is the number of rows.
    far_rows = dmu_counts[:, 3].sum()
    return {
        "slope_log_distance": float(slope),
        "ratio_64_over_4": float(rg[4] / rg[0]) if rg[0] > 0 and np.isfinite(rg[4]) else float("nan"),
        "dmu_ge32": float(dmu_sums[:, 3:].sum() / far_rows) if far_rows > 0 else float("nan"),
    }


def d1_distance_profile(pairs: list[dict], resamples: int, seed: int, d_min: int = 4) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    for view in band_views(pairs[0]):
        terms = [routing_gap_by_bin(*band_views(p)[view], p["rows"], DISTANCE_BINS,
                                    far_target_mask(p["P"].size(-1), d_min)) for p in pairs]
        rg_s, rg_c = _pooled(terms, "rg")
        mu_s, mu_c = _pooled(terms, "dmu")
        rg, dmu = _profile(rg_s, rg_c), _profile(mu_s, mu_c)
        point = _summaries(rg, mu_s, mu_c)
        boot = {key: [] for key in point}
        boot_rg, boot_mu = [], []
        for _ in range(resamples):
            idx = rng.integers(0, len(pairs), len(pairs))
            b_rg, b_mu = _profile(rg_s[idx], rg_c[idx]), _profile(mu_s[idx], mu_c[idx])
            boot_rg.append(b_rg)
            boot_mu.append(b_mu)
            for key, value in _summaries(b_rg, mu_s[idx], mu_c[idx]).items():
                boot[key].append(value)
        ci = lambda values: [float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))]  # noqa: E731
        out[view] = {
            "bins": BIN_NAMES,
            "RG": rg.tolist(), "RG_ci": [ci(np.asarray(boot_rg)[:, k]) for k in range(len(BIN_NAMES))],
            "RG_rows": rg_c.sum(0).tolist(),
            "delta_mu": dmu.tolist(), "delta_mu_ci": [ci(np.asarray(boot_mu)[:, k]) for k in range(len(BIN_NAMES))],
            **{key: value for key, value in point.items()},
            **{f"{key}_ci": ci(values) for key, values in boot.items()},
        }
    return out


def gate_g1(d1: dict, view: str = "avg") -> dict:
    """G1: RG(>=64)/RG([4,8)) >= 1.5 with a CI excluding 1, or Delta-mu(>=32) < 0 significantly.
    Also reports Hypothesis 1's falsification test (slope CI contains 0, or ratio < 1.2)."""
    r = d1[view]
    ratio_ok = r["ratio_64_over_4"] >= 1.5 and r["ratio_64_over_4_ci"][0] > 1.0
    dmu_ok = r["dmu_ge32_ci"][1] < 0
    slope_ci = r["slope_log_distance_ci"]
    undetermined = not np.isfinite(r["ratio_64_over_4"])  # no row reaches the >=64 bin
    return {
        "G1": bool(ratio_ok or dmu_ok), "ratio_criterion": bool(ratio_ok), "delta_mu_criterion": bool(dmu_ok),
        "H1_rejected": None if undetermined else bool((slope_ci[0] <= 0 <= slope_ci[1]) or r["ratio_64_over_4"] < 1.2),
    }


# ---------------------------------------------------------------- D2

def ndcg_at_k(relevance: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """nDCG@k of the ranking by `predicted`, with graded relevance = the teacher's vertical score."""
    finite = np.isfinite(relevance) & np.isfinite(predicted)
    relevance, predicted = relevance[finite], predicted[finite]
    if relevance.size == 0:
        return float("nan")
    relevance = relevance - relevance.min()
    discounts = 1.0 / np.log2(np.arange(2, min(k, relevance.size) + 2))
    dcg = (relevance[np.argsort(-predicted)][: discounts.size] * discounts).sum()
    ideal = (np.sort(relevance)[::-1][: discounts.size] * discounts).sum()
    return float(dcg / ideal) if ideal > 0 else float("nan")


def d2_anchors(pairs: list[dict], labels: dict[str, list[int]], kinds: dict[str, list[str]], d_min: int, k: int, seed: int) -> dict:
    out = {}
    for view in band_views(pairs[0]):
        row_diffs, target_diffs, ndcgs = [], [], []
        for pair in pairs:
            anchor = labels.get(pair["id"])
            if anchor is None:
                continue
            P, Q = band_views(pair)[view]
            rows = pair["rows"] & (P.sum(-1) > 0)
            js = js_divergence(P, Q)
            anchor_t = torch.tensor(anchor, dtype=torch.bool)
            compute_t = torch.tensor([kind == "active_computation" for kind in kinds[pair["id"]]])
            a_rows, c_rows = rows & anchor_t, rows & compute_t
            if a_rows.any() and c_rows.any():
                row_diffs.append(float(js[a_rows].mean() - js[c_rows].mean()))
            dev = (P - Q).abs() * rows.unsqueeze(-1).float()
            a_cols, c_cols = anchor_t & (dev.sum(0) > 0), compute_t & (dev.sum(0) > 0)
            if a_cols.any() and c_cols.any():
                target_diffs.append(float(dev[:, a_cols].sum(0).mean() - dev[:, c_cols].sum(0).mean()))
            nu_t = vertical_scores(P.unsqueeze(0), rows, d_min)[0].numpy()
            nu_s = vertical_scores(Q.unsqueeze(0), rows, d_min)[0].numpy()
            ndcgs.append(ndcg_at_k(nu_t, nu_s, k))
        one_sided = lambda diffs: paired_permutation_test(diffs, [0.0] * len(diffs), seed=seed) / 2 if diffs else float("nan")  # noqa: E731
        out[view] = {
            "traces_row_test": len(row_diffs),
            "row_gap_anchor_minus_compute": float(np.mean(row_diffs)) if row_diffs else float("nan"),
            "row_p_one_sided": one_sided(row_diffs) if row_diffs and np.mean(row_diffs) > 0 else 1.0,
            "target_gap_anchor_minus_compute": float(np.mean(target_diffs)) if target_diffs else float("nan"),
            "target_p_one_sided": one_sided(target_diffs) if target_diffs and np.mean(target_diffs) > 0 else 1.0,
            f"ndcg@{k}": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
        }
    out["G2"] = bool(out["avg"]["row_p_one_sided"] < 0.05 or out["avg"]["target_p_one_sided"] < 0.05)
    return out


# ---------------------------------------------------------------- D3

def d3_error_prediction(pairs: list[dict], records: dict[str, dict], labels_path: Path, seed: int) -> dict:
    """Does the routing gap predict wrong answers beyond length, NLL and reflection markers?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    wrong = {row["id"]: 1 - int(row["correct"]) for row in map(json.loads, open(labels_path))}
    X_ctrl, X_gap, y = [], [], []
    for pair in pairs:
        if pair["id"] not in wrong or pair["id"] not in records:
            continue
        record = records[pair["id"]]
        P, Q = band_views(pair)["avg"]
        rows = pair["rows"] & (P.sum(-1) > 0)
        gap = float(js_divergence(P, Q)[rows].mean()) if rows.any() else 0.0
        X_ctrl.append([math.log(record["n_tokens"]), pair["nll"] or 0.0, len(REFLECTION_MARKERS.findall(record["response"]))])
        X_gap.append(gap)
        y.append(wrong[pair["id"]])
    y = np.asarray(y)
    if len(y) < 20 or y.min() == y.max():
        return {"n": int(len(y)), "note": "too few rollouts or a single class"}
    X_ctrl = np.asarray(X_ctrl)
    X_full = np.column_stack([X_ctrl, np.asarray(X_gap)])

    def cv_auc(X):
        scores = np.zeros(len(y))
        folds = StratifiedKFold(n_splits=min(5, int(min(y.sum(), len(y) - y.sum()))), shuffle=True, random_state=seed)
        for train, test in folds.split(X, y):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[train], y[train])
            scores[test] = model.predict_proba(X[test])[:, 1]
        return float(roc_auc_score(y, scores))

    auc_ctrl, auc_full = cv_auc(X_ctrl), cv_auc(X_full)
    return {"n": int(len(y)), "wrong_rate": float(y.mean()), "auc_controls": auc_ctrl, "auc_with_gap": auc_full,
            "delta_auc": auc_full - auc_ctrl, "G3": bool(auc_full - auc_ctrl >= 0.03),
            "note": "the 'local gap before the first error' criterion needs first-error annotation; not computed"}


# ---------------------------------------------------------------- D4

def d4_attention_causal(teacher_dir: Path, causal_dir: Path, d_min: int) -> dict:
    from scipy.stats import spearmanr

    per_band, per_head = {}, {}
    for c_path in sorted(causal_dir.glob("*.npz")):
        t_path = teacher_dir / c_path.name
        if not t_path.exists():
            continue
        t, c = np.load(t_path), np.load(c_path)
        C, J = torch.from_numpy(c["C"]), torch.from_numpy(c["J"])
        far = far_target_mask(C.size(0), d_min)
        c_tilde, support = causal_target(C, J, far)
        sources = {f"band{b}": torch.from_numpy(t["P"][b]) for b in range(t["P"].shape[0])}
        sources["avg"] = torch.from_numpy(t["P"]).mean(0)
        if "R_heads" in t:
            for (layer, head), R in zip(t["heads"].tolist(), t["R_heads"]):
                per_head.setdefault(f"L{layer}H{head}", [])
                sources[f"head:L{layer}H{head}"] = torch.from_numpy(R.astype(np.float32))
        for i in range(C.size(0)):
            cols = support[i].nonzero().squeeze(-1)
            if cols.numel() < 3:
                continue
            for name, P in sources.items():
                rho = spearmanr(P[i, cols].numpy(), c_tilde[i, cols].numpy()).statistic
                if np.isfinite(rho):
                    (per_head[name[5:]] if name.startswith("head:") else per_band.setdefault(name, [])).append(float(rho))
    summary = {name: {"median_spearman": float(np.median(v)), "rows": len(v)} for name, v in per_band.items() if v}
    heads = {name: float(np.median(v)) for name, v in per_head.items() if v}
    return {"bands": summary, "heads": heads, "G4": bool(summary.get("avg", {}).get("median_spearman", 0.0) >= 0.3)}


# ---------------------------------------------------------------- D5

def d5_lora_norms(adapter_dir: Path) -> dict:
    """||Delta W||_F = (alpha/r) ||B A||_F per module type and layer; QK vs VO ratio per layer."""
    from safetensors.torch import load_file

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    tensors = load_file(adapter_dir / "adapter_model.safetensors")
    norms: dict[int, dict[str, float]] = {}
    for key, A in tensors.items():
        if "lora_A" not in key:
            continue
        B = tensors[key.replace("lora_A", "lora_B")]
        layer = int(re.search(r"layers\.(\d+)\.", key).group(1))
        module = re.search(r"\.(\w+_proj)\.lora_A", key).group(1)
        norms.setdefault(layer, {})[module] = float((B.float() @ A.float()).norm() * scaling)
    per_layer = {}
    for layer, mods in sorted(norms.items()):
        qk = math.hypot(mods.get("q_proj", 0.0), mods.get("k_proj", 0.0))
        vo = math.hypot(mods.get("v_proj", 0.0), mods.get("o_proj", 0.0))
        per_layer[layer] = {**mods, "qk_over_vo": qk / vo if vo > 0 else float("nan")}
    return {"per_layer": per_layer}


def d5_attention_distance(before: Path | None, after: Path) -> dict:
    """Mean attention distance of *every* head (routing/mean-distance.npy from the calibrate stage), before vs
    after training, per layer and averaged over the heads of each depth band. The mean runs over the query
    tokens of rows with |F(i)| >= 2 (the tokens every routing quantity is defined on)."""
    from receiver_heads import band_layers

    def load(path):
        return np.load(path) if path and path.exists() else None

    a, b = load(after), load(before)
    if a is None:
        return {"note": f"{after} not found"}
    bands = band_layers(a.shape[0])
    out = {"after_per_layer": a.mean(1).tolist(), "after_band": [float(a[layers].mean()) for layers in bands]}
    if b is not None and b.shape == a.shape:
        out.update(before_per_layer=b.mean(1).tolist(), before_band=[float(b[layers].mean()) for layers in bands],
                   delta_band=[float(a[layers].mean() - b[layers].mean()) for layers in bands])
    return out


def write_causal_heads(per_head: dict[str, float], num_layers: int | None, k_per_band: int, path: Path) -> None:
    """CSRD-C head selection: per depth band, the heads whose routing best agrees with suppression effects."""
    from receiver_heads import DEPTH_BANDS, band_layers

    if not per_head or not num_layers:
        raise SystemExit("--write-causal-heads needs per-head D4 results (--save-per-head targets) and --num-layers")
    parsed = [(int(m.group(1)), int(m.group(2)), rho) for name, rho in per_head.items()
              if (m := re.fullmatch(r"L(\d+)H(\d+)", name))]
    layers = band_layers(num_layers)
    heads = [[[l, h] for l, h, _ in sorted([x for x in parsed if x[0] in band], key=lambda x: -x[2])[:k_per_band]]
             for band in layers]
    path.write_text(json.dumps({"score": "causal", "num_layers": num_layers, "bands": [list(b) for b in DEPTH_BANDS],
                                "band_layers": layers, "k_per_band": k_per_band, "heads": heads,
                                "median_spearman": {f"L{l}H{h}": rho for l, h, rho in parsed}}, indent=2))
    print(f"causal head selection -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-targets", required=True)
    parser.add_argument("--student-targets", required=True)
    parser.add_argument("--records", help="records with node_labels/anchor (anchor_labels.py), for D2/D3")
    parser.add_argument("--causal-dir", help="causal targets for D4")
    parser.add_argument("--adapter", help="trained LoRA adapter for D5")
    parser.add_argument("--distance-after", help="routing/mean-distance.npy of this student (D5)")
    parser.add_argument("--distance-before", help="routing/mean-distance.npy of the untrained student (D5)")
    parser.add_argument("--rollout-labels", help="JSONL {id, correct} of student rollouts -> D3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-min", type=int, default=4)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-causal-heads", help="CSRD-C: write heads.json with the top --k-per-band heads per band "
                        "by median Spearman(R_head, C~) (needs targets extracted with --save-per-head on all band heads)")
    parser.add_argument("--k-per-band", type=int, default=16)
    parser.add_argument("--num-layers", type=int, help="depth of the model the heads belong to (for --write-causal-heads)")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(Path(args.teacher_targets), Path(args.student_targets))
    records = {}
    if args.records:
        records = {r["id"]: r for r in map(json.loads, open(args.records))}

    report, gates = {"traces": len(pairs)}, {}
    report["D1"] = d1_distance_profile(pairs, args.bootstrap, args.seed, args.d_min)
    gates.update(gate_g1(report["D1"]))
    if records and all("anchor" in r for r in records.values()):
        labels = {rid: r["anchor"] for rid, r in records.items()}
        kinds = {rid: r["node_labels"] for rid, r in records.items()}
        report["D2"] = d2_anchors(pairs, labels, kinds, args.d_min, args.ndcg_k, args.seed)
        gates["G2"] = report["D2"]["G2"]
    if args.rollout_labels and records:
        report["D3"] = d3_error_prediction(pairs, records, Path(args.rollout_labels), args.seed)
        if "G3" in report["D3"]:
            gates["G3"] = report["D3"]["G3"]
    if args.causal_dir:
        report["D4"] = d4_attention_causal(Path(args.teacher_targets), Path(args.causal_dir), args.d_min)
        gates["G4"] = report["D4"]["G4"]
    if args.adapter and (Path(args.adapter) / "adapter_config.json").exists():
        report["D5"] = {"lora_norms": d5_lora_norms(Path(args.adapter))}
    if args.distance_after:
        report.setdefault("D5", {})["attention_distance"] = d5_attention_distance(
            Path(args.distance_before) if args.distance_before else None, Path(args.distance_after))
    report["gates"] = gates
    if args.write_causal_heads:
        write_causal_heads(report.get("D4", {}).get("heads", {}), args.num_layers, args.k_per_band, Path(args.write_causal_heads))

    (output / "diagnostics.json").write_text(json.dumps(report, indent=2))
    avg = report["D1"]["avg"]
    lines = [f"# Diagnostics ({len(pairs)} traces)", "", "| bin | RG | 95% CI | Delta-mu |", "|---|---|---|---|"]
    lines += [f"| {name} | {rg:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {mu:+.4f} |"
              for name, rg, ci, mu in zip(BIN_NAMES, avg["RG"], avg["RG_ci"], avg["delta_mu"])]
    lines += ["", f"slope over log distance: {avg['slope_log_distance']:.4f} {avg['slope_log_distance_ci']}",
              f"RG(>=64)/RG([4,8)): {avg['ratio_64_over_4']:.3f} {avg['ratio_64_over_4_ci']}",
              f"Delta-mu(>=32): {avg['dmu_ge32']:+.4f} {avg['dmu_ge32_ci']}", "",
              "gates: " + ", ".join(f"{k}={v}" for k, v in gates.items())]
    (output / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
