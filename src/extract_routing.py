"""Offline routing extraction from a frozen model reading traces (teacher targets, Sec. 4.3 / 4.8).

    --stage calibrate  every head of every layer on the first --n-traces records (D_cal, 200):
                       vertical scores nu_j -> per-trace receiver scores (both "excess_bg" and
                       "kurtosis") and mean attention distance -> calib-shard*.npz
    --stage select     (CPU) merge the calibration shards, keep the top-K heads per depth band ->
                       heads-<score>.json for both scores plus a random-K control (A2), with
                       split-half stability
    --stage targets    the selected heads on every record -> <output-dir>/<id>.npz with
                       P [bands, N, N] (band-averaged far routing, lower triangular),
                       Z [bands, N] (far mass), rows [N], char_spans [N, 2]

The same script reads the *student* for diagnostics (--adapter, --qk-restore, its own heads.json).
Queries: the full token set of every row (Q_T(i) = I_T(v_i)). Each layer's attention is recomputed
from the captured q/k in query blocks with an exact full-row softmax, so no T x T matrix is ever
stored and SDPA/FlashAttention still runs the forward.
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import attention_capture
from model_utils import load_causal_lm, num_layers_and_heads
from receiver_heads import DEPTH_BANDS, SCORE_MODES, band_layers, receiver_scores, select_heads, split_half_stability, vertical_scores
from routing import far_target_mask, head_row_routing_blockwise, valid_rows
from step_nodes import token_node_ids


def load_records(path: str, limit: int | None = None) -> list[dict]:
    records = []
    with open(path) as handle:
        for line in handle:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def record_char_spans(record: dict) -> np.ndarray:
    return np.asarray([[n["char_start"], n["char_end"]] for n in record["nodes"]], dtype=np.int64)


@torch.no_grad()
def response_nll(model, hidden: torch.Tensor, record: dict, chunk: int = 2048) -> float:
    """Mean next-token NLL of the response tokens (D3 control variable), chunked over the LM head."""
    ids = torch.tensor(record["input_ids"], device=hidden.device)
    start, end = record["response_token_span"]
    positions = torch.arange(start - 1, end - 1, device=hidden.device)
    head = model.get_output_embeddings()
    total = 0.0
    for offset in range(0, positions.numel(), chunk):
        pos = positions[offset : offset + chunk]
        logits = head(hidden[pos]).float()
        total += float(torch.nn.functional.cross_entropy(logits, ids[pos + 1], reduction="sum"))
    return total / max(1, positions.numel())


class RoutingExtractor:
    """Runs one teacher-forced forward and computes routing of chosen heads inside the layer hooks."""

    def __init__(self, model, d_min: int, query_block: int = 1024, head_chunk: int = 8):
        self.model = model
        self.modules = attention_capture.attention_modules(model)
        attention_capture.check_attn_implementation(model)
        self.num_layers, self.num_heads, self.head_dim = num_layers_and_heads(model)
        self.d_min = d_min
        self.query_block = query_block
        self.head_chunk = head_chunk
        self.device = next(model.parameters()).device
        self.want_nll = False
        self.last_nll = None

    @torch.no_grad()
    def run(self, record: dict, heads_by_layer: dict[int, list[int]] | None, reduce):
        """reduce(layer, heads, out) consumes each head chunk's routing dict (R, Z, mean_distance)."""
        input_ids = torch.tensor([record["input_ids"]], device=self.device)
        spans = [(n["token_start"], n["token_end"]) for n in record["nodes"]]
        key_nodes = torch.tensor(token_node_ids(spans, input_ids.size(1)), device=self.device)
        far = far_target_mask(len(spans), self.d_min, self.device)

        def make(layer):
            heads = list(range(self.num_heads)) if heads_by_layer is None else heads_by_layer.get(layer)
            if not heads:
                return None

            def hook(module, query, key, value, attention_mask, **kwargs):
                scale = kwargs.get("scaling") or module.scaling
                groups = query.size(1) // key.size(1)
                for start in range(0, len(heads), self.head_chunk):
                    chunk = heads[start : start + self.head_chunk]
                    index = torch.tensor(chunk, device=query.device)
                    q = query[0].index_select(0, index)
                    k = key[0].index_select(0, index // groups)
                    reduce(layer, chunk, head_row_routing_blockwise(q, k, key_nodes, far, scale, self.query_block))
                return None

            return hook

        attention_capture.set_hooks(self.modules, make)
        try:
            hidden = self.model.model(input_ids=input_ids).last_hidden_state[0]
        finally:
            attention_capture.clear_hooks(self.modules)
        self.last_nll = response_nll(self.model, hidden, record) if self.want_nll else None
        return far


def calibrate(extractor: RoutingExtractor, records: list[dict], output: Path, shard: str) -> None:
    L, H = extractor.num_layers, extractor.num_heads
    scores = {mode: [] for mode in SCORE_MODES}
    distances, ids = [], []
    for record in tqdm(records, desc="calibrate", unit="trace"):
        num_nodes = len(record["nodes"])
        nu = torch.full((L, H, num_nodes), float("nan"), device=extractor.device)
        md = torch.zeros(L, H, device=extractor.device)

        def reduce(layer, heads, out):
            idx = torch.tensor(heads, device=extractor.device)
            # rows that own queries are exactly the valid rows (|F(i)| >= 2)
            nu[layer, idx] = vertical_scores(out["R"], out["counts"] > 0, extractor.d_min)
            md[layer, idx] = out["mean_distance"]

        extractor.run(record, None, reduce)
        flat = nu.view(L * H, num_nodes)
        for mode in SCORE_MODES:
            scores[mode].append(receiver_scores(flat, mode).view(L, H).cpu().numpy())
        distances.append(md.cpu().numpy())
        ids.append(record["id"])
    np.savez(
        output / f"calib-{shard}.npz",
        ids=np.asarray(ids),
        mean_distance=np.stack(distances),
        **{f"scores_{mode}": np.stack(values) for mode, values in scores.items()},
    )


def select(output: Path, k_per_band: int, seed: int) -> None:
    from scipy.stats import spearmanr

    shards = sorted(glob.glob(str(output / "calib-*.npz")))
    if not shards:
        raise SystemExit(f"no calib-*.npz in {output}")
    data = [np.load(path) for path in shards]
    per_mode = {mode: np.concatenate([d[f"scores_{mode}"] for d in data]) for mode in SCORE_MODES}
    distance = np.concatenate([d["mean_distance"] for d in data]).mean(0)
    num_layers, num_heads = distance.shape
    layers = band_layers(num_layers)
    summary = {}
    for mode, per_trace in per_mode.items():
        mean = np.nanmean(per_trace, axis=0)
        heads = select_heads(mean, k_per_band)
        payload = {
            "score": mode,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "bands": [list(b) for b in DEPTH_BANDS],
            "band_layers": layers,
            "k_per_band": k_per_band,
            "heads": [[list(pair) for pair in band] for band in heads],
            "calibration_traces": int(per_trace.shape[0]),
            "stability": split_half_stability(per_trace, k_per_band, seed=seed),
            "mean_score_selected": [float(np.mean([mean[l, h] for l, h in band])) for band in heads],
            "mean_distance_selected": [float(np.mean([distance[l, h] for l, h in band])) for band in heads],
            "mean_distance_band": [float(distance[band].mean()) for band in layers],
        }
        (output / f"heads-{mode}.json").write_text(json.dumps(payload, indent=2))
        np.save(output / f"mean-scores-{mode}.npy", mean)
        summary[mode] = payload["stability"]
    # A2 control: K random heads per band.
    rng = random.Random(seed)
    random_heads = []
    for band in layers:
        pool = [(l, h) for l in band for h in range(num_heads)]
        random_heads.append([list(pair) for pair in rng.sample(pool, min(k_per_band, len(pool)))])
    base = json.loads((output / "heads-excess_bg.json").read_text())
    (output / "heads-random.json").write_text(json.dumps({**base, "score": "random", "heads": random_heads}, indent=2))
    # Every head of each band: A2/A5 "whole band", and the candidate pool for causal selection (CSRD-C).
    all_band = [[[l, h] for l in band for h in range(num_heads)] for band in layers]
    (output / "heads-allband.json").write_text(json.dumps(
        {**base, "score": "allband", "k_per_band": max(len(b) for b in all_band), "heads": all_band}, indent=2))
    np.save(output / "mean-distance.npy", distance)
    # Table 6 week 1 "xác nhận receiver heads bằng cả hai điểm receiver": how much the two selections agree.
    by_mode = {mode: json.loads((output / f"heads-{mode}.json").read_text())["heads"] for mode in SCORE_MODES}
    summary["agreement_excess_bg_vs_kurtosis"] = {
        "topk_overlap": [len({tuple(x) for x in a} & {tuple(x) for x in b}) / max(1, k_per_band)
                         for a, b in zip(by_mode["excess_bg"], by_mode["kurtosis"])],
        "spearman_mean_scores": float(spearmanr(np.nanmean(per_mode["excess_bg"], 0).ravel(),
                                                np.nanmean(per_mode["kurtosis"], 0).ravel(), nan_policy="omit").statistic),
    }
    (output / "selection-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def extract_targets(extractor: RoutingExtractor, records: list[dict], heads_json: dict, output: Path,
                    save_per_head: bool) -> None:
    bands = [[tuple(pair) for pair in band] for band in heads_json["heads"]]
    heads_by_layer: dict[int, list[int]] = {}
    for band in bands:
        for layer, head in band:
            heads_by_layer.setdefault(layer, []).append(head)
    band_of = {pair: b for b, band in enumerate(bands) for pair in band}
    order = [pair for band in bands for pair in band]

    for record in tqdm(records, desc="targets", unit="trace"):
        path = output / f"{record['id']}.npz"
        if path.exists():
            continue
        num_nodes = len(record["nodes"])
        P = torch.zeros(len(bands), num_nodes, num_nodes, device=extractor.device)
        Z = torch.zeros(len(bands), num_nodes, device=extractor.device)
        per_head_R, per_head_md = {}, {}

        def reduce(layer, heads, out):
            for index, head in enumerate(heads):
                b = band_of[(layer, head)]
                P[b] += out["R"][index] / len(bands[b])
                Z[b] += out["Z"][index] / len(bands[b])
                per_head_md[(layer, head)] = float(out["mean_distance"][index])
                if save_per_head:
                    per_head_R[(layer, head)] = out["R"][index].half().cpu().numpy()

        far = extractor.run(record, heads_by_layer, reduce)
        payload = {
            "P": P.cpu().numpy().astype(np.float32),
            "Z": Z.cpu().numpy().astype(np.float32),
            "rows": valid_rows(far).cpu().numpy(),
            "char_spans": record_char_spans(record),
            "heads": np.asarray(order, dtype=np.int64),
            "mean_distance": np.asarray([per_head_md[pair] for pair in order], dtype=np.float32),
        }
        if save_per_head:
            payload["R_heads"] = np.stack([per_head_R[pair] for pair in order])
        if extractor.last_nll is not None:
            payload["nll"] = np.float32(extractor.last_nll)
        np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibrate", "select", "targets"), required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--adapter", help="LoRA adapter dir (student diagnostics)")
    parser.add_argument("--qk-restore", action="store_true", help="zero the adapter's q/k update (D6)")
    parser.add_argument("--data-path", help="records from data_prep.py for this model's tokenizer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heads-json", help="head selection for --stage targets")
    parser.add_argument("--n-traces", type=int, default=200, help="calibration set size |D_cal|")
    parser.add_argument("--limit", type=int, help="targets: first N records only")
    parser.add_argument("--d-min", type=int, default=4)
    parser.add_argument("--k-per-band", type=int, default=16)
    parser.add_argument("--query-block", type=int, default=1024)
    parser.add_argument("--head-chunk", type=int, default=8)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--save-per-head", action="store_true", help="also store per-head R (float16) for D4 by head")
    parser.add_argument("--save-nll", action="store_true", help="also store the response's mean NLL (D3 control)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "select":
        select(output, args.k_per_band, args.seed)
        return
    if not (args.model_name and args.data_path):
        parser.error(f"--stage {args.stage} needs --model-name and --data-path")

    model = load_causal_lm(args.model_name, args.adapter, args.qk_restore, args.attn_implementation)
    extractor = RoutingExtractor(model, args.d_min, args.query_block, args.head_chunk)
    extractor.want_nll = args.save_nll
    shard = f"shard{args.shard_index}of{args.num_shards}"
    if args.stage == "calibrate":
        records = load_records(args.data_path, args.n_traces)[args.shard_index :: args.num_shards]
        calibrate(extractor, records, output, shard)
    else:
        if not args.heads_json:
            parser.error("--stage targets needs --heads-json")
        heads_json = json.loads(Path(args.heads_json).read_text())
        if heads_json["num_layers"] != extractor.num_layers:
            raise ValueError(f"heads.json is for {heads_json['num_layers']} layers, model has {extractor.num_layers}")
        records = load_records(args.data_path, args.limit)[args.shard_index :: args.num_shards]
        extract_targets(extractor, records, heads_json, output, args.save_per_head)
        (output / "targets-config.json").write_text(json.dumps(
            {"model": args.model_name, "adapter": args.adapter, "qk_restore": args.qk_restore,
             "heads_json": args.heads_json, "d_min": args.d_min, "score": heads_json["score"]}, indent=2))


if __name__ == "__main__":
    main()
