"""Causal routing targets by attention suppression (proposal Sec. 4.4, Eq. 3).

    C[i, j] = (1 / |I(v_i)|) sum_{t in I(v_i)} KL( p_T(. | x_<t) || p_T^{not j}(. | x_<t) )

where p^{not j} blocks every later position from attending to v_j at every layer. One suppressed forward
per target, J = top-24 nodes by teacher vertical score, on a seeded 20% subset (Table 3). Writes
<output-dir>/<id>.npz with C [N, N] (NaN outside J), J and floor; row normalization happens at use.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import attention_capture
from extract_routing import load_records, record_char_spans, record_hashes
from model_utils import load_causal_lm
from receiver_heads import vertical_scores
from signal_bank import SignalSource
from step_nodes import token_node_ids


def choose_targets(data: dict, top_j: int, d_min: int) -> np.ndarray:
    """Top-|J| nodes by vertical score of the band-averaged teacher routing (answer node excluded)."""
    P = torch.from_numpy(data["P"]).mean(0, keepdim=True)
    nu = vertical_scores(P, torch.from_numpy(data["rows"].astype(bool)), d_min)[0].numpy()
    nu[-1] = np.nan  # nothing reads the answer node back
    order = [j for j in np.argsort(-np.nan_to_num(nu, nan=-np.inf)) if np.isfinite(nu[j])]
    return np.asarray(sorted(order[:top_j]), dtype=np.int64)


@torch.no_grad()
def final_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model.model(input_ids=input_ids).last_hidden_state[0]


@torch.no_grad()
def token_kl(model, clean: torch.Tensor, perturbed: torch.Tensor, positions: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """KL(p_clean || p_perturbed) of the next-token distribution at each position, chunked over vocab-sized logits."""
    head = model.get_output_embeddings()
    out = []
    for start in range(0, positions.numel(), chunk):
        pos = positions[start : start + chunk]
        log_p = F.log_softmax(head(clean[pos].to(head.weight.device)).float(), dim=-1)
        log_q = F.log_softmax(head(perturbed[pos].to(head.weight.device)).float(), dim=-1)
        out.append((log_p.exp() * (log_p - log_q)).sum(-1).to(clean.device))
    return torch.cat(out) if out else clean.new_zeros(0)


def hidden_with_suppression(model, modules, input_ids, start: int, end: int, query_block: int) -> torch.Tensor:
    attention_capture.set_hooks(modules, lambda layer: attention_capture.suppression_hook(start, end, query_block))
    try:
        return final_hidden(model, input_ids)
    finally:
        attention_capture.clear_hooks(modules)


def causal_matrix(model, record: dict, targets: np.ndarray, query_block: int) -> tuple[np.ndarray, np.ndarray]:
    """(C [N, N], floor [|J|]); floor = mean KL before node j, which must be ~0.

    The clean run uses the same explicit-mask path with an empty block, so it differs from the suppressed
    run only by the suppression (the stock SDPA kernel would add a bf16 mismatch KL everywhere).
    """
    input_ids = torch.tensor([record["input_ids"]], device=next(model.parameters()).device)
    spans = [(n["token_start"], n["token_end"]) for n in record["nodes"]]
    num_nodes, length = len(spans), input_ids.size(1)
    modules = attention_capture.attention_modules(model)
    clean = hidden_with_suppression(model, modules, input_ids, 0, 0, query_block)
    device = clean.device  # the last layer's device when the teacher is sharded (device_map)
    node_of = torch.tensor(token_node_ids(spans, length), device=device)

    C = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)
    floor = np.zeros(len(targets), dtype=np.float32)
    for index, j in enumerate(targets):
        start, end = spans[j]
        perturbed = hidden_with_suppression(model, modules, input_ids, start, end, query_block)
        before = torch.arange(max(0, start - 256), start, device=device)
        floor[index] = float(token_kl(model, clean, perturbed, before).mean()) if before.numel() else 0.0
        # token t is predicted at position t - 1; score every token of a later node
        tokens = torch.arange(end, length, device=device)
        tokens = tokens[node_of[tokens] > j]
        kl = token_kl(model, clean, perturbed, tokens - 1)
        rows = node_of[tokens]
        sums = torch.zeros(num_nodes, device=device).index_add_(0, rows, kl)
        counts = torch.bincount(rows, minlength=num_nodes).float()
        column = (sums / counts.clamp_min(1)).cpu().numpy()
        has = (counts > 0).cpu().numpy()
        C[has, j] = column[has]
    return C, floor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True, help="teacher records (data_prep.py)")
    parser.add_argument("--targets-dir", required=True, help="teacher routing targets (extract_routing.py)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.2, help="share of traces with causal targets (D_c)")
    parser.add_argument("--limit", type=int, help="cap on the number of traces (D4 pilot uses 20)")
    parser.add_argument("--top-j", type=int, default=24)
    parser.add_argument("--d-min", type=int, default=4)
    parser.add_argument("--query-block", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device-map", help="'auto' shards a large teacher over the visible GPUs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    records = load_records(args.data_path)
    chosen = sorted(random.Random(args.seed).sample(range(len(records)), max(1, round(args.fraction * len(records)))))
    if args.limit:
        chosen = chosen[: args.limit]
    records = [records[i] for i in chosen][args.shard_index :: args.num_shards]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_causal_lm(args.model_name, attn_implementation=args.attn_implementation, device_map=args.device_map)
    signals = SignalSource(args.targets_dir)
    for record in tqdm(records, desc="causal", unit="trace"):
        path = output / f"{record['id']}.npz"
        if path.exists() or record["id"] not in signals:
            continue
        targets = choose_targets(signals.get(record["id"]), args.top_j, args.d_min)
        C, floor = causal_matrix(model, record, targets, args.query_block)
        if floor.max() > 1e-4:
            print(f"WARNING {record['id']}: KL before the suppressed node is {floor.max():.2e} (should be ~0)")
        np.savez_compressed(path, C=C, J=targets, floor=floor, char_spans=record_char_spans(record),
                            hash=record_hashes(record))
    (output / "causal-config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
