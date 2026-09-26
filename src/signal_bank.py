"""Reusable teacher signal bank: one .safetensors file per (teacher, dataset, segmentation, d_min, heads).

    python src/signal_bank.py pack --targets-dir T [--causal-dir C] --output signals/<name>.safetensors
    python src/signal_bank.py info signals/<name>.safetensors
    python src/signal_bank.py heads signals/<name>.safetensors <heads.json>   # the teacher head selection

Signals are keyed by trace id and by node *text hashes*, not by token positions, so a bank written once
by an expensive teacher (e.g. DeepSeek-R1-Distill-Qwen-32B) serves any later student, whatever its
tokenizer or chat template: the student's own records (data_prep.py --style sgl) only have to carry
the same node hashes. Per trace:

    <id>/P      float32 [bands, N, N]  band-averaged far routing (rows over F(i), lower triangular)
    <id>/Z      float32 [bands, N]     far mass
    <id>/rows   uint8   [N]            rows with |F(i)| >= 2
    <id>/hash   int64   [N]            node text hashes (prompting.text_hash)
    <id>/C, <id>/J, <id>/floor         causal targets, when the trace is in the causal subset
The header metadata holds the teacher, source, style, segmentation, d_min, receiver score and head list.

`SignalSource` reads either a packed bank or the per-trace .npz directories the extraction writes
(resumable, shardable), with the same interface.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

PER_TRACE = ("P", "Z", "rows", "hash")
CAUSAL = ("C", "J", "floor")
FORMAT = "crsd-signal-bank/1"


class SignalSource:
    """ids() / get(id) over a packed .safetensors bank or a targets dir (+ optional causal dir)."""

    def __init__(self, path: str, causal_dir: str | None = None):
        self.path = Path(path)
        self.causal_dir = Path(causal_dir) if causal_dir else None
        self._handle = None
        if self.path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(self.path), framework="np") as handle:
                meta = handle.metadata() or {}
                keys = list(handle.keys())
            if meta.get("format") != FORMAT:
                raise ValueError(f"{self.path} is not a CSRD signal bank")
            self.info = json.loads(meta.get("info", "{}"))
            self._ids = sorted({key.rsplit("/", 1)[0] for key in keys})
            self._keys = set(keys)
        else:
            config = self.path / "targets-config.json"
            self.info = json.loads(config.read_text()) if config.exists() else {}
            self._ids = sorted(p.stem for p in self.path.glob("*.npz"))
            self._keys = None

    def ids(self) -> list[str]:
        return self._ids

    def __contains__(self, trace_id: str) -> bool:
        if not hasattr(self, "_id_set"):
            self._id_set = set(self._ids)
        return trace_id in self._id_set

    def has_causal(self, trace_id: str) -> bool:
        if self._keys is not None:
            return f"{trace_id}/C" in self._keys
        return self.causal_dir is not None and (self.causal_dir / f"{trace_id}.npz").exists()

    def get(self, trace_id: str) -> dict[str, np.ndarray]:
        if self._keys is not None:
            if self._handle is None:  # opened lazily: one handle per process (DataLoader workers)
                from safetensors import safe_open

                self._handle = safe_open(str(self.path), framework="np")
            return {name: self._handle.get_tensor(f"{trace_id}/{name}")
                    for name in PER_TRACE + CAUSAL if f"{trace_id}/{name}" in self._keys}
        data = dict(np.load(self.path / f"{trace_id}.npz"))
        if self.causal_dir is not None and (self.causal_dir / f"{trace_id}.npz").exists():
            causal = np.load(self.causal_dir / f"{trace_id}.npz")
            if "hash" in causal and not np.array_equal(causal["hash"], data["hash"]):
                raise ValueError(f"{trace_id}: causal and routing targets have different nodes")
            data.update({name: causal[name] for name in CAUSAL if name in causal})
        return data


def pack(targets_dir: str, causal_dir: str | None, output: str) -> None:
    from safetensors.numpy import save_file

    source = SignalSource(targets_dir, causal_dir)
    tensors, n_causal = {}, 0
    for trace_id in source.ids():
        data = source.get(trace_id)
        tensors[f"{trace_id}/P"] = data["P"].astype(np.float32)
        tensors[f"{trace_id}/Z"] = data["Z"].astype(np.float32)
        tensors[f"{trace_id}/rows"] = data["rows"].astype(np.uint8)
        tensors[f"{trace_id}/hash"] = data["hash"].astype(np.int64)
        if "C" in data:
            n_causal += 1
            tensors[f"{trace_id}/C"] = data["C"].astype(np.float32)
            tensors[f"{trace_id}/J"] = data["J"].astype(np.int64)
            if "floor" in data:
                tensors[f"{trace_id}/floor"] = data["floor"].astype(np.float32)
    info = {**source.info, "traces": len(source.ids()), "causal_traces": n_causal,
            "packed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "targets_dir": str(targets_dir),
            "causal_dir": str(causal_dir) if causal_dir else None}
    heads_json = source.info.get("heads_json")
    if heads_json and Path(heads_json).exists():
        info["heads"] = json.loads(Path(heads_json).read_text())["heads"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata={"format": FORMAT, "info": json.dumps(info)})
    size = Path(output).stat().st_size / 2**20
    print(f"packed {len(source.ids())} traces ({n_causal} with causal targets), {size:.1f} MB -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--targets-dir", required=True)
    p_pack.add_argument("--causal-dir")
    p_pack.add_argument("--output", required=True)
    p_info = sub.add_parser("info")
    p_info.add_argument("bank")
    p_heads = sub.add_parser("heads", help="write the bank's teacher head selection as a heads.json")
    p_heads.add_argument("bank")
    p_heads.add_argument("output")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.targets_dir, args.causal_dir, args.output)
    elif args.command == "heads":
        info = SignalSource(args.bank).info
        if "heads" not in info:
            raise SystemExit(f"{args.bank} carries no head list")
        payload = {key: info.get(key) for key in ("score", "num_layers", "num_heads", "bands", "band_layers")}
        payload.update(heads=info["heads"], k_per_band=max(len(band) for band in info["heads"]))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"heads -> {args.output}")
    else:
        source = SignalSource(args.bank)
        info = {k: v for k, v in source.info.items() if k != "heads"}
        print(json.dumps({**info, "ids": len(source.ids())}, indent=2))


if __name__ == "__main__":
    main()
