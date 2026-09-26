"""Training dataset and collator: SFT fields plus, per sample, the CSRD routing targets.

A record (data_prep.py --style sgl, student tokenizer) supplies input_ids, the response span and the
node token spans; the teacher signals come from a signal bank or a targets dir (signal_bank.SignalSource),
optionally with a causal dir. The node hashes stored with the signals must equal the record's -- the
check that teacher rows and student rows are the same steps, whatever tokenizer each side used.

The SFT arm uses the same dataset without targets, so both arms read identical samples.
"""

from dataclasses import dataclass

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from masked_loss import IGNORE_INDEX
from signal_bank import SignalSource

CSRD_KEY = "csrd"


class CSRDDataset(Dataset):
    def __init__(self, path: str, signals: str | None = None, causal_dir: str | None = None,
                 max_seq_len: int | None = None):
        with open(path) as handle:
            self.records = [json.loads(line) for line in handle]
        if max_seq_len is not None:
            before = len(self.records)
            self.records = [r for r in self.records if len(r["input_ids"]) <= max_seq_len]
            if len(self.records) < before:
                print(f"--max-seq-len {max_seq_len}: dropped {before - len(self.records)}/{before} samples")
        self.signals = SignalSource(signals, causal_dir) if signals else None
        if self.signals is not None:
            # Fail instead of dropping: the SFT arm would keep those records, and the arms must see the same
            # samples (Sec. 6.3 "cùng số token đã thấy"). Rerun extract_routing.py --stage targets (it resumes).
            available = set(self.signals.ids())
            missing = [r["id"] for r in self.records if r["id"] not in available]
            if missing:
                raise FileNotFoundError(f"no teacher signal in {signals} for {len(missing)} records (e.g. {missing[:3]})")

    def __len__(self) -> int:
        return len(self.records)

    def supervised_token_count(self) -> int:
        # response tokens + the stop token after them
        return sum(r["response_token_span"][1] - r["response_token_span"][0] + 1 for r in self.records)

    def causal_count(self) -> int:
        if self.signals is None:
            return 0
        return sum(self.signals.has_causal(r["id"]) for r in self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        start, end = record["response_token_span"]
        loss_mask = [0] * start + [1] * (end - start) + [1] * (len(record["input_ids"]) - end)
        item = {"input_ids": record["input_ids"], "loss_mask": loss_mask}
        if self.signals is None:
            return item

        target = self.signals.get(record["id"])
        hashes = np.asarray([n["hash"] for n in record["nodes"]], dtype=np.int64)
        if not np.array_equal(target["hash"], hashes):
            raise ValueError(f"{record['id']}: teacher signal nodes differ from the student record's nodes "
                             "(same canonical content and segmentation on both sides?)")
        csrd = {
            "node_spans": torch.tensor([[n["token_start"], n["token_end"]] for n in record["nodes"]], dtype=torch.long),
            "P": torch.from_numpy(np.asarray(target["P"], dtype=np.float32)),
            "Z": torch.from_numpy(np.asarray(target["Z"], dtype=np.float32)),
            "rows": torch.from_numpy(np.asarray(target["rows"]).astype(bool)),
            "anchor": torch.tensor(record.get("anchor") or [0] * len(record["nodes"]), dtype=torch.float32),
        }
        if "C" in target:
            csrd["C"] = torch.from_numpy(np.asarray(target["C"], dtype=np.float32))
            csrd["J"] = torch.from_numpy(np.asarray(target["J"], dtype=np.int64))
        item[CSRD_KEY] = csrd
        return item


@dataclass
class CSRDCollator:
    """Pads input_ids / labels; CSRD targets stay a per-sample list (node counts differ)."""

    pad_token_id: int

    def __call__(self, examples: list[dict]) -> dict:
        max_length = max(len(e["input_ids"]) for e in examples)
        input_ids, attention_mask, labels = [], [], []
        for example in examples:
            ids, mask = list(example["input_ids"]), list(example["loss_mask"])
            if len(ids) != len(mask):
                raise ValueError(f"loss_mask length {len(mask)} != input_ids length {len(ids)}")
            padding = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append([t if m else IGNORE_INDEX for t, m in zip(ids, mask)] + [IGNORE_INDEX] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if any(CSRD_KEY in e for e in examples):
            if not all(CSRD_KEY in e for e in examples):
                raise ValueError("a batch must carry routing targets for every sample or for none")
            batch[CSRD_KEY] = [e[CSRD_KEY] for e in examples]
        return batch
