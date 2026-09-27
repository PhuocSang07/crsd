"""Masked next-token cross-entropy, full-logit and chunked (memory-bounded) forms.

    L_CE = -(1/Z) * sum_t M_t * log P(y_t | y_<t),   Z = supervised tokens of the whole optimizer step

At 32k tokens the logits alone are ~19 GB fp32 (Sec. 4.8), so training uses `chunked_cross_entropy`:
the LM head runs on `chunk` positions at a time under activation checkpointing (one chunk of logits alive).
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    denominator: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Reference form on full (B, T, V) logits; labels unshifted with IGNORE_INDEX where masked."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    total = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )
    if denominator is None:
        denominator = (shift_labels != IGNORE_INDEX).sum()
    return total / torch.as_tensor(denominator, device=total.device).float().clamp_min(1.0)


def _chunk_loss(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight, bias).float()
    return F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")


def chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    denominator: torch.Tensor | int | None = None,
    chunk: int = 4096,
) -> torch.Tensor:
    """Same value and gradient as masked_cross_entropy(lm_head(hidden), labels), chunk by chunk.

    hidden: (B, T, D) final hidden states (after the last norm); only supervised positions are projected.
    """
    shift_hidden = hidden[:, :-1].reshape(-1, hidden.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    keep = torch.nonzero(shift_labels != IGNORE_INDEX, as_tuple=False).squeeze(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    weight, bias = lm_head.weight, getattr(lm_head, "bias", None)
    for start in range(0, keep.numel(), chunk):
        index = keep[start : start + chunk]
        h, y = shift_hidden.index_select(0, index), shift_labels.index_select(0, index)
        if torch.is_grad_enabled():
            total = total + checkpoint(_chunk_loss, h, weight, bias, y, use_reentrant=False)
        else:
            total = total + _chunk_loss(h, weight, bias, y)
    if keep.numel() == 0:
        total = total + hidden.sum() * 0.0  # keep the graph connected for DDP/ZeRO
    if denominator is None:
        denominator = keep.numel()
    return total / torch.as_tensor(denominator, device=total.device).float().clamp_min(1.0)
