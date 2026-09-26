"""Small training invariants shared by custom selective-loss trainers."""

import torch
from transformers import set_seed


def compensate_global_token_mean(
    loss: torch.Tensor,
    *,
    average_tokens_across_devices: bool,
    num_items_in_batch: torch.Tensor | int | None,
    num_processes: int,
) -> torch.Tensor:
    """Mirror Trainer's DDP compensation for a globally gathered token count."""
    if average_tokens_across_devices and num_items_in_batch is not None:
        return loss * num_processes
    return loss


def set_training_seed(seed: int) -> None:
    """Seed before model/LoRA construction so every experimental arm starts identically."""
    set_seed(seed)
