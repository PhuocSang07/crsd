"""Access to the exact q/k a HF decoder attends with, without giving up SDPA/FlashAttention.

`install()` wraps the sdpa/flash-attention entries of transformers' AttentionInterface; for a module
carrying a hook in CALLBACK_ATTR the wrapper first calls `hook(module, query, key, value, attention_mask, **kwargs)`:
    query [B, H, T, d]   post q_norm + RoPE (what the softmax actually sees)
    key   [B, KVH, T, d] post k_norm + RoPE, before GQA repetition
Returning None runs the normal kernel (capture); returning (attn_output, None) replaces it (suppression).
Registry names are unchanged, so the mask builders keyed on them keep working.
"""

import sys

import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

CALLBACK_ATTR = "_csrd_attention_hook"
WRAPPED_NAMES = ("sdpa", "flash_attention_2", "flash_attention_3")
_WRAPPED_FLAG = "_csrd_wrapped"


def _wrap(base):
    def forward(module, query, key, value, attention_mask, *args, **kwargs):
        hook = getattr(module, CALLBACK_ATTR, None)
        if hook is not None:
            replaced = hook(module, query, key, value, attention_mask, **kwargs)
            if replaced is not None:
                return replaced
        return base(module, query, key, value, attention_mask, *args, **kwargs)

    setattr(forward, _WRAPPED_FLAG, True)
    return forward


def install() -> None:
    """Idempotently wrap the registered attention kernels (process-wide)."""
    for name in WRAPPED_NAMES:
        if name not in ALL_ATTENTION_FUNCTIONS:
            continue
        base = ALL_ATTENTION_FUNCTIONS[name]
        if not getattr(base, _WRAPPED_FLAG, False):
            AttentionInterface.register(name, _wrap(base))


def attention_modules(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """{layer_idx: self-attention module} of a (possibly PEFT/DeepSpeed-wrapped) causal LM."""
    found = {}
    for _, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "layer_idx"):
            found[int(module.layer_idx)] = module
    if not found:
        raise ValueError("no attention module with q_proj/k_proj/layer_idx found")
    return dict(sorted(found.items()))


def set_hooks(modules: dict[int, torch.nn.Module], hook_for_layer) -> None:
    """hook_for_layer(layer_idx) -> callable or None."""
    for layer, module in modules.items():
        setattr(module, CALLBACK_ATTR, hook_for_layer(layer))


def clear_hooks(modules: dict[int, torch.nn.Module]) -> None:
    for module in modules.values():
        if hasattr(module, CALLBACK_ATTR):
            delattr(module, CALLBACK_ATTR)


def check_attn_implementation(model) -> None:
    impl = getattr(model.config, "_attn_implementation", None)
    if impl not in WRAPPED_NAMES:
        raise ValueError(
            f"attn_implementation={impl!r} cannot be hooked; use one of {WRAPPED_NAMES} "
            "(eager has no registry entry to wrap)"
        )


class QKCapture:
    """Armed for one forward: keeps q and k of the wanted layers; disarm() slices them into
    `q[layer]` [Hsel, Nq, d] (selected heads at `query_positions`, sample `batch_index`) and
    `k[layer]` [Hsel, T, d] (matching KV head), with their graph unless `detach=True`.

    The hook only keeps references and slicing waits for disarm(), outside the checkpointed region:
    a tensor op inside a layer under non-reentrant gradient checkpointing would be saved in the
    forward but not in the (disarmed) recomputation, which checkpointing rejects. Each layer disarms
    after its first capture, so the backward re-forward neither overwrites nor duplicates anything.
    """

    def __init__(self, modules: dict[int, torch.nn.Module]):
        self.modules = modules
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self._raw: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending: set[int] = set()
        self._spec = None

    def arm(self, wanted: dict[int, list[int]], query_positions: torch.Tensor, detach: bool, batch_index: int = 0):
        self.q, self.k, self._raw = {}, {}, {}
        self._pending = {layer for layer, heads in wanted.items() if heads}
        self._spec = (wanted, query_positions, detach, batch_index)

        def make(layer):
            if layer not in self._pending:
                return None

            def hook(module, query, key, value, attention_mask, **kwargs):
                if layer in self._pending:
                    self._raw[layer] = (query, key)
                    self._pending.discard(layer)
                return None

            return hook

        set_hooks(self.modules, make)

    def disarm(self):
        self._pending = set()
        clear_hooks(self.modules)
        if self._spec is None:
            return
        wanted, positions, detach, batch_index = self._spec
        self._spec = None
        for layer, (query, key) in self._raw.items():
            groups = query.size(1) // key.size(1)
            head_idx = torch.tensor(wanted[layer], device=query.device)
            q = query[batch_index].index_select(0, head_idx).index_select(1, positions.to(query.device))
            k = key[batch_index].index_select(0, head_idx // groups)
            self.q[layer], self.k[layer] = (q.detach(), k.detach()) if detach else (q, k)
        self._raw = {}


def suppression_hook(blocked_start: int, blocked_end: int, query_block: int = 2048):
    """Attention suppression of one node (Sec. 4.4): queries at >= blocked_end see no key in
    [blocked_start, blocked_end). Masked SDPA per query block; returns [B, T, H, d] like the kernels."""

    def hook(module, query, key, value, attention_mask, **kwargs):
        batch, heads, length, dim = query.shape
        groups = heads // key.size(1)
        key_r = key.repeat_interleave(groups, dim=1)
        value_r = value.repeat_interleave(groups, dim=1)
        scale = kwargs.get("scaling") or getattr(module, "scaling", dim**-0.5)
        key_pos = torch.arange(length, device=query.device)
        blocked_key = (key_pos >= blocked_start) & (key_pos < blocked_end)
        outputs = []
        for start in range(0, length, query_block):
            q_pos = torch.arange(start, min(start + query_block, length), device=query.device)
            allowed = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            allowed &= ~(blocked_key.unsqueeze(0) & (q_pos.unsqueeze(1) >= blocked_end))
            outputs.append(
                F.scaled_dot_product_attention(
                    query[:, :, start : start + q_pos.numel()], key_r, value_r, attn_mask=allowed, scale=scale
                )
            )
        return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None

    return hook


def recompute_qk(module: torch.nn.Module, hidden: torch.Tensor, position_embeddings) -> tuple[torch.Tensor, torch.Tensor]:
    """q [B, H, T, d], k [B, KVH, T, d] (post-norm, post-RoPE) of one Qwen3/Llama-style attention
    module from its (possibly detached) input; CSRD-QK detaches it so no gradient reaches lower layers."""
    rope = getattr(sys.modules[type(module).__module__], "apply_rotary_pos_emb")
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    q = module.q_proj(hidden).view(shape)
    k = module.k_proj(hidden).view(shape)
    if hasattr(module, "q_norm"):
        q, k = module.q_norm(q), module.k_norm(k)
    cos, sin = position_embeddings
    return rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)


class AttentionInputCapture:
    """Pre-hooks storing detached (hidden_states, position_embeddings) of armed layers for one forward (CSRD-QK)."""

    def __init__(self, modules: dict[int, torch.nn.Module]):
        self.modules = modules
        self.inputs: dict[int, tuple] = {}
        self._pending: set[int] = set()
        self._handles = [
            module.register_forward_pre_hook(self._make(layer), with_kwargs=True) for layer, module in modules.items()
        ]

    def _make(self, layer):
        def pre_hook(module, args, kwargs):
            if layer in self._pending:
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                self.inputs[layer] = (hidden.detach(), kwargs.get("position_embeddings"))
                self._pending.discard(layer)

        return pre_hook

    def arm(self, layers):
        self.inputs = {}
        self._pending = set(layers)

    def disarm(self):
        self._pending = set()
