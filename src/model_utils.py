"""Model loading shared by extraction, causal targets and diagnostics, plus QK-Restore (D6)."""

import json
from pathlib import Path

import torch

QK_MODULES = ("q_proj", "k_proj")


def load_causal_lm(model_name: str, adapter: str | None = None, qk_restore: bool = False,
                   attn_implementation: str = "sdpa", device: str | None = None, device_map: str | None = None):
    """Frozen bf16 (fp32 on CPU) causal LM for teacher-forced reading, adapter merged if given.

    qk_restore zeroes the LoRA update of q_proj/k_proj before merging: the model then routes
    with the *pre-training* W_Q, W_K while keeping every other learned update (Zhou et al., 2026).
    device_map="auto" spreads a teacher too large for one GPU (DeepSeek-R1-Distill-Qwen-32B at 32k
    tokens) over the visible GPUs; the attention hooks follow each layer's device.
    """
    from transformers import AutoModelForCausalLM

    import attention_capture

    attention_capture.install()
    on_gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if on_gpu else torch.float32,
        attn_implementation=attn_implementation,
        device_map=device_map if on_gpu else None,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if qk_restore:
            zero_qk_lora(model)
        model = model.merge_and_unload()
    elif qk_restore:
        raise ValueError("--qk-restore needs an --adapter (for full FT restore W_Q/W_K from the base instead)")
    model.config.use_cache = False
    model.eval().requires_grad_(False)
    if device_map and on_gpu:
        return model
    return model.to(device or ("cuda" if on_gpu else "cpu"))


@torch.no_grad()
def zero_qk_lora(model) -> int:
    """Zero lora_B of every q_proj/k_proj LoRA layer (all adapters); returns how many were zeroed."""
    zeroed = 0
    for name, module in model.named_modules():
        if name.split(".")[-1] in QK_MODULES and hasattr(module, "lora_B"):
            for linear in module.lora_B.values():
                linear.weight.zero_()
                zeroed += 1
    if not zeroed:
        raise ValueError("no q_proj/k_proj LoRA layers found to restore")
    return zeroed


def write_qk_restored_adapter(adapter_dir: str, output_dir: str) -> None:
    """Copy of a PEFT adapter with q/k lora_B zeroed, loadable by vLLM for the QK-Restore eval."""
    from safetensors.torch import load_file, save_file

    src, dst = Path(adapter_dir), Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    tensors = load_file(src / "adapter_model.safetensors")
    zeroed = 0
    for key, value in tensors.items():
        if "lora_B" in key and any(f".{m}." in key for m in QK_MODULES):
            tensors[key] = torch.zeros_like(value)
            zeroed += 1
    if not zeroed:
        raise ValueError(f"{adapter_dir}: no q_proj/k_proj lora_B tensors")
    save_file(tensors, dst / "adapter_model.safetensors")
    for extra in src.iterdir():
        if extra.suffix in (".json", ".txt", ".jinja") or extra.name.startswith("tokenizer"):
            (dst / extra.name).write_bytes(extra.read_bytes())
    (dst / "qk-restore.json").write_text(json.dumps({"source": str(src), "zeroed_tensors": zeroed}, indent=2))


def num_layers_and_heads(model) -> tuple[int, int, int]:
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return config.num_hidden_layers, config.num_attention_heads, head_dim
