"""LoRA SFT with the optional CSRD routing losses (proposal Sec. 4.5-4.7, Table 3).

Every arm (SFT baseline B1 and all CSRD variants) uses this entry point with the same data, LoRA,
schedule and seed; only the --csrd-* flags differ. --csrd-lambda 0 (default) is plain SFT.

    L = L_CE + lambda_r * (L_route + (lambda_m/lambda_r) L_mass + (lambda_c/lambda_r) L_causal)

CE is chunked over the final hidden states (masked_loss.chunked_cross_entropy).
Settings come from CLI flags or --config (yaml).
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import attention_capture
from csrd_data import CSRD_KEY, CSRDCollator, CSRDDataset
from csrd_trainer import HEADS_FILE, CSRDLossMixin
from masked_loss import chunked_cross_entropy
from receiver_heads import band_layers
from training_utils import compensate_global_token_mean, set_training_seed

ALL_LINEAR = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
QK_ADAPTER = "qk"


class FinalHiddenCapture:
    """Forward hook on the final norm: the (B, T, D) hidden states the LM head would read."""

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        norm = getattr(getattr(base, "model", None), "norm", None)
        if norm is None:
            raise ValueError("could not locate <model>.model.norm")
        self._hidden = None
        self._armed = False
        norm.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self._armed:
            self._hidden, self._armed = output, False

    def arm(self):
        self._hidden, self._armed = None, True

    def take(self) -> torch.Tensor:
        if self._hidden is None:
            raise RuntimeError("final hidden state not captured")
        hidden, self._hidden = self._hidden, None
        return hidden


class CESFTTrainer(Trainer):
    """Masked CE over the response (+ stop token), normalized by the step's supervised-token count.

    `model_accepts_loss_kwargs=True` makes Trainer pass the token count of the whole accumulation
    step as num_items_in_batch, used as Z.
    """

    def __init__(self, *args, ce_chunk: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True
        self.ce_chunk = ce_chunk
        self._final = FinalHiddenCapture(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop(CSRD_KEY, None)
        self._final.arm()
        outputs = model(**inputs, logits_to_keep=1)
        hidden = self._final.take()
        loss = chunked_cross_entropy(hidden, self.model.get_output_embeddings(), labels,
                                     denominator=num_items_in_batch, chunk=self.ce_chunk)
        loss = compensate_global_token_mean(
            loss,
            average_tokens_across_devices=self.args.average_tokens_across_devices,
            num_items_in_batch=num_items_in_batch,
            num_processes=self.accelerator.num_processes,
        )
        return (loss, outputs) if return_outputs else loss


class CSRDTrainer(CSRDLossMixin, CESFTTrainer):
    """CESFTTrainer + the routing losses."""


def build_training_arguments(config: dict) -> TrainingArguments:
    on_gpu = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        # min_lr_rate, not min_lr: min_lr reads optimizer.defaults["lr"], which DeepSpeed ZeRO doesn't expose.
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": config["min_learning_rate"] / config["learning_rate"]},
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config.get("weight_decay", 0.0),
        bf16=on_gpu,
        use_cpu=not on_gpu,
        gradient_checkpointing=config.get("gradient_checkpointing", on_gpu),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.get("logging_steps", 5),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 6),
        report_to=[],
        seed=config.get("seed", 42),
        remove_unused_columns=False,
        deepspeed=config.get("deepspeed_config"),
        ddp_find_unused_parameters=False,
    )


def attach_lora(model, config: dict):
    model = get_peft_model(model, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"].split(","), bias="none", task_type="CAUSAL_LM",
    ))
    if config.get("csrd_qk_rank"):
        # CSRD-QK: a second adapter on q_proj/k_proj of the band layers, trained by the routing loss.
        layers = sorted({l for band in band_layers(model.config.num_hidden_layers) for l in band})
        model.add_adapter(QK_ADAPTER, LoraConfig(
            r=config["csrd_qk_rank"], lora_alpha=config["csrd_qk_rank"], lora_dropout=0.0,
            target_modules=["q_proj", "k_proj"], layers_to_transform=layers, bias="none", task_type="CAUSAL_LM",
        ))
        model.base_model.set_adapter(["default", QK_ADAPTER])
        for name, param in model.named_parameters():  # set_adapter may freeze non-first adapters
            if "lora_" in name:
                param.requires_grad_(True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true", help="8 samples, 1 epoch")
    parser.add_argument("--model-name")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float, help="cosine floor (SGL: 1e-5)")
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--attn-implementation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--metrics-log", help="write trainer.state.log_history + config to this JSON")
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--ce-chunk", type=int, help="positions per LM-head chunk in the CE (default 4096)")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"])
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    # LoRA (defaults match the SpectralGuidedLearning baselines)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules")
    # CSRD
    parser.add_argument("--csrd-lambda", type=float, help="lambda_r (0/unset = plain SFT)")
    # not "--signals": under torchrun that is an ambiguous prefix of its own --signals-to-handle / --signals_to_handle
    parser.add_argument("--signal-bank", "--signals", dest="signals", help="teacher signals: a packed bank (.safetensors, "
                        "signal_bank.py) or a targets dir (extract_routing.py --stage targets)")
    parser.add_argument("--causal-dir", help="causal targets dir (causal_targets.py), when --signal-bank is a targets dir")
    parser.add_argument("--csrd-mass-ratio", type=float, help="lambda_m / lambda_r (default 0.1; 0 = A7)")
    parser.add_argument("--csrd-causal-ratio", type=float, help="lambda_c / lambda_r (default 1.0; 0 = ignore the causal "
                        "targets a signal bank carries -- every arm except CSRD-C)")
    parser.add_argument("--csrd-route-ratio", type=float, help="L_route weight / lambda_r (default 1.0; 0 = A3 causal only)")
    parser.add_argument("--csrd-bands", help="bands in the loss: '0,1' (default), '0' = B1, '1' = B2 (A4)")
    parser.add_argument("--csrd-d-min", type=int, help="far threshold in steps (default 4; must match targets)")
    parser.add_argument("--csrd-queries", type=int, help="m queries per row (default 8; <=0 = all)")
    parser.add_argument("--csrd-k-student", type=int, help="K_S heads per band (default 16)")
    parser.add_argument("--csrd-head-mode", choices=("receiver", "band", "fixed"))
    parser.add_argument("--csrd-student-heads", help="heads.json for --csrd-head-mode fixed")
    parser.add_argument("--csrd-score", choices=("excess_bg", "kurtosis"))
    parser.add_argument("--csrd-anchor-beta", type=float, help="CSRD-A: row weight 1 + beta * anchor")
    parser.add_argument("--csrd-loss-form", choices=("pooled", "per_query"), help="per_query = CSRD-PQ")
    parser.add_argument("--csrd-warmup-frac", type=float)
    parser.add_argument("--csrd-ramp-frac", type=float)
    parser.add_argument("--csrd-qk-rank", type=int, help="CSRD-QK: rank of the separate Q/K adapter (0 = off)")
    parser.add_argument("--csrd-grad-log-interval", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    config.update({key: value for key, value in vars(args).items()
                   if value is not None and key not in ("config", "smoke")})
    defaults = {
        "epochs": 3, "learning_rate": 5e-5, "min_learning_rate": 1e-5, "warmup_ratio": 0.1, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 32, "attn_implementation": "sdpa", "seed": 42, "ce_chunk": 4096,
        "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05, "lora_target_modules": ALL_LINEAR,
        "csrd_lambda": 0.0, "csrd_mass_ratio": 0.1, "csrd_causal_ratio": 1.0, "csrd_route_ratio": 1.0,
        "csrd_bands": "0,1", "csrd_d_min": 4,
        "csrd_queries": 8, "csrd_k_student": 16, "csrd_head_mode": "receiver", "csrd_score": "excess_bg",
        "csrd_anchor_beta": 0.0, "csrd_loss_form": "pooled", "csrd_warmup_frac": 0.1, "csrd_ramp_frac": 0.1,
        "csrd_qk_rank": 0, "csrd_grad_log_interval": 50,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    missing = [key for key in ("model_name", "data_path", "output_dir") if key not in config]
    if missing:
        parser.error(f"missing required settings: {missing}")
    use_csrd = config["csrd_lambda"] > 0
    if use_csrd and not config.get("signals"):
        parser.error("--csrd-lambda > 0 needs --signal-bank")
    if use_csrd:
        from signal_bank import SignalSource

        info = SignalSource(config["signals"]).info
        if info.get("d_min") is not None and info["d_min"] != config["csrd_d_min"]:
            parser.error(f"--csrd-d-min {config['csrd_d_min']} differs from the signals' d_min ({info['d_min']})")
        print(f"teacher signals: {info.get('model')} score={info.get('score')} style={info.get('style')} "
              f"source={info.get('source')} d_min={info.get('d_min')}")

    set_training_seed(config["seed"])
    attention_capture.install()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = CSRDDataset(
        config["data_path"],
        signals=config.get("signals") if use_csrd else None,
        causal_dir=config.get("causal_dir") if use_csrd else None,
        max_seq_len=config.get("max_seq_len"),
    )
    if args.smoke:
        dataset.records = dataset.records[:8]
        config["epochs"] = 1
    print(f"{len(dataset)} samples, {dataset.supervised_token_count():,} supervised tokens"
          + (f", causal targets on {dataset.causal_count()}" if use_csrd else ""))

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=config["attn_implementation"],
    )
    model.config.use_cache = False
    model = attach_lora(model, config)

    trainer_kwargs = {"ce_chunk": config["ce_chunk"]}
    trainer_cls = CESFTTrainer
    if use_csrd:
        heads_path = config.get("csrd_student_heads")
        resumed_heads = Path(config["output_dir"]) / HEADS_FILE
        if not heads_path and config.get("resume") and resumed_heads.exists():
            heads_path = str(resumed_heads)  # keep H_S fixed across a restart
        trainer_cls = CSRDTrainer
        trainer_kwargs.update({
            key: config[key] for key in (
                "csrd_lambda", "csrd_mass_ratio", "csrd_causal_ratio", "csrd_route_ratio", "csrd_d_min", "csrd_queries",
                "csrd_k_student", "csrd_head_mode", "csrd_score", "csrd_anchor_beta", "csrd_loss_form",
                "csrd_warmup_frac", "csrd_ramp_frac", "csrd_grad_log_interval",
            )
        })
        trainer_kwargs["csrd_student_heads"] = json.loads(Path(heads_path).read_text()) if heads_path else None
        trainer_kwargs["csrd_bands"] = tuple(int(b) for b in str(config["csrd_bands"]).split(","))
        trainer_kwargs["csrd_qk_adapter"] = QK_ADAPTER if config["csrd_qk_rank"] else None
        print("CSRD: " + " ".join(f"{k}={v}" for k, v in trainer_kwargs.items() if k != "csrd_student_heads"))

    trainer = trainer_cls(
        model=model,
        args=build_training_arguments(config),
        train_dataset=dataset,
        data_collator=CSRDCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
        processing_class=tokenizer,
        **trainer_kwargs,
    )

    resume_from = None
    if config.get("resume"):
        from transformers.trainer_utils import get_last_checkpoint

        out_dir = config["output_dir"]
        resume_from = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
        print(f"--resume: {'continuing from ' + resume_from if resume_from else 'no checkpoint, starting fresh'}")
    result = trainer.train(resume_from_checkpoint=resume_from)

    if not config["csrd_qk_rank"]:
        trainer.save_model(config["output_dir"])  # collective-safe: Trainer writes on the main process only
    # Rank 0 only below: every torchrun rank writing the same files concurrently can corrupt them.
    if not trainer.is_world_process_zero():
        return
    if config.get("metrics_log"):
        path = Path(config["metrics_log"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": config, "final_metrics": result.metrics,
                                    "log_history": trainer.state.log_history}, indent=2, default=str))

    # CSRD-QK: both adapters under adapters-separate/ plus a fully merged checkpoint for vLLM.
    if config["csrd_qk_rank"]:
        trainer.model.save_pretrained(os.path.join(config["output_dir"], "adapters-separate"))
        trainer.model.base_model.set_adapter(["default", QK_ADAPTER])
        trainer.model.merge_and_unload().save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    if use_csrd and trainer.student_heads is not None:
        trainer.save_student_heads(os.path.join(config["output_dir"], HEADS_FILE))
    (Path(config["output_dir"]) / "run-summary.json").write_text(json.dumps({
        "data_path": config["data_path"], "samples": len(dataset),
        "supervised_tokens": dataset.supervised_token_count(), "epochs": config["epochs"],
        "seed": config["seed"], "learning_rate": config["learning_rate"],
        "csrd_lambda": config["csrd_lambda"], "csrd_qk_rank": config["csrd_qk_rank"],
        "train_runtime_s": result.metrics.get("train_runtime"), "final_train_loss": result.metrics.get("train_loss"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        # Sec. 6.7 efficiency (rank 0)
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
    }, indent=2))


if __name__ == "__main__":
    main()
