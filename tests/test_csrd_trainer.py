"""Attention capture and the CSRD trainer on a tiny random Qwen3 (CPU)."""

import copy
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

import attention_capture
from csrd_data import CSRDCollator, CSRDDataset
from csrd_trainer import key_node_ids, sample_queries
from routing import attention_probs, far_target_mask, valid_rows
from train_sft import CSRDTrainer, attach_lora

LAYERS, HEADS, KV, DIM = 4, 4, 2, 8


def _config():
    return Qwen3Config(vocab_size=64, hidden_size=HEADS * DIM, intermediate_size=64, num_hidden_layers=LAYERS,
                       num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=DIM, max_position_embeddings=512)


def _model(impl="sdpa", seed=0):
    attention_capture.install()
    torch.manual_seed(seed)
    return Qwen3ForCausalLM._from_config(copy.deepcopy(_config()), attn_implementation=impl)


def test_capture_matches_eager_attention():
    sdpa = _model("sdpa").eval()
    eager = _model("eager").eval()
    eager.load_state_dict(sdpa.state_dict())
    ids = torch.randint(0, 64, (1, 40))
    capture = attention_capture.QKCapture(attention_capture.attention_modules(sdpa))
    positions = torch.tensor([2, 17, 39])
    capture.arm({1: [0, 3], 3: [2]}, positions, detach=True)
    logits = sdpa(ids).logits
    capture.disarm()
    reference = eager(ids, output_attentions=True)
    assert torch.allclose(logits, reference.logits, atol=1e-5)
    for layer, heads in {1: [0, 3], 3: [2]}.items():
        for slot, head in enumerate(heads):
            probs = attention_probs(capture.q[layer][slot], capture.k[layer][slot], positions, DIM**-0.5)
            assert torch.allclose(probs, reference.attentions[layer][0, head, positions], atol=1e-5)


def test_suppression_changes_only_later_positions():
    model = _model().eval()
    ids = torch.randint(0, 64, (1, 30))
    clean = model(ids).logits
    modules = attention_capture.attention_modules(model)
    attention_capture.set_hooks(modules, lambda layer: attention_capture.suppression_hook(5, 11, query_block=8))
    suppressed = model(ids).logits
    attention_capture.clear_hooks(modules)
    assert torch.allclose(clean[0, :11], suppressed[0, :11], atol=1e-5)
    assert (clean[0, 11:] - suppressed[0, 11:]).abs().max() > 1e-4


def test_sample_queries_includes_last_token_and_respects_rows():
    spans = torch.tensor([[0, 4], [4, 20], [20, 23], [23, 40]])
    rows = torch.tensor([False, True, True, False])
    positions, owners = sample_queries(spans, rows, 5, torch.Generator().manual_seed(0))
    assert set(owners.tolist()) == {1, 2}
    assert 19 in positions.tolist() and 22 in positions.tolist()
    assert (owners == 1).sum() == 5 and (owners == 2).sum() == 3  # short node: all of its tokens
    assert all(spans[o, 0] <= p < spans[o, 1] for p, o in zip(positions.tolist(), owners.tolist()))
    ids = key_node_ids(spans, 42)
    assert ids[:4].tolist() == [0] * 4 and ids[40:].tolist() == [-1, -1]


@pytest.fixture()
def csrd_data(tmp_path):
    """Two synthetic records + random teacher targets on the student's own node layout."""
    rng = np.random.default_rng(0)
    records, targets = tmp_path / "records.jsonl", tmp_path / "targets"
    targets.mkdir()
    with records.open("w") as handle:
        for index in range(2):
            lengths = [5] + [int(x) for x in rng.integers(3, 6, size=12)] + [3]
            nodes, cursor = [], 3
            for k, length in enumerate(lengths):
                nodes.append({"kind": "step", "char_start": cursor * 10, "char_end": (cursor + length) * 10,
                              "token_start": cursor, "token_end": cursor + length})
                cursor += length
            input_ids = rng.integers(0, 64, size=cursor + 1).tolist()
            handle.write(json.dumps({"id": f"r{index}", "input_ids": input_ids, "response_token_span": [8, cursor],
                                     "nodes": nodes, "anchor": [0, 1] * (len(nodes) // 2)}) + "\n")
            N = len(nodes)
            far = far_target_mask(N, 2).numpy()
            P = rng.random((2, N, N)) * far
            P = P / np.maximum(P.sum(-1, keepdims=True), 1e-12)
            np.savez(targets / f"r{index}.npz", P=P.astype(np.float32), Z=rng.random((2, N)).astype(np.float32),
                     rows=valid_rows(torch.from_numpy(far)).numpy(),
                     char_spans=np.asarray([[n["char_start"], n["char_end"]] for n in nodes]))
    return records, targets


def _trainer(tmp_path, records, targets, qk_rank=0, lam=1.0, checkpointing=False):
    config = {"lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.0,
              "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", "csrd_qk_rank": qk_rank}
    model = attach_lora(_model(seed=1), config)
    args = TrainingArguments(output_dir=str(tmp_path / "out"), per_device_train_batch_size=1, max_steps=10,
                             use_cpu=True, report_to=[], gradient_checkpointing=checkpointing,
                             gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False)
    trainer = CSRDTrainer(model=model, args=args, train_dataset=CSRDDataset(str(records), str(targets)),
                          data_collator=CSRDCollator(pad_token_id=0), csrd_lambda=lam, csrd_d_min=2, csrd_queries=3,
                          csrd_head_mode="band", csrd_warmup_frac=0.0, csrd_ramp_frac=0.0, csrd_grad_log_interval=0,
                          csrd_qk_adapter="qk" if qk_rank else None)
    trainer.state.max_steps = 10
    trainer.current_gradient_accumulation_steps = 1  # set by train() in real runs
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return trainer


def _batch(trainer):
    return trainer.data_collator([trainer.train_dataset[0]])


def _grads(model, keyword):
    return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for n, p in model.named_parameters() if p.requires_grad and keyword in n}


@pytest.mark.parametrize("checkpointing", [False, True])
def test_routing_loss_reaches_query_key_lora(tmp_path, csrd_data, checkpointing):
    records, targets = csrd_data
    trainer = _trainer(tmp_path, records, targets, checkpointing=checkpointing)
    loss = trainer.compute_loss(trainer.model, _batch(trainer))
    loss.backward()
    with_aux = _grads(trainer.model, "q_proj.lora_B")
    trainer.model.zero_grad()
    trainer.csrd_lambda = 0.0
    trainer.compute_loss(trainer.model, _batch(trainer)).backward()
    ce_only = _grads(trainer.model, "q_proj.lora_B")
    band_layers = {f"layers.{l}." for l in (2, 3)}
    assert any((with_aux[n] - ce_only[n]).abs().max() > 1e-7 for n in with_aux if any(b in n for b in band_layers))
    assert all(torch.allclose(with_aux[n], ce_only[n], atol=1e-9) for n in with_aux if "layers.3." not in n
               and "layers.2." not in n and "layers.1." not in n and "layers.0." not in n)
    assert "loss_route" in trainer._metric_sums and trainer._metric_sums["loss_mass"] >= 0


def test_csrd_qk_isolates_gradients(tmp_path, csrd_data):
    """Appendix D: the Q/K adapter gets only routing gradients, the main adapter only CE gradients."""
    records, targets = csrd_data
    runs = {}
    for lam in (0.0, 1.0):
        trainer = _trainer(tmp_path, records, targets, qk_rank=2, lam=lam)
        trainer.training_step(trainer.model, _batch(trainer))
        runs[lam] = (_grads(trainer.model, ".qk."), _grads(trainer.model, ".default."))
    qk_ce_only, main_ce_only = runs[0.0]
    qk_route, main_route = runs[1.0]
    assert all(g.abs().max() == 0 for g in qk_ce_only.values())  # CE never trains A_QK
    assert any(g.abs().max() > 0 for g in qk_route.values())  # routing loss does
    for name in main_ce_only:  # main adapter identical with or without the routing loss
        assert torch.allclose(main_ce_only[name], main_route[name], atol=1e-7), name
