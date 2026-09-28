"""MC-CSRD (proposal v4): the probability identities of Sec. 4-5, the raw-mass cache and the trainer objectives (CPU)."""

import copy
import itertools
import json

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM, TrainingArguments

import attention_capture
from csrd_data import CSRDCollator, CSRDDataset
from routing import (
    attention_probs,
    bernoulli_kl,
    far_target_mask,
    head_query_far_mass,
    head_query_routing,
    head_row_routing_blockwise,
    hierarchical_mc_kl,
    kl_rows,
    mass_gap_by_bin,
    mass_groups,
    mc_kl_rows,
    mc_loss,
    node_mass,
    query_weights,
    row_average,
    synthetic_far,
    valid_rows,
    weighted_row_sum,
)
from train_sft import CSRDTrainer, attach_lora


def _far_pair(num_nodes=80, d_min=4, seed=0, dtype=torch.float64, grad=False):
    """Teacher/student far masses over F(i) from softmaxes over [REST, nodes] (the non-far node mass joins REST)."""
    gen = torch.Generator().manual_seed(seed)
    far = far_target_mask(num_nodes, d_min)
    theta_T = torch.randn(num_nodes, num_nodes + 1, generator=gen, dtype=dtype)
    theta_S = torch.randn(num_nodes, num_nodes + 1, generator=gen, dtype=dtype, requires_grad=grad)
    far_T = torch.softmax(theta_T, -1)[:, 1:] * far
    far_S = torch.softmax(theta_S, -1)[:, 1:] * far
    return far_T, far_S, theta_S, far


def test_binary_chain_rule():
    far_T, far_S, _, far = _far_pair()
    rows = valid_rows(far)
    z_T, z_S = far_T.sum(-1), far_S.sum(-1)
    expected = bernoulli_kl(z_T, z_S) + z_T * kl_rows(far_T / z_T.clamp_min(1e-300).unsqueeze(-1),
                                                     far_S / z_S.clamp_min(1e-300).unsqueeze(-1))
    assert torch.allclose(mc_kl_rows(far_T, far_S)[rows], expected[rows], atol=1e-12)


def test_hierarchical_equals_flat_loss_and_gradient():
    far_T, far_S, theta, far = _far_pair(grad=True)
    names, groups = mass_groups(far)
    assert names == ["q", "[4,8)", "[8,16)", "[16,32)", "[32,64)", "[64,inf)"]
    assert torch.equal(groups.sum(0), far.long())  # a partition of every F(i)
    rows = valid_rows(far)
    flat = mc_kl_rows(far_T, far_S)[rows].sum()
    top, within = hierarchical_mc_kl(far_T, far_S, groups)
    hier = (top + within.sum(0))[rows].sum()
    (g_flat,) = torch.autograd.grad(flat, theta, retain_graph=True)
    (g_hier,) = torch.autograd.grad(hier, theta)
    assert abs(float(flat - hier)) < 1e-10
    assert torch.allclose(g_flat, g_hier, atol=1e-12)


def test_mass_groups_residual_when_d_min_below_first_bin():
    names, groups = mass_groups(far_target_mask(20, 2))
    assert names[-1] == "other" and torch.equal(groups.sum(0), far_target_mask(20, 2).long())


def test_mc_synthetic_is_mass_plus_teacher_weighted_route():
    far_T, far_S, _, far = _far_pair(seed=1)
    rows = valid_rows(far)
    P, Q = far_T / far_T.sum(-1, keepdim=True).clamp_min(1e-300), far_S / far_S.sum(-1, keepdim=True).clamp_min(1e-300)
    Z_T, Z_S = torch.rand(far.size(0), dtype=torch.float64) * 0.5, torch.rand(far.size(0), dtype=torch.float64) * 0.5
    per_row = mc_kl_rows(synthetic_far(P, Z_T), synthetic_far(Q, Z_S))
    assert torch.allclose(per_row[rows], (bernoulli_kl(Z_T, Z_S) + Z_T * kl_rows(P, Q))[rows], atol=1e-12)


def test_mean_of_products_is_not_product_of_means():
    """v4 Prop. covariance: the old cache (mean z, mean r) cannot recover the mean mass."""
    z = torch.tensor([0.9, 0.1], dtype=torch.float64)
    r = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    raw = (z.unsqueeze(-1) * r).mean(0)
    assert torch.allclose(raw, torch.tensor([0.41, 0.09], dtype=torch.float64))
    assert torch.allclose(z.mean() * r.mean(0), torch.tensor([0.25, 0.25], dtype=torch.float64))
    swapped = (z.unsqueeze(-1) * r.flip(0)).mean(0)
    assert torch.allclose(swapped, torch.tensor([0.09, 0.41], dtype=torch.float64))


def test_route_only_update_moves_far_mass():
    """v4 Sec. 5.3: zero far gradient sum, REST logit untouched, yet Z changes."""
    logits = torch.log(torch.tensor([0.8, 0.16, 0.04], dtype=torch.float64)).requires_grad_(True)
    A = torch.softmax(logits, -1)
    r = A[1:] / A[1:].sum()
    P = torch.tensor([0.95, 0.05], dtype=torch.float64)
    loss = (P * (P.log() - r.log())).sum()
    (grad,) = torch.autograd.grad(loss, logits)
    assert torch.allclose(grad, torch.tensor([0.0, -0.15, 0.15], dtype=torch.float64), atol=1e-12)
    Z_new = torch.softmax(logits.detach() - 0.1 * grad, -1)[1:].sum()
    assert abs(float(Z_new) - 0.2014554018) < 1e-9


def test_group_gradient_is_mass_difference():
    """v4 Eq. fullgrad: for one component, sum_{u in K_g} dL/de_u = S_g - T_g (REST included)."""
    torch.manual_seed(0)
    key_nodes = torch.tensor([-1, -1, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6])
    far = far_target_mask(7, 2)
    row, num_nodes = 6, 7
    e = torch.randn(key_nodes.numel(), dtype=torch.float64, requires_grad=True)
    A = torch.softmax(e, -1)
    far_S = node_mass(A.unsqueeze(0), key_nodes, num_nodes)[0] * far[row]
    T = torch.softmax(torch.randn(num_nodes, dtype=torch.float64), -1) * far[row] * 0.6
    loss = mc_kl_rows(T.unsqueeze(0), far_S.unsqueeze(0)).sum()
    (grad,) = torch.autograd.grad(loss, e)
    for j in torch.nonzero(far[row]).squeeze(-1).tolist():
        assert abs(float(grad[key_nodes == j].sum() - (far_S[j] - T[j]))) < 1e-12
    rest = ~torch.isin(key_nodes, torch.nonzero(far[row]).squeeze(-1))
    assert abs(float(grad[rest].sum() - ((1 - far_S.sum()) - (1 - T.sum())))) < 1e-12


def test_mc_loss_gradcheck_through_attention():
    torch.manual_seed(0)
    key_nodes = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6])
    num_nodes, T = 7, key_nodes.numel()
    far = far_target_mask(num_nodes, 2)
    positions, owners = torch.tensor([15, 16, 13, 14]), torch.tensor([6, 6, 5, 5])
    weights = query_weights(torch.tensor([[1, 3], [3, 5], [5, 8], [8, 10], [10, 12], [12, 15], [15, 17]]),
                            positions, owners, "unbiased").double()
    q = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(T, 8, dtype=torch.float64, requires_grad=True)
    rows = valid_rows(far)
    far_T = torch.rand(num_nodes, num_nodes, dtype=torch.float64) * far * 0.05

    def loss(q, k):
        m = head_query_far_mass(q, k, positions, key_nodes, owners, far, 0.5)
        return mc_loss(far_T, weighted_row_sum(m, owners, weights, num_nodes), rows)

    assert torch.autograd.gradcheck(loss, (q, k), eps=1e-6, atol=1e-7)


def test_query_weights_unbiased_for_the_token_mean():
    n, m = 6, 3
    spans = torch.tensor([[10, 10 + n]])
    x = torch.randn(n, dtype=torch.float64)
    estimates = []
    for others in itertools.combinations(range(n - 1), m - 1):
        positions = torch.tensor([10 + o for o in others] + [10 + n - 1])
        w = query_weights(spans, positions, torch.zeros(m, dtype=torch.long), "unbiased").double()
        assert abs(float(w.sum()) - 1) < 1e-6
        estimates.append(float((w * x[positions - 10]).sum()))
    assert abs(np.mean(estimates) - float(x.mean())) < 1e-6  # exact over all equally likely samples
    full = query_weights(spans, torch.arange(10, 16), torch.zeros(6, dtype=torch.long), "unbiased")
    assert torch.allclose(full, torch.full((6,), 1 / 6))
    single = query_weights(spans, torch.tensor([15]), torch.zeros(1, dtype=torch.long), "unbiased")
    assert float(single) == 1.0
    uniform = query_weights(spans, torch.tensor([11, 13, 15]), torch.zeros(3, dtype=torch.long), "uniform")
    assert torch.allclose(uniform, torch.full((3,), 1 / 3))


def test_uniform_weights_reproduce_v3_pooling():
    torch.manual_seed(0)
    key_nodes = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6])
    far = far_target_mask(7, 2)
    positions, owners = torch.tensor([15, 16, 12, 14, 11]), torch.tensor([6, 6, 5, 5, 4])
    spans = torch.tensor([[1, 3], [3, 5], [5, 8], [8, 10], [10, 12], [12, 15], [15, 17]])
    q, k = torch.randn(5, 8), torch.randn(17, 8)
    r, z = head_query_routing(q, k, positions, key_nodes, owners, far, 0.5)
    m = head_query_far_mass(q, k, positions, key_nodes, owners, far, 0.5)
    w = query_weights(spans, positions, owners, "uniform")
    assert torch.allclose(weighted_row_sum(m, owners, w, 7).sum(-1), row_average(z, owners, 7)[0], atol=1e-6)
    r_new = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(weighted_row_sum(r_new, owners, w, 7), row_average(r, owners, 7)[0], atol=1e-6)


def test_blockwise_raw_mass_is_the_mean_mass():
    torch.manual_seed(0)
    key_nodes = torch.tensor([-1, -1] + [0] * 3 + sum(([j] * 3 for j in range(1, 9)), []))
    far = far_target_mask(9, 2)
    q, k = torch.randn(2, key_nodes.numel(), 8), torch.randn(2, key_nodes.numel(), 8)
    out = head_row_routing_blockwise(q, k, key_nodes, far, 0.5, block=5)
    for h in range(2):
        attn = attention_probs(q[h], k[h], torch.arange(key_nodes.numel()), 0.5)
        mass = node_mass(attn, key_nodes, 9)
        for i in valid_rows(far).nonzero().squeeze(-1).tolist():
            tokens = (key_nodes == i).nonzero().squeeze(-1)
            expected = (mass[tokens] * far[i]).mean(0)
            assert torch.allclose(out["M"][h, i], expected, atol=1e-6)
    assert torch.allclose(out["M"].sum(-1), out["Z"], atol=1e-6)
    rows = valid_rows(far)
    assert (out["M"] / out["Z"].unsqueeze(-1).clamp_min(1e-12) - out["R"])[:, rows].abs().max() > 1e-4  # Cov != 0


def test_mass_gap_by_bin_pools():
    far_T, far_S, _, far = _far_pair(seed=2)
    names, groups = mass_groups(far)
    gaps = mass_gap_by_bin(far_T, far_S, valid_rows(far), groups, names)
    total_rows = int(valid_rows(far).sum())
    assert gaps[0]["bin"] == "q" and gaps[0]["count"] == total_rows
    manual = float((far_S[:, 0] - far_T[:, 0])[valid_rows(far)].sum())
    assert abs(gaps[0]["dM_sum"] - manual) < 1e-9


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_capture_matches_eager_attention_for_student_and_teacher_families(family):
    """R1-Distill-Qwen is Qwen2 (q/k bias, no q/k norm); Qwen3 has q/k norm: the hook sees post-RoPE q/k in both."""
    attention_capture.install()
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=256)
    config = Qwen2Config(**common) if family == "qwen2" else Qwen3Config(head_dim=8, **common)
    cls = Qwen2ForCausalLM if family == "qwen2" else Qwen3ForCausalLM
    torch.manual_seed(0)
    sdpa = cls._from_config(copy.deepcopy(config), attn_implementation="sdpa").eval()
    eager = cls._from_config(copy.deepcopy(config), attn_implementation="eager").eval()
    eager.load_state_dict(sdpa.state_dict())
    ids = torch.randint(0, 64, (1, 33))
    capture = attention_capture.QKCapture(attention_capture.attention_modules(sdpa))
    positions = torch.tensor([0, 9, 32])
    capture.arm({1: [1, 2]}, positions, detach=True)
    sdpa(ids)
    capture.disarm()
    reference = eager(ids, output_attentions=True).attentions[1][0]
    for slot, head in enumerate([1, 2]):
        probs = attention_probs(capture.q[1][slot], capture.k[1][slot], positions, 8**-0.5)
        assert torch.allclose(probs, reference[head, positions], atol=1e-5)
        assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------- trainer

LAYERS, HEADS, KV, DIM = 4, 4, 2, 8


def _model(seed=1):
    attention_capture.install()
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=64, hidden_size=HEADS * DIM, intermediate_size=64, num_hidden_layers=LAYERS,
                         num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=DIM, max_position_embeddings=512)
    return Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa")


@pytest.fixture()
def mc_data(tmp_path):
    """Two synthetic records + random teacher targets with raw mass M (row sums = Z) on the student's layout."""
    rng = np.random.default_rng(0)
    records, targets = tmp_path / "records.jsonl", tmp_path / "targets"
    targets.mkdir()
    with records.open("w") as handle:
        for index in range(2):
            lengths = [5] + [int(x) for x in rng.integers(3, 6, size=12)] + [3]
            nodes, cursor = [], 3
            for k, length in enumerate(lengths):
                nodes.append({"kind": "step", "char_start": cursor * 10, "char_end": (cursor + length) * 10,
                              "token_start": cursor, "token_end": cursor + length, "hash": 1000 * index + k})
                cursor += length
            handle.write(json.dumps({"id": f"r{index}", "input_ids": rng.integers(0, 64, size=cursor + 1).tolist(),
                                     "response_token_span": [8, cursor], "nodes": nodes}) + "\n")
            N = len(nodes)
            far = far_target_mask(N, 2).numpy()
            M = rng.random((2, N, N)) * far
            M = 0.4 * M / np.maximum(M.sum(-1, keepdims=True), 1e-12)
            Z = M.sum(-1)
            P = M / np.maximum(Z[..., None], 1e-12)
            np.savez(targets / f"r{index}.npz", P=P.astype(np.float32), Z=Z.astype(np.float32), M=M.astype(np.float32),
                     rows=valid_rows(torch.from_numpy(far)).numpy(),
                     char_spans=np.asarray([[n["char_start"], n["char_end"]] for n in nodes]),
                     hash=np.asarray([n["hash"] for n in nodes], dtype=np.int64))
    return records, targets


def _trainer(tmp_path, records, targets, objective, lam=1.0, **kwargs):
    config = {"lora_r": 4, "lora_alpha": 4, "lora_dropout": 0.0, "csrd_qk_rank": 0,
              "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"}
    model = attach_lora(_model(), config)
    args = TrainingArguments(output_dir=str(tmp_path / "out"), per_device_train_batch_size=1, max_steps=10,
                             use_cpu=True, report_to=[], remove_unused_columns=False)
    options = dict(csrd_lambda=lam, csrd_d_min=2, csrd_queries=3, csrd_head_mode="band", csrd_warmup_frac=0.0,
                   csrd_ramp_frac=0.0, csrd_grad_log_interval=0, csrd_objective=objective,
                   csrd_query_weighting="unbiased")
    options.update(kwargs)
    trainer = CSRDTrainer(model=model, args=args, train_dataset=CSRDDataset(str(records), signals=str(targets)),
                          data_collator=CSRDCollator(pad_token_id=0), **options)
    trainer.state.max_steps = 10
    trainer.current_gradient_accumulation_steps = 1
    model.train()
    return trainer


def _step(trainer, seed=0):
    trainer._generator = torch.Generator().manual_seed(seed)
    trainer.model.zero_grad()
    trainer.compute_loss(trainer.model, trainer.data_collator([trainer.train_dataset[0]])).backward()
    return {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.requires_grad and p.grad is not None}


def test_objective_none_is_ce_only_but_logs_metrics(tmp_path, mc_data):
    shadow = _trainer(tmp_path, *mc_data, "none")
    ce_only = _trainer(tmp_path, *mc_data, "route_mass", lam=0.0)
    g_shadow, g_ce = _step(shadow), _step(ce_only)
    assert g_shadow.keys() == g_ce.keys() and all(torch.equal(g_shadow[n], g_ce[n]) for n in g_ce)
    for key in ("loss_mc_raw", "loss_mc_syn", "loss_route", "csrd_zS_b1", "csrd_ez_b0", "csrd_dM_q", "csrd_mcber_b0"):
        assert key in shadow._metric_sums, key


@pytest.mark.parametrize("objective", ["mc_raw", "mc_syn", "route_mass"])
def test_objectives_train_the_band_query_key_lora(tmp_path, mc_data, objective):
    with_aux = _step(_trainer(tmp_path, *mc_data, objective))
    ce_only = _step(_trainer(tmp_path, *mc_data, objective, lam=0.0))
    band_qk = [n for n in with_aux if ("layers.2." in n or "layers.3." in n) and "q_proj.lora_B" in n]
    assert any((with_aux[n] - ce_only[n]).abs().max() > 1e-7 for n in band_qk)


def test_mc_raw_chain_rule_in_trainer_metrics(tmp_path, mc_data):
    trainer = _trainer(tmp_path, *mc_data, "mc_raw", csrd_band_reduction="sum")
    _step(trainer)
    sums = trainer._metric_sums
    parts = sum(sums[f"csrd_mcber_b{b}"] + sums[f"csrd_mccond_b{b}"] for b in (0, 1))
    assert abs(sums["loss_mc_raw"] - parts) < 1e-5


def test_band_mean_halves_the_band_sum(tmp_path, mc_data):
    summed = _trainer(tmp_path, *mc_data, "mc_raw", csrd_band_reduction="sum")
    mean = _trainer(tmp_path, *mc_data, "mc_raw", csrd_band_reduction="mean")
    _step(summed)
    _step(mean)
    assert abs(summed._metric_sums["loss_mc_raw"] - 2 * mean._metric_sums["loss_mc_raw"]) < 1e-6


def test_mc_raw_needs_raw_mass(tmp_path, mc_data):
    records, targets = mc_data
    for path in targets.glob("*.npz"):
        data = dict(np.load(path))
        data.pop("M")
        np.savez(path, **data)
    trainer = _trainer(tmp_path, records, targets, "mc_raw")
    with pytest.raises(ValueError, match="raw mass"):
        _step(trainer)


def test_norm_probe_writes_ratios(tmp_path, mc_data):
    trainer = _trainer(tmp_path, *mc_data, "none", lam=0.0, csrd_probe_microbatches=2)
    for seed in range(3):
        _step(trainer, seed)
    assert len(trainer._probe_records) == 2
    trainer.write_norm_probe()
    probe = json.loads((tmp_path / "out" / "csrd-norm-probe.json").read_text())
    assert set(probe["median_ratio"]) >= {"mc_raw/route_mass0.1", "mc_syn/route_mass0.1"}
    record = probe["records"][0]
    assert record["norms"]["mc_raw"] > 0 and set(record["groups"]["mc_raw"]) == {"qk", "vo", "mlp"}


def test_signal_bank_packs_raw_mass_and_quality(tmp_path, mc_data):
    from signal_bank import SignalSource, pack

    records, targets = mc_data
    bank = tmp_path / "bank.safetensors"
    pack(str(targets), None, str(bank))
    source = SignalSource(str(bank))
    assert source.info["raw_mass_traces"] == 2
    assert source.info["raw_mass_quality"]["max_abs_row_sum_error"] < 1e-5
    item = CSRDDataset(str(records), signals=str(bank))[0]["csrd"]
    assert torch.allclose(item["M"].sum(-1), item["Z"], atol=1e-5)


def test_arm_b0_selects_heads_after_warmup_then_probes(tmp_path, mc_data):
    """B0 flow: receiver statistics during the CE-only warmup, heads fixed at its end, probe on that same step."""
    trainer = _trainer(tmp_path, *mc_data, "none", lam=0.0, csrd_head_mode="receiver", csrd_k_student=2,
                       csrd_warmup_frac=0.2, csrd_probe_microbatches=1)
    assert trainer._phase_steps()[0] == 2
    for step in range(3):
        trainer.state.global_step = step
        _step(trainer, step)
        assert (trainer.student_heads is None) == (step < 2)
        assert ("csrd_zS_b0" in trainer._metric_sums) == (step == 2)
    assert len(trainer._probe_records) == 1 and trainer._probe_records[0]["global_step"] == 2
    heads = json.loads((tmp_path / "out" / "csrd-student-heads.json").read_text())
    assert [len(band) for band in heads["heads"]] == [2, 2]


def test_mass_diagnostics_end_to_end(tmp_path, mc_data, monkeypatch):
    """Teacher vs two synthetic students: the student equal to the teacher has zero error and wins the contrast."""
    import sys

    import mass_diagnostics

    _, teacher = mc_data
    students = {}
    for tag, noise in (("same", 0.0), ("off", 0.5)):
        folder = tmp_path / tag
        folder.mkdir()
        rng = np.random.default_rng(1)
        for path in teacher.glob("*.npz"):
            data = dict(np.load(path))
            M = data["M"] * (1 + noise * rng.random(data["M"].shape)) * (1 + noise)
            data.update(M=M.astype(np.float32), Z=M.sum(-1).astype(np.float32),
                        P=(M / np.maximum(M.sum(-1, keepdims=True), 1e-12)).astype(np.float32))
            np.savez(folder / path.name, **data)
        students[tag] = folder
    out = tmp_path / "diag"
    monkeypatch.setattr(sys, "argv", ["mass_diagnostics.py", "--teacher", str(teacher), "--student", f"same={students['same']}",
                                      "--student", f"off={students['off']}", "--reference", "off", "--output-dir", str(out),
                                      "--d-min", "2", "--bootstrap", "50"])
    mass_diagnostics.main()
    result = json.loads((out / "mass-diagnostics.json").read_text())
    assert result["traces"] == 2
    same = result["arms"]["same"]["avg"]
    assert same["ez"]["value"] < 1e-6 and same["kl_raw"]["value"] < 1e-6
    assert result["arms"]["off"]["avg"]["dMq"]["value"] > 0  # inflated student mass on the question
    assert result["contrasts"]["same-off"]["avg"]["ez"]["value"] < 0
