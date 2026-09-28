"""Trainer plumbing for L = L_CE + lambda(t) L_aux, with L_aux one of (csrd_objective):

    route_mass  v3 Eq. 7: L_route + (lambda_m/lambda_r) L_mass + (lambda_c/lambda_r) L_causal
    mc_syn      v4 MC-synthetic: KL over {REST} U F(i) of D^syn = Z * R, built from the cached means (P, Z)
    mc_raw      v4 MC-CSRD: KL(D_T || D_S) over {REST} U F(i) of the raw pooled mass (bank field M)
    none        CE only; every routing/mass metric is still logged (no graph), the SFT-only trajectory of v4 Stage A

Used as `class CSRDTrainer(CSRDLossMixin, CESFTTrainer)` (train_sft.py). One step (Algorithm 1):
    1. sample m query tokens per row with |F(i)| >= 2: the step's last token + m-1 random ones, weighted
       uniformly or so that the row estimate is unbiased for the uniform token mean (csrd_query_weighting)
    2. one forward gives L_CE; the attention wrapper keeps q (sampled positions) and k of the heads H_S
    3. per head: exact causal softmax over the full support -> raw far mass m_t(j) per node; components
       (head, query) pooled per depth band -> D_S [N, N] (raw mass, row sums Z_S) and Q (mean of m_t / Z_t)
    4. losses against the teacher's P_i, Z_T,i, M_i (and C~_i on the causal subset); bands summed or averaged
H_S: receiver scores accumulate during the CE-only warmup, then the top K_S per band are fixed
(A5: all heads of the band, or a given heads.json); lambda_r then ramps linearly. With fixed heads the metrics are
logged from step 0, so the warmup itself is on record.
Variants: CSRD-A (anchor row weights), CSRD-PQ (per-query KL, A14), CSRD-QK (separate Q/K adapter, A12).
"""

import json
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
from transformers import TrainerCallback

import attention_capture
from csrd_data import CSRD_KEY
from receiver_heads import band_layers, receiver_scores, select_heads, vertical_scores
from routing import (
    EPS,
    MASS_EPS,
    causal_loss,
    causal_target,
    far_target_mask,
    head_query_far_mass,
    head_query_routing,
    mass_gap_by_bin,
    mass_groups,
    mass_loss,
    mc_loss,
    mc_parts,
    per_query_route_loss,
    query_weights,
    route_loss,
    row_average,
    synthetic_far,
    valid_rows,
    weighted_row_sum,
)

HEADS_FILE = "csrd-student-heads.json"
PROBE_FILE = "csrd-norm-probe.json"
OBJECTIVES = ("route_mass", "mc_syn", "mc_raw", "none")
PARAM_GROUPS = {"q_proj": "qk", "k_proj": "qk", "v_proj": "vo", "o_proj": "vo",
                "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
BACKWARD_EPILOGUE_METHODS = ("_backward_epilogue", "run_grad_acc_post_hooks")


def _noop(*args, **kwargs):
    return None


def _scalar_state(obj):
    return {k: v for k, v in obj.__dict__.items() if v is None or isinstance(v, (bool, int, float, str))}


def key_node_ids(node_spans: torch.Tensor, length: int) -> torch.Tensor:
    ids = torch.full((length,), -1, dtype=torch.long, device=node_spans.device)
    for index, (start, end) in enumerate(node_spans.tolist()):
        ids[start:end] = index
    return ids


def sample_queries(node_spans: torch.Tensor, rows: torch.Tensor, m: int, generator: torch.Generator | None = None):
    """Q_S(i): last token of each valid row's node + m-1 others without replacement (m <= 0: all, A6).
    Returns (positions [Nq], rows [Nq]) on node_spans' device."""
    positions, owners = [], []
    for i in torch.nonzero(rows, as_tuple=False).squeeze(-1).tolist():
        start, end = node_spans[i].tolist()
        if m <= 0 or end - start <= m:
            chosen = list(range(start, end))
        else:
            others = torch.randperm(end - start - 1, generator=generator)[: m - 1] + start
            chosen = others.tolist() + [end - 1]
        positions += chosen
        owners += [i] * len(chosen)
    device = node_spans.device
    return torch.tensor(positions, dtype=torch.long, device=device), torch.tensor(owners, dtype=torch.long, device=device)


def param_group(name: str) -> str:
    """qk / vo / mlp / other by the projection a (LoRA) parameter belongs to."""
    return next((group for module, group in PARAM_GROUPS.items() if f".{module}." in name), "other")


class HeadsCheckpointCallback(TrainerCallback):
    """Write the fixed student heads into every checkpoint-N/, so --resume keeps the same H_S; write the
    gradient-norm probe once its step is over."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.trainer.student_heads is not None:
            path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", HEADS_FILE)
            self.trainer.save_student_heads(path)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > self.trainer._phase_steps()[0]:
            self.trainer.write_norm_probe()

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.write_norm_probe()


class CSRDLossMixin:
    """Constructor kwargs (popped before Trainer; see train_sft.py --csrd-* for the rest):
        csrd_lambda            lambda_r at full strength
        csrd_route_ratio       L_route weight / lambda_r (0 with causal targets = A3 "causal only")
        csrd_head_mode         "receiver" (select after warmup) | "band" (all heads of each band) | "fixed"
        csrd_student_heads     dict (heads.json payload) for "fixed" / resume
        csrd_warmup_frac       CE-only share of steps; csrd_ramp_frac: linear ramp share after it
        csrd_qk_adapter        name of the separate Q/K adapter (CSRD-QK) or None
        csrd_head_checkpoint   recompute per-head score matrices in backward (memory)
        csrd_grad_log_interval every N steps log ||grad CE||, ||grad aux|| (total and per qk/vo/mlp) and their cosine
        csrd_objective         route_mass | mc_syn | mc_raw | none (module doc)
        csrd_band_reduction    "sum" (v3) or "mean" (v4) of the per-band losses
        csrd_query_weighting   "uniform" (v3) or "unbiased" (v4 Eq. querysampling) pooling of the sampled queries
        csrd_probe_microbatches  on the first K microbatches of the first post-warmup step, the gradient norm of
                               every candidate objective (unweighted) -> <output_dir>/csrd-norm-probe.json (0 = off)
    """

    def __init__(self, *args, **kwargs):
        self.csrd_lambda = float(kwargs.pop("csrd_lambda"))
        self.csrd_mass_ratio = float(kwargs.pop("csrd_mass_ratio", 0.1))
        self.csrd_causal_ratio = float(kwargs.pop("csrd_causal_ratio", 1.0))
        self.csrd_route_ratio = float(kwargs.pop("csrd_route_ratio", 1.0))
        self.csrd_bands = tuple(kwargs.pop("csrd_bands", (0, 1)))
        self.csrd_d_min = int(kwargs.pop("csrd_d_min", 4))
        self.csrd_queries = int(kwargs.pop("csrd_queries", 8))
        self.csrd_k_student = int(kwargs.pop("csrd_k_student", 16))
        self.csrd_head_mode = kwargs.pop("csrd_head_mode", "receiver")
        preset = kwargs.pop("csrd_student_heads", None)
        self.csrd_score = kwargs.pop("csrd_score", "excess_bg")
        self.csrd_anchor_beta = float(kwargs.pop("csrd_anchor_beta", 0.0))
        self.csrd_loss_form = kwargs.pop("csrd_loss_form", "pooled")
        self.csrd_warmup_frac = float(kwargs.pop("csrd_warmup_frac", 0.1))
        self.csrd_ramp_frac = float(kwargs.pop("csrd_ramp_frac", 0.1))
        self.csrd_qk_adapter = kwargs.pop("csrd_qk_adapter", None)
        self.csrd_head_checkpoint = bool(kwargs.pop("csrd_head_checkpoint", True))
        self.csrd_grad_log_interval = int(kwargs.pop("csrd_grad_log_interval", 50))
        self.csrd_objective = kwargs.pop("csrd_objective", "route_mass")
        self.csrd_band_reduction = kwargs.pop("csrd_band_reduction", "sum")
        self.csrd_query_weighting = kwargs.pop("csrd_query_weighting", "uniform")
        self.csrd_probe_microbatches = int(kwargs.pop("csrd_probe_microbatches", 0))
        super().__init__(*args, **kwargs)
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("CSRD samples queries per sequence: use --per-device-batch-size 1 (Table 3)")
        if self.csrd_loss_form not in ("pooled", "per_query"):
            raise ValueError(f"unknown csrd_loss_form {self.csrd_loss_form!r}")
        if self.csrd_objective not in OBJECTIVES:
            raise ValueError(f"unknown csrd_objective {self.csrd_objective!r}; expected one of {OBJECTIVES}")
        if self.csrd_band_reduction not in ("sum", "mean"):
            raise ValueError(f"unknown csrd_band_reduction {self.csrd_band_reduction!r}")

        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.bands = band_layers(self.num_layers)
        self.modules_by_layer = attention_capture.attention_modules(self.model)
        attention_capture.check_attn_implementation(self.model)
        self._capture = attention_capture.QKCapture(self.modules_by_layer)
        self._input_capture = attention_capture.AttentionInputCapture(self.modules_by_layer) if self.csrd_qk_adapter else None

        self.student_heads: list[list[tuple[int, int]]] | None = None
        if preset is not None:
            self.student_heads = [[tuple(pair) for pair in band] for band in preset["heads"]]
            if preset.get("num_layers", self.num_layers) != self.num_layers:
                raise ValueError("student heads.json was built for a different depth")
        elif self.csrd_head_mode == "band":
            self.student_heads = [[(l, h) for l in band for h in range(self.num_heads)] for band in self.bands]
        elif self.csrd_head_mode == "fixed":
            raise ValueError("csrd_head_mode='fixed' needs csrd_student_heads")
        self._score_sum = torch.zeros(self.num_layers, self.num_heads, dtype=torch.float64)
        self._score_count = 0

        self._metric_sums = defaultdict(float)
        self._metric_counts = defaultdict(int)
        self._grad_logged_step = -1
        self._generator = torch.Generator().manual_seed(self.args.seed + 7919 * self.args.process_index)
        self._shared_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self._probe_records, self._probe_written = [], False
        self._setup_qk_adapter()
        names = {id(p): n for n, p in self.model.named_parameters()}
        self._shared_groups = [param_group(names[id(p)]) for p in self._shared_params]
        self.add_callback(HeadsCheckpointCallback(self))

    # ---------------------------------------------------------------- schedule
    def _phase_steps(self) -> tuple[int, int]:
        total = max(1, self.state.max_steps or 1)
        return round(self.csrd_warmup_frac * total), round(self.csrd_ramp_frac * total)

    def current_lambda(self) -> float:
        warmup, ramp = self._phase_steps()
        step = self.state.global_step
        if step < warmup:
            return 0.0
        return self.csrd_lambda * min(1.0, (step - warmup + 1) / max(1, ramp))

    # ---------------------------------------------------------------- CSRD-QK
    def _setup_qk_adapter(self):
        """CSRD-QK: a tensor hook on each Q/K adapter parameter *replaces* its CE gradient with the routing
        gradient from compute_loss; DDP then reduces the replaced value (its hooks run after ours)."""
        self._qk_params, self._qk_pending, self._qk_replace = [], {}, False
        if not self.csrd_qk_adapter:
            return
        if self.is_deepspeed_enabled:
            raise ValueError("CSRD-QK replaces gradients with tensor hooks; run it with DDP, not DeepSpeed")
        tag = f".{self.csrd_qk_adapter}."
        for name, param in self.model.named_parameters():
            if tag in name and param.requires_grad:
                self._qk_params.append(param)
                param.register_hook(self._make_qk_hook(param))
        if not self._qk_params:
            raise ValueError(f"no trainable parameters of adapter {self.csrd_qk_adapter!r}")
        qk_ids = {id(p) for p in self._qk_params}
        self._shared_params = [p for p in self._shared_params if id(p) not in qk_ids]

    def _make_qk_hook(self, param):
        def hook(grad):
            if not self._qk_replace:
                return grad
            pending = self._qk_pending.get(id(param))
            return torch.zeros_like(grad) if pending is None else pending.to(grad.dtype)

        return hook

    # ---------------------------------------------------------------- student heads
    def _all_heads(self) -> dict[int, list[int]]:
        return {layer: list(range(self.num_heads)) for layer in range(self.num_layers)}

    def _selected_by_layer(self) -> dict[int, list[int]]:
        by_layer: dict[int, list[int]] = {}
        for b, band in enumerate(self.student_heads):
            if b not in self.csrd_bands:
                continue
            for layer, head in band:
                if head not in by_layer.setdefault(layer, []):
                    by_layer[layer].append(head)
        return by_layer

    def _arm_head_stats(self, key_nodes, far, positions, owners):
        """Warmup: receiver statistics nu [L, H, N] of every head, computed without grad inside the layer hooks."""
        num_nodes = far.size(0)
        nu = torch.full((self.num_layers, self.num_heads, num_nodes), float("nan"), device=positions.device)
        pending = set(range(self.num_layers))
        rows_with_queries = torch.bincount(owners, minlength=num_nodes) > 0

        def make(layer):
            def hook(module, query, key, value, attention_mask, **kwargs):
                if layer not in pending:
                    return None
                pending.discard(layer)
                with torch.no_grad():
                    groups = query.size(1) // key.size(1)
                    q = query[0][:, positions]
                    k = key[0].repeat_interleave(groups, dim=0)
                    scale = kwargs.get("scaling") or module.scaling
                    R_rows = []
                    for h in range(q.size(0)):
                        r, _ = head_query_routing(q[h], k[h], positions, key_nodes, owners, far, scale)
                        R_rows.append(row_average(r, owners, num_nodes)[0])
                    nu[layer] = vertical_scores(torch.stack(R_rows), rows_with_queries, self.csrd_d_min)
                return None

            return hook

        attention_capture.set_hooks(self.modules_by_layer, make)
        return nu

    def _finish_head_stats(self, nu):
        attention_capture.clear_hooks(self.modules_by_layer)
        if torch.isnan(nu).all():
            return
        scores = receiver_scores(nu.view(self.num_layers * self.num_heads, -1), self.csrd_score)
        self._score_sum += scores.view(self.num_layers, self.num_heads).double().cpu()
        self._score_count += 1

    def _select_student_heads(self):
        sums = self._score_sum.clone()
        count = torch.tensor([float(self._score_count)])
        if dist.is_available() and dist.is_initialized():
            device = next(self.model.parameters()).device
            sums, count = sums.to(device), count.to(device)
            dist.all_reduce(sums)
            dist.all_reduce(count)
            sums, count = sums.cpu(), count.cpu()
        if count.item() == 0:
            raise RuntimeError("no receiver statistics collected before student head selection")
        mean = (sums / count).numpy()
        self.student_heads = [[tuple(pair) for pair in band] for band in select_heads(mean, self.csrd_k_student)]
        self._head_selection_scores = mean
        if self.is_world_process_zero():
            print(f"CSRD: fixed student heads after {int(count.item())} warmup sequences: {self.student_heads}")
            self.save_student_heads(os.path.join(self.args.output_dir, HEADS_FILE))

    def save_student_heads(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump({
                "score": self.csrd_score if self.csrd_head_mode == "receiver" else self.csrd_head_mode,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "band_layers": self.bands,
                "k_per_band": self.csrd_k_student,
                "heads": [[list(pair) for pair in band] for band in self.student_heads],
                "warmup_sequences": self._score_count,
            }, handle, indent=2)

    # ---------------------------------------------------------------- losses
    def _band_routing(self, sample_q, sample_k, positions, owners, weights, key_nodes, far, num_nodes, scale):
        """Per used band: {"band", "M": pooled raw far mass D_S [N, N] (row sums Z_S), "Q": pooled far-normalized
        routing [N, N], "per_query": per-head R_t (CSRD-PQ only)}. Component (head, query) weight: 1/|band| x w_t."""
        out = []
        for b, band in enumerate(self.student_heads):
            if b not in self.csrd_bands:
                continue
            M = Q = 0.0
            per_query = []
            for layer, head in band:
                slot = self._slot[layer][head]
                m = head_query_far_mass(sample_q[layer][slot], sample_k[layer][slot], positions, key_nodes, owners,
                                        far, scale, use_checkpoint=self.csrd_head_checkpoint)
                r = m / m.sum(-1, keepdim=True).clamp_min(EPS)
                M = M + weighted_row_sum(m, owners, weights, num_nodes)
                Q = Q + weighted_row_sum(r, owners, weights, num_nodes)
                if self.csrd_loss_form == "per_query":
                    per_query.append(r)
            out.append({"band": b, "M": M / len(band), "Q": Q / len(band), "per_query": per_query})
        return out

    def _reduce_bands(self, values: list[torch.Tensor]) -> torch.Tensor:
        total = torch.stack(values).sum()
        return total / len(values) if self.csrd_band_reduction == "mean" else total

    def _aux_losses(self, targets, band_routing, owners, far, rows):
        """Every candidate objective on the same pooled routing ({name: band-reduced loss}) and the metrics:
        Z_T / Z_S / E_Z per band, the chain-rule parts of the raw KL, unconditional mass gaps per distance bin."""
        device = far.device
        weights = 1.0 + self.csrd_anchor_beta * targets["anchor"].to(device)
        C_tilde = support = None
        # causal targets are used only by CSRD-C (lambda_c > 0)
        if "C" in targets and self.csrd_causal_ratio > 0:
            C_tilde, support = causal_target(targets["C"].to(device), targets["J"].to(device), far)
        has_raw = "M" in targets
        if self.csrd_objective == "mc_raw" and not has_raw:
            raise ValueError("csrd_objective mc_raw needs a signal bank with raw mass M (extract_routing.py of v4)")
        names, groups = mass_groups(far) if has_raw else (None, None)
        parts, stats, gaps = defaultdict(list), {}, defaultdict(list)
        for band in band_routing:
            b, M_S, Q = band["band"], band["M"], band["Q"]
            Z_S = M_S.sum(-1)
            P, Z_T = targets["P"][b].to(device), targets["Z"][b].to(device)
            if self.csrd_loss_form == "per_query":
                parts["route"].append(torch.stack([per_query_route_loss(P, r, owners, rows, weights)
                                                   for r in band["per_query"]]).mean())
            else:
                parts["route"].append(route_loss(P, Q, rows, weights))
            parts["mass"].append(mass_loss(Z_T, Z_S, rows, weights))
            parts["mc_syn"].append(mc_loss(synthetic_far(P, Z_T), synthetic_far(Q, Z_S), rows, weights))
            if C_tilde is not None:
                parts["causal"].append(causal_loss(C_tilde, Q, support))
            if has_raw:
                M_T = targets["M"][b].to(device)
                parts["mc_raw"].append(mc_loss(M_T, M_S, rows, weights))
            with torch.no_grad():
                stats[f"csrd_zT_b{b}"] = float(Z_T[rows].mean())
                stats[f"csrd_zS_b{b}"] = float(Z_S[rows].mean())
                stats[f"csrd_ez_b{b}"] = float((Z_S - Z_T)[rows].abs().mean())
                if has_raw:
                    ber, cond = mc_parts(M_T, M_S.detach(), rows)
                    stats[f"csrd_mcber_b{b}"], stats[f"csrd_mccond_b{b}"] = float(ber), float(cond)
                    clamped = (M_T > 0) & (M_S < MASS_EPS) & rows.unsqueeze(-1)
                    stats["csrd_clamped"] = stats.get("csrd_clamped", 0) + int(clamped.sum())
                    for gap in mass_gap_by_bin(M_T, M_S.detach(), rows, groups, names):
                        if gap["count"]:
                            gaps[gap["bin"]].append(gap["dM_sum"] / gap["count"])
        losses = {name: self._reduce_bands(values) for name, values in parts.items()}
        losses.setdefault("causal", torch.zeros((), device=device))
        losses["route_mass"] = (self.csrd_route_ratio * losses["route"] + self.csrd_mass_ratio * losses["mass"]
                                + self.csrd_causal_ratio * losses["causal"])
        stats.update({f"csrd_dM_{name}": sum(values) / len(values) for name, values in gaps.items()})
        return losses, stats

    # ---------------------------------------------------------------- compute_loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        targets_list = inputs.pop(CSRD_KEY, None)
        if targets_list is None or not self.model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch, **kwargs)
        targets = targets_list[0]
        length = inputs["input_ids"].size(1)
        device = inputs["input_ids"].device
        node_spans = targets["node_spans"].to(device)
        num_nodes = node_spans.size(0)
        key_nodes = key_node_ids(node_spans, length)
        far = far_target_mask(num_nodes, self.csrd_d_min, device)
        rows = valid_rows(far) & targets["rows"].to(device).bool()
        positions, owners = sample_queries(node_spans.cpu(), rows.cpu(), self.csrd_queries, self._generator)
        weights = query_weights(node_spans.cpu(), positions, owners, self.csrd_query_weighting)
        positions, owners, weights = positions.to(device), owners.to(device), weights.to(device)

        lam = self.current_lambda()
        in_warmup = self.state.global_step < self._phase_steps()[0]
        collect_stats = self.student_heads is None and self.csrd_head_mode == "receiver"
        if collect_stats and not in_warmup:
            # warmup ended (or was empty): select from what was collected, else from this batch first
            if self._score_count == 0:
                with torch.no_grad():
                    nu = self._arm_head_stats(key_nodes, far, positions, owners)
                    model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), logits_to_keep=1)
                    self._finish_head_stats(nu)
            self._select_student_heads()
            collect_stats = False

        has_queries = positions.numel() > 0
        # measure: routing of H_S on this sequence (graph only when a loss is applied or probed)
        measure = self.student_heads is not None and has_queries
        apply_aux = measure and lam > 0 and self.csrd_objective != "none"
        probe = measure and self._probe_due()
        need_graph = apply_aux or probe
        nu = None
        if collect_stats:
            nu = self._arm_head_stats(key_nodes, far, positions, owners)
        elif measure:
            wanted = self._selected_by_layer()
            self._slot = {layer: {h: s for s, h in enumerate(heads)} for layer, heads in wanted.items()}
            if self._input_capture is not None:
                self._input_capture.arm(wanted.keys())
            else:
                self._capture.arm(wanted, positions, detach=not need_graph)

        try:
            loss_ce, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                    num_items_in_batch=num_items_in_batch, **kwargs)
        finally:
            if nu is not None:
                self._finish_head_stats(nu)
            self._capture.disarm()
            if self._input_capture is not None:
                self._input_capture.disarm()

        accumulation = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
        loss = loss_ce
        values = {"loss_ce": float(loss_ce.detach()) * accumulation, "csrd_lambda": lam, "csrd_queries": positions.numel()}
        if measure:
            scale = next(iter(self.modules_by_layer.values())).scaling
            with torch.set_grad_enabled(need_graph and torch.is_grad_enabled()):
                if self._input_capture is not None:
                    sample_q, sample_k = self._recomputed_qk(wanted, positions)
                else:
                    sample_q, sample_k = self._capture.q, self._capture.k
                band_routing = self._band_routing(sample_q, sample_k, positions, owners, weights, key_nodes, far,
                                                  num_nodes, scale)
                losses, stats = self._aux_losses(targets, band_routing, owners, far, rows)
            if probe:
                self._probe(loss_ce, losses)
            values.update({f"loss_{name}": float(value.detach()) for name, value in losses.items()}, **stats)
        if apply_aux:
            aux = lam * losses[self.csrd_objective]
            # aux is a per-sequence mean, loss_ce a share of the step sum/Z: divide or lambda scales with accumulation.
            aux_scaled = aux / accumulation
            if self._should_log_grads():
                self._log_grad_stats(loss_ce, aux_scaled)
            if self.csrd_qk_adapter:
                grads = torch.autograd.grad(aux_scaled, self._qk_params, allow_unused=True)
                # the replaced gradient bypasses accelerator.backward's 1/accumulation scaling, so apply it here.
                ga = self.accelerator.gradient_accumulation_steps
                self._qk_pending = {id(p): (g.detach() / ga if g is not None else None)
                                    for p, g in zip(self._qk_params, grads)}
                self._qk_replace = True
            else:
                loss = loss_ce + aux_scaled
        elif self.csrd_qk_adapter:
            self._qk_pending, self._qk_replace = {}, True  # CE never trains the Q/K adapter

        for key, value in values.items():
            self._metric_sums[key] += value
            self._metric_counts[key] += 1
        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        try:
            return super().training_step(*args, **kwargs)
        finally:
            self._qk_replace, self._qk_pending = False, {}

    def _recomputed_qk(self, wanted, positions):
        """CSRD-QK: q/k recomputed from the *detached* attention inputs (gradients reach only this layer's Q/K)."""
        sample_q, sample_k = {}, {}
        for layer, heads in wanted.items():
            hidden, position_embeddings = self._input_capture.inputs[layer]
            q, k = attention_capture.recompute_qk(self.modules_by_layer[layer], hidden, position_embeddings)
            groups = q.size(1) // k.size(1)
            index = torch.tensor(heads, device=q.device)
            sample_q[layer] = q[0].index_select(0, index)[:, positions]
            sample_k[layer] = k[0].index_select(0, index // groups)
        return sample_q, sample_k

    # ---------------------------------------------------------------- diagnostics
    def _should_log_grads(self) -> bool:
        interval, step = self.csrd_grad_log_interval, self.state.global_step
        if interval <= 0 or step % interval != 0 or step == self._grad_logged_step:
            return False
        self._grad_logged_step = step
        return True

    @contextmanager
    def _backward_hooks_muted(self):
        """Hide the probe backward passes from DeepSpeed (see SpectralGuidedLearning transition_trainer)."""
        engine = next((obj for obj in (getattr(self, "deepspeed", None), self.model_wrapped)
                       if hasattr(obj, "_backward_epilogue")), None)
        if engine is None:
            yield
            return
        objects = [obj for obj in (engine, getattr(engine, "optimizer", None)) if obj is not None]
        saved = [(obj, _scalar_state(obj)) for obj in objects]
        muted = [(obj, name) for obj in objects for name in BACKWARD_EPILOGUE_METHODS if hasattr(obj, name)]
        for obj, name in muted:
            setattr(obj, name, _noop)
        try:
            yield
        finally:
            for obj, name in muted:
                obj.__dict__.pop(name, None)
            for obj, state in saved:
                obj.__dict__.update(state)

    def _flat_grads(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """{name: flattened fp32 gradient of that loss on the shared LoRA parameters}, graph kept."""
        params, out = self._shared_params, {}
        with self._backward_hooks_muted():
            for name, loss in losses.items():
                if not loss.requires_grad:
                    out[name] = torch.cat([torch.zeros_like(p, dtype=torch.float32).flatten() for p in params])
                    continue
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                out[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).float().flatten()
                                       for g, p in zip(grads, params)])
        return out

    def _group_norms(self, flat: torch.Tensor) -> dict[str, float]:
        if not hasattr(self, "_group_index"):
            sizes = torch.tensor([p.numel() for p in self._shared_params])
            owner = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
            self._group_index = {group: torch.nonzero(torch.tensor([g == group for g in self._shared_groups])[owner])
                                 .squeeze(-1).to(flat.device) for group in sorted(set(self._shared_groups))}
        return {group: float(flat.index_select(0, index).norm()) for group, index in self._group_index.items()}

    def _log_grad_stats(self, loss_ce, aux_scaled):
        """||grad CE||, ||grad aux|| (total and per qk / vo / mlp) and their cosine on the shared LoRA parameters.

        Both are what this microbatch adds to the step's gradient before clipping: loss_ce is its share of the
        step's token-mean CE, aux_scaled is lambda * L_aux / accumulation (grad_route_* keep their v3 names)."""
        try:
            grads = self._flat_grads({"ce": loss_ce, "route": aux_scaled})
        except Exception as exc:  # never let a diagnostic kill the run
            self.csrd_grad_log_interval = 0
            if self.is_world_process_zero():
                print(f"grad diagnostic failed ({type(exc).__name__}: {exc}); disabled")
            return
        g_ce, g_route = grads["ce"], grads["route"]
        n_ce, n_route = float(g_ce.norm()), float(g_route.norm())
        cos = float(g_ce @ g_route) / max(n_ce * n_route, 1e-20)
        entries = [("grad_ce_lora", n_ce), ("grad_route_lora", n_route), ("grad_cos_ce_route", cos),
                   ("grad_route_ratio", n_route / max(n_ce, 1e-20))]
        for name, flat in (("ce", g_ce), ("route", g_route)):
            entries += [(f"grad_{name}_{group}", value) for group, value in self._group_norms(flat).items()]
        for key, value in entries:
            self._metric_sums[key] += value
            self._metric_counts[key] += 1

    # ---------------------------------------------------------------- norm probe (B4's lambda)
    def _probe_due(self) -> bool:
        if self.csrd_probe_microbatches <= 0 or self._probe_written:
            return False
        return (self.state.global_step == self._phase_steps()[0]
                and len(self._probe_records) < self.csrd_probe_microbatches)

    def _probe(self, loss_ce, losses):
        """v4 Sec. 7.2: gradient norms of every candidate objective on the same microbatch and parameters (the
        first post-warmup step, identical in every arm with the same seed), before lambda and accumulation."""
        names = [name for name in ("route", "mass", "mc_syn", "mc_raw") if name in losses]
        try:
            grads = self._flat_grads({"ce": loss_ce, **{name: losses[name] for name in names}})
        except Exception as exc:  # never let a diagnostic kill the run
            self.csrd_probe_microbatches = 0
            if self.is_world_process_zero():
                print(f"norm probe failed ({type(exc).__name__}: {exc}); disabled")
            return
        norms = {name: float(g.norm()) for name, g in grads.items()}
        for alpha in sorted({0.1, 1.0, self.csrd_mass_ratio}):
            norms[f"route_mass{alpha:g}"] = float((grads["route"] + alpha * grads["mass"]).norm())
        cosine = {f"ce_{name}": float(grads["ce"] @ grads[name]) / max(norms["ce"] * norms[name], 1e-20) for name in names}
        self._probe_records.append({"global_step": self.state.global_step, "norms": norms, "cos": cosine,
                                    "groups": {name: self._group_norms(grads[name]) for name in grads}})

    def write_norm_probe(self) -> None:
        if self._probe_written or not self._probe_records:
            return
        self._probe_written = True
        if not self.is_world_process_zero():
            return
        reference = "route_mass0.1"
        ratios = {}
        for name in ("mc_raw", "mc_syn", "route_mass1", "route", "mass"):
            values = [r["norms"][name] / r["norms"][reference] for r in self._probe_records
                      if name in r["norms"] and r["norms"][reference] > 0]
            if values:
                ratios[f"{name}/{reference}"] = float(sorted(values)[len(values) // 2])
        path = os.path.join(self.args.output_dir, PROBE_FILE)
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({
                "note": "gradient L2 norms on the shared LoRA parameters of each objective, unweighted (no lambda, no "
                        "1/accumulation), band reduction " + self.csrd_band_reduction + "; 'ce' is the microbatch's "
                        "share of the step CE. median_ratio: median over microbatches (upper median).",
                "band_reduction": self.csrd_band_reduction,
                "query_weighting": self.csrd_query_weighting,
                "records": self._probe_records,
                "median_ratio": ratios,
            }, handle, indent=2)
        print(f"CSRD: gradient-norm probe over {len(self._probe_records)} microbatches -> {path}: {ratios}")

    def log(self, logs, *args, **kwargs):
        for key, total in self._metric_sums.items():
            logs[key] = round(total / max(1, self._metric_counts[key]), 6)
        self._metric_sums.clear()
        self._metric_counts.clear()
        super().log(logs, *args, **kwargs)
