"""Trainer plumbing for L = L_CE + lambda_r L_route + lambda_m L_mass + lambda_c L_causal (Eq. 7).

Mixed in before the CE trainer (train_sft.py): `class CSRDTrainer(CSRDLossMixin, CESFTTrainer)`.
One training step (Algorithm 1):
    1. sample m query tokens per row with |F(i)| >= 2: the last token of the step + m-1 random ones
    2. one forward (SDPA/FlashAttention) gives L_CE; the attention wrapper keeps q (post-RoPE, at
       the sampled positions) and k of the student heads H_S
    3. per head: exact causal softmax of the sampled queries over all keys -> node masses ->
       far-normalized R_t, Z_t -> row means; heads averaged per depth band -> Q_i, Z_S,i
    4. KL losses against the teacher's P_i, Z_T,i (and C~_i on the causal subset)

Student heads H_S: during the CE-only warmup (lambda_r = 0 for the first 10% of steps) receiver
scores of every student head are accumulated from the sampled queries; at the end of warmup the
top K_S per band are fixed (A5 alternatives: all heads of the band, or a given heads.json).
lambda_r then ramps linearly to its target over the next 10% of steps.

Variants: CSRD-A (row weights 1 + beta * anchor), CSRD-PQ (per-query, per-head KL, A14), CSRD-QK
(the routing loss trains only a separate Q/K adapter, from detached inputs; A12).
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
    causal_loss,
    causal_target,
    far_target_mask,
    head_query_routing,
    mass_loss,
    per_query_route_loss,
    route_loss,
    row_average,
    valid_rows,
)

HEADS_FILE = "csrd-student-heads.json"
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
    """Q_S(i): the last token of each valid row's node plus m-1 other tokens of it drawn without
    replacement (all of them if the node is shorter). m <= 0 means every token (A6 "all").
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


class HeadsCheckpointCallback(TrainerCallback):
    """Write the fixed student heads into every checkpoint-N/, so --resume keeps the same H_S."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.trainer.student_heads is not None:
            path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", HEADS_FILE)
            self.trainer.save_student_heads(path)


class CSRDLossMixin:
    """Constructor kwargs (popped before Trainer):
        csrd_lambda            lambda_r at full strength
        csrd_mass_ratio        lambda_m / lambda_r (0.1; 0 = ablation A7)
        csrd_causal_ratio      lambda_c / lambda_r (1.0; used only on samples with causal targets)
        csrd_route_ratio       weight of L_route relative to lambda_r (1.0; 0 with causal targets = A3 "causal only")
        csrd_bands             depth bands entering the losses, e.g. (0, 1) both (default), (0,) = B1 only (A4)
        csrd_d_min             far threshold in steps (must match the targets)
        csrd_queries           m, queries per row (A6; <= 0 = all tokens)
        csrd_k_student         K_S heads per band
        csrd_head_mode         "receiver" (select after warmup) | "band" (all heads of each band) | "fixed"
        csrd_student_heads     dict (heads.json payload) for "fixed" / resume
        csrd_score             receiver score for student selection ("excess_bg" | "kurtosis")
        csrd_anchor_beta       CSRD-A row weight 1 + beta * anchor (0 = uniform)
        csrd_loss_form         "pooled" (Eq. 4) | "per_query" (CSRD-PQ)
        csrd_warmup_frac       CE-only share of steps (0.1)
        csrd_ramp_frac         linear ramp share after warmup (0.1)
        csrd_qk_adapter        name of the separate Q/K adapter (CSRD-QK) or None
        csrd_head_checkpoint   recompute per-head score matrices in backward (memory)
        csrd_grad_log_interval every N steps log ||grad CE||, ||grad route|| and their cosine (0 = off)
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
        super().__init__(*args, **kwargs)
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("CSRD samples queries per sequence: use --per-device-batch-size 1 (Table 3)")
        if self.csrd_loss_form not in ("pooled", "per_query"):
            raise ValueError(f"unknown csrd_loss_form {self.csrd_loss_form!r}")

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
        self._setup_qk_adapter()
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
        """Grad routing for CSRD-QK: CE gradients never reach the Q/K adapter and routing-loss gradients
        reach nothing else. A tensor hook on each Q/K adapter parameter *replaces* the gradient of the
        Trainer's (CE-only) backward with the routing gradient computed in compute_loss; DDP/ZeRO then
        reduce the replaced value as usual (their hooks run on accumulation, after ours)."""
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
        """Warmup: receiver statistics of every head from the sampled queries, computed inside the
        layer hooks without grad (nothing is stored across layers but nu [H, N])."""
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
    def _band_routing(self, sample_q, sample_k, positions, owners, key_nodes, far, num_nodes, scale):
        """Per used band: (band index, pooled Q [N, N], Z_S [N] (row means averaged over heads), per-head R_t)."""
        out = []
        for b, band in enumerate(self.student_heads):
            if b not in self.csrd_bands:
                continue
            Qs, Zs, per_query = [], [], []
            for layer, head in band:
                slot = self._slot[layer][head]
                r, z = head_query_routing(sample_q[layer][slot], sample_k[layer][slot], positions, key_nodes, owners,
                                          far, scale, use_checkpoint=self.csrd_head_checkpoint)
                Qs.append(row_average(r, owners, num_nodes)[0])
                Zs.append(row_average(z, owners, num_nodes)[0])
                per_query.append(r)
            out.append((b, torch.stack(Qs).mean(0), torch.stack(Zs).mean(0), per_query))
        return out

    def _aux_losses(self, targets, band_routing, owners, far, rows):
        device = far.device
        weights = 1.0 + self.csrd_anchor_beta * targets["anchor"].to(device)
        C_tilde = support = None
        if "C" in targets:
            C_tilde, support = causal_target(targets["C"].to(device), targets["J"].to(device), far)
        L_route = L_mass = L_causal = torch.zeros((), device=device)
        stats = defaultdict(float)
        for b, Q, Z_S, per_query in band_routing:
            P, Z_T = targets["P"][b].to(device), targets["Z"][b].to(device)
            if self.csrd_loss_form == "per_query":
                L_route = L_route + torch.stack([per_query_route_loss(P, r, owners, rows, weights) for r in per_query]).mean()
            else:
                L_route = L_route + route_loss(P, Q, rows, weights)
            L_mass = L_mass + mass_loss(Z_T, Z_S, rows, weights)
            if C_tilde is not None:
                L_causal = L_causal + causal_loss(C_tilde, Q, support)
            with torch.no_grad():
                stats[f"csrd_zT_b{b}"] = float(Z_T[rows].mean())
                stats[f"csrd_zS_b{b}"] = float(Z_S[rows].mean())
        return L_route, L_mass, L_causal, stats

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
        positions, owners = positions.to(device), owners.to(device)

        lam = self.current_lambda()
        collect_stats = self.student_heads is None and self.csrd_head_mode == "receiver"
        if collect_stats and lam > 0:
            # warmup ended (or was empty): select from what was collected, else from this batch first
            if self._score_count == 0:
                with torch.no_grad():
                    nu = self._arm_head_stats(key_nodes, far, positions, owners)
                    model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), logits_to_keep=1)
                    self._finish_head_stats(nu)
            self._select_student_heads()
            collect_stats = False

        compute_aux = lam > 0 and positions.numel() > 0
        nu = None
        if collect_stats:
            nu = self._arm_head_stats(key_nodes, far, positions, owners)
        elif compute_aux:
            wanted = self._selected_by_layer()
            self._slot = {layer: {h: s for s, h in enumerate(heads)} for layer, heads in wanted.items()}
            if self._input_capture is not None:
                self._input_capture.arm(wanted.keys())
            else:
                self._capture.arm(wanted, positions, detach=False)

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
        if compute_aux:
            scale = next(iter(self.modules_by_layer.values())).scaling
            if self._input_capture is not None:
                sample_q, sample_k = self._recomputed_qk(wanted, positions)
            else:
                sample_q, sample_k = self._capture.q, self._capture.k
            band_routing = self._band_routing(sample_q, sample_k, positions, owners, key_nodes, far, num_nodes, scale)
            L_route, L_mass, L_causal, stats = self._aux_losses(targets, band_routing, owners, far, rows)
            aux = lam * (self.csrd_route_ratio * L_route + self.csrd_mass_ratio * L_mass
                         + self.csrd_causal_ratio * L_causal)
            # loss_ce is already a share of the step-level sum/Z; the auxiliary term is a per-sequence
            # mean, so divide by the accumulation count or lambda would scale with it.
            aux_scaled = aux / accumulation
            if self._should_log_grads():
                self._log_grad_stats(loss_ce, aux_scaled)
            if self.csrd_qk_adapter:
                grads = torch.autograd.grad(aux_scaled, self._qk_params, allow_unused=True)
                # accelerator.backward divides the returned loss by its accumulation count; the replaced
                # gradient bypasses that, so apply it here to keep the shared-adapter scaling of L_route.
                ga = self.accelerator.gradient_accumulation_steps
                self._qk_pending = {id(p): (g.detach() / ga if g is not None else None)
                                    for p, g in zip(self._qk_params, grads)}
                self._qk_replace = True
            else:
                loss = loss_ce + aux_scaled
            values.update({"loss_route": float(L_route.detach()), "loss_mass": float(L_mass.detach()),
                           "loss_causal": float(L_causal.detach()), **stats})
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
        """CSRD-QK: q/k recomputed from the *detached* attention inputs, so routing gradients reach only the
        same layer's projections (and, via the grad hook, only its Q/K adapter)."""
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

    def _log_grad_stats(self, loss_ce, aux_scaled):
        """||grad CE||, ||grad route|| and cos(grad CE, grad route) on the shared LoRA parameters (Sec. 4.7:
        a persistently negative cosine means switching to CSRD-QK or PCGrad)."""
        params = self._shared_params

        def grads_of(loss):
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).float().flatten() for g, p in zip(grads, params)])

        try:
            with self._backward_hooks_muted():
                g_ce = grads_of(loss_ce)
                g_route = grads_of(aux_scaled) if aux_scaled.requires_grad else torch.zeros_like(g_ce)
        except Exception as exc:  # a diagnostic is never worth taking the run down for
            self.csrd_grad_log_interval = 0
            if self.is_world_process_zero():
                print(f"grad diagnostic failed ({type(exc).__name__}: {exc}); disabled")
            return
        n_ce, n_route = float(g_ce.norm()), float(g_route.norm())
        cos = float(g_ce @ g_route) / max(n_ce * n_route, 1e-20)
        for key, value in (("grad_ce_lora", n_ce), ("grad_route_lora", n_route), ("grad_cos_ce_route", cos),
                           ("grad_route_ratio", n_route / max(n_ce, 1e-20))):
            self._metric_sums[key] += value
            self._metric_counts[key] += 1

    def log(self, logs, *args, **kwargs):
        for key, total in self._metric_sums.items():
            logs[key] = round(total / max(1, self._metric_counts[key]), 6)
        self._metric_sums.clear()
        self._metric_counts.clear()
        super().log(logs, *args, **kwargs)
