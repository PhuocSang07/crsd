"""Step routing: node mass, far-normalized routing, CSRD losses and gap metrics (Sec. 4.2, 4.5, 5).

For a head (l, h) and a query token t of step i (Definition 1):

    m_t(j)  = sum_{u in I(v_j)} A_{t,u}                   attention mass of t on node j
    F(i)    = {0} U {j : 1 <= j <= i - d_min}              far targets of step i
    Z_t     = sum_{j in F(i)} m_t(j),   R_t(j) = m_t(j) / Z_t   (j in F(i))

Rows average *already normalized* per-query distributions, R[i] = mean_{t in Q(i)} R_t, which is
what makes Lemma 1 exact per query. Only rows with |F(i)| >= 2 enter any loss.

Framework-agnostic torch (no model, no Trainer) so every formula is unit-testable on CPU.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

EPS = 1e-12


def _full(x: torch.Tensor) -> torch.Tensor:
    """Upcast bf16/fp16 to fp32; leave fp32/fp64 alone (fp64 keeps gradcheck meaningful)."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()
# Distance bins of Definition 2, in steps; the last one is open-ended.
DISTANCE_BINS = ((4, 8), (8, 16), (16, 32), (32, 64), (64, math.inf))


def far_target_mask(num_nodes: int, d_min: int, device=None) -> torch.Tensor:
    """[N, N] bool, row i marks F(i). Node 0 (q) is always far; the answer node is a row like a step."""
    i = torch.arange(num_nodes, device=device).unsqueeze(1)
    j = torch.arange(num_nodes, device=device).unsqueeze(0)
    mask = (j == 0) | ((j >= 1) & (j <= i - d_min))
    mask[0] = False  # the question is never a query row
    return mask


def valid_rows(far_mask: torch.Tensor) -> torch.Tensor:
    """[N] bool: rows with at least two far targets."""
    return far_mask.sum(-1) >= 2


def node_mass(attn: torch.Tensor, key_nodes: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """[..., Nq, T] attention probabilities -> [..., Nq, N] mass per node (keys in no node dropped)."""
    keep = key_nodes >= 0
    mass = attn.new_zeros((*attn.shape[:-1], num_nodes))
    return mass.index_add(attn.dim() - 1, key_nodes[keep], attn[..., keep])


def query_routing(
    mass: torch.Tensor, query_rows: torch.Tensor, far_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query far mass Z_t [..., Nq] and far-normalized routing R_t [..., Nq, N] (zero outside F(i))."""
    far = far_mask[query_rows].to(mass.dtype)
    far_mass = mass * far
    z = far_mass.sum(-1)
    return far_mass / z.clamp_min(EPS).unsqueeze(-1), z


def row_average(values: torch.Tensor, query_rows: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of per-query `values` [Nq, ...] over each row's queries -> ([N, ...], counts [N])."""
    sums = values.new_zeros((num_nodes, *values.shape[1:])).index_add(0, query_rows, values)
    counts = torch.bincount(query_rows, minlength=num_nodes).to(values.dtype)
    shape = (num_nodes,) + (1,) * (values.dim() - 1)
    return sums / counts.clamp_min(1).view(shape), counts


def attention_probs(
    q: torch.Tensor, k: torch.Tensor, query_pos: torch.Tensor, scale: float
) -> torch.Tensor:
    """Exact causal softmax(q k^T * scale) of queries [..., Nq, d] over all keys [..., T, d], fp32.

    Autocast is switched off: the warmup head statistics run inside the model forward, which
    accelerate wraps in bf16 autocast, and a bf16 q k^T would round the logits more coarsely than
    the attention kernel itself (bf16 inputs, fp32 accumulation) does.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (_full(q) @ _full(k).transpose(-1, -2)) * scale
        key_pos = torch.arange(k.size(-2), device=k.device)
        scores = scores.masked_fill(key_pos.unsqueeze(0) > query_pos.unsqueeze(1), float("-inf"))
        return torch.softmax(scores, dim=-1)


def _head_query_routing(q, k, query_pos, key_nodes, query_rows, far_mask, scale, num_nodes: int):
    attn = attention_probs(q, k, query_pos, scale)
    return query_routing(node_mass(attn, key_nodes, num_nodes), query_rows, far_mask)


def head_query_routing(
    q: torch.Tensor,
    k: torch.Tensor,
    query_pos: torch.Tensor,
    key_nodes: torch.Tensor,
    query_rows: torch.Tensor,
    far_mask: torch.Tensor,
    scale: float,
    use_checkpoint: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(R_t [Nq, N], Z_t [Nq]) of one head for sampled queries, differentiable in q and k.

    use_checkpoint recomputes the [Nq, T] score matrix in backward instead of storing it, so the
    student's autograd graph holds only q/k per head (~150 MB/head at 2.4k queries x 32k keys).
    """
    num_nodes = far_mask.size(0)
    args = (q, k, query_pos, key_nodes, query_rows, far_mask, scale, num_nodes)
    if use_checkpoint and torch.is_grad_enabled() and (q.requires_grad or k.requires_grad):
        return checkpoint(_head_query_routing, *args, use_reentrant=False)
    return _head_query_routing(*args)


@torch.no_grad()
def head_row_routing_blockwise(
    q: torch.Tensor,
    k: torch.Tensor,
    key_nodes: torch.Tensor,
    far_mask: torch.Tensor,
    scale: float,
    block: int = 1024,
    query_nodes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Teacher-side routing over *every* query token of every row, block by block.

    q: [H, T, d] (post-RoPE) and k: [H, T, d] (already matched to the query heads), or [T, d]
    for a single head; the query set of row i is all of I(v_i) (Sec. 4.3). Never materializes
    T x T: each block of queries computes its exact full-row softmax, reduces it to node masses
    and is dropped (peak memory H x block x T fp32). Also returns each head's mean attention
    distance over those queries (D5: d_h = E_t sum_u (t - u) A_{t,u}).
    Returns R [H, N, N], Z [H, N], counts [N], mean_distance [H] (no H dim for 2-D input).
    """
    single = q.dim() == 2
    if single:
        q, k = q.unsqueeze(0), k.unsqueeze(0)
    heads = q.size(0)
    query_nodes = key_nodes if query_nodes is None else query_nodes
    num_nodes = far_mask.size(0)
    rows_ok = valid_rows(far_mask)
    positions = torch.nonzero((query_nodes >= 1) & rows_ok[query_nodes.clamp_min(0)], as_tuple=False).squeeze(-1)
    r_sum = q.new_zeros(heads, num_nodes, num_nodes, dtype=torch.float32)
    z_sum = q.new_zeros(heads, num_nodes, dtype=torch.float32)
    counts = q.new_zeros(num_nodes, dtype=torch.float32)
    dist_sum = q.new_zeros(heads, dtype=torch.float32)
    key_pos = torch.arange(k.size(-2), device=k.device, dtype=torch.float32)
    for start in range(0, positions.numel(), block):
        pos = positions[start : start + block]
        rows = query_nodes[pos]
        attn = attention_probs(q[:, pos], k, pos, scale)  # [H, b, T]
        r, z = query_routing(node_mass(attn, key_nodes, num_nodes), rows, far_mask)
        r_sum.index_add_(1, rows, r)
        z_sum.index_add_(1, rows, z)
        counts.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
        dist_sum += (pos.float().unsqueeze(0) - attn @ key_pos).sum(-1)
        del attn
    denom = counts.clamp_min(1)
    out = {
        "R": r_sum / denom.view(1, -1, 1),
        "Z": z_sum / denom.view(1, -1),
        "counts": counts,
        "mean_distance": dist_sum / max(1, positions.numel()),
    }
    if single:
        out.update(R=out["R"][0], Z=out["Z"][0], mean_distance=out["mean_distance"][0])
    return out


def _bool_rows(rows: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return rows.to(torch.bool) & (target.sum(-1) > 0)


def kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) along the last dim, 0 log 0 = 0, q clamped (the student can put ~0 mass)."""
    p, q = _full(p), _full(q)
    terms = torch.where(p > 0, p * (torch.log(p.clamp_min(EPS)) - torch.log(q.clamp_min(EPS))), torch.zeros_like(p))
    return terms.sum(-1)


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(Ber(p) || Ber(q)) elementwise."""
    p = _full(p).clamp(EPS, 1 - EPS)
    q = _full(q).clamp(EPS, 1 - EPS)
    return p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())


def weighted_row_mean(values: torch.Tensor, rows: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    w = rows.to(values.dtype) if weights is None else rows.to(values.dtype) * weights.to(values.dtype)
    return (values * w).sum() / w.sum().clamp_min(EPS)


def route_loss(P: torch.Tensor, Q: torch.Tensor, rows: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. 4 for one band: sum_i w_i KL(P_i || Q_i) / sum_i w_i over the given rows."""
    rows = _bool_rows(rows, P)
    return weighted_row_mean(kl_rows(P, Q), rows, weights)


def mass_loss(z_teacher: torch.Tensor, z_student: torch.Tensor, rows: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Eq. 5 for one band: far-mass matching, the term Lemma 1 (iii) says L_route cannot replace."""
    return weighted_row_mean(bernoulli_kl(z_teacher, z_student), rows.to(torch.bool), weights)


def restrict(dist: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Restrict row distributions to a boolean support ([N, N] or [N]) and renormalize."""
    kept = dist * support.to(dist.dtype)
    return kept / kept.sum(-1, keepdim=True).clamp_min(EPS)


def causal_target(C: torch.Tensor, J: torch.Tensor, far_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw suppression effects C [N, N] (NaN = not measured) -> (C~ row-normalized over F(i) & J, support)."""
    in_j = torch.zeros(C.size(-1), dtype=torch.bool, device=C.device)
    in_j[J] = True
    support = far_mask & in_j.unsqueeze(0) & torch.isfinite(C)
    values = torch.where(support, C.clamp_min(0), torch.zeros_like(C))
    support = support & (values.sum(-1, keepdim=True) > 0)
    return restrict(values, support), support


def causal_loss(c_tilde: torch.Tensor, Q: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Eq. 6 for one band: KL(C~_i || Q_i|J) over rows whose causal target has >= 2 entries."""
    support = support.to(torch.bool)
    rows = support.sum(-1) >= 2
    if not rows.any():
        return Q.sum() * 0.0
    return kl_rows(c_tilde[rows], restrict(Q[rows], support[rows])).mean()


def per_query_route_loss(
    P: torch.Tensor, R_queries: torch.Tensor, query_rows: torch.Tensor, rows: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CSRD-PQ (A14): E_t KL(P_i || R_t) per query -- an upper bound of Eq. 4 by convexity."""
    per_query = kl_rows(P[query_rows], R_queries)
    keep = _bool_rows(rows, P)[query_rows]
    w = keep.float() if weights is None else keep.float() * weights[query_rows].float()
    return (per_query * w).sum() / w.sum().clamp_min(EPS)


# ---------------------------------------------------------------- diagnostics (Sec. 5)

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_rows(p, m) + 0.5 * kl_rows(q, m)


def distance_bin_mask(num_nodes: int, low: float, high: float, device=None) -> torch.Tensor:
    """[N, N] bool: F_b(i) = {j >= 1 : low <= i - j < high} (node 0 excluded, Definition 2)."""
    i = torch.arange(num_nodes, device=device).unsqueeze(1)
    j = torch.arange(num_nodes, device=device).unsqueeze(0)
    distance = (i - j).float()
    return (j >= 1) & (distance >= low) & (distance < high)


def routing_gap_by_bin(
    P: torch.Tensor, Q: torch.Tensor, rows: torch.Tensor, bins=DISTANCE_BINS, far_mask: torch.Tensor | None = None
) -> list[dict]:
    """Per-bin RG(b) and Delta-mu(b) terms of one trace (sums + counts, so traces pool exactly).

    RG(b): mean over rows with |F_b(i)| >= 2 of JS(P_i|F_b || Q_i|F_b).
    Delta-mu(b): mean over rows with |F_b(i)| >= 1 of sum_{j in F_b(i)} (Q_i(j) - P_i(j)).
    F_b(i) is intersected with F(i) (far_mask), so with d_min > 4 (A1) the nearest bin is simply
    empty instead of contributing rows where P = Q = 0.
    """
    rows = _bool_rows(rows, P)
    out = []
    for low, high in bins:
        support = distance_bin_mask(P.size(0), low, high, P.device) & rows.unsqueeze(-1)
        if far_mask is not None:
            support = support & far_mask
        size = support.sum(-1)
        js_rows = size >= 2
        js = js_divergence(restrict(P[js_rows], support[js_rows]), restrict(Q[js_rows], support[js_rows]))
        mu_rows = size >= 1
        delta = ((Q - P) * support.float()).sum(-1)[mu_rows]
        out.append({
            "bin": [low, high],
            "rg_sum": float(js.sum()),
            "rg_count": int(js_rows.sum()),
            "dmu_sum": float(delta.sum()),
            "dmu_count": int(mu_rows.sum()),
        })
    return out
