"""Pure metric cores for the IM-LAM Stage-1 diagnostics.

Kept free of model / data / checkpoint loading so they unit-test on synthetic inputs. The three
diagnostics are built from these:

- Held-out object-prediction error (S7.4.1): ``object_prediction_error`` (= the training's
  area-normalized masked MSE, restricted to the object mask), the copy baseline, and their ratio.
- Object-dynamics probe (S7.4.2): ``fit_linear_probe`` / ``probe_scores`` (R², NMSE vs Var(Delta s^O)),
  and ``separation_ratio``.
- Agent-path dependence (S7.4.3) reuses ``object_prediction_error`` under different agent_ctx_modes.
"""
from typing import Dict

import torch
from torch import Tensor

from ifo.modules.slapo.utils import area_normalized_masked_mse

EPS = 1e-8


def object_prediction_error(pred: Tensor, target: Tensor, object_mask: Tensor) -> Tensor:
    """E_O: area-normalized masked MSE over the object region (training-consistent).

    Args:
        pred (Tensor): Predicted next observation ``(B, C, H, W)``.
        target (Tensor): True next observation ``(B, C, H, W)``.
        object_mask (Tensor): Object mask ``(B, 1, H, W)`` in ``{0, 1}``.
    """
    return area_normalized_masked_mse((pred - target) ** 2, object_mask)


def copy_baseline_error(current: Tensor, target: Tensor, object_mask: Tensor) -> Tensor:
    """E_O^copy: the error of the trivial copy predictor ``o_{t+1} = o_t`` over the object region.

    A model only demonstrates it has *learned object motion* by beating this - copying is free.
    """
    return object_prediction_error(current, target, object_mask)


def object_prediction_ratio(e_model: Tensor, e_copy: Tensor, eps: float = EPS) -> Tensor:
    """``E_O^model / (E_O^copy + eps)``. < 1 means the model beats copy; >= 1 means it does not."""
    return e_model / (e_copy + eps)


def masked_sq_error_and_area(pred: Tensor, target: Tensor, object_mask: Tensor):
    """Pooled-estimator building blocks: ``(sum masked squared error, sum mask area)`` for one batch.

    Accumulate both across batches and divide once at the end for the S7.4.1 **pooled** estimator
    ``E_O = sum_error / sum_area``. This weights each object pixel equally (as the proposal defines it),
    unlike averaging per-batch area-normalized errors, which would weight whole batches equally regardless
    of how much object was visible in each.
    """
    err = ((pred - target) ** 2) * object_mask
    return err.sum(), object_mask.sum()


def quat_geodesic_angle(q_a: Tensor, q_b: Tensor) -> Tensor:
    """Geodesic rotation angle (radians, in ``[0, pi]``) between unit quaternions ``q_a, q_b`` ``(..., 4)``.

    Orientation-delta magnitude for the object-dynamics-probe target: position is Euclidean, but
    quaternions are not, so a raw quaternion difference is not a meaningful regression target - the
    rotation angle is. ``|<q_a, q_b>|`` (absolute) treats ``q`` and ``-q`` as the same rotation.
    """
    q_a = q_a / q_a.norm(dim=-1, keepdim=True).clamp(min=EPS)
    q_b = q_b / q_b.norm(dim=-1, keepdim=True).clamp(min=EPS)
    dot = (q_a * q_b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.arccos(dot)


def fit_linear_probe(features: Tensor, targets: Tensor, ridge: float = 1e-3) -> Tensor:
    """Closed-form ridge regression ``features -> targets`` with a bias term.

    Args:
        features (Tensor): ``(N, d)`` probe inputs (e.g. pooled O_t).
        targets (Tensor): ``(N, k)`` regression targets (e.g. Delta s^O).
        ridge (float): L2 regularization on the weights (not the bias).
    Returns:
        Tensor ``(d + 1, k)``: fitted weights, last row is the bias.
    """
    n = features.shape[0]
    x = torch.cat([features, torch.ones(n, 1, dtype=features.dtype, device=features.device)], dim=1)
    d = x.shape[1]
    reg = ridge * torch.eye(d, dtype=x.dtype, device=x.device)
    reg[-1, -1] = 0.0  # do not regularize the bias
    w = torch.linalg.solve(x.T @ x + reg, x.T @ targets)
    return w


def probe_scores(features: Tensor, targets: Tensor, weights: Tensor) -> Dict[str, Tensor]:
    """Score a fitted probe: aggregate R², NMSE (vs Var(targets)), and Var(targets).

    R² and NMSE are pooled over all target dimensions, so a single scalar summarizes how much of the
    object motion the features linearly explain. NMSE is normalized by ``Var(targets)`` computed here from
    the evaluation targets - NOT by the expert *action* variance (a different quantity).
    """
    n = features.shape[0]
    x = torch.cat([features, torch.ones(n, 1, dtype=features.dtype, device=features.device)], dim=1)
    pred = x @ weights
    ss_res = ((targets - pred) ** 2).sum()
    ss_tot = ((targets - targets.mean(0, keepdim=True)) ** 2).sum()
    var = targets.var(0, unbiased=False).mean()
    return {
        "r2": 1.0 - ss_res / (ss_tot + EPS),
        "nmse": ((targets - pred) ** 2).mean() / (var + EPS),
        "var": var,
    }


def linear_probe_r2(
    features: Tensor,
    targets: Tensor,
    ridge: float = 1e-3,
    eval_features: Tensor = None,
    eval_targets: Tensor = None,
) -> Dict[str, Tensor]:
    """Fit a ridge probe on ``(features, targets)`` and score it (held-out if an eval set is given,
    else in-sample)."""
    w = fit_linear_probe(features, targets, ridge)
    ef = eval_features if eval_features is not None else features
    et = eval_targets if eval_targets is not None else targets
    return probe_scores(ef, et, w)


def separation_ratio(r2_agent: Tensor, r2_object: Tensor, eps: float = EPS) -> Tensor:
    """``R2(A_t -> Delta s^O) / (R2(O_t -> Delta s^O) + eps)``.

    Low = the object read-out predicts object motion far better than the agent read-out does (good entity
    separation - the object branch carries object-specific dynamics). ~1 = the agent read-out predicts it
    just as well (poor separation - the two read-outs are redundant).
    """
    return r2_agent / (r2_object + eps)
