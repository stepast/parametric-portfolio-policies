from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch


def build_value_weight_benchmark(df_month: pd.DataFrame, me_col: str = "me") -> np.ndarray:
    me = df_month[me_col].to_numpy(dtype=np.float64)
    me = np.where(np.isfinite(me) & (me > 0), me, 0.0)
    s = me.sum()
    if s <= 0:
        n = len(me)
        return np.ones(n, dtype=np.float64) / max(n, 1)
    return me / s


def _normalize_simplex(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize so sum(w)=1. Falls back to equal weights if sum is bad."""
    s = w.sum()
    if torch.isfinite(s) and s.abs() > eps:
        return w / s
    return torch.ones_like(w) / max(int(w.numel()), 1)


def scores_to_weights(
    scores: torch.Tensor,
    w_bench: torch.Tensor,
    *,
    mode: Literal["long_only_softmax", "long_short_tilt"] = "long_only_softmax",
    short_budget: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    scores = scores.view(-1)
    w_bench = _normalize_simplex(w_bench.view(-1), eps=eps)

    n = int(scores.numel())
    if n == 0:
        return w_bench

    if mode == "long_only_softmax":
        logits = torch.log(torch.clamp(w_bench, min=eps)) + scores
        logits = logits - logits.max()
        return torch.softmax(logits, dim=0)

    if mode == "long_short_tilt":
        tilt = (scores - scores.mean()) / float(n)
        w = w_bench + tilt  # sums to 1 automatically

        if short_budget is None:
            return w
        if short_budget <= 0:
            raise ValueError("short_budget must be positive or None.")

        short_b = (-w_bench).clamp_min(0).sum()
        if float(short_b) > float(short_budget) + 1e-12:
            raise ValueError(
                f"short_budget={short_budget} is below benchmark short side="
                f"{float(short_b):.6f}. Infeasible."
            )

        short_w = (-w).clamp_min(0).sum()
        denom = (short_w - short_b).clamp_min(1e-12)
        alpha = ((short_budget - short_b) / denom).clamp(0.0, 1.0)
        binds = (short_w > short_budget).to(alpha.dtype)
        alpha = binds * alpha + (1.0 - binds) * torch.ones_like(alpha)
        return w_bench + alpha * tilt

    raise ValueError(f"Unknown mode: {mode!r}")


def portfolio_return(weights: torch.Tensor, asset_returns: torch.Tensor) -> torch.Tensor:
    w = weights.view(-1)
    r = asset_returns.view(-1)
    return (w * r).sum()


def _safe_portfolio_return(
    weights: torch.Tensor,
    asset_returns: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Portfolio return that renormalizes over names with finite returns.

    For names with NaN return, weight is redistributed pro-rata across the
    valid universe rather than scoring those positions at 0%. Stays in-graph
    via differentiable indexing / sum.
    """
    w = weights.view(-1)
    r = asset_returns.view(-1)
    valid = torch.isfinite(r)
    if bool(valid.all()):
        return (w * r).sum()
    w_v = w * valid.to(w.dtype)
    denom = w_v.sum().clamp_min(eps)
    r_safe = torch.where(valid, r, torch.zeros_like(r))
    return (w_v * r_safe).sum() / denom


def turnover_by_id(
    prev_ids: np.ndarray,
    prev_w: torch.Tensor,
    ids: np.ndarray,
    w: torch.Tensor,
) -> torch.Tensor:
    prev_ids = np.asarray(prev_ids)
    ids = np.asarray(ids)

    _, prev_idx, curr_idx = np.intersect1d(
        prev_ids, ids, return_indices=True, assume_unique=False
    )
    prev_only_mask = np.ones(prev_ids.shape[0], dtype=bool)
    prev_only_mask[prev_idx] = False
    curr_only_mask = np.ones(ids.shape[0], dtype=bool)
    curr_only_mask[curr_idx] = False

    to = torch.zeros((), dtype=w.dtype, device=w.device)
    if len(prev_idx):
        prev_idx_t = torch.as_tensor(prev_idx, dtype=torch.long, device=w.device)
        curr_idx_t = torch.as_tensor(curr_idx, dtype=torch.long, device=w.device)
        to = to + (w[curr_idx_t] - prev_w[prev_idx_t]).abs().sum()
    if np.any(prev_only_mask):
        prev_only_idx_t = torch.as_tensor(
            np.flatnonzero(prev_only_mask), dtype=torch.long, device=w.device
        )
        to = to + prev_w[prev_only_idx_t].abs().sum()
    if np.any(curr_only_mask):
        curr_only_idx_t = torch.as_tensor(
            np.flatnonzero(curr_only_mask), dtype=torch.long, device=w.device
        )
        to = to + w[curr_only_idx_t].abs().sum()
    return to


def transaction_cost_by_id(
    prev_ids: np.ndarray,
    prev_w: torch.Tensor,
    prev_tc_oneway: torch.Tensor,
    ids: np.ndarray,
    w: torch.Tensor,
    tc_oneway: torch.Tensor,
) -> torch.Tensor:
    prev_ids = np.asarray(prev_ids)
    ids = np.asarray(ids)

    _, prev_idx, curr_idx = np.intersect1d(
        prev_ids, ids, return_indices=True, assume_unique=False
    )
    prev_only_mask = np.ones(prev_ids.shape[0], dtype=bool)
    prev_only_mask[prev_idx] = False
    curr_only_mask = np.ones(ids.shape[0], dtype=bool)
    curr_only_mask[curr_idx] = False

    tc = torch.zeros((), dtype=w.dtype, device=w.device)
    if len(prev_idx):
        prev_idx_t = torch.as_tensor(prev_idx, dtype=torch.long, device=w.device)
        curr_idx_t = torch.as_tensor(curr_idx, dtype=torch.long, device=w.device)
        tc = tc + ((w[curr_idx_t] - prev_w[prev_idx_t]).abs() * tc_oneway[curr_idx_t]).sum()
    if np.any(prev_only_mask):
        prev_only_idx_t = torch.as_tensor(
            np.flatnonzero(prev_only_mask), dtype=torch.long, device=w.device
        )
        tc = tc + (prev_w[prev_only_idx_t].abs() * prev_tc_oneway[prev_only_idx_t]).sum()
    if np.any(curr_only_mask):
        curr_only_idx_t = torch.as_tensor(
            np.flatnonzero(curr_only_mask), dtype=torch.long, device=w.device
        )
        tc = tc + (w[curr_only_idx_t].abs() * tc_oneway[curr_only_idx_t]).sum()
    return tc


@dataclass
class MonthlyBatch:
    date: object
    label_date: object | None
    ids: np.ndarray
    X: torch.Tensor
    r_next: torch.Tensor
    r_exc_next: torch.Tensor
    w_bench: torch.Tensor
    tc_oneway: torch.Tensor | None = None


def run_model_on_batches(
    model,
    batches: list[MonthlyBatch],
    policy_mode: str = "long_only_softmax",
    short_budget: float | None = None,
    compute_turnover: bool = True,
    transaction_cost_multiplier: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    r_list: list[torch.Tensor] = []
    r_exc_list: list[torch.Tensor] = []
    to_list: list[torch.Tensor] = []
    tc_list: list[torch.Tensor] = []

    prev_ids: np.ndarray | None = None
    prev_w: torch.Tensor | None = None
    prev_tc_oneway: torch.Tensor | None = None

    for b in batches:
        scores = model(b.X).view(-1)
        scores = scores - scores.mean()
        w = scores_to_weights(
            scores,
            b.w_bench,
            mode=policy_mode,
            short_budget=short_budget,
        )

        r_t = _safe_portfolio_return(w, b.r_next)
        r_list.append(r_t)
        r_exc_list.append(_safe_portfolio_return(w, b.r_exc_next))

        if compute_turnover:
            if prev_w is not None and prev_ids is not None:
                to_list.append(turnover_by_id(prev_ids, prev_w, b.ids, w))
                if (
                    transaction_cost_multiplier > 0.0
                    and b.tc_oneway is not None
                    and prev_tc_oneway is not None
                ):
                    tc = transaction_cost_by_id(
                        prev_ids, prev_w, prev_tc_oneway, b.ids, w, b.tc_oneway
                    )
                    tc_list.append(float(transaction_cost_multiplier) * tc)
            valid = torch.isfinite(b.r_next)
            r_safe = torch.where(valid, b.r_next, torch.zeros_like(b.r_next))
            drift_factor = torch.where(
                valid,
                (1.0 + r_safe) / (1.0 + r_t),
                torch.ones_like(r_safe),
            )
            prev_ids = b.ids
            prev_w = w * drift_factor
            prev_tc_oneway = b.tc_oneway

    device = batches[0].X.device if len(batches) else torch.device("cpu")
    r_p = torch.stack(r_list) if len(r_list) else torch.empty(0, device=device)
    r_p_exc = torch.stack(r_exc_list) if len(r_exc_list) else torch.empty(0, device=device)

    if compute_turnover:
        turnover = torch.stack(to_list) if len(to_list) else torch.empty(0, dtype=torch.float32, device=device)
        tc_series = torch.stack(tc_list) if len(tc_list) else torch.empty(0, dtype=torch.float32, device=device)
        return r_p, r_p_exc, turnover, tc_series

    return r_p, r_p_exc, None, None


def run_ensemble_on_batches(
    models: list[torch.nn.Module],
    batches: list[MonthlyBatch],
    *,
    policy_mode: str,
    short_budget: float | None,
    compute_turnover: bool = True,
    transaction_cost_multiplier: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(models) == 0:
        raise ValueError("models must be a non-empty list")

    for m in models:
        m.eval()

    r_list: list[torch.Tensor] = []
    r_exc_list: list[torch.Tensor] = []
    to_list: list[torch.Tensor] = []
    tc_list: list[torch.Tensor] = []

    prev_ids: np.ndarray | None = None
    prev_w: torch.Tensor | None = None
    prev_tc_oneway: torch.Tensor | None = None

    for b in batches:
        score_sum: torch.Tensor | None = None
        for m in models:
            s = m(b.X).view(-1)
            score_sum = s if score_sum is None else (score_sum + s)
        scores = score_sum / float(len(models))
        scores = scores - scores.mean()

        w = scores_to_weights(
            scores,
            b.w_bench,
            mode=policy_mode,
            short_budget=short_budget,
        )

        r_t = _safe_portfolio_return(w, b.r_next)
        r_list.append(r_t)
        r_exc_list.append(_safe_portfolio_return(w, b.r_exc_next))

        if compute_turnover:
            if prev_w is not None and prev_ids is not None:
                to_list.append(turnover_by_id(prev_ids, prev_w, b.ids, w))
                if (
                    transaction_cost_multiplier > 0.0
                    and b.tc_oneway is not None
                    and prev_tc_oneway is not None
                ):
                    tc = transaction_cost_by_id(
                        prev_ids, prev_w, prev_tc_oneway, b.ids, w, b.tc_oneway
                    )
                    tc_list.append(float(transaction_cost_multiplier) * tc)
            valid = torch.isfinite(b.r_next)
            r_safe = torch.where(valid, b.r_next, torch.zeros_like(b.r_next))
            drift_factor = torch.where(
                valid,
                (1.0 + r_safe) / (1.0 + r_t),
                torch.ones_like(r_safe),
            )
            prev_ids = b.ids
            prev_w = w * drift_factor
            prev_tc_oneway = b.tc_oneway

    device = batches[0].X.device if len(batches) else torch.device("cpu")
    r_p = torch.stack(r_list) if len(r_list) else torch.empty(0, device=device)
    r_p_exc = torch.stack(r_exc_list) if len(r_exc_list) else torch.empty(0, device=device)

    if compute_turnover:
        turnover = torch.stack(to_list) if len(to_list) else torch.empty(0, dtype=torch.float32, device=device)
        tc_series = torch.stack(tc_list) if len(tc_list) else torch.empty(0, dtype=torch.float32, device=device)
        return r_p, r_p_exc, turnover, tc_series

    return r_p, r_p_exc, None, None
