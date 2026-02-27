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
    gross_leverage: float | None = None,
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

        if gross_leverage is None:
            return w

        L = gross_leverage
        if L <= 0:
            raise ValueError("gross_leverage must be positive or None.")

        gross_b = w_bench.abs().sum().item()
        if L < gross_b - 1e-12:
            raise ValueError(f"gross_leverage={L} is below benchmark gross={gross_b:.6f}. Infeasible.")

        gross = w.abs().sum().item()
        if gross <= L + 1e-12:
            return w

        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            w_mid = w_bench + mid * tilt
            g_mid = w_mid.abs().sum().item()
            if g_mid > L:
                hi = mid
            else:
                lo = mid

        return w_bench + lo * tilt

    raise ValueError(f"Unknown mode: {mode!r}")


def portfolio_return(weights: torch.Tensor, asset_returns: torch.Tensor) -> torch.Tensor:
    w = weights.view(-1)
    r = asset_returns.view(-1)
    return (w * r).sum()


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
    gross_leverage: float | None = None,
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
            gross_leverage=gross_leverage,
        )

        valid_mask = torch.isfinite(b.r_next) & torch.isfinite(b.r_exc_next)
        if valid_mask.all():
            r_list.append(portfolio_return(w, b.r_next))
            r_exc_list.append(portfolio_return(w, b.r_exc_next))
        else:
            r_next_eff = torch.where(valid_mask, b.r_next, torch.zeros_like(b.r_next))
            r_exc_eff = torch.where(valid_mask, b.r_exc_next, torch.zeros_like(b.r_exc_next))
            r_list.append(portfolio_return(w, r_next_eff))
            r_exc_list.append(portfolio_return(w, r_exc_eff))

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
            prev_ids = b.ids
            prev_w = w
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
    gross_leverage: float | None,
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
            gross_leverage=gross_leverage,
        )

        valid_mask = torch.isfinite(b.r_next) & torch.isfinite(b.r_exc_next)
        if valid_mask.all():
            r_list.append(portfolio_return(w, b.r_next))
            r_exc_list.append(portfolio_return(w, b.r_exc_next))
        else:
            r_next_eff = torch.where(valid_mask, b.r_next, torch.zeros_like(b.r_next))
            r_exc_eff = torch.where(valid_mask, b.r_exc_next, torch.zeros_like(b.r_exc_next))
            r_list.append(portfolio_return(w, r_next_eff))
            r_exc_list.append(portfolio_return(w, r_exc_eff))

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
            prev_ids = b.ids
            prev_w = w
            prev_tc_oneway = b.tc_oneway

    device = batches[0].X.device if len(batches) else torch.device("cpu")
    r_p = torch.stack(r_list) if len(r_list) else torch.empty(0, device=device)
    r_p_exc = torch.stack(r_exc_list) if len(r_exc_list) else torch.empty(0, device=device)

    if compute_turnover:
        turnover = torch.stack(to_list) if len(to_list) else torch.empty(0, dtype=torch.float32, device=device)
        tc_series = torch.stack(tc_list) if len(tc_list) else torch.empty(0, dtype=torch.float32, device=device)
        return r_p, r_p_exc, turnover, tc_series

    return r_p, r_p_exc, None, None
