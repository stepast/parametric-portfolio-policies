from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np
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

    prev_w_np = prev_w.detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()

    prev_map = {int(i): float(v) for i, v in zip(prev_ids, prev_w_np)}
    curr_map = {int(i): float(v) for i, v in zip(ids, w_np)}

    all_ids = set(prev_map.keys()) | set(curr_map.keys())
    to = 0.0
    for i in all_ids:
        to += abs(curr_map.get(i, 0.0) - prev_map.get(i, 0.0))

    return torch.tensor(to, dtype=torch.float32, device=w.device)


@dataclass
class MonthlyBatch:
    date: object             
    ids: np.ndarray          
    X: torch.Tensor          
    r_next: torch.Tensor     
    r_exc_next: torch.Tensor 
    w_bench: torch.Tensor    


def run_model_on_batches(
    model,
    batches: list[MonthlyBatch],
    policy_mode: str = "long_only_softmax",
    gross_leverage: float | None = None,
    compute_turnover: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Shared evaluator for any score model.

    Returns:
      r_p:      (T,)   portfolio raw returns
      r_p_exc:  (T,)   portfolio excess returns (for Sharpe using ret_exc_lead1m)
      turnover: (T-1,) if compute_turnover else None
    """
    r_list: list[torch.Tensor] = []
    r_exc_list: list[torch.Tensor] = []
    to_list: list[torch.Tensor] = []

    prev_ids: np.ndarray | None = None
    prev_w: torch.Tensor | None = None

    for b in batches:
        scores = model(b.X)
        scores = scores.view(-1)  # force (N,) to avoid broadcasting surprises

        scores = scores - scores.mean()        
        w = scores_to_weights(
            scores,
            b.w_bench,
            mode=policy_mode,              # "long_only_softmax" or "long_short_tilt"
            gross_leverage=gross_leverage, # float or None
        )

        r_list.append(portfolio_return(w, b.r_next))
        r_exc_list.append(portfolio_return(w, b.r_exc_next))

        if compute_turnover:
            if prev_w is not None and prev_ids is not None:
                to_list.append(turnover_by_id(prev_ids, prev_w, b.ids, w))
            prev_ids = b.ids
            prev_w = w

    device = batches[0].X.device if len(batches) else torch.device("cpu")

    r_p = torch.stack(r_list) if len(r_list) else torch.empty(0, device=device)
    r_p_exc = torch.stack(r_exc_list) if len(r_exc_list) else torch.empty(0, device=device)

    if compute_turnover:
        turnover = torch.stack(to_list) if len(to_list) else torch.empty(0, dtype=torch.float32, device=device)
        return r_p, r_p_exc, turnover

    return r_p, r_p_exc, None

def run_ensemble_on_batches(
    models: list[torch.nn.Module],
    batches: list["MonthlyBatch"],
    *,
    policy_mode: str,
    gross_leverage: float | None,
    compute_turnover: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Evaluate an ensemble by averaging scores across members, then mapping once to weights.

    """
    if len(models) == 0:
        raise ValueError("models must be a non-empty list")

    # put all models in eval mode (caller may also do this)
    for m in models:
        m.eval()

    r_list: list[torch.Tensor] = []
    r_exc_list: list[torch.Tensor] = []
    to_list: list[torch.Tensor] = []

    prev_ids: np.ndarray | None = None
    prev_w: torch.Tensor | None = None

    for b in batches:
        # ---- average scores across members ----
        score_sum: torch.Tensor | None = None
        for m in models:
            s = m(b.X).view(-1)
            score_sum = s if score_sum is None else (score_sum + s)
        scores = score_sum / float(len(models))

        # Keep consistent with your single-model evaluation:
        scores = scores - scores.mean()

        w = scores_to_weights(
            scores,
            b.w_bench,
            mode=policy_mode,
            gross_leverage=gross_leverage,
        )

        r_list.append(portfolio_return(w, b.r_next))
        r_exc_list.append(portfolio_return(w, b.r_exc_next))

        if compute_turnover:
            if prev_w is not None and prev_ids is not None:
                to_list.append(turnover_by_id(prev_ids, prev_w, b.ids, w))
            prev_ids = b.ids
            prev_w = w

    device = batches[0].X.device if len(batches) else torch.device("cpu")

    r_p = torch.stack(r_list) if len(r_list) else torch.empty(0, device=device)
    r_p_exc = torch.stack(r_exc_list) if len(r_exc_list) else torch.empty(0, device=device)

    if compute_turnover:
        turnover = torch.stack(to_list) if len(to_list) else torch.empty(0, dtype=torch.float32, device=device)
        return r_p, r_p_exc, turnover

    return r_p, r_p_exc, None
