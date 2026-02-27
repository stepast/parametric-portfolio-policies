from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .portfolio import build_value_weight_benchmark, MonthlyBatch


def make_monthly_batches(
    df_split: pd.DataFrame,
    X_split: np.ndarray,
    date_col: str = "eom",
    label_date_col: str | None = None,
    id_col: str = "id",
    ret_next_col: str = "ret_lead1m",
    ret_exc_next_col: str = "ret_exc_lead1m",
) -> list[MonthlyBatch]:
    if len(df_split) != X_split.shape[0]:
        raise ValueError("df_split and X_split must have the same number of rows.")

    # Sort entire split by (date, id) once; keep alignment with X_split
    order = np.lexsort((df_split[id_col].to_numpy(), df_split[date_col].to_numpy()))
    df_split = df_split.iloc[order].reset_index(drop=True)
    X_split = X_split[order]

    batches: list[MonthlyBatch] = []

    for date, g in df_split.groupby(date_col, sort=True):
        # Since df_split is already sorted by (date, id), g is in correct order.
        idx = g.index.to_numpy()          
        X_t = torch.as_tensor(X_split[idx], dtype=torch.float32)

        r_next = g[ret_next_col].to_numpy(dtype=np.float32)
        r_exc_next = g[ret_exc_next_col].to_numpy(dtype=np.float32)

        r_next_t = torch.as_tensor(r_next, dtype=torch.float32)
        r_exc_next_t = torch.as_tensor(r_exc_next, dtype=torch.float32)

        w_b = np.asarray(build_value_weight_benchmark(g), dtype=np.float32)
        s = float(w_b.sum()) if len(w_b) else 0.0
        if len(w_b) and abs(s - 1.0) > 1e-4:
            raise ValueError(f"Benchmark weights do not sum to 1 for month {date}: sum={s:.6f}")

        w_b_t = torch.as_tensor(w_b, dtype=torch.float32)

        ids = g[id_col].to_numpy()

        label_date = None
        if label_date_col is not None and label_date_col in g.columns:
            label_vals = g[label_date_col].to_numpy()
            label_date = label_vals[0] if len(label_vals) else None
            if len(label_vals) and not np.all(label_vals == label_vals[0]):
                raise ValueError(f"Non-constant label dates within month {date}.")

        batches.append(MonthlyBatch(
            date=date,
            label_date=label_date,
            ids=ids,
            X=X_t,
            r_next=r_next_t,
            r_exc_next=r_exc_next_t,
            w_bench=w_b_t,
        ))

    return batches

def benchmark_vw_excess_series(
    df: pd.DataFrame,
    date_col: str = "eom",
    me_col: str = "me",
    ret_exc_col: str = "ret_exc_lead1m",
) -> pd.Series:
    """
    Computes the value-weighted benchmark's monthly excess return series:
      r_bench_t = sum_i w_it * ret_exc_it
    where w_it = me_it / sum_j me_j within month t.
    """
    d = df[[date_col, me_col, ret_exc_col]].copy()

    # keep only usable rows
    d = d.dropna(subset=[date_col, me_col, ret_exc_col])
    d = d[np.isfinite(d[me_col]) & np.isfinite(d[ret_exc_col])]
    d = d[d[me_col] > 0]

    # compute month sums and weights
    me_sum = d.groupby(date_col)[me_col].transform("sum")
    w = d[me_col] / me_sum

    # monthly VW excess return
    r_bench = (w * d[ret_exc_col]).groupby(d[date_col]).sum()
    r_bench.name = "vw_benchmark_excess"
    return r_bench.sort_index()


def sharpe_ratio_from_excess(monthly_excess: np.ndarray, eps: float = 1e-12) -> float:
    """Compute annualized Sharpe ratio from a monthly excess return series."""
    m = float(np.mean(monthly_excess))
    s = float(np.std(monthly_excess, ddof=0))
    return (m / max(s, eps)) * np.sqrt(12.0)

def _ann_mean(r: np.ndarray) -> float:
    return float(np.mean(r) * 12.0)

def _ann_vol(r: np.ndarray) -> float:
    return float(np.std(r, ddof=0) * np.sqrt(12.0))


def _compute_overall_metrics(
    r: np.ndarray,
    r_exc: np.ndarray,
    turnover: np.ndarray | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out["ann_mean"] = _ann_mean(r)
    out["ann_vol"] = _ann_vol(r)
    out["ann_sharpe_excess"] = sharpe_ratio_from_excess(r_exc)
    out["mean_turnover"] = np.mean(turnover) if turnover is not None and turnover.size else float("nan")
    out["n_oos_months"] = r.shape[0]
    return out
