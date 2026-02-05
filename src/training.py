from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .schemas import TrainConfig
from .splitting import iter_gkx_splits
from .backtest import make_monthly_batches, _ann_mean, _ann_vol, _compute_overall_metrics, sharpe_ratio_from_excess
from .portfolio import run_model_on_batches, run_ensemble_on_batches


# ---------------------------------------------------------------------
# Loss functions and regularization
# ---------------------------------------------------------------------

def mean_variance_loss(r: torch.Tensor, risk_aversion: float = 5.0) -> torch.Tensor:
    """Mean-variance objective: -E[r] + 0.5 * gamma * Var[r]"""
    mu = r.mean()
    var = r.var(unbiased=False)  # Population variance for stability
    return -mu + 0.5 * risk_aversion * var


def l1_penalty_no_bias(model: torch.nn.Module) -> torch.Tensor:
    """L1 penalty on weight matrices only (excludes biases)."""
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    
    penalty = torch.zeros((), device=device)
    for p in params:
        if p.ndim > 1:  # Weight matrices only
            penalty = penalty + p.abs().sum()
    return penalty


def l2_penalty_no_bias(model: torch.nn.Module) -> torch.Tensor:
    """L2 penalty on weight matrices only (excludes biases)."""
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    
    penalty = torch.zeros((), device=device)
    for p in params:
        if p.ndim > 1:  # Weight matrices only
            penalty = penalty + (p * p).sum()
    return penalty


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_split_policy(
    *,
    build_model: Callable[[], torch.nn.Module],
    train_batches,
    val_batches,
    cfg,
    seed: int = 0,
    train_verbose: bool = False,
    log_every: int = 1,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model()
    opt = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            )
    
    scheduler = None
    if getattr(cfg, "use_plateau_scheduler", False):
        thr = float(cfg.min_delta if cfg.plateau_threshold is None else cfg.plateau_threshold)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(cfg.plateau_factor),
            patience=int(cfg.plateau_patience),
            threshold=thr,
            min_lr=float(cfg.plateau_min_lr),
        )

    best_val_obj = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None

    bad_epochs = 0
    last_epoch = -1

    for epoch in range(int(cfg.epochs)):
        last_epoch = epoch

        # ----- TRAIN STEP -----
        model.train()
        opt.zero_grad(set_to_none=True)

        r_train, r_train_exc, _ = run_model_on_batches(
            model,
            train_batches,
            policy_mode=cfg.policy_mode,
            gross_leverage=cfg.gross_leverage,
            compute_turnover=False,
        )

        train_obj = mean_variance_loss(r_train, risk_aversion=float(cfg.risk_aversion))

        pen_l1 = torch.zeros((), device=r_train.device)
        pen_l2 = torch.zeros((), device=r_train.device)
        if cfg.l1 > 0.0:
            pen_l1 = cfg.l1 * l1_penalty_no_bias(model)
        if cfg.l2 > 0.0:
            pen_l2 = cfg.l2 * l2_penalty_no_bias(model)

        loss_total = train_obj + pen_l1 + pen_l2
        loss_total.backward()
        opt.step()

        # ----- VALIDATION -----
        model.eval()
        with torch.no_grad():
            r_val, r_val_exc, _ = run_model_on_batches(
                model,
                val_batches,
                policy_mode=cfg.policy_mode,
                gross_leverage=cfg.gross_leverage,
                compute_turnover=False,
            )
            val_obj_t = mean_variance_loss(r_val, risk_aversion=float(cfg.risk_aversion))
            val_obj = float(val_obj_t.detach().cpu().item())
            
            if scheduler is not None and np.isfinite(val_obj):
                scheduler.step(float(val_obj))

        # ----- LOGGING -----
        if train_verbose and (epoch % int(log_every) == 0 or epoch == int(cfg.epochs) - 1):
            r_tr_exc_np = r_train_exc.detach().cpu().numpy()
            r_val_exc_np = r_val_exc.detach().cpu().numpy()

            print(
                f"Epoch {epoch:03d} | "
                f"train_obj={float(train_obj.detach().cpu().item()):.6f} | "
                f"train_loss_total={float(loss_total.detach().cpu().item()):.6f} | "
                f"val_obj={val_obj:.6f} | "
                f"train_SR={sharpe_ratio_from_excess(r_tr_exc_np):.3f} | "
                f"val_SR={sharpe_ratio_from_excess(r_val_exc_np):.3f} | "
                f"bad_epochs={bad_epochs}"
            )

        # ----- EARLY STOPPING -----
        improved = (val_obj < best_val_obj - float(cfg.min_delta))
        if improved:
            best_val_obj = float(val_obj)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > int(cfg.patience):
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation metrics with best weights
    model.eval()
    with torch.no_grad():
        r_val, r_val_exc, _ = run_model_on_batches(
            model,
            val_batches,
            policy_mode=cfg.policy_mode,
            gross_leverage=cfg.gross_leverage,
            compute_turnover=False,
        )

    r_val_np = r_val.detach().cpu().numpy()
    r_val_exc_np = r_val_exc.detach().cpu().numpy()

    val_metrics = {
        "val_obj": float(best_val_obj),
        "val_ann_mean": _ann_mean(r_val_np),
        "val_ann_vol": _ann_vol(r_val_np),
        "val_ann_sharpe_excess": float(sharpe_ratio_from_excess(r_val_exc_np)),
        "best_epoch": int(best_epoch) if best_epoch is not None else -1,
        "last_epoch": int(last_epoch),
        "stopped_early": bool(best_state is not None and last_epoch < int(cfg.epochs) - 1),
    }
    return model, val_metrics


# ---------------------------------------------------------------------
# Tuning helper (small grid search)
# ---------------------------------------------------------------------

def tune_policy_hyperparams(
    *,
    build_model: Callable[[], torch.nn.Module],
    train_batches,
    val_batches,
    base_cfg,
    lr_grid: Sequence[float],
    l1_grid: Sequence[float],
    l2_grid: Sequence[float],
    seed: int,
    train_verbose: bool = False,
    log_every: int = 1,
) -> Tuple[object, Dict[str, float]]:
    """
    Grid search hyperparameters by MINIMIZING validation objective.

    """
    if len(lr_grid) == 0 or len(l1_grid) == 0 or len(l2_grid) == 0:
        raise ValueError("lr_grid, l1_grid, l2_grid must be non-empty when tuning.")

    best_cfg = base_cfg
    best_val_metrics: Dict[str, float] = {}
    best_val_obj = float("inf")

    trial = 0
    for lr in lr_grid:
        for l1 in l1_grid:
            for l2 in l2_grid:
                trial += 1

                # Preserve everything, override only tuned fields
                cfg_try = replace(base_cfg, lr=float(lr), l1=float(l1), l2=float(l2))

                model, val_metrics = train_one_split_policy(
                    build_model=build_model,
                    train_batches=train_batches,
                    val_batches=val_batches,
                    cfg=cfg_try,
                    seed=int(seed + trial),
                    train_verbose=train_verbose,
                    log_every=log_every,
                )
                del model

                val_obj = float(val_metrics["val_obj"])
                if val_obj < best_val_obj:
                    best_val_obj = val_obj
                    best_cfg = cfg_try
                    best_val_metrics = val_metrics

    return best_cfg, best_val_metrics


# ---------------------------------------------------------------------
# GKX loop runner (analogue of run_gkx_mlp / run_gkx_sklearn...)
# ---------------------------------------------------------------------

def run_portfolio_policy_with_features(
    *,
    df: pd.DataFrame,
    feature_builder: Callable[..., Dict[str, Any]],
    model_cls: type[torch.nn.Module],
    model_kwargs: Dict[str, Any],
    model_name: str,
    base_cfg: "TrainConfig",
    tune_hyperparams: bool = False,
    tune_grids: Optional[Dict[str, Sequence[float]]] = None,
    seed: int = 0,
    verbose: bool = True,
    train_verbose: bool = False,
    log_every: int = 1,
    # ---- NEW: ensemble options ----
    ensemble_n: int = 1,
    ensemble_seed_stride: int = 10_000,
) -> Dict[str, Any]:
    """
    GKX-style rolling OOS evaluation for portfolio policies.

    - Split-safe feature construction via feature_builder(...)
    - Train on TRAIN; select best epoch by validation objective on VAL
    - Optional tuning on VAL for (lr, l1, l2)
    - Evaluate on TEST, store per-year metrics
    - Compute "overall" metrics on concatenated OOS monthly series
    - Optional ensemble averaging of *scores* (ensemble_n > 1)

    Returns:
      {
        "per_year": pd.DataFrame,
        "overall": dict,
        "chosen_params": dict[int, dict]   # if tuning=True
      }
    """
    if int(ensemble_n) < 1:
        raise ValueError("ensemble_n must be >= 1")

    per_year_records: List[Dict[str, Any]] = []
    chosen_params: Dict[int, Dict[str, float]] = {}

    # Full OOS monthly series for global metrics
    oos_r: List[float] = []
    oos_r_exc: List[float] = []
    oos_to: List[float] = []
    
    train_years = base_cfg.train_years
    val_years = base_cfg.val_years
    test_start_year = base_cfg.test_start_year
    test_end_year = base_cfg.test_end_year

    for split in iter_gkx_splits(
        df=df,
        date_col="eom",
        train_years=train_years,
        val_years=val_years,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
    ):
        test_year = int(split["test_year"])
        tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue

        # -------------------------
        # Features (split-safe)
        # -------------------------
        feats = feature_builder(tr, va, te)

        X_train = np.asarray(feats["X_train"], dtype=np.float32)
        X_val   = np.asarray(feats["X_val"], dtype=np.float32)
        X_test  = np.asarray(feats["X_test"], dtype=np.float32)

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            continue

        # -------------------------
        # Split frames (aligned)
        # -------------------------
        df_train = df.iloc[tr].reset_index(drop=True)
        df_val   = df.iloc[va].reset_index(drop=True)
        df_test  = df.iloc[te].reset_index(drop=True)

        K = int(X_train.shape[1])

        train_batches = make_monthly_batches(df_train, X_train)
        val_batches   = make_monthly_batches(df_val,   X_val)
        test_batches  = make_monthly_batches(df_test,  X_test)

        def build_model() -> torch.nn.Module:
            return model_cls(K, **model_kwargs)

        # -------------------------
        # Choose cfg (tuned or base)
        # -------------------------
        cfg_used = base_cfg
        val_info: Dict[str, float] = {}

        if tune_hyperparams:
            grids = tune_grids or {}
            lr_grid = [float(x) for x in grids.get("lr", [base_cfg.lr])]
            l1_grid = [float(x) for x in grids.get("l1", [base_cfg.l1])]
            l2_grid = [float(x) for x in grids.get("l2", [base_cfg.l2])]

            cfg_used, val_info = tune_policy_hyperparams(
                build_model=build_model,
                train_batches=train_batches,
                val_batches=val_batches,
                base_cfg=base_cfg,
                lr_grid=lr_grid,
                l1_grid=l1_grid,
                l2_grid=l2_grid,
                seed=int(seed + 100_000 * test_year),
            )

            chosen_params[test_year] = {
                "lr": float(cfg_used.lr),
                "l1": float(cfg_used.l1),
                "l2": float(cfg_used.l2),
            }

        # -------------------------
        # Train model
        # -------------------------
        models: List[torch.nn.Module] = []
        member_metrics: List[Dict[str, float]] = []

        for m in range(int(ensemble_n)):
            member_seed = int(seed + 1_000_000 * test_year + m * int(ensemble_seed_stride))

            model_m, metrics_m = train_one_split_policy(
                build_model=build_model,
                train_batches=train_batches,
                val_batches=val_batches,
                cfg=cfg_used,
                seed=member_seed,
                # If ensembling, avoid spamming logs per member unless you really want it:
                train_verbose=(train_verbose and int(ensemble_n) == 1),
                log_every=log_every,
            )
            model_m.eval()
            models.append(model_m)
            member_metrics.append(metrics_m)

        # -------------------------
        # Test evaluation
        # -------------------------
        with torch.no_grad():
            if int(ensemble_n) == 1:
                r_test, r_test_exc, to_test = run_model_on_batches(
                    models[0],
                    test_batches,
                    policy_mode=cfg_used.policy_mode,
                    gross_leverage=cfg_used.gross_leverage,
                    compute_turnover=True,
                )
            else:
                r_test, r_test_exc, to_test = run_ensemble_on_batches(
                    models,
                    test_batches,
                    policy_mode=cfg_used.policy_mode,
                    gross_leverage=cfg_used.gross_leverage,
                    compute_turnover=True,
                )

        # free models (important for GPU memory)
        for mm in models:
            del mm

        r_np = r_test.detach().cpu().numpy()
        r_exc_np = r_test_exc.detach().cpu().numpy()
        to_np = to_test.detach().cpu().numpy() if to_test is not None else np.array([])

        ann_mean = _ann_mean(r_np)
        ann_vol  = _ann_vol(r_np)
        ann_sr   = float(sharpe_ratio_from_excess(r_exc_np))
        mean_to  = float(np.mean(to_np)) if to_np.size else float("nan")

        # training diagnostics (for ensemble: use member 0 or averages)
        if int(ensemble_n) == 1:
            train_val_metrics = member_metrics[0]
            best_epoch = float(train_val_metrics.get("best_epoch", np.nan))
            val_obj_diag = float(train_val_metrics.get("val_obj", np.nan))
            val_sr_diag  = float(train_val_metrics.get("val_ann_sharpe_excess", np.nan))
        else:
            best_epoch = float(np.mean([m.get("best_epoch", np.nan) for m in member_metrics]))
            val_obj_diag = float(np.mean([m.get("val_obj", np.nan) for m in member_metrics]))
            val_sr_diag  = float(np.mean([m.get("val_ann_sharpe_excess", np.nan) for m in member_metrics]))

        rec: Dict[str, Any] = {
            "test_year": test_year,
            "ann_mean": ann_mean,
            "ann_vol": ann_vol,
            "ann_sharpe_excess": ann_sr,
            "mean_turnover": mean_to,
            "lr": float(cfg_used.lr),
            "l1": float(cfg_used.l1),
            "l2": float(cfg_used.l2),
            "policy_mode": str(cfg_used.policy_mode),
            "gross_leverage": float(cfg_used.gross_leverage) if cfg_used.gross_leverage is not None else np.nan,
            "n_vars": int(K),
            "ensemble_n": int(ensemble_n),
            # diagnostics from training
            "best_epoch": best_epoch,
            "val_obj": val_obj_diag,
            "val_ann_sharpe_excess": val_sr_diag,
        }

        # (Optional) overwrite with tuning val_info if you prefer those numbers
        if tune_hyperparams and val_info:
            rec["val_obj"] = float(val_info.get("val_obj", rec["val_obj"]))
            rec["val_ann_sharpe_excess"] = float(val_info.get("val_ann_sharpe_excess", rec["val_ann_sharpe_excess"]))

        per_year_records.append(rec)

        # append global OOS series
        oos_r.extend(list(r_np))
        oos_r_exc.extend(list(r_exc_np))
        if to_np.size:
            oos_to.extend(list(to_np))

        SR_prov = sharpe_ratio_from_excess(np.asarray(oos_r_exc, dtype=np.float64))

        if verbose:
            print(
                f"{test_year} {model_name} | mean={ann_mean:.3f} vol={ann_vol:.3f}  "
                f"SR(excess)={ann_sr:.3f} TO={mean_to:.3f} SR_tot={SR_prov:.3f} "
                f"l1={float(cfg_used.l1)} l2={float(cfg_used.l2)}"
            )

    if not per_year_records:
        raise ValueError("No test years were evaluated (all splits were skipped).")

    per_year = pd.DataFrame(per_year_records).sort_values("test_year").reset_index(drop=True)

    oos_r_arr = np.asarray(oos_r, dtype=np.float64)
    oos_r_exc_arr = np.asarray(oos_r_exc, dtype=np.float64)
    oos_to_arr = np.asarray(oos_to, dtype=np.float64) if len(oos_to) else None

    overall = _compute_overall_metrics(oos_r_arr, oos_r_exc_arr, oos_to_arr)

    out: Dict[str, Any] = {"per_year": per_year, "overall": overall}
    if tune_hyperparams:
        out["chosen_params"] = chosen_params
    return out