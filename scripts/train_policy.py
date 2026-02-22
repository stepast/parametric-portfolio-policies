from __future__ import annotations

"""Train PPP-style portfolio policies (Brandt–Santa-Clara–Valkanov).


The script expects you have already created the processed panel:
    python scripts/prepare_data.py

Outputs are written to:
    data/processed/results/
"""

from dataclasses import asdict, replace

import pandas as pd

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works when running from Spyder or directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.schemas import FeatureConfig, TrainConfig


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

MACRO_INTERACTIONS = False
USE_JKP_CHARACTERISTICS = True
SIZE_FILTER_QUANTILE = 0.50

TRAIN_YEARS=25
VAL_YEARS=5
TEST_START_YEAR=2000

TRAIN_CFG = TrainConfig(
    train_years=TRAIN_YEARS,
    val_years=VAL_YEARS,
    test_start_year=TEST_START_YEAR,
    seed=0,
    patience=10,
    min_delta=1e-4,
    use_plateau_scheduler=True,
)

# Policy training choices
RISK_AVERSION = 5.0
EPOCHS_LINEAR = 200
EPOCHS_NN = 200

POLICY_MODE = "long_short_tilt"
GROSS_LEVERAGE = 2.0

TUNE_HYPERPARAMS = False
LINEAR_GRID = {
    "lr": [1e-2, 3e-3],
    "l1": [0.0, 1e-6, 1e-5, 1e-4],
    "l2": [0.0, 1e-6, 1e-5],
}
NN_GRID = {
    "lr": [1e-3, 3e-4],
    "l1": [0.0, 1e-7, 1e-6],
    "l2": [0.0, 1e-6, 1e-5],
}


def load_jkp_characteristics(raw_dir: Path) -> list[str]:
    """Load JKP characteristic list from Factor_Details.xlsx."""
    path = raw_dir / "Factor_Details.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Factor_Details.xlsx not found at {path}")

    df = pd.read_excel(path)
    if "abr_jkp" not in df.columns:
        raise KeyError("Column 'abr_jkp' not found")

    chars = df.loc[df["abr_jkp"].notna(), "abr_jkp"].astype(str).unique().tolist()
    print(f"Loaded {len(chars)} JKP characteristics")
    return chars


def main() -> None:
    from src.config import get_processed_dir, get_raw_dir, get_results_dir
    from src.models import LinearPolicy, MLPPolicy
    from src.preprocessing import make_feature_builder, prepare_features_and_target
    from src.results import (
        make_run_id,
        save_overall_json,
        save_per_year,
        save_run_metadata,
        save_summary,
        print_summary_table
    )
    from src.training import run_portfolio_policy_with_features

    raw_dir = get_raw_dir()
    proc_dir = get_processed_dir()
    results_dir = get_results_dir()

    feature_set_name = (
        f"{'jkp_chars' if USE_JKP_CHARACTERISTICS else 'all_chars'}__"
        f"mktcap_above_p{int(SIZE_FILTER_QUANTILE*100)}__"
        f"{'macro_int' if MACRO_INTERACTIONS else 'no_macro_int'}"
    )


    print("Project root:", REPO_ROOT)
    print("Raw dir      :", raw_dir)
    print("Processed dir:", proc_dir)
    print("Results dir  :", results_dir)

    # -----------------------------------------------------------------------
    # Load processed data
    # -----------------------------------------------------------------------
    processed_path = proc_dir / "JKP_US_processed.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed panel not found: {processed_path}\n"
            "Run scripts/prepare_data.py first."
        )

    df = pd.read_parquet(processed_path)
    print(f"\nLoaded: {len(df):,} rows, {len(df.columns)} columns")

    # -------------------------------------------------------------------
    # Filter by size
    # -------------------------------------------------------------------
    if SIZE_FILTER_QUANTILE > 0:
        cutoffs = df.groupby("eom")["me"].transform(lambda x: x.quantile(SIZE_FILTER_QUANTILE))
        df = df[df["me"] >= cutoffs].copy()
        print(f"After ME >= p{SIZE_FILTER_QUANTILE:.0%}: {len(df):,} rows")
        
    # -------------------------------------------------------------------
    # Identify macro / characteristic columns
    # -------------------------------------------------------------------
    macro_csv = raw_dir / "macro_predictors.csv"
    if not macro_csv.exists():
        raise FileNotFoundError(
            f"macro_predictors.csv not found at {macro_csv}. "
            "Place it in data/raw."
        )

    macro_sample = pd.read_csv(macro_csv, nrows=1)
    macro_cols = [c for c in macro_sample.columns if c != "date"]

    id_vars = {
        "id", "permno", "permco", "gvkey", "iid",
        "date", "eom",
        "curcd", "fx", "excntry",
        "me", "size_grp", "me_company",
        "common", "exch_main", "primary_sec", "obs_main",
        "comp_exchg", "comp_tpci",
        "source_crsp", "crsp_exchcd", "crsp_shrcd",
        "ff49", "sic", "naics", "gics",
    }

    ret_vars = {
        "ret_exc_lead1m", "ret_exc", "ret", "ret_lead1m",
        "ret_local", "ret_lag_dif",
        "prc_local", "prc_high", "prc_low",
        "bidask", "dolvol",
        "shares", "tvol", "adjfct",
    }

    excluded = id_vars | ret_vars | set(macro_cols)
    char_cols_all = sorted(set(df.columns) - excluded)

    # -----------------------------------------------------------------------
    # Select characteristics
    # -----------------------------------------------------------------------
    if USE_JKP_CHARACTERISTICS:
        jkp_chars = load_jkp_characteristics(raw_dir)
        char_cols = [c for c in char_cols_all if c in jkp_chars]
        print(f"Using {len(char_cols)} JKP characteristics")
    else:
        char_cols = char_cols_all
        print(f"Using all {len(char_cols)} characteristics")

    print(f"Macro predictors: {len(macro_cols)}")

    # -----------------------------------------------------------------------
    # Prepare features
    # -----------------------------------------------------------------------
    feature_cols = char_cols + macro_cols if MACRO_INTERACTIONS else char_cols
    
    X, y, df_clean = prepare_features_and_target(
        df,
        feature_cols=feature_cols,
        target_col = "ret_lead1m",
        required_cols = ["ret_exc_lead1m"]
    )
    
    assert len(df_clean) == X.shape[0] == y.shape[0], "X/y/df_clean misalignment"

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: {y.shape}")
    
    min_date = df_clean["eom"].min()
    max_date = df_clean["eom"].max()
    n_obs = len(df_clean)
    n_permno = df_clean["permno"].nunique() if "permno" in df_clean.columns else None

    feat_cfg = FeatureConfig(
        macro_cols=macro_cols,
        char_cols=char_cols,
        interactions=bool(MACRO_INTERACTIONS),
        ff49_col="ff49",
        date_col=TRAIN_CFG.date_col,
    )
    feature_builder = make_feature_builder(
        X_base=X,
        y=y,
        df_aligned=df_clean,
        X_columns=list(X.columns),
        cfg=feat_cfg,
    )

    print("\nRun configuration")
    print("Feature set        :", feature_set_name)
    print("Size filter (ME q) :", SIZE_FILTER_QUANTILE)
    print("JKP restricted set :", USE_JKP_CHARACTERISTICS)
    print("n_chars            :", len(char_cols))
    print("n_macro            :", len(macro_cols))
    print("macro interactions :", MACRO_INTERACTIONS)
    print(
        f"Split              : train={TRAIN_CFG.train_years}y, val={TRAIN_CFG.val_years}y, "
        f"test starts {TRAIN_CFG.test_start_year}"
    )
    print("Policy mode        :", POLICY_MODE)
    print("Risk aversion      :", RISK_AVERSION)

    # -------------------------------------------------------------------
    # Run policies 
    # -------------------------------------------------------------------
    run_id = make_run_id()
    rows: list[dict] = []

    def record(model_name: str, res: dict) -> None:
        save_per_year(res["per_year"], model_name=model_name, feature_set=feature_set_name, run_id=run_id)
        save_overall_json(res["overall"], model_name=model_name, feature_set=feature_set_name, run_id=run_id)
        rows.append({"model": model_name, **res["overall"]})

    # Linear policy
    train_lin = replace(
        TRAIN_CFG,
        risk_aversion=RISK_AVERSION,
        policy_mode=POLICY_MODE,
        gross_leverage=GROSS_LEVERAGE,
        epochs=EPOCHS_LINEAR,
        lr=1e-2,
        l1=0.0,
        l2=0.0,
    )

    res_lin = run_portfolio_policy_with_features(
        df=df_clean,
        feature_builder=feature_builder,
        model_cls=LinearPolicy,
        model_kwargs={"bias": True},
        model_name = "linear",
        base_cfg=train_lin,
        tune_hyperparams=TUNE_HYPERPARAMS,
        tune_grids=LINEAR_GRID,
        verbose=True,
    )
    record("Linear", res_lin)

    # MLP policy
    train_nn = replace(
        TRAIN_CFG,
        risk_aversion=RISK_AVERSION,
        policy_mode=POLICY_MODE,
        gross_leverage=GROSS_LEVERAGE,
        epochs=EPOCHS_NN,
        lr=1e-3,
        l1=0.0,
        l2=0.0,
    )

    res_nn = run_portfolio_policy_with_features(
        df=df_clean,
        feature_builder=feature_builder,
        model_cls=MLPPolicy,
        model_kwargs={"hidden_dims": (32,16,8),
                      "dropout": 0.0,
                      "use_batchnorm": True},
        model_name = "MLP",
        base_cfg=train_nn,
        tune_hyperparams=TUNE_HYPERPARAMS,
        tune_grids=NN_GRID,
        verbose=True,
    )
    record("MLP", res_nn)

    # -------------------------------------------------------------------
    # Summary + metadata
    # -------------------------------------------------------------------
    summary_df = pd.DataFrame(rows)
    summary_df.insert(0, "feature_set", feature_set_name)

    summary_df["train_years"] = TRAIN_CFG.train_years
    summary_df["val_years"] = TRAIN_CFG.val_years
    summary_df["test_start_year"] = TRAIN_CFG.test_start_year
    if TRAIN_CFG.test_end_year is not None:
        summary_df["test_end_year"] = TRAIN_CFG.test_end_year

    summary_df["size_filter_quantile"] = SIZE_FILTER_QUANTILE
    summary_df["use_jkp_characteristics"] = USE_JKP_CHARACTERISTICS
    summary_df["n_chars"] = len(char_cols)
    summary_df["n_macro"] = len(macro_cols)
    summary_df["macro_interactions"] = MACRO_INTERACTIONS

    summary_df["policy_mode"] = POLICY_MODE
    summary_df["risk_aversion"] = RISK_AVERSION
    summary_df["gross_leverage"] = GROSS_LEVERAGE
    summary_df["tune_hyperparams"] = TUNE_HYPERPARAMS
    
    summary_df["n_obs"] = n_obs
    summary_df["min_eom"] = str(min_date)
    summary_df["max_eom"] = str(max_date)
    if n_permno is not None:
        summary_df["n_permno"] = n_permno


    summary_path = save_summary(summary_df, feature_set=feature_set_name, run_id=run_id)

    run_metadata = {
        "run_id": run_id,
        "feature_set": feature_set_name,
        "script": "train_policy.py",
        "config": {
            "train": asdict(TRAIN_CFG),
            "size_filter_quantile": SIZE_FILTER_QUANTILE,
            "macro_interactions": MACRO_INTERACTIONS,
            "use_jkp_characteristics": USE_JKP_CHARACTERISTICS,
            "n_chars": len(char_cols),
            "n_macro": len(macro_cols),
            "test_end_year": TRAIN_CFG.test_end_year,
            "n_obs": n_obs,
            "min_eom": str(min_date),
            "max_eom": str(max_date),
            "n_permno": n_permno,
            "risk_aversion": RISK_AVERSION,
            "policy_mode": POLICY_MODE,
            "gross_leverage": GROSS_LEVERAGE,
            "tune_hyperparams": TUNE_HYPERPARAMS,
       },
    }
    meta_path = save_run_metadata(run_metadata, feature_set=feature_set_name, run_id=run_id)
    
    summary_ranked = summary_df.sort_values('ann_sharpe_excess', ascending=False).reset_index(drop=True)

    print_summary_table(
            summary_ranked,
            title="\n" + "="*60 + "\nResults\n" + "="*60,
            common_cols=[
                "feature_set",
                "train_years",
                "val_years",
                "test_start_year",
                "size_filter_quantile",
                "use_jkp_characteristics",
                "n_chars",
                "n_macro",
                "macro_interactions",
                "risk_aversion",
                "policy_mode",
                "gross_leverage",
                "tune_hyperparams"                
            ],
            value_cols=["model", "ann_mean", "ann_vol", "ann_sharpe_excess", "mean_turnover"],
        )
    
    print("\nSaved summary results to:", summary_path.name)
    print("Saved run metadata to:   ", meta_path.name)


if __name__ == "__main__":
    main()