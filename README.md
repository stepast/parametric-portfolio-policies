## Parametric portfolio policies based on stock characteristics

This repository implements and extends the portfolio optimization framework proposed by "Parametric Portfolio Policies: Exploiting Characteristics in the Cross-Section of Equity Returns" by Michael W. Brandt, Pedro Santa-Clara and Rossen Valkanov (BSV). It trains characteristic-driven portfolio policies on a processed U.S. equity panel, with optional macro interactions and an optional restricted JKP characteristic set. Portfolio weights are chosen to maximize mean-variance preferences, and can be either a linear function of stock characteristics (following BSV) or a neural network.

Data note: Raw data are not included. The equity panel is typically accessed via WRDS (credentials required).

## Project structure

```text

.
├── data/
│   ├── raw/                   # raw downloads (ignored by git; .gitkeep included)
│   └── processed/             # processed panel + outputs (ignored by git; .gitkeep included)
│       └── results/           # saved experiment outputs
├── scripts/
│   ├── data_download.py       # downloads raw inputs
│   ├── prepare_data.py        # builds processed parquet panel
│   └── train_policy.py        # trains portfolio policies and saves results
└── src/
    ├── backtest.py            # portfolio backtest utilities
    ├── config.py              # project, data, and results paths
    ├── models.py              # policy model definitions
    ├── portfolio.py           # portfolio construction / constraints helpers
    ├── preprocessing.py       # feature preparation + split-safe transforms
    ├── results.py             # result saving utilities + run metadata
    ├── schemas.py             # FeatureConfig / TrainConfig dataclasses
    ├── splitting.py           # rolling train/val/test splits
    └── training.py            # policy training + (optional) tuning

```

Setup

Option A: Conda (recommended)

conda env create -f environment.yml
conda activate bsv-ppp

---

Run the pipeline

From the repository root:

1) Download raw inputs
python scripts/data_download.py

2) Build processed panel
python scripts/prepare_data.py

Expected output:
- data/processed/JKP_US_processed.parquet

3) Train portfolio policies
python scripts/train_policy.py

Outputs are written to:
data/processed/results/

---

Configuration (feature set choices)

At the top of scripts/train_policy.py you can control the feature set using three switches:

- SIZE_FILTER_QUANTILE
  Cross-sectional size filter applied each month. Example: 0.50 keeps stocks with market equity (ME) above the monthly median. Use 0.0 to disable.

- MACRO_INTERACTIONS
  If True, includes interactions between firm characteristics and macro predictors (constructed in a split-safe way).

- USE_JKP_CHARACTERISTICS
  If True, restricts characteristics to the JKP set loaded from data/raw/Factor_Details.xlsx

---

Configuration (policy training)

Key policy-training knobs in scripts/train_policy.py:

- RISK_AVERSION
  Risk aversion parameter used in the policy objective (e.g., mean-variance style tradeoff).

- POLICY_MODE
  Portfolio constraint / mapping mode (e.g., long-short tilt). See scripts/train_policy.py for supported values.

- GROSS_LEVERAGE
  Gross exposure constraint used by the policy.

- EPOCHS_LINEAR / EPOCHS_NN
  Number of training epochs for the linear and neural policy versions.

Optional hyperparameter tuning:

- TUNE_HYPERPARAMS
  If True, performs a small grid search using LINEAR_GRID and NN_GRID.

---

Results and outputs

Results are saved to: data/processed/results/

For each run, the script writes:

- summary__<feature_set>__<run_id>.csv
  One-row-per-model summary table.

- per_year__<model>__<feature_set>__<run_id>.csv
  Year-by-year performance metrics.

- overall__<model>__<feature_set>__<run_id>.json
  Overall metrics per model.

- run_metadata__<feature_set>__<run_id>.json
  Full run configuration, script name, paths, and key package versions.

---

Models included

The default script trains and evaluates:
- Linear policy
- Neural policy
---

References

- Brandt, Michael W.; Santa-Clara, Pedro; Valkanov, Rossen. "Parametric Portfolio Policies: Exploiting Characteristics in the Cross-Section of Equity Returns". The Review of Financial Studies, 22 (9), 2009, 3411–3447
- Jensen, Theis Ingerslev; Kelly, Bryan; Pedersen, Lasse Heje. Is There a Replication Crisis in Finance? The Journal of Finance, 78(5), 2023, 2465-2518.

License
This repository is intended for educational and research demonstration purposes.
