"""Benchmark on the public PaySim data set.

PaySim (Lopez-Rojas, Elmir and Axelsson, 2016) is a well-known synthetic mobile-money data set,
published on Kaggle under CC BY-SA 4.0: https://www.kaggle.com/datasets/ealaxi/paysim1
The file is large (about 470 MB) and needs a Kaggle login, so it is not part of this repository.
Download it once and point `fraudwatch paysim --csv PATH` at it.

What this benchmark is for: showing how much of the headline performance comes from simulator
artefacts. PaySim fraud always empties the origin account and only ever happens in TRANSFER and
CASH_OUT, so balance columns make the problem close to trivial. Every system is therefore
evaluated twice: with behavioural features only, and with the balance features added.

Citation:
    E. A. Lopez-Rojas, A. Elmir and S. Axelsson. "PaySim: A financial mobile money simulator
    for fraud detection". 28th European Modeling and Simulation Symposium, 2016.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

BEHAVIOURAL = [
    "amount",
    "log_amount",
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_cash_in",
    "is_debit",
    "hour_of_day",
    "dest_is_merchant",
    "dest_cnt_24h",
    "dest_sum_24h",
    "dest_cnt_7d",
]
BALANCE = [
    "orig_balance_before",
    "orig_emptied",
    "amount_to_balance",
    "orig_balance_error",
    "dest_balance_before",
    "dest_balance_error",
]
REVIEW_BUDGET = 0.005  # analysts can review 0.5% of transactions, as in the main project
TRAIN_NEGATIVE_SHARE = 0.05  # legitimate rows kept for training; calibration restores the prior


def build_features(con: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    """Load the CSV into DuckDB and run the PaySim feature SQL. Returns the row count."""
    con.execute(
        "CREATE OR REPLACE TABLE paysim_raw AS SELECT * FROM read_csv_auto(?, header = true)",
        [str(csv_path)],
    )
    sql = resources.files("fraudwatch").joinpath("sql/paysim_features.sql").read_text()
    con.execute(sql)
    return con.execute("SELECT COUNT(*) FROM paysim_features").fetchone()[0]


def time_split(con: duckdb.DuckDBPyConnection, train_share=0.70, valid_share=0.15) -> tuple[int, int]:
    """Step boundaries so that train, valid and test hold roughly 70%, 15% and 15% of rows."""
    steps = con.execute("SELECT step, COUNT(*) AS n FROM paysim_features GROUP BY step ORDER BY step").df()
    cum = steps["n"].cumsum() / steps["n"].sum()
    train_end = int(steps.loc[cum >= train_share, "step"].iloc[0])
    valid_end = int(steps.loc[cum >= train_share + valid_share, "step"].iloc[0])
    return train_end, valid_end


def _load(con, columns: list[str], where: str, sample_negatives: bool = False) -> pd.DataFrame:
    cols = ", ".join(sorted(set(columns + ["is_fraud", "is_flagged", "amount", "type", "step"])))
    extra = f" AND (is_fraud = 1 OR random() < {TRAIN_NEGATIVE_SHARE})" if sample_negatives else ""
    return con.execute(f"SELECT {cols} FROM paysim_features WHERE {where}{extra}").df()


def _flag_stats(frame: pd.DataFrame, flagged: pd.Series) -> dict:
    y = frame["is_fraud"].astype(bool)
    value = frame.loc[y, "amount"].sum()
    return {
        "alert_rate": float(flagged.mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "recall": float(y[flagged].sum() / max(y.sum(), 1)),
        "fraud_value_caught": float(frame.loc[flagged & y, "amount"].sum() / max(value, 1)),
    }


def _top_budget(score: pd.Series, budget: float) -> pd.Series:
    k = max(int(round(budget * len(score))), 1)
    cutoff = score.nlargest(k).min()
    return score >= cutoff


def run_benchmark(csv_path: Path, out_path: Path | None = None, seed: int = 7) -> dict:
    con = duckdb.connect()
    rows = build_features(con, csv_path)
    train_end, valid_end = time_split(con)
    where = {
        "train": f"step <= {train_end}",
        "valid": f"step > {train_end} AND step <= {valid_end}",
        "test": f"step > {valid_end}",
    }
    report: dict = {
        "source": "PaySim, Kaggle ealaxi/paysim1, CC BY-SA 4.0",
        "rows": rows,
        "train_end_step": train_end,
        "valid_end_step": valid_end,
        "review_budget": REVIEW_BUDGET,
        "splits": {},
        "systems": {},
    }
    test_all = _load(con, BEHAVIOURAL + BALANCE, where["test"])
    for name in ("train", "valid", "test"):
        n, f = con.execute(
            f"SELECT COUNT(*), SUM(is_fraud) FROM paysim_features WHERE {where[name]}"
        ).fetchone()
        report["splits"][name] = {
            "rows": int(n),
            "fraud": int(f or 0),
            "fraud_rate": float((f or 0) / max(n, 1)),
        }

    # Baseline 1: the data set's own rule (isFlaggedFraud: transfers above 200,000).
    flagged = test_all["is_flagged"].astype(bool)
    report["systems"]["dataset_flag_rule"] = {
        **_flag_stats(test_all, flagged),
        "pr_auc": None,
        "roc_auc": None,
    }
    # Baseline 2: a one-line rule an analyst would write: transfer or cash-out emptying the account.
    rule = (test_all["is_transfer"] + test_all["is_cash_out"] > 0) & (test_all["orig_emptied"] == 1)
    report["systems"]["rule_account_emptied"] = {
        **_flag_stats(test_all, rule),
        "pr_auc": None,
        "roc_auc": None,
    }

    for label, features in (
        ("ml_behavioural_only", BEHAVIOURAL),
        ("ml_with_balance_features", BEHAVIOURAL + BALANCE),
    ):
        train = _load(con, features, where["train"], sample_negatives=True)
        valid = _load(con, features, where["valid"])
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        ).fit(train[features], train["is_fraud"])
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(valid[features], valid["is_fraud"])
        score = pd.Series(calibrated.predict_proba(test_all[features])[:, 1], index=test_all.index)
        y = test_all["is_fraud"]
        entry = {
            "pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "features": len(features),
        }
        entry.update(_flag_stats(test_all, _top_budget(score, REVIEW_BUDGET)))
        report["systems"][label] = entry

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
        )
    return report


def format_report(report: dict) -> str:
    lines = [
        f"PaySim: {report['rows']:,} rows, time split at steps {report['train_end_step']} / "
        f"{report['valid_end_step']}, review budget {report['review_budget'] * 100:.1f}%",
        "split   rows        fraud   fraud rate",
    ]
    for name, s in report["splits"].items():
        lines.append(f"{name:6s}  {s['rows']:>10,}  {s['fraud']:>6,}  {s['fraud_rate'] * 100:.3f}%")
    lines.append("")
    lines.append(f"{'system':28s} {'PR-AUC':>7} {'alert%':>7} {'precis':>7} {'recall':>7} {'value':>7}")
    for name, s in report["systems"].items():
        pr = "" if s["pr_auc"] is None else f"{s['pr_auc']:.3f}"
        lines.append(
            f"{name:28s} {pr:>7} {s['alert_rate'] * 100:>6.2f}% {s['precision'] * 100:>6.0f}% "
            f"{s['recall'] * 100:>6.0f}% {s['fraud_value_caught'] * 100:>6.0f}%"
        )
    return "\n".join(lines)
