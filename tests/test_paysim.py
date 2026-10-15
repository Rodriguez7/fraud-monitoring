"""PaySim benchmark on a tiny file in the same format as the Kaggle CSV."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from fraudwatch import paysim

COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def mini_paysim(n=6000, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 120, n))
    kind = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n, p=[0.5, 0.1, 0.25, 0.1, 0.05]
    )
    amount = np.round(rng.lognormal(7, 1.2, n), 2)
    old = np.round(rng.lognormal(9, 1.5, n), 2)
    fraud = np.isin(kind, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.08)
    amount = np.where(fraud, old, amount)
    new = np.where(fraud, 0.0, np.maximum(old - amount, 0))
    dest = np.where(
        fraud, rng.choice([f"C{i}" for i in range(5)], n), [f"C{i}" for i in rng.integers(100, 4000, n)]
    )
    return pd.DataFrame(
        {
            "step": step,
            "type": kind,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": old,
            "newbalanceOrig": new,
            "nameDest": dest,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": fraud.astype(int),
            "isFlaggedFraud": 0,
        }
    )[COLUMNS]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "paysim.csv"
    mini_paysim().to_csv(path, index=False)
    return path


def small_csv(tmp_path, rows):
    path = tmp_path / "small.csv"
    base = {
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": "C1",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "nameDest": "C9",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 100.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    pd.DataFrame([{**base, **r} for r in rows])[COLUMNS].to_csv(path, index=False)
    return path


def test_destination_history_only_counts_earlier_steps(tmp_path):
    rows = [
        {"step": 1, "nameOrig": "A"},
        {"step": 1, "nameOrig": "B"},
        {"step": 2, "nameOrig": "C"},
        {"step": 5, "nameOrig": "D"},
    ]
    con = duckdb.connect()
    paysim.build_features(con, small_csv(tmp_path, rows))
    got = con.execute("SELECT step, dest_cnt_24h FROM paysim_features ORDER BY rid").fetchall()
    # same-step rows never see each other; later steps see strictly earlier ones
    assert got == [(1, 0), (1, 0), (2, 2), (5, 3)]


def test_account_emptied_and_balance_error_flags(tmp_path):
    rows = [
        {"step": 1, "oldbalanceOrg": 300.0, "amount": 300.0, "newbalanceOrig": 0.0},
        {"step": 2, "oldbalanceOrg": 300.0, "amount": 100.0, "newbalanceOrig": 200.0},
    ]
    con = duckdb.connect()
    paysim.build_features(con, small_csv(tmp_path, rows))
    f = con.execute("SELECT orig_emptied, orig_balance_error FROM paysim_features ORDER BY rid").fetchall()
    assert f == [(1, 0.0), (0, 0.0)]


def test_labels_are_not_model_features():
    assert not {"is_fraud", "is_flagged"} & set(paysim.BEHAVIOURAL + paysim.BALANCE)


def test_time_split_covers_every_step_once(csv_path):
    con = duckdb.connect()
    paysim.build_features(con, csv_path)
    train_end, valid_end = paysim.time_split(con)
    assert 1 <= train_end < valid_end <= 119


def test_benchmark_runs_and_balance_features_help(csv_path, tmp_path):
    out = tmp_path / "paysim.json"
    report = paysim.run_benchmark(csv_path, out)
    assert out.exists()
    assert set(report["systems"]) == {
        "dataset_flag_rule",
        "rule_account_emptied",
        "ml_behavioural_only",
        "ml_with_balance_features",
    }
    assert all(report["splits"][s]["rows"] > 0 for s in ("train", "valid", "test"))
    behav = report["systems"]["ml_behavioural_only"]["pr_auc"]
    full = report["systems"]["ml_with_balance_features"]["pr_auc"]
    assert full >= behav  # the simulator artefact is the stronger signal
    assert "PR-AUC" in paysim.format_report(report)
