import duckdb
import pandas as pd
import pytest

from fraudwatch.config import Paths, SplitConfig
from fraudwatch.generate import COLUMNS
from fraudwatch.pipeline import Params

SMALL_SPLIT = SplitConfig(burn_in=10, train_end=40, valid_end=50, total=60, review_budget=0.01)


@pytest.fixture
def small_params() -> Params:
    return Params(n_customers=400, seed=3, victim_share=0.1, split=SMALL_SPLIT)


@pytest.fixture
def tmp_paths(tmp_path) -> Paths:
    return Paths(base=tmp_path)


def make_txns(rows: list[dict]) -> pd.DataFrame:
    """Build a transactions frame from partial rows; missing columns get neutral defaults."""
    defaults = {
        "txn_type": "card_payment",
        "amount_eur": 10.0,
        "merchant_id": "m1",
        "merchant_category": "grocery",
        "merchant_country": "AT",
        "channel": "card_present",
        "device_id": None,
        "payee_id": None,
        "is_fraud": False,
        "fraud_type": None,
        "fraud_event_id": None,
        "customer_id": "C1",
    }
    out = []
    for i, row in enumerate(rows):
        full = {**defaults, **row, "txn_id": f"T{i}"}
        full["ts"] = pd.Timestamp(full["ts"])
        out.append(full)
    return pd.DataFrame(out)[COLUMNS]


@pytest.fixture
def features_con():
    """Factory: load rows into an in-memory DuckDB and return the computed features frame."""
    from fraudwatch.features import build_features, load_inputs

    def run(rows: list[dict]) -> pd.DataFrame:
        txns = make_txns(rows)
        customers = pd.DataFrame({"customer_id": sorted(txns["customer_id"].unique())}).assign(
            home_country="AT", account_age_days=500
        )
        con = duckdb.connect()
        load_inputs(con, customers, txns)
        build_features(con)
        return con.execute("SELECT * FROM features ORDER BY ts").df()

    return run
