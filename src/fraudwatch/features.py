"""Run the SQL feature pipeline in DuckDB."""

from __future__ import annotations

from importlib import resources

import duckdb
import pandas as pd

# Columns the model may use. Labels (is_fraud, fraud_type, fraud_event_id) are never features.
NUMERIC_FEATURES = [
    "amount_eur",
    "is_transfer",
    "is_online",
    "is_card_present",
    "is_cross_border",
    "hour",
    "dow",
    "is_night",
    "category_risk",
    "is_near_threshold",
    "is_new_merchant",
    "is_new_device",
    "is_new_country",
    "is_new_payee",
    "account_age_days",
    "customer_txn_seq",
    "secs_since_prev",
    "cnt_10m",
    "cnt_1h",
    "cnt_24h",
    "sum_24h",
    "avg_30d",
    "std_30d",
    "max_30d",
    "amount_z",
    "amount_ratio_max",
    "transfers_24h",
    "new_payee_transfers_24h",
    "near_threshold_48h",
    "country_changed_fast",
]
LABEL_COLUMNS = ["is_fraud", "fraud_type", "fraud_event_id"]


def feature_sql() -> str:
    return resources.files("fraudwatch").joinpath("sql/features.sql").read_text()


def build_features(con: duckdb.DuckDBPyConnection) -> int:
    """Create the `features` table from `transactions` and `customers`. Returns the row count."""
    con.execute(feature_sql())
    return con.execute("SELECT COUNT(*) FROM features").fetchone()[0]


def load_inputs(con: duckdb.DuckDBPyConnection, customers: pd.DataFrame, txns: pd.DataFrame) -> None:
    con.register("_customers_df", customers)
    con.register("_txns_df", txns)
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM _customers_df")
    con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM _txns_df")
    con.unregister("_customers_df")
    con.unregister("_txns_df")
