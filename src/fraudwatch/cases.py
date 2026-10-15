"""Case management: the analyst's review queue and decisions, stored in DuckDB."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

STATUSES = ("open", "confirmed_fraud", "false_positive")


def init_cases(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)create the case table with one open case per alert."""
    con.execute("DROP TABLE IF EXISTS cases")
    con.execute(
        """
        CREATE TABLE cases (
            txn_id VARCHAR PRIMARY KEY,
            status VARCHAR NOT NULL DEFAULT 'open',
            analyst VARCHAR,
            note VARCHAR,
            decided_at TIMESTAMP
        )
        """
    )
    con.execute("INSERT INTO cases (txn_id) SELECT txn_id FROM alerts")


def decide(
    con: duckdb.DuckDBPyConnection,
    txn_id: str,
    status: str,
    analyst: str = "",
    note: str = "",
) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    con.execute(
        "UPDATE cases SET status = ?, analyst = ?, note = ?, decided_at = ? WHERE txn_id = ?",
        [status, analyst, note, datetime.now() if status != "open" else None, txn_id],
    )


def queue(con: duckdb.DuckDBPyConnection, status: str = "open", limit: int = 500) -> pd.DataFrame:
    """Alerts joined with their case status, highest risk first."""
    return con.execute(
        """
        SELECT a.*, c.status, c.analyst, c.note, c.decided_at
        FROM alerts a JOIN cases c USING (txn_id)
        WHERE c.status = ?
        ORDER BY a.risk_score DESC, a.rule_prob DESC, a.amount_eur DESC, a.ts
        LIMIT ?
        """,
        [status, limit],
    ).df()


def summary(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute(
        """
        SELECT
            COUNT(*)                                          AS alerts,
            COUNT(*) FILTER (status = 'open')                 AS open_cases,
            COUNT(*) FILTER (status = 'confirmed_fraud')      AS confirmed,
            COUNT(*) FILTER (status = 'false_positive')       AS false_positives
        FROM cases
        """
    ).fetchone()
    keys = ("alerts", "open_cases", "confirmed", "false_positives")
    out = dict(zip(keys, row, strict=True))
    decided = out["confirmed"] + out["false_positives"]
    out["analyst_precision"] = out["confirmed"] / decided if decided else None
    return out


def customer_history(
    con: duckdb.DuckDBPyConnection, customer_id: str, before: pd.Timestamp, limit: int = 15
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT ts, txn_type, channel, amount_eur, merchant_category, merchant_country, device_id, payee_id
        FROM features
        WHERE customer_id = ? AND ts <= ?
        ORDER BY ts DESC
        LIMIT ?
        """,
        [customer_id, before, limit],
    ).df()
