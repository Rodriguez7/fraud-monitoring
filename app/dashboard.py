"""Streamlit dashboard and case-review screen.

Run with `fraudwatch dashboard` after `fraudwatch run`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudwatch import cases  # noqa: E402
from fraudwatch.config import Paths  # noqa: E402

PATHS = Paths()
BAND_COLORS = {"block": "#dc2626", "review": "#f59e0b"}

st.set_page_config(page_title="Fraudwatch", layout="wide", page_icon=":mag:")


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(PATHS.db), read_only=read_only)


def ready() -> bool:
    if not PATHS.db.exists() or not PATHS.metrics.exists():
        st.error("No data yet. Run `fraudwatch run` in the project folder, then reload this page.")
        return False
    return True


@st.cache_data(show_spinner=False)
def load_metrics(mtime: float) -> dict:
    return json.loads(PATHS.metrics.read_text())


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


# --------------------------------------------------------------------------- pages


def overview() -> None:
    metrics = load_metrics(PATHS.metrics.stat().st_mtime)
    with connect() as con:
        s = cases.summary(con)
        scored = con.execute(
            "SELECT COUNT(*) FROM features WHERE ts >= (SELECT MIN(ts) FROM alerts)"
        ).fetchone()[0]
        daily = con.execute(
            "SELECT CAST(ts AS DATE) AS day, band, COUNT(*) AS alerts FROM alerts GROUP BY 1, 2 ORDER BY 1"
        ).df()
        sources = con.execute("SELECT alert_source, COUNT(*) AS alerts FROM alerts GROUP BY 1").df()
        scores = con.execute("SELECT risk_score, band FROM alerts").df()

    st.title("Fraud and transaction monitoring")
    st.caption(
        "Synthetic transactions. Alerts combine an ML model and a rules engine; "
        "every alert carries plain-language reasons."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions scored", f"{scored:,}")
    c2.metric(
        "Alerts raised",
        f"{s['alerts']:,}",
        help=f"{pct(s['alerts'] / max(scored, 1), 2)} of scored transactions",
    )
    c3.metric("Open cases", f"{s['open_cases']:,}")
    c4.metric("Confirmed fraud", f"{s['confirmed']:,}")
    c5.metric(
        "Analyst precision",
        "n/a" if s["analyst_precision"] is None else pct(s["analyst_precision"], 0),
        help="Confirmed fraud / decided cases",
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Alerts per day")
        fig = px.bar(daily, x="day", y="alerts", color="band", color_discrete_map=BAND_COLORS)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Why alerts fire")
        fig = px.pie(sources, names="alert_source", values="alerts", hole=0.5)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Rules: how often they fire and how often they are right")
    rules = pd.DataFrame(metrics["rules"])
    rules["precision"] = rules["precision"].map(pct)
    rules["event_recall"] = rules["event_recall"].map(pct)
    rules["escalate"] = rules["escalate"].map({True: "yes", False: ""})
    st.dataframe(
        rules[["id", "name", "escalate", "fires", "precision", "event_recall", "description"]],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Measured on the held-out test period. Rules marked `escalate` open a case on their own; "
        "the others only raise the score."
    )

    st.subheader("Risk score distribution of alerts")
    fig = px.histogram(scores, x="risk_score", color="band", nbins=30, color_discrete_map=BAND_COLORS)
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def queue_page() -> None:
    st.title("Case review")
    with connect() as con:
        open_df = cases.queue(con, "open")
        total_open = len(open_df)
    f1, f2, f3 = st.columns([1, 1, 2])
    status = f1.selectbox("Status", ["open", "confirmed_fraud", "false_positive"])
    bands = f2.multiselect("Band", ["block", "review"], default=["block", "review"])
    with connect() as con:
        df = cases.queue(con, status)
    df = df[df["band"].isin(bands)]
    f3.markdown(f"**{len(df)}** cases shown, **{total_open}** open in total")
    if df.empty:
        st.info("No cases match.")
        return

    view = df[
        [
            "txn_id",
            "risk_score",
            "band",
            "ts",
            "customer_id",
            "amount_eur",
            "merchant_category",
            "merchant_country",
            "fired_rules",
            "alert_source",
        ]
    ].copy()
    st.dataframe(view, hide_index=True, width="stretch", height=260)

    chosen = st.selectbox(
        "Open a case",
        df["txn_id"].tolist(),
        format_func=lambda t: _label(df.loc[df["txn_id"] == t].iloc[0]),
    )
    detail(df.loc[df["txn_id"] == chosen].iloc[0])


def _label(row: pd.Series) -> str:
    return (
        f"{row['txn_id']} | risk {row['risk_score']:.1f} | EUR {row['amount_eur']:,.2f} | "
        f"{row['merchant_category']} | {row['merchant_country']}"
    )


def detail(row: pd.Series) -> None:
    st.divider()
    a, b = st.columns([3, 2])
    with a:
        st.subheader(f"Case {row['txn_id']}")
        color = BAND_COLORS.get(row["band"], "#64748b")
        st.markdown(
            f"<span style='background:{color};color:white;padding:2px 10px;border-radius:10px'>"
            f"{row['band'].upper()}</span> &nbsp; **risk score {row['risk_score']:.1f} / 100** "
            f"&nbsp; (model {min(row['ml_prob'] * 100, 99.9):.1f}%, rules {row['rule_prob'] * 100:.0f}%)",
            unsafe_allow_html=True,
        )
        st.markdown("**Why this alert fired**")
        for reason in json.loads(row["reasons"]):
            st.markdown(f"- {reason}")
        st.markdown("**Transaction**")
        st.table(
            pd.DataFrame(
                {
                    "time": [row["ts"]],
                    "customer": [row["customer_id"]],
                    "type": [row["txn_type"]],
                    "channel": [row["channel"]],
                    "amount (EUR)": [f"{row['amount_eur']:,.2f}"],
                    "category": [row["merchant_category"]],
                    "country": [row["merchant_country"]],
                    "home": [row["home_country"]],
                    "device": [row["device_id"]],
                    "payee": [row["payee_id"]],
                }
            ).T.rename(columns={0: ""})
        )
    with b:
        st.markdown("**Customer's recent activity**")
        with connect() as con:
            hist = cases.customer_history(con, row["customer_id"], row["ts"], 14)
        st.dataframe(hist, hide_index=True, width="stretch", height=330)
        with st.expander("Simulation ground truth (not available in production)"):
            st.write({"is_fraud": bool(row["is_fraud"]), "fraud_type": row["fraud_type"]})

    st.markdown("**Decision**")
    d1, d2 = st.columns([1, 3])
    analyst = d1.text_input("Analyst", value=st.session_state.get("analyst", ""))
    note = d2.text_input("Note", value=row.get("note") or "", key=f"note-{row['txn_id']}")
    st.session_state["analyst"] = analyst
    b1, b2, b3 = st.columns(3)
    for col, label, status in (
        (b1, "Confirm fraud", "confirmed_fraud"),
        (b2, "False positive", "false_positive"),
        (b3, "Reopen", "open"),
    ):
        if col.button(label, key=f"{status}-{row['txn_id']}", width="stretch"):
            with connect(read_only=False) as con:
                cases.decide(con, row["txn_id"], status, analyst, note)
            st.rerun()


def model_page() -> None:
    m = load_metrics(PATHS.metrics.stat().st_mtime)
    st.title("Model and rules evaluation")
    st.caption(
        f"Time-based split: days 0-{m['split_days']['burn_in']} build history, "
        f"{m['split_days']['burn_in']}-{m['split_days']['train_end']} train, "
        f"{m['split_days']['train_end']}-{m['split_days']['valid_end']} validate (thresholds, calibration), "
        f"{m['split_days']['valid_end']}-{m['split_days']['total']} test. "
        f"Analysts can review about {pct(m['review_budget'])} of transactions."
    )
    systems = pd.DataFrame(m["systems"]).T
    systems.index.name = "system"
    table = systems[
        ["pr_auc", "roc_auc", "alert_rate", "precision", "txn_recall", "event_recall", "fraud_value_caught"]
    ].copy()
    for col in ("alert_rate", "precision", "txn_recall", "event_recall", "fraud_value_caught"):
        table[col] = table[col].map(pct)
    table[["pr_auc", "roc_auc"]] = table[["pr_auc", "roc_auc"]].round(3)
    st.subheader("Test period")
    st.dataframe(table, width="stretch")
    st.caption(
        f"{m['fraud_events_test']} fraud events in the test period: small sample, expect wide "
        "uncertainty. Synthetic data, so absolute numbers are optimistic."
    )

    st.subheader("Recall by fraud typology")
    typ = pd.DataFrame(m["by_typology"]).T
    for col in ("rules", "ml", "hybrid"):
        typ[col] = typ[col].map(pct)
    st.dataframe(typ, width="stretch")

    unseen = PATHS.base / "reports" / "unseen_typology.json"
    if unseen.exists():
        st.subheader("When the model has never seen the typology")
        u = pd.DataFrame(json.loads(unseen.read_text()))
        for col in ("rules_only", "ml_only", "hybrid"):
            u[col] = u[col].map(pct)
        st.dataframe(u, hide_index=True, width="stretch")
        st.caption(
            "Model retrained with all transactions of one typology removed. "
            "Generate with `python -m fraudwatch.experiments`."
        )


# --------------------------------------------------------------------------- main

if ready():
    page = st.sidebar.radio("Page", ["Overview", "Case review", "Model & rules"])
    with connect() as _c:
        _s = cases.summary(_c)
    st.sidebar.metric("Open cases", _s["open_cases"])
    {"Overview": overview, "Case review": queue_page, "Model & rules": model_page}[page]()
