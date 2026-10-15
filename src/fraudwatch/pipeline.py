"""End-to-end pipeline: generate -> SQL features -> train -> score -> alerts and cases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from . import cases, generate
from .config import Paths, SplitConfig
from .features import build_features, load_inputs
from .model import evaluate, load_bundle, save_bundle, split_frames, train_model
from .rules import escalated, evaluate_rules, load_rules
from .scoring import combine, explain_rows, occlusion_contributions, reasons_to_json, risk_band

START = pd.Timestamp("2026-01-01")  # day 0 of the simulation (see generate._EPOCH)


@dataclass
class Params:
    n_customers: int = 5000
    seed: int = 42
    victim_share: float = 0.05
    split: SplitConfig = field(default_factory=SplitConfig)


def run_generate(paths: Paths, params: Params) -> dict:
    paths.data.mkdir(parents=True, exist_ok=True)
    customers, txns = generate.generate(
        n_customers=params.n_customers,
        days=params.split.total,
        burn_in=params.split.burn_in,
        victim_share=params.victim_share,
        seed=params.seed,
    )
    customers.to_parquet(paths.customers, index=False)
    txns.to_parquet(paths.transactions, index=False)
    return {
        "customers": len(customers),
        "transactions": len(txns),
        "fraud_events": int(txns["fraud_event_id"].nunique()),
        "fraud_rate": float(txns["is_fraud"].mean()),
    }


def run_features(paths: Paths) -> dict:
    with duckdb.connect(str(paths.db)) as con:
        load_inputs(con, pd.read_parquet(paths.customers), pd.read_parquet(paths.transactions))
        n = build_features(con)
    return {"feature_rows": n}


def _features_df(paths: Paths) -> pd.DataFrame:
    with duckdb.connect(str(paths.db), read_only=True) as con:
        return con.execute("SELECT * FROM features").df()


def run_train(paths: Paths, params: Params) -> dict:
    df = _features_df(paths)
    bundle = train_model(df, params.split, START, seed=params.seed)
    save_bundle(bundle, paths)
    report = evaluate(df, bundle, START)
    paths.metrics.parent.mkdir(parents=True, exist_ok=True)
    paths.metrics.write_text(json.dumps(report, indent=2))
    return report


def run_score(paths: Paths, params: Params) -> dict:
    """Score the validation and test periods, keep alerts above the review threshold."""
    bundle = load_bundle(paths)
    report = json.loads(paths.metrics.read_text())
    review_thr, block_thr = report["thresholds"]["review"], report["thresholds"]["block"]
    df = _features_df(paths)
    _, valid, test = split_frames(df, params.split, START)
    scored = pd.concat([valid, test]).copy()

    rules = load_rules()
    fired, rule_p = evaluate_rules(scored, rules)
    scored["rule_prob"] = rule_p
    scored["ml_prob"] = bundle.predict_proba(scored)
    scored["score"] = combine(scored["rule_prob"], scored["ml_prob"])
    forced = escalated(fired, rules)
    by_score = scored["score"] >= review_thr
    alerts = scored[by_score | forced].copy()
    alerts["alert_source"] = pd.Series(
        [
            "both" if s and f else "model/score" if s else "escalated rule"
            for s, f in zip(by_score[alerts.index], forced[alerts.index], strict=True)
        ],
        index=alerts.index,
    )

    contrib = occlusion_contributions(bundle, alerts)
    reasons = explain_rows(alerts, fired.loc[alerts.index], rules, contrib, bundle.baseline)
    alerts["reasons"] = reasons_to_json(reasons)
    alerts["fired_rules"] = [",".join(fired.columns[fired.loc[i].to_numpy()]) for i in alerts.index]
    # Probabilities saturate near 1.0; cap the displayed score so it never claims certainty.
    alerts["risk_score"] = (alerts["score"].clip(upper=0.999) * 100).round(1)
    # Every alert needs a human look; "block" is reserved for the very highest scores.
    alerts["band"] = risk_band(alerts["score"], review_thr, block_thr).replace("allow", "review")
    keep = [
        "txn_id",
        "ts",
        "customer_id",
        "txn_type",
        "channel",
        "amount_eur",
        "merchant_category",
        "merchant_country",
        "home_country",
        "device_id",
        "payee_id",
        "risk_score",
        "band",
        "ml_prob",
        "rule_prob",
        "alert_source",
        "fired_rules",
        "reasons",
        "is_fraud",
        "fraud_type",
    ]
    with duckdb.connect(str(paths.db)) as con:
        con.register("_alerts_df", alerts[keep])
        con.execute("CREATE OR REPLACE TABLE alerts AS SELECT * FROM _alerts_df")
        con.unregister("_alerts_df")
        cases.init_cases(con)
    return {
        "scored": len(scored),
        "alerts": len(alerts),
        "alert_rate": len(alerts) / len(scored),
        "blocked": int((alerts["band"] == "block").sum()),
    }


def run_all(paths: Paths, params: Params) -> dict:
    return {
        "generate": run_generate(paths, params),
        "features": run_features(paths),
        "train": {k: v for k, v in run_train(paths, params).items() if k == "systems"},
        "score": run_score(paths, params),
    }
