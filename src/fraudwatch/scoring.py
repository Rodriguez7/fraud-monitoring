"""Hybrid scoring and human-readable explanations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .rules import Rule

# How to phrase a feature in an analyst-facing sentence. {v} = this transaction, {b} = typical.
PHRASES = {
    "cnt_10m": "{v:.0f} other transactions in the previous 10 minutes (typical: {b:.0f})",
    "cnt_1h": "{v:.0f} other transactions in the previous hour (typical: {b:.0f})",
    "cnt_24h": "{v:.0f} other transactions in the previous 24 hours (typical: {b:.0f})",
    "amount_z": "amount unusual for this customer ({v:+.1f} standard deviations from their 30-day pattern)",
    "amount_eur": "amount of EUR {v:,.2f} (typical: EUR {b:,.2f})",
    "amount_ratio_max": "amount is {v:.1f}x the customer's largest payment in the last 30 days",
    "is_new_device": "device first seen for this customer in the last 24 hours",
    "is_new_country": "first activity in this country in the last 24 hours",
    "is_new_payee": "payee first seen for this customer in the last 24 hours",
    "is_new_merchant": "first payment to this merchant",
    "is_cross_border": "payment outside the customer's home country",
    "country_changed_fast": "different country than the previous card payment, less than 2 hours earlier",
    "is_night": "payment between 00:00 and 05:00",
    "category_risk": "higher-risk merchant category (risk level {v:.0f} of 3)",
    "is_near_threshold": "transfer just below the 10,000 EUR reporting threshold",
    "near_threshold_48h": "{v:.0f} earlier transfers just below 10,000 EUR within 48 hours",
    "transfers_24h": "{v:.0f} earlier transfers in the previous 24 hours",
    "new_payee_transfers_24h": "{v:.0f} earlier transfers to new payees in the previous 24 hours",
    "sum_24h": "EUR {v:,.0f} spent in the previous 24 hours (typical: EUR {b:,.0f})",
    "secs_since_prev": "only {v:.0f} seconds since the previous transaction",
}


def combine(rule_p: pd.Series, ml_p: pd.Series) -> pd.Series:
    """Noisy-OR of rule evidence and ML probability: either source alone can raise the score."""
    return 1.0 - (1.0 - rule_p) * (1.0 - ml_p)


def occlusion_contributions(bundle, rows: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution to the ML probability.

    Each feature is replaced by its training median, one at a time. The drop in predicted
    probability is that feature's contribution. Model-agnostic and exact for the question
    "what changes if this value looked normal?". Correlated features (cnt_10m / cnt_1h)
    share credit, which is a known limitation.
    """
    base_p = bundle.predict_proba(rows)
    out = {}
    for feat in bundle.features:
        probe = rows[bundle.features].copy()
        probe[feat] = bundle.baseline[feat]
        out[feat] = base_p - bundle.model.predict_proba(probe)[:, 1]
    return pd.DataFrame(out, index=rows.index)


def explain_rows(
    rows: pd.DataFrame,
    fired: pd.DataFrame,
    rules: list[Rule],
    contributions: pd.DataFrame,
    baseline: dict[str, float],
    top_n: int = 3,
    min_contribution: float = 0.02,
) -> list[list[str]]:
    """One list of plain-language reasons per row: fired rules first, then top ML drivers."""
    rule_by_id = {r.id: r for r in rules}
    reasons: list[list[str]] = []
    for idx in rows.index:
        items = [
            f"Rule {rid} ({rule_by_id[rid].name}): {rule_by_id[rid].description}"
            for rid in fired.columns[fired.loc[idx].to_numpy()]
        ]
        contrib = contributions.loc[idx].sort_values(ascending=False)
        for feat, value in contrib.head(top_n).items():
            if value < min_contribution or feat not in PHRASES:
                continue
            text = PHRASES[feat].format(v=float(rows.loc[idx, feat]), b=baseline.get(feat, 0.0))
            items.append(f"Model: {text} (+{value * 100:.0f} pts)")
        reasons.append(items or ["No single dominant driver; score comes from many small signals"])
    return reasons


def reasons_to_json(reasons: list[list[str]]) -> list[str]:
    return [json.dumps(r) for r in reasons]


def risk_band(score: pd.Series, review_threshold: float, block_threshold: float) -> pd.Series:
    return pd.Series(
        np.select([score >= block_threshold, score >= review_threshold], ["block", "review"], "allow"),
        index=score.index,
    )
