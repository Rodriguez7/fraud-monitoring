"""Train, calibrate and evaluate the ML model with a strict time-based split."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Paths, SplitConfig
from .features import NUMERIC_FEATURES
from .rules import escalated, evaluate_rules, load_rules
from .scoring import combine


@dataclass
class ModelBundle:
    """Everything needed to score new transactions."""

    model: object
    features: list[str]
    baseline: dict[str, float]  # training medians, used as the "typical" value in explanations
    split: SplitConfig

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]


def split_frames(df: pd.DataFrame, split: SplitConfig, start: pd.Timestamp):
    day = (df["ts"] - start).dt.total_seconds() / 86400
    train = df[(day >= split.burn_in) & (day < split.train_end)]
    valid = df[(day >= split.train_end) & (day < split.valid_end)]
    test = df[day >= split.valid_end]
    return train, valid, test


def train_model(df: pd.DataFrame, split: SplitConfig, start: pd.Timestamp, seed: int = 7):
    """Fit on train, calibrate on validation. Test is never touched here."""
    train, valid, _ = split_frames(df, split, start)
    x_train, y_train = train[NUMERIC_FEATURES], train["is_fraud"].astype(int)
    base = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=24,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )
    base.fit(x_train, y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(valid[NUMERIC_FEATURES], valid["is_fraud"].astype(int))
    baseline = train[NUMERIC_FEATURES].median().to_dict()
    return ModelBundle(calibrated, list(NUMERIC_FEATURES), baseline, split)


def save_bundle(bundle: ModelBundle, paths: Paths) -> None:
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths.model)


def load_bundle(paths: Paths) -> ModelBundle:
    return joblib.load(paths.model)


# --------------------------------------------------------------------------- evaluation


def _flag_stats(frame: pd.DataFrame, flagged: pd.Series) -> dict:
    y = frame["is_fraud"].astype(bool)
    events = frame.loc[y, "fraud_event_id"]
    caught_events = frame.loc[flagged & y, "fraud_event_id"].nunique()
    value = frame.loc[y, "amount_eur"].sum()
    return {
        "alert_rate": float(flagged.mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "txn_recall": float(y[flagged].sum() / max(y.sum(), 1)),
        "event_recall": float(caught_events / max(events.nunique(), 1)),
        "fraud_value_caught": float(frame.loc[flagged & y, "amount_eur"].sum() / max(value, 1)),
    }


def review_threshold(score: pd.Series, forced: pd.Series, budget: float) -> float:
    """Score threshold so that forced alerts (escalated rules) plus score alerts fit the budget.

    Escalated transactions always alert. What is left of the review budget goes to the
    highest-scoring of the rest. If escalations alone exceed the budget, nothing else alerts.
    """
    allowed = int(budget * len(score))
    remaining = max(allowed - int(forced.sum()), 0)
    rest = score[~forced].sort_values(ascending=False)
    if remaining == 0 or rest.empty:
        return float(np.nextafter(score.max(), np.inf))
    return float(max(rest.iloc[min(remaining, len(rest)) - 1], 1e-9))


def evaluate(df: pd.DataFrame, bundle: ModelBundle, start: pd.Timestamp) -> dict:
    """Compare rules only, ML only and the hybrid on the held-out test period.

    Thresholds are fixed on the validation period so analysts review about `review_budget`
    of transactions, then applied unchanged to test.
    """
    split = bundle.split
    _, valid, test = split_frames(df, split, start)
    rules = load_rules()
    parts = {}
    for name, frame in (("valid", valid), ("test", test)):
        fired, rule_p = evaluate_rules(frame, rules)
        ml_p = pd.Series(bundle.predict_proba(frame), index=frame.index)
        parts[name] = {
            "rules": rule_p,
            "ml": ml_p,
            "hybrid": combine(rule_p, ml_p),
            "forced": escalated(fired, rules),
        }

    report: dict = {
        "split_days": {
            "burn_in": split.burn_in,
            "train_end": split.train_end,
            "valid_end": split.valid_end,
            "total": split.total,
        },
        "rows": {"valid": len(valid), "test": len(test)},
        "fraud_rate_test": float(test["is_fraud"].mean()),
        "fraud_events_test": int(test.loc[test["is_fraud"], "fraud_event_id"].nunique()),
        "review_budget": split.review_budget,
        "systems": {},
    }
    no_force = {
        k: pd.Series(False, index=v.index)
        for k, v in (("valid", parts["valid"]["rules"]), ("test", parts["test"]["rules"]))
    }
    y_test = test["is_fraud"].astype(int)
    flagged_test: dict[str, pd.Series] = {}
    for system in ("rules", "ml", "hybrid"):
        use_forced = system == "hybrid"
        f_valid = parts["valid"]["forced"] if use_forced else no_force["valid"]
        f_test = parts["test"]["forced"] if use_forced else no_force["test"]
        thr = review_threshold(parts["valid"][system], f_valid, split.review_budget)
        s_test = parts["test"][system]
        flagged = (s_test >= thr) | f_test
        flagged_test[system] = flagged
        entry = {
            "threshold": float(thr),
            "roc_auc": float(roc_auc_score(y_test, s_test)),
            "pr_auc": float(average_precision_score(y_test, s_test)),
        }
        entry.update(_flag_stats(test, flagged))
        report["systems"][system] = entry
    hybrid_valid = parts["valid"]["hybrid"]
    report["thresholds"] = {
        "review": report["systems"]["hybrid"]["threshold"],
        "block": float(max(np.quantile(hybrid_valid, 0.999), report["systems"]["hybrid"]["threshold"])),
    }
    report["rules"] = _rule_table(test, rules)
    report["by_typology"] = _by_typology(test, flagged_test)
    return report


def _by_typology(test: pd.DataFrame, flagged: dict[str, pd.Series]) -> dict:
    """Share of each fraud typology's transactions flagged by each system."""
    fraud = test[test["is_fraud"].astype(bool)]
    out: dict = {}
    for kind, group in fraud.groupby("fraud_type"):
        out[kind] = {"transactions": int(len(group))}
        for system, mask in flagged.items():
            out[kind][system] = float(mask.loc[group.index].mean())
    return out


def _rule_table(test: pd.DataFrame, rules) -> list[dict]:
    """Per-rule behaviour on the test period: how often it fires and how often it is right."""
    fired, _ = evaluate_rules(test, rules)
    y = test["is_fraud"].astype(bool)
    all_events = max(test.loc[y, "fraud_event_id"].nunique(), 1)
    table = []
    for rule in rules:
        mask = fired[rule.id]
        table.append(
            {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "weight": rule.weight,
                "escalate": rule.escalate,
                "fires": int(mask.sum()),
                "precision": float(y[mask].mean()) if mask.any() else 0.0,
                "event_recall": float(test.loc[mask & y, "fraud_event_id"].nunique() / all_events),
            }
        )
    return table
