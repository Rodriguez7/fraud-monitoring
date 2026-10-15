import pandas as pd

from fraudwatch.model import review_threshold
from fraudwatch.rules import Condition, Rule
from fraudwatch.scoring import combine, explain_rows, risk_band


def test_combine_is_noisy_or():
    rule_p = pd.Series([0.0, 0.5, 0.0])
    ml_p = pd.Series([0.0, 0.5, 0.9])
    assert combine(rule_p, ml_p).round(3).tolist() == [0.0, 0.75, 0.9]


def test_combine_never_lowers_either_source():
    rule_p = pd.Series([0.1, 0.6, 0.3])
    ml_p = pd.Series([0.2, 0.1, 0.9])
    out = combine(rule_p, ml_p)
    assert (out >= rule_p).all() and (out >= ml_p).all() and (out <= 1).all()


def test_review_threshold_fits_the_budget():
    score = pd.Series(range(1000), dtype=float) / 1000
    forced = pd.Series(False, index=score.index)
    thr = review_threshold(score, forced, budget=0.05)
    assert (score >= thr).sum() == 50


def test_escalated_alerts_consume_the_budget_first():
    score = pd.Series(range(1000), dtype=float) / 1000
    forced = pd.Series(False, index=score.index)
    forced.iloc[:30] = True  # 30 escalated low-score transactions
    thr = review_threshold(score, forced, budget=0.05)
    by_score = ((score >= thr) & ~forced).sum()
    assert by_score == 20


def test_when_escalations_exceed_budget_nothing_else_alerts():
    score = pd.Series([0.1, 0.9, 0.5])
    forced = pd.Series([True, True, True])
    thr = review_threshold(score, forced, budget=0.0001)
    assert (score >= thr).sum() == 0


def test_risk_band_thresholds():
    s = pd.Series([0.2, 0.7, 0.99])
    assert risk_band(s, 0.5, 0.9).tolist() == ["allow", "review", "block"]


def test_explanations_name_fired_rules_and_top_drivers():
    rule = Rule("R1", "velocity", "Many transactions in 10 minutes", 0.5, (Condition("cnt_10m", ">=", 5),))
    rows = pd.DataFrame({"cnt_10m": [7.0], "is_new_device": [1.0]}, index=[0])
    fired = pd.DataFrame({"R1": [True]}, index=[0])
    contrib = pd.DataFrame({"cnt_10m": [0.4], "is_new_device": [0.3]}, index=[0])
    reasons = explain_rows(rows, fired, [rule], contrib, {"cnt_10m": 0.0, "is_new_device": 0.0})[0]
    assert reasons[0].startswith("Rule R1")
    assert any("7 other transactions" in r for r in reasons)
    assert any("device first seen" in r for r in reasons)


def test_small_contributions_are_not_reported():
    rows = pd.DataFrame({"cnt_10m": [1.0]}, index=[0])
    fired = pd.DataFrame({"R1": [False]}, index=[0])
    contrib = pd.DataFrame({"cnt_10m": [0.001]}, index=[0])
    reasons = explain_rows(rows, fired, [], contrib, {"cnt_10m": 0.0})[0]
    assert len(reasons) == 1 and "No single dominant driver" in reasons[0]
