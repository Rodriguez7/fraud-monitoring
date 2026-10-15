import pandas as pd
import pytest

from fraudwatch.rules import Condition, Rule, escalated, evaluate_rules, load_rules


def frame(**cols):
    return pd.DataFrame(cols)


def test_default_rules_load_and_are_valid():
    rules = load_rules()
    assert len(rules) >= 6
    assert all(0 < r.weight < 1 for r in rules)
    assert len({r.id for r in rules}) == len(rules)


def test_conditions_cover_every_operator():
    df = frame(x=[1, 5, 10])
    assert Condition("x", ">=", 5).evaluate(df).tolist() == [False, True, True]
    assert Condition("x", "<", 5).evaluate(df).tolist() == [True, False, False]
    assert Condition("x", "==", 5).evaluate(df).tolist() == [False, True, False]
    assert Condition("x", "!=", 5).evaluate(df).tolist() == [True, False, True]
    assert Condition("x", "in", [1, 10]).evaluate(df).tolist() == [True, False, True]


def test_unsupported_operator_is_rejected():
    with pytest.raises(ValueError):
        Condition("x", "~", 1).evaluate(frame(x=[1]))


def test_rule_requires_all_conditions():
    rule = Rule("T", "t", "d", 0.5, (Condition("a", ">=", 1), Condition("b", "==", 1)))
    df = frame(a=[1, 1, 0], b=[1, 0, 1])
    assert rule.evaluate(df).tolist() == [True, False, False]


def test_noisy_or_combines_evidence_and_stays_below_one():
    r1 = Rule("A", "a", "d", 0.5, (Condition("x", "==", 1),))
    r2 = Rule("B", "b", "d", 0.5, (Condition("y", "==", 1),))
    df = frame(x=[0, 1, 1], y=[0, 0, 1])
    fired, prob = evaluate_rules(df, [r1, r2])
    assert prob.round(4).tolist() == [0.0, 0.5, 0.75]
    assert fired.sum().to_dict() == {"A": 2, "B": 1}


def test_escalation_flag():
    r1 = Rule("A", "a", "d", 0.5, (Condition("x", "==", 1),), escalate=True)
    r2 = Rule("B", "b", "d", 0.5, (Condition("y", "==", 1),))
    df = frame(x=[1, 0, 0], y=[0, 1, 0])
    fired, _ = evaluate_rules(df, [r1, r2])
    assert escalated(fired, [r1, r2]).tolist() == [True, False, False]


def test_invalid_weight_is_rejected(tmp_path):
    bad = tmp_path / "rules.yaml"
    bad.write_text(
        "rules:\n  - {id: X, name: x, description: d, weight: 1.5, all: [{feature: a, op: '>', value: 0}]}\n"
    )
    with pytest.raises(ValueError):
        load_rules(bad)


def test_duplicate_rule_ids_are_rejected(tmp_path):
    cond = "all: [{feature: a, op: '>', value: 0}]"
    bad = tmp_path / "rules.yaml"
    bad.write_text(
        f"rules:\n  - {{id: X, name: x, description: d, weight: 0.5, {cond}}}\n"
        f"  - {{id: X, name: y, description: d, weight: 0.5, {cond}}}\n"
    )
    with pytest.raises(ValueError):
        load_rules(bad)


def test_every_default_rule_references_real_features():
    from fraudwatch.features import NUMERIC_FEATURES

    for rule in load_rules():
        for cond in rule.conditions:
            assert cond.feature in NUMERIC_FEATURES, f"{rule.id}: unknown feature {cond.feature}"
