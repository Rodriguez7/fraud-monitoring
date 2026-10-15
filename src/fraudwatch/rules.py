"""A small, declarative rules engine.

Rules live in `rules.yaml` so a fraud analyst can read and change them without touching Python.
Conditions are evaluated as vectorised pandas comparisons; there is no `eval`.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: object

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.feature]
        if self.op == "in":
            return col.isin(self.value)
        if self.op not in _OPS:
            raise ValueError(f"Unsupported operator: {self.op!r}")
        return _OPS[self.op](col, self.value)


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    description: str
    weight: float
    conditions: tuple[Condition, ...]
    escalate: bool = False

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for cond in self.conditions:
            mask &= cond.evaluate(df)
        return mask


def load_rules(path: Path | None = None) -> list[Rule]:
    text = path.read_text() if path else resources.files("fraudwatch").joinpath("rules.yaml").read_text()
    rules = []
    for raw in yaml.safe_load(text)["rules"]:
        weight = float(raw["weight"])
        if not 0 < weight < 1:
            raise ValueError(f"Rule {raw['id']}: weight must be between 0 and 1")
        conds = tuple(Condition(c["feature"], c["op"], c["value"]) for c in raw["all"])
        rules.append(
            Rule(raw["id"], raw["name"], raw["description"], weight, conds, bool(raw.get("escalate", False)))
        )
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate rule ids")
    return rules


def evaluate_rules(df: pd.DataFrame, rules: list[Rule]) -> tuple[pd.DataFrame, pd.Series]:
    """Return a boolean frame (one column per rule id) and the noisy-OR rule probability."""
    fired = pd.DataFrame({r.id: r.evaluate(df).to_numpy() for r in rules}, index=df.index)
    weights = np.array([r.weight for r in rules])
    survive = np.where(fired.to_numpy(), 1.0 - weights, 1.0).prod(axis=1)
    return fired, pd.Series(1.0 - survive, index=df.index, name="rule_prob")


def escalated(fired: pd.DataFrame, rules: list[Rule]) -> pd.Series:
    """True where at least one `escalate: true` rule fired: those transactions always get a case."""
    ids = [r.id for r in rules if r.escalate]
    if not ids:
        return pd.Series(False, index=fired.index)
    return fired[ids].any(axis=1)
