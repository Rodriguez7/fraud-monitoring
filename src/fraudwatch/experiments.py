"""Experiment: how does each system cope with a fraud typology the model never saw in training?

New fraud patterns appear faster than models are retrained. Here the model is trained with all
transactions of one typology removed. Rules written by an analyst from a threat report can
still catch it; the ML model cannot know what it was never shown.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import Paths, SplitConfig
from .model import evaluate, train_model
from .pipeline import START, _features_df

TYPOLOGIES = ["account_takeover", "card_testing", "cash_out", "structuring", "impossible_travel"]


def unseen_typology(paths: Paths, split: SplitConfig, seed: int = 7) -> pd.DataFrame:
    df = _features_df(paths)
    rows = []
    for kind in TYPOLOGIES:
        bundle = train_model(df[df["fraud_type"] != kind], split, START, seed=seed)
        report = evaluate(df, bundle, START)
        stats = report["by_typology"].get(kind)
        if not stats:
            continue
        rows.append(
            {
                "unseen_typology": kind,
                "test_transactions": stats["transactions"],
                "rules_only": stats["rules"],
                "ml_only": stats["ml"],
                "hybrid": stats["hybrid"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    paths = Paths()
    table = unseen_typology(paths, SplitConfig())
    out = paths.base / "reports" / "unseen_typology.json"
    out.write_text(json.dumps(table.to_dict("records"), indent=2))
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
