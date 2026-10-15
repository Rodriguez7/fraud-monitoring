"""End-to-end run on a small simulation."""

import json

import duckdb
import pandas as pd
import pytest

from fraudwatch import cases
from fraudwatch.features import LABEL_COLUMNS, NUMERIC_FEATURES
from fraudwatch.model import load_bundle
from fraudwatch.pipeline import run_features, run_generate, run_score, run_train


@pytest.fixture
def pipeline(tmp_paths, small_params):
    run_generate(tmp_paths, small_params)
    run_features(tmp_paths)
    report = run_train(tmp_paths, small_params)
    summary = run_score(tmp_paths, small_params)
    return tmp_paths, small_params, report, summary


def test_labels_are_never_model_features():
    assert not set(NUMERIC_FEATURES) & set(LABEL_COLUMNS)


def test_metrics_report_has_all_systems_and_sane_ranges(pipeline):
    _, _, report, _ = pipeline
    assert set(report["systems"]) == {"rules", "ml", "hybrid"}
    for entry in report["systems"].values():
        assert 0 <= entry["precision"] <= 1 and 0 <= entry["txn_recall"] <= 1
        assert 0.5 <= entry["roc_auc"] <= 1
    assert report["thresholds"]["block"] >= report["thresholds"]["review"]


def test_model_beats_a_random_ranking(pipeline):
    _, _, report, _ = pipeline
    assert report["systems"]["ml"]["roc_auc"] > 0.9


def test_split_has_no_overlap_in_time(pipeline):
    paths, params, _, _ = pipeline
    con = duckdb.connect(str(paths.db), read_only=True)
    day = con.execute("SELECT MIN(ts), MAX(ts) FROM alerts").fetchone()
    start = pd.Timestamp("2026-01-01")
    assert (pd.Timestamp(day[0]) - start).days >= params.split.train_end  # alerts only after training


def test_alerts_and_cases_are_consistent(pipeline):
    paths, _, _, summary = pipeline
    con = duckdb.connect(str(paths.db), read_only=True)
    n_alerts = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    n_cases = con.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    assert n_alerts == n_cases == summary["alerts"] > 0
    assert con.execute("SELECT COUNT(*) FROM alerts WHERE band NOT IN ('review','block')").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM alerts WHERE risk_score > 99.9").fetchone()[0] == 0


def test_every_alert_has_readable_reasons(pipeline):
    paths, _, _, _ = pipeline
    con = duckdb.connect(str(paths.db), read_only=True)
    for (raw,) in con.execute("SELECT reasons FROM alerts").fetchall():
        reasons = json.loads(raw)
        assert reasons and all(isinstance(r, str) and r for r in reasons)


def test_case_decisions_persist_and_summary_updates(pipeline):
    paths, _, _, _ = pipeline
    con = duckdb.connect(str(paths.db))
    first = cases.queue(con, "open").iloc[0]["txn_id"]
    cases.decide(con, first, "confirmed_fraud", "analyst", "looks like card testing")
    assert cases.summary(con)["confirmed"] == 1
    assert first not in cases.queue(con, "open")["txn_id"].tolist()
    cases.decide(con, first, "open")
    assert cases.summary(con)["confirmed"] == 0
    with pytest.raises(ValueError):
        cases.decide(con, first, "maybe")


def test_queue_is_sorted_by_risk(pipeline):
    paths, _, _, _ = pipeline
    con = duckdb.connect(str(paths.db), read_only=True)
    scores = cases.queue(con, "open")["risk_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_saved_model_scores_new_rows(pipeline):
    paths, _, _, _ = pipeline
    bundle = load_bundle(paths)
    con = duckdb.connect(str(paths.db), read_only=True)
    sample = con.execute("SELECT * FROM features LIMIT 50").df()
    p = bundle.predict_proba(sample)
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()
