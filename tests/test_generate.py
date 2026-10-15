import pandas as pd

from fraudwatch.generate import FRAUD_TYPES, generate


def test_generation_is_deterministic():
    _, a = generate(n_customers=150, days=60, burn_in=10, seed=11)
    _, b = generate(n_customers=150, days=60, burn_in=10, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    _, a = generate(n_customers=150, days=60, burn_in=10, seed=1)
    _, b = generate(n_customers=150, days=60, burn_in=10, seed=2)
    assert not a["amount_eur"].equals(b["amount_eur"])


def test_customer_timestamps_are_unique():
    """Window features rely on (customer, ts) being unique so no same-instant peers leak."""
    _, t = generate(n_customers=200, days=60, burn_in=10, seed=5)
    assert not t.duplicated(["customer_id", "ts"]).any()


def test_no_fraud_during_burn_in():
    _, t = generate(n_customers=300, days=60, burn_in=15, seed=5)
    cutoff = pd.Timestamp("2026-01-01") + pd.Timedelta(days=15)
    assert t.loc[t["is_fraud"], "ts"].min() >= cutoff


def test_fraud_rate_and_typologies_are_plausible():
    _, t = generate(n_customers=600, days=60, burn_in=10, victim_share=0.05, seed=9)
    assert 0.0005 < t["is_fraud"].mean() < 0.03
    assert set(t["fraud_type"].dropna()) <= set(FRAUD_TYPES)
    assert t.loc[t["is_fraud"], "fraud_event_id"].notna().all()


def test_ids_do_not_leak_the_label():
    _, t = generate(n_customers=300, days=60, burn_in=10, seed=5)
    devices = t["device_id"].dropna()
    assert devices.str.startswith("D").all()
    assert t["payee_id"].dropna().str.startswith("P").all()


def test_amounts_are_positive_and_rounded():
    _, t = generate(n_customers=150, days=60, burn_in=10, seed=5)
    assert (t["amount_eur"] > 0).all()
    assert (t["amount_eur"].round(2) == t["amount_eur"]).all()
