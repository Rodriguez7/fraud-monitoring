"""The SQL feature pipeline: correctness on hand-built cases and absence of look-ahead."""

import pandas as pd


def row(ts, **kw):
    return {"ts": ts, **kw}


def test_velocity_counts_only_previous_transactions(features_con):
    rows = [row(f"2026-02-01 10:0{m}:00") for m in range(5)]
    f = features_con(rows)
    # the k-th transaction sees k previous ones, never itself or later ones
    assert f["cnt_10m"].tolist() == [0, 1, 2, 3, 4]


def test_window_boundary_excludes_old_transactions(features_con):
    f = features_con([row("2026-02-01 10:00:00"), row("2026-02-01 10:30:00")])
    assert f["cnt_10m"].tolist() == [0, 0]
    assert f["cnt_1h"].tolist() == [0, 1]


def test_features_do_not_change_when_future_rows_are_added(features_con):
    """Appending rows seconds after a transaction must not alter that transaction's features."""
    base = [row(f"2026-02-0{d} 12:00:00", amount_eur=20.0 + d) for d in range(1, 6)]
    before = features_con(base)
    future = [
        row("2026-02-05 12:01:00", amount_eur=9000.0, merchant_country="US"),
        row("2026-02-05 12:03:00", amount_eur=9000.0, merchant_country="US"),
        row("2026-02-05 12:08:00", amount_eur=9000.0, merchant_country="US"),
    ]
    after = features_con(base + future).iloc[: len(base)].reset_index(drop=True)
    cols = [c for c in before.columns if c != "txn_id"]
    pd.testing.assert_frame_equal(before[cols], after[cols], check_exact=False)


def test_new_device_is_true_only_within_24_hours_of_first_sight(features_con):
    rows = [
        row("2026-02-01 10:00:00", channel="online", device_id="D1"),
        row("2026-02-01 18:00:00", channel="online", device_id="D1"),
        row("2026-02-03 10:00:00", channel="online", device_id="D1"),
        row("2026-02-03 10:05:00", channel="online", device_id="D2"),
    ]
    f = features_con(rows)
    assert f["is_new_device"].tolist() == [1, 1, 0, 1]


def test_card_present_has_no_device_flag(features_con):
    f = features_con([row("2026-02-01 10:00:00")])
    assert f["is_new_device"].iloc[0] == 0


def test_cross_border_and_transfer_flags(features_con):
    rows = [
        row("2026-02-01 10:00:00", merchant_country="DE"),
        row(
            "2026-02-01 11:00:00",
            txn_type="bank_transfer",
            channel="app",
            merchant_country="AT",
            merchant_id=None,
            payee_id="P1",
            amount_eur=500.0,
        ),
    ]
    f = features_con(rows)
    assert f["is_cross_border"].tolist() == [1, 0]
    assert f["is_transfer"].tolist() == [0, 1]
    assert f["is_new_payee"].tolist() == [0, 1]


def test_impossible_travel_needs_a_country_change_within_two_hours(features_con):
    rows = [
        row("2026-02-01 10:00:00", merchant_country="AT"),
        row("2026-02-01 10:45:00", merchant_country="US"),
        row("2026-02-01 18:00:00", merchant_country="AT"),
    ]
    f = features_con(rows)
    assert f["country_changed_fast"].tolist() == [0, 1, 0]


def test_near_threshold_structuring_features(features_con):
    rows = [
        row(
            f"2026-02-0{d} 10:00:00",
            txn_type="bank_transfer",
            channel="app",
            merchant_id=None,
            payee_id="P9",
            amount_eur=a,
        )
        for d, a in ((1, 9500.0), (2, 9400.0), (3, 12000.0))
    ]
    f = features_con(rows)
    assert f["is_near_threshold"].tolist() == [1, 1, 0]
    assert f["near_threshold_48h"].tolist() == [0, 1, 2]


def test_amount_z_is_high_for_outlier_and_zero_history_is_safe(features_con):
    rows = [row(f"2026-02-0{d} 12:00:00", amount_eur=20.0) for d in range(1, 6)]
    rows.append(row("2026-02-07 12:00:00", amount_eur=2000.0))
    f = features_con(rows)
    assert f["amount_z"].iloc[0] == 0  # first transaction: no history, no divide-by-zero
    assert f["amount_z"].iloc[-1] > 20


def test_customers_are_isolated(features_con):
    rows = [row("2026-02-01 10:00:00", customer_id="A"), row("2026-02-01 10:01:00", customer_id="B")]
    f = features_con(rows)
    assert f["cnt_10m"].tolist() == [0, 0]
