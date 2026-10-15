"""Synthetic transaction generator.

Real payment data is confidential and the public data sets either need a login (PaySim)
or have anonymised features (ULB credit card data) that cannot be explained to an analyst.
This generator creates business-readable transactions with seeded randomness so every run
is reproducible.

Two things make the problem realistic instead of trivially separable:

* **Legitimate look-alikes.** Customers travel abroad, replace their phone, make one-off large
  purchases and go on shopping sprees. Each of these trips a naive rule.
* **Mild fraud.** A share of fraud events use smaller amounts or a known device, so no single
  signal catches everything.

Fraud typologies (the `fraud_type` column is for analysis only and is never used as a feature):

======================  ==============================================================
account_takeover        New device and country, high-value electronics / gift cards / crypto
card_testing            Many tiny online payments at new merchants within minutes
cash_out                Several transfers to new payees within hours
structuring             Repeated transfers just below the 10,000 EUR reporting threshold
impossible_travel       Card-present payments in two distant countries within the hour
======================  ==============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORIES, COUNTRIES

HOME_COUNTRIES = ["AT", "DE", "IT", "FR", "ES", "NL", "CZ", "HU"]
HOME_WEIGHTS = np.array([0.72, 0.12, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02])
TRIP_COUNTRIES = ["IT", "FR", "ES", "NL", "CZ", "HU", "GB", "US", "TR"]
FRAUD_COUNTRIES = ["US", "TR", "NG", "GB"]

CATEGORY_BASE_P = np.array([0.30, 0.17, 0.12, 0.05, 0.04, 0.05, 0.09, 0.12, 0.03, 0.03])
CATEGORY_AMOUNT_MULT = {
    "grocery": 1.0,
    "restaurant": 1.2,
    "transport": 0.6,
    "utilities": 3.0,
    "travel": 6.0,
    "electronics": 8.0,
    "fashion": 2.5,
    "online_services": 0.8,
    "gift_cards": 1.5,
    "crypto_exchange": 4.0,
}
ONLINE_CATEGORIES = {"online_services", "electronics", "gift_cards", "crypto_exchange"}
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 7, 8, 7, 6, 6, 6, 7, 8, 8, 6, 4, 3, 2], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()
FRAUD_TYPES = ["account_takeover", "card_testing", "cash_out", "structuring", "impossible_travel"]
FRAUD_SHARES = [0.35, 0.25, 0.15, 0.10, 0.15]

COLUMNS = [
    "txn_id",
    "ts",
    "customer_id",
    "txn_type",
    "amount_eur",
    "merchant_id",
    "merchant_category",
    "merchant_country",
    "channel",
    "device_id",
    "payee_id",
    "is_fraud",
    "fraud_type",
    "fraud_event_id",
]


def generate(
    n_customers: int = 5000,
    days: int = 120,
    burn_in: int = 30,
    victim_share: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(customers, transactions)`.

    Fraud is only injected after the `burn_in` days so every fraud transaction has history.
    """
    rng = np.random.default_rng(seed)
    customers = _make_customers(rng, n_customers, days)
    normal = _normal_transactions(rng, customers, days)
    fraud = _fraud_transactions(rng, customers, normal, days, burn_in, victim_share)
    txns = pd.concat([normal, fraud], ignore_index=True)
    txns = _unique_timestamps(txns)
    txns = txns.sort_values("ts", kind="stable").reset_index(drop=True)
    txns["txn_id"] = [f"T{i:08d}" for i in range(len(txns))]
    cust_out = customers[["customer_id", "home_country", "account_age_days"]].copy()
    return cust_out, txns[COLUMNS]


# --------------------------------------------------------------------------- customers


def _make_customers(rng: np.random.Generator, n: int, days: int) -> pd.DataFrame:
    idx = np.arange(n)
    home = rng.choice(HOME_COUNTRIES, size=n, p=HOME_WEIGHTS)
    has_trip = rng.random(n) < 0.12
    trip_start = np.where(has_trip, rng.integers(10, days - 12, n), -1)
    trip_len = rng.integers(4, 10, n)
    trip_country = rng.choice(TRIP_COUNTRIES, size=n)
    replaces_phone = rng.random(n) < 0.04
    cat_p = CATEGORY_BASE_P * rng.lognormal(0, 0.5, (n, len(CATEGORIES)))
    cat_p /= cat_p.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        {
            "customer_idx": idx,
            "customer_id": [f"C{i:06d}" for i in idx],
            "home_country": home,
            "account_age_days": rng.integers(30, 3000, n),
            "daily_rate": np.clip(rng.lognormal(np.log(0.6), 0.6, n), 0.12, 3.5),
            "amount_mu": rng.normal(3.2, 0.35, n),
            "has_second_device": rng.random(n) < 0.15,
            "replace_day": np.where(replaces_phone, rng.integers(20, days - 5, n), 10**6),
            "trip_start": trip_start,
            "trip_end": np.where(has_trip, trip_start + trip_len, -1),
            "trip_country": np.where(has_trip, trip_country, ""),
            "cat_p": list(cat_p),
        }
    )


# --------------------------------------------------------------------------- normal behaviour


def _normal_transactions(rng: np.random.Generator, c: pd.DataFrame, days: int) -> pd.DataFrame:
    n_tx = rng.poisson(c["daily_rate"].to_numpy() * days)
    idx = np.repeat(c["customer_idx"].to_numpy(), n_tx)
    n = len(idx)
    day = rng.integers(0, days, n)
    sec = _random_seconds(rng, n)
    ts = _to_ts(day, sec)

    cat_p = np.vstack(c["cat_p"].to_numpy())
    cat_idx = _sample_rows(rng, cat_p[idx])
    category = np.array(CATEGORIES)[cat_idx]
    is_transfer = rng.random(n) < 0.06

    mult = pd.Series(category).map(CATEGORY_AMOUNT_MULT).to_numpy()
    amount = rng.lognormal(c["amount_mu"].to_numpy()[idx], 0.75) * mult
    big = rng.random(n) < 0.012  # one-off large purchase
    amount = np.where(big, amount * rng.uniform(4, 12, n), amount)
    transfer_amount = rng.lognormal(c["amount_mu"].to_numpy()[idx] + 1.8, 0.9)
    amount = np.where(is_transfer, np.clip(transfer_amount, 5, 4000), np.clip(amount, 0.5, 6000))

    home = c["home_country"].to_numpy()[idx]
    in_trip = (c["trip_start"].to_numpy()[idx] <= day) & (day < c["trip_end"].to_numpy()[idx])
    location = np.where(in_trip, c["trip_country"].to_numpy()[idx], home)

    online = np.isin(category, list(ONLINE_CATEGORIES))
    online = np.where(online, rng.random(n) < 0.9, rng.random(n) < 0.15)
    channel = np.where(is_transfer, "app", np.where(online, "online", "card_present"))
    foreign_online = (channel == "online") & (rng.random(n) < 0.15)
    merchant_country = np.where(foreign_online, rng.choice(COUNTRIES, n), location)

    k_main = (idx * 7 + cat_idx * 13) % 40
    habit = rng.random(n) < 0.85
    k = np.where(habit, k_main + rng.integers(0, 3, n), rng.integers(0, 40, n))
    merchant_id = np.array(
        [f"{CATEGORIES[ci][:4]}_{co}_{kk}" for ci, co, kk in zip(cat_idx, merchant_country, k, strict=True)]
    )

    use_second = c["has_second_device"].to_numpy()[idx] & (rng.random(n) < 0.15)
    replaced = day >= c["replace_day"].to_numpy()[idx]
    suffix = np.where(replaced, 3, np.where(use_second, 2, 1))
    device = np.where(
        channel == "card_present", None, [f"D{i:06d}-{s}" for i, s in zip(idx, suffix, strict=True)]
    )
    payee_slot = rng.integers(0, 4, n)
    new_payee = rng.random(n) < 0.05
    payee = np.where(
        is_transfer,
        np.where(
            new_payee,
            [f"P{i:06d}-n{j}" for i, j in zip(idx, rng.integers(0, 10**6, n), strict=True)],
            [f"P{i:06d}-{s}" for i, s in zip(idx, payee_slot, strict=True)],
        ),
        None,
    )
    merchant_id = np.where(is_transfer, None, merchant_id)
    category = np.where(is_transfer, "transfer", category)
    merchant_country = np.where(is_transfer, home, merchant_country)

    base = pd.DataFrame(
        {
            "ts": ts,
            "customer_id": c["customer_id"].to_numpy()[idx],
            "txn_type": np.where(is_transfer, "bank_transfer", "card_payment"),
            "amount_eur": np.round(amount, 2),
            "merchant_id": merchant_id,
            "merchant_category": category,
            "merchant_country": merchant_country,
            "channel": channel,
            "device_id": device,
            "payee_id": payee,
            "is_fraud": False,
            "fraud_type": None,
            "fraud_event_id": None,
        }
    )
    extras = [_shopping_sprees(rng, c, days), _legit_confounders(rng, c, days)]
    return pd.concat([base, *extras], ignore_index=True)


def _shopping_sprees(rng: np.random.Generator, c: pd.DataFrame, days: int) -> pd.DataFrame:
    """Legitimate bursts: 5-8 card-present payments within ~20 minutes. Trips velocity rules."""
    n_sprees = int(len(c) * 0.08)
    rows = []
    for cust in rng.choice(len(c), n_sprees, replace=False):
        row = c.iloc[cust]
        start = _to_ts(np.array([rng.integers(5, days)]), np.array([rng.integers(36000, 68000)]))[0]
        for j in range(rng.integers(5, 9)):
            rows.append(
                {
                    "ts": start + pd.Timedelta(seconds=int(j * rng.integers(60, 240))),
                    "customer_id": row["customer_id"],
                    "txn_type": "card_payment",
                    "amount_eur": round(float(rng.lognormal(3.4, 0.5)), 2),
                    "merchant_id": f"fash_{row['home_country']}_{rng.integers(0, 6)}",
                    "merchant_category": "fashion",
                    "merchant_country": row["home_country"],
                    "channel": "card_present",
                    "device_id": None,
                    "payee_id": None,
                    "is_fraud": False,
                    "fraud_type": None,
                    "fraud_event_id": None,
                }
            )
    return pd.DataFrame(rows)


def _legit_row(ts, cust, **kw) -> dict:
    base = {
        "ts": ts,
        "customer_id": cust["customer_id"],
        "txn_type": "card_payment",
        "amount_eur": 0.0,
        "merchant_id": None,
        "merchant_category": None,
        "merchant_country": cust["home_country"],
        "channel": "online",
        "device_id": None,
        "payee_id": None,
        "is_fraud": False,
        "fraud_type": None,
        "fraud_event_id": None,
    }
    base.update(kw)
    return base


def _pick(rng: np.random.Generator, c: pd.DataFrame, share: float) -> list[pd.Series]:
    return [c.iloc[i] for i in rng.choice(len(c), int(len(c) * share), replace=False)]


def _legit_confounders(rng: np.random.Generator, c: pd.DataFrame, days: int) -> pd.DataFrame:
    """Genuine behaviour that looks like fraud to a single rule. These create false positives."""
    rows: list[dict] = []

    def when(low=8):
        return _to_ts(np.array([rng.integers(low, days - 2)]), np.array([rng.integers(30000, 75000)]))[0]

    # Micro-payment bursts: app-store or gaming top-ups, 3-6 small online payments in minutes.
    for cust in _pick(rng, c, 0.04):
        t = when()
        for _ in range(int(rng.integers(3, 7))):
            t += pd.Timedelta(seconds=int(rng.integers(25, 150)))
            rows.append(
                _legit_row(
                    t,
                    cust,
                    amount_eur=round(float(rng.uniform(0.9, 9.5)), 2),
                    merchant_id=f"onli_{cust['home_country']}_{rng.integers(40, 150)}",
                    merchant_category="online_services",
                    device_id=f"D{int(cust['customer_idx']):06d}-1",
                )
            )
    # Bill batches: 3-5 transfers to new payees within a couple of hours (moving house, rent).
    for cust in _pick(rng, c, 0.03):
        t = when()
        for _ in range(int(rng.integers(3, 6))):
            t += pd.Timedelta(minutes=int(rng.integers(8, 40)))
            rows.append(
                _legit_row(
                    t,
                    cust,
                    txn_type="bank_transfer",
                    channel="app",
                    amount_eur=round(float(rng.uniform(120, 1600)), 2),
                    merchant_category="transfer",
                    device_id=f"D{int(cust['customer_idx']):06d}-1",
                    payee_id=f"P{int(cust['customer_idx']):06d}-b{rng.integers(0, 10**6)}",
                )
            )
    # Border day trips: home payment, then a payment in a neighbouring country within 2 hours.
    for cust in _pick(rng, c, 0.05):
        if cust["home_country"] != "AT":
            continue
        t = when()
        near = str(rng.choice(["DE", "CZ", "HU"]))
        rows.append(
            _legit_row(
                t,
                cust,
                channel="card_present",
                amount_eur=round(float(rng.uniform(8, 60)), 2),
                merchant_id=f"groc_AT_{rng.integers(0, 3)}",
                merchant_category="grocery",
            )
        )
        t += pd.Timedelta(minutes=int(rng.integers(55, 110)))
        rows.append(
            _legit_row(
                t,
                cust,
                channel="card_present",
                amount_eur=round(float(rng.uniform(15, 220)), 2),
                merchant_id=f"fash_{near}_{rng.integers(0, 6)}",
                merchant_category="fashion",
                merchant_country=near,
            )
        )
    # New phone bought abroad: a young device, a foreign electronics purchase, high value.
    for cust in _pick(rng, c, 0.012):
        t = when()
        country = str(rng.choice(TRIP_COUNTRIES))
        for _ in range(int(rng.integers(1, 3))):
            t += pd.Timedelta(minutes=int(rng.integers(5, 90)))
            rows.append(
                _legit_row(
                    t,
                    cust,
                    amount_eur=round(float(rng.uniform(300, 1400)), 2),
                    merchant_id=f"elec_{country}_{rng.integers(0, 12)}",
                    merchant_category="electronics",
                    merchant_country=country,
                    device_id=f"D{int(cust['customer_idx']):06d}-9",
                )
            )
    # Property purchase: two large transfers just below 10,000 EUR on the same day.
    for cust in _pick(rng, c, 0.006):
        t = when()
        for _ in range(2):
            t += pd.Timedelta(hours=int(rng.integers(1, 6)))
            rows.append(
                _legit_row(
                    t,
                    cust,
                    txn_type="bank_transfer",
                    channel="app",
                    amount_eur=round(float(rng.uniform(9050, 9900)), 2),
                    merchant_category="transfer",
                    device_id=f"D{int(cust['customer_idx']):06d}-1",
                    payee_id=f"P{int(cust['customer_idx']):06d}-prop",
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fraud


def _fraud_transactions(
    rng: np.random.Generator,
    c: pd.DataFrame,
    normal: pd.DataFrame,
    days: int,
    burn_in: int,
    victim_share: float,
) -> pd.DataFrame:
    n_events = max(1, int(len(c) * victim_share))
    victims = rng.choice(len(c), n_events, replace=False)
    kinds = rng.choice(FRAUD_TYPES, size=n_events, p=FRAUD_SHARES)
    builders = {
        "account_takeover": _account_takeover,
        "card_testing": _card_testing,
        "cash_out": _cash_out,
        "structuring": _structuring,
        "impossible_travel": _impossible_travel,
    }
    rows: list[dict] = []
    for event_no, (victim, kind) in enumerate(zip(victims, kinds, strict=True)):
        cust = c.iloc[victim]
        day = int(rng.integers(burn_in + 2, days - 1))
        start = _to_ts(np.array([day]), np.array([int(rng.choice(86400, p=_hour_second_p()))]))[0]
        mild = rng.random() < 0.5
        rows.extend(builders[kind](rng, cust, start, mild, f"E{event_no:05d}"))
    return pd.DataFrame(rows)


def _fraud_device(event: str) -> str:
    """Looks like any customer device id, so the id itself carries no label."""
    return f"D{900000 + int(event[1:])}-1"


def _fraud_payee(event: str, i: int) -> str:
    return f"P{900000 + int(event[1:])}-{i}"


def _hour_second_p() -> np.ndarray:
    p = np.repeat(HOUR_WEIGHTS / 3600, 3600)
    return p / p.sum()


def _row(ts, cust, kind, event, **kw) -> dict:
    base = {
        "ts": ts,
        "customer_id": cust["customer_id"],
        "txn_type": "card_payment",
        "amount_eur": 0.0,
        "merchant_id": None,
        "merchant_category": None,
        "merchant_country": None,
        "channel": "online",
        "device_id": None,
        "payee_id": None,
        "is_fraud": True,
        "fraud_type": kind,
        "fraud_event_id": event,
    }
    base.update(kw)
    return base


def _account_takeover(rng, cust, start, mild, event):
    country = str(rng.choice(FRAUD_COUNTRIES if not mild else TRIP_COUNTRIES))
    device = _fraud_device(event)
    if mild:
        device = f"D{int(cust['customer_idx']):06d}-1"  # known device: harder to catch
    out, t = [], start
    for _ in range(int(rng.integers(3, 9))):
        t += pd.Timedelta(seconds=int(rng.integers(60, 1500)))
        cat = str(rng.choice(["electronics", "gift_cards", "crypto_exchange"]))
        amount = rng.lognormal(4.5 if mild else 6.0, 0.5)
        out.append(
            _row(
                t,
                cust,
                "account_takeover",
                event,
                amount_eur=round(float(np.clip(amount, 40 if mild else 150, 3000)), 2),
                merchant_id=f"{cat[:4]}_{country}_{rng.integers(200, 900)}",
                merchant_category=cat,
                merchant_country=country,
                device_id=device,
            )
        )
    return out


def _card_testing(rng, cust, start, mild, event):
    out, t = [], start
    n = int(rng.integers(3, 5) if mild else rng.integers(6, 16))
    for _ in range(n):
        t += pd.Timedelta(seconds=int(rng.integers(20, 90)))
        cat = str(rng.choice(["online_services", "gift_cards"]))
        out.append(
            _row(
                t,
                cust,
                "card_testing",
                event,
                amount_eur=round(float(rng.uniform(3, 9) if mild else rng.uniform(0.5, 5)), 2),
                merchant_id=f"{cat[:4]}_{rng.choice(COUNTRIES)}_{rng.integers(200, 900)}",
                merchant_category=cat,
                merchant_country=str(rng.choice(COUNTRIES)),
                device_id=_fraud_device(event),
            )
        )
    if not mild and rng.random() < 0.8:
        t += pd.Timedelta(seconds=int(rng.integers(60, 400)))
        out.append(
            _row(
                t,
                cust,
                "card_testing",
                event,
                amount_eur=round(float(rng.uniform(150, 900)), 2),
                merchant_id=f"elec_{rng.choice(COUNTRIES)}_{rng.integers(200, 900)}",
                merchant_category="electronics",
                merchant_country=str(rng.choice(COUNTRIES)),
                device_id=_fraud_device(event),
            )
        )
    return out


def _cash_out(rng, cust, start, mild, event):
    out, t = [], start
    payees = [_fraud_payee(event, i) for i in range(int(rng.integers(2, 4)))]
    device = _fraud_device(event) if rng.random() < 0.6 else f"D{int(cust['customer_idx']):06d}-1"
    for _ in range(int(rng.integers(5, 11))):
        t += pd.Timedelta(minutes=int(rng.integers(10, 60)))
        out.append(
            _row(
                t,
                cust,
                "cash_out",
                event,
                txn_type="bank_transfer",
                channel="app",
                amount_eur=round(float(rng.uniform(250, 900) if mild else rng.uniform(400, 1800)), 2),
                merchant_category="transfer",
                merchant_country=cust["home_country"],
                device_id=device,
                payee_id=str(rng.choice(payees)),
            )
        )
    return out


def _structuring(rng, cust, start, mild, event):
    out, t = [], start
    payee = _fraud_payee(event, 0)
    for _ in range(int(rng.integers(4, 9))):
        t += pd.Timedelta(hours=int(rng.integers(3, 14)))
        amount = rng.uniform(7000, 8900) if mild else rng.uniform(9000, 9990)
        out.append(
            _row(
                t,
                cust,
                "structuring",
                event,
                txn_type="bank_transfer",
                channel="app",
                amount_eur=round(float(amount), 2),
                merchant_category="transfer",
                merchant_country=cust["home_country"],
                device_id=f"D{int(cust['customer_idx']):06d}-1",
                payee_id=payee,
            )
        )
    return out


def _impossible_travel(rng, cust, start, mild, event):
    home = cust["home_country"]
    far = str(rng.choice(FRAUD_COUNTRIES))
    cat0 = str(rng.choice(["fashion", "restaurant", "electronics"]))
    # The first payment is the genuine cardholder: legitimate, but it anchors the travel check.
    out = [
        _row(
            start,
            cust,
            "impossible_travel",
            event,
            channel="card_present",
            amount_eur=round(float(rng.uniform(20, 120)), 2),
            merchant_id=f"{cat0[:4]}_{home}_{rng.integers(0, 40)}",
            merchant_category=cat0,
            merchant_country=home,
            is_fraud=False,
            fraud_type=None,
            fraud_event_id=None,
        )
    ]
    t = start
    for _ in range(int(rng.integers(1, 4))):
        t += pd.Timedelta(minutes=int(rng.integers(25, 100 if mild else 70)))
        cat = str(rng.choice(["electronics", "fashion", "travel"]))
        out.append(
            _row(
                t,
                cust,
                "impossible_travel",
                event,
                channel="card_present",
                amount_eur=round(float(rng.uniform(80, 600)), 2),
                merchant_id=f"{cat[:4]}_{far}_{rng.integers(200, 900)}",
                merchant_category=cat,
                merchant_country=far,
            )
        )
    return out


# --------------------------------------------------------------------------- helpers

_EPOCH = pd.Timestamp("2026-01-01")


def _to_ts(day: np.ndarray, sec: np.ndarray) -> pd.Series:
    return pd.Series(_EPOCH + pd.to_timedelta(day, unit="D") + pd.to_timedelta(sec, unit="s"))


def _random_seconds(rng: np.random.Generator, n: int) -> np.ndarray:
    hour = rng.choice(24, size=n, p=HOUR_WEIGHTS)
    return hour * 3600 + rng.integers(0, 3600, n)


def _sample_rows(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    """Sample one column index per row from a matrix of row-wise probabilities."""
    cum = p.cumsum(axis=1)
    u = rng.random((len(p), 1))
    return np.minimum((u > cum).sum(axis=1), p.shape[1] - 1)


def _unique_timestamps(txns: pd.DataFrame) -> pd.DataFrame:
    """Guarantee unique (customer, timestamp) so window features never see 'peers'."""
    txns = txns.copy()
    txns["ts"] = pd.to_datetime(txns["ts"])
    for _ in range(5):
        dup = txns.duplicated(["customer_id", "ts"], keep="first")
        if not dup.any():
            break
        txns.loc[dup, "ts"] += pd.to_timedelta(
            txns[dup].groupby(["customer_id", "ts"]).cumcount() + 1, unit="ms"
        )
    return txns
