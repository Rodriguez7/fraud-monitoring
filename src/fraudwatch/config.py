"""Paths and shared constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = [
    "grocery",
    "restaurant",
    "transport",
    "utilities",
    "travel",
    "electronics",
    "fashion",
    "online_services",
    "gift_cards",
    "crypto_exchange",
]
HIGH_RISK_CATEGORIES = ("electronics", "gift_cards", "crypto_exchange")
COUNTRIES = ["AT", "DE", "IT", "FR", "ES", "NL", "CZ", "HU", "US", "GB", "TR", "NG"]


@dataclass(frozen=True)
class Paths:
    """Where artefacts live. Everything under `data/`, `models/`, `reports/` is git-ignored."""

    base: Path = ROOT

    @property
    def data(self) -> Path:
        return self.base / "data"

    @property
    def transactions(self) -> Path:
        return self.data / "transactions.parquet"

    @property
    def customers(self) -> Path:
        return self.data / "customers.parquet"

    @property
    def db(self) -> Path:
        return self.data / "fraudwatch.duckdb"

    @property
    def model(self) -> Path:
        return self.base / "models" / "model.joblib"

    @property
    def metrics(self) -> Path:
        return self.base / "reports" / "metrics.json"

    @property
    def figures(self) -> Path:
        return self.base / "docs" / "figures"


@dataclass(frozen=True)
class SplitConfig:
    """Time-based split in days since the start of the simulation.

    Days before `burn_in` build customer history only: no training, no evaluation.
    """

    burn_in: int = 30
    train_end: int = 90
    valid_end: int = 105
    total: int = 120
    review_budget: float = 0.005  # share of transactions analysts can review
