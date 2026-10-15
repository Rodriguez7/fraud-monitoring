"""Command line interface: `fraudwatch run` builds everything, `fraudwatch dashboard` opens the UI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ROOT, Paths
from .pipeline import (
    Params,
    run_all,
    run_features,
    run_generate,
    run_score,
    run_train,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudwatch", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, helptext in (
        ("generate", "create synthetic customers and transactions"),
        ("features", "run the SQL feature pipeline in DuckDB"),
        ("train", "train and evaluate the model (time-based split)"),
        ("score", "score the validation and test periods and open cases"),
        ("run", "all of the above"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--customers", type=int, default=5000)
        p.add_argument("--seed", type=int, default=42)
    sub.add_parser("dashboard", help="open the Streamlit dashboard and case review")
    ps = sub.add_parser("paysim", help="benchmark on the public PaySim data set (needs the CSV)")
    ps.add_argument("--csv", type=Path, required=True, help="path to the PaySim CSV from Kaggle")

    args = parser.parse_args(argv)
    paths = Paths()
    if args.cmd == "dashboard":
        app = ROOT / "app" / "dashboard.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])

    if args.cmd == "paysim":
        from .paysim import format_report, run_benchmark

        report = run_benchmark(args.csv, paths.base / "reports" / "paysim.json")
        print(format_report(report))
        return 0

    params = Params(n_customers=args.customers, seed=args.seed)
    steps = {
        "generate": lambda: run_generate(paths, params),
        "features": lambda: run_features(paths),
        "train": lambda: {k: v for k, v in run_train(paths, params).items() if k == "systems"},
        "score": lambda: run_score(paths, params),
        "run": lambda: run_all(paths, params),
    }
    print(json.dumps(steps[args.cmd](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
