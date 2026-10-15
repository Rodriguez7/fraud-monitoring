.PHONY: install demo run dashboard test lint experiments clean

PY := .venv/bin/python

install:
	python3 -m venv .venv
	.venv/bin/pip install -q -e ".[dev]"

run:            ## generate data, build SQL features, train, score
	.venv/bin/fraudwatch run

experiments:    ## unseen-typology experiment (about 20 s)
	$(PY) -m fraudwatch.experiments

dashboard:
	.venv/bin/fraudwatch dashboard

demo: run experiments dashboard

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

clean:
	rm -rf data/*.parquet data/*.duckdb models/*.joblib reports/*.json
