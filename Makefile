PYTHON ?= python

.PHONY: help pipeline test sql ci

help:
	@echo "pipeline  run 00-09 from raw data"
	@echo "test      run the 9 contract tests"
	@echo "sql       run the SQL reconciliation only"
	@echo "ci        pipeline + tests (about 4-5 minutes)"

pipeline:
	$(PYTHON) src/run_all.py

test:
	$(PYTHON) -m unittest discover -s tests -v

sql:
	$(PYTHON) src/07_sql_pipeline.py

ci: pipeline test
