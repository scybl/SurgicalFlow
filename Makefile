.PHONY: check results test clean

PYTHON ?= python
PYTEST ?= pytest

check:
	bash scripts/check_project.sh

results:
	mkdir -p docs/results
	$(PYTHON) -B scripts/generate_project_summary.py --output-dir docs/results
	bash scripts/check_project.sh > docs/results/structure_check.txt

test:
	$(PYTEST) tests/ -q

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__
