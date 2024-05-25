.PHONY: install
install:
    pdm install

.PHONY: lint
lint:
    pdm run flake8 .
    pdm run black --check .
    pdm run mypy .

.PHONY: format
format:
    pdm run black .
    pdm run isort .

.PHONY: test
test:
    pdm run pytest

.PHONY: bench
bench:
    pdm run python src/bench/bench.py

.PHONY: pre-commit-install
pre-commit-install:
    pdm run pre-commit install

.PHONY: pre-commit-uninstall
pre-commit-uninstall:
    pdm run pre-commit uninstall