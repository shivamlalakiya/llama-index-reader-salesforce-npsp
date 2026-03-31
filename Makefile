.PHONY: install format lint test build

install:
	pip install -e ".[dev]"

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	mypy .

test:
	pytest tests/

build:
	python -m build
