.PHONY: install run test lint typecheck

install:
	pip install -e .[dev]

run:
	uvicorn tradeengine.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

