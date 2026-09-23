.PHONY: install test lint bench clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -q

lint:
	ruff check .

bench:
	@echo "No benchmark in this repo."

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info .ruff_cache
