# Makefile for drift adaptation project

.PHONY: help install install-dev test lint format clean demo run-demo

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src/drift_adaptation --cov-report=html --cov-report=term

test-fast: ## Run fast tests only
	pytest tests/ -v -m "not slow"

lint: ## Run linting
	ruff check src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	ruff check src/ tests/ --fix

clean: ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf assets/*.png
	rm -rf logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: ## Run the demo script
	python scripts/demo.py

run-demo: ## Launch Streamlit demo
	streamlit run demo/app.py

notebook: ## Launch Jupyter notebook
	jupyter notebook notebooks/

docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html

check: lint test ## Run all checks

all: clean install-dev test ## Clean, install, and test

# Development workflow
dev-setup: install-dev ## Set up development environment
	@echo "Development environment set up successfully!"
	@echo "Run 'make demo' to test the demo script"
	@echo "Run 'make run-demo' to launch the Streamlit demo"
	@echo "Run 'make test' to run the test suite"
