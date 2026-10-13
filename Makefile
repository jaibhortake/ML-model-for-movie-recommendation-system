.PHONY: install install-dev train serve web test lint format docker-build docker-up clean all help

PYTHON ?= python
PIP ?= pip

help:
	@echo "Available Makefile targets:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  train         - Run model training pipeline"
	@echo "  serve         - Start FastAPI backend server"
	@echo "  web           - Start Streamlit frontend app"
	@echo "  test          - Run tests with pytest and coverage"
	@echo "  lint          - Run ruff linter and mypy type checks"
	@echo "  format        - Format codebase with black and ruff"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-up     - Start services via docker-compose"
	@echo "  clean         - Clean build, cache, and test artifacts"
	@echo "  all           - Run install, train, and test"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

train:
	$(PYTHON) scripts/train.py

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	streamlit run src/web/app.py

test:
	pytest tests/ --cov=src --cov=config --cov-report=term-missing

lint:
	ruff check .
	mypy src config

format:
	black .
	ruff format .

docker-build:
	docker build -t movie-rec-system:latest .

docker-up:
	docker compose up --build -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage coverage.xml 2>/dev/null || true

all: install train test
