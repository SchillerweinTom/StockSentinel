PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
FLAKE8 := flake8

help:
	@echo "StockSentinel - Makefile Commands"
	@echo "=================================="
	@echo "install        - Install dependencies"
	@echo "test           - Run tests with pytest"
	@echo "lint           - Run flake8 linter"
	@echo "format         - Format code with black"
	@echo "format-check   - Check code formatting"
	@echo "run-api        - Start FastAPI server"
	@echo "run-frontend   - Start Streamlit app"
	@echo "docker-build   - Build Docker image"
	@echo "docker-compose - Run with docker-compose"
	@echo "analyze        - Run CLI analysis (example)"
	@echo "ci        	  - Run CI (format, lint, test)"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PYTEST) tests/ -v

lint:
	$(FLAKE8) src/ tests/ api/ --max-line-length=100 --ignore=E203,W503,W293

format:
	$(BLACK) src/ tests/ api/ frontend/ --line-length=100

format-check:
	$(BLACK) src/ tests/ api/ frontend/ --check --line-length=100

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	streamlit run frontend/app.py

docker-build:
	docker build -t stocksentinel:latest .

docker-compose:
	docker-compose up --build

analyze:
	$(PYTHON) -m src.cli --ticker AAPL --days 7

ci: format-check lint test
	@echo "✅ CI Pipeline completed successfully!"