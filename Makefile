# Makefile for Attendance-Management-System-Using-Face-Recognition

.PHONY: all setup run clean lint test migrate train evaluate report reproduce install-hooks demo docs-screenshots

# Default target
all: setup run

# Project setup: install dependencies and run migrations
setup:
	@echo "Setting up the project..."
	pip install -r requirements.txt
	python manage.py migrate
	@echo "Setup complete."

# Install pre-commit hooks
install-hooks:
	@echo "Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed."

# Run linting with black, isort, and flake8
lint:
	@echo "Running code quality checks..."
	black --check --line-length=100 .
	isort --check-only --profile=black --line-length=100 .
	flake8 --max-line-length=100 --ignore=E203,W503,E501 --exclude=migrations,__pycache__,.venv,venv .
	@echo "Lint checks complete."

# Format code with black and isort
format:
	@echo "Formatting code..."
	black --line-length=100 .
	isort --profile=black --line-length=100 .
	@echo "Code formatting complete."

# Run tests
test:
	@echo "Running tests..."
	python manage.py test
	@echo "Tests complete."

# Run database migrations
migrate:
	@echo "Running database migrations..."
	python manage.py makemigrations
	python manage.py migrate
	@echo "Migrations complete."

# Run the development server
run:
	@echo "Starting the development server at http://127.0.0.1:8000/"
	python manage.py runserver

# Bootstrap a self-contained demo with synthetic data
demo:
	@echo "=== Preparing demo environment ==="
	python manage.py migrate --noinput
	python scripts/bootstrap_demo.py
	@echo "--- Demo credentials ---"
	@echo "Admin: demo_admin / demo_admin_pass"
	@echo "Users: user_001, user_002, user_003 (password: demo_user_pass)"
	@echo "Launch the server with: python manage.py runserver"

# Train target retained for compatibility but training happens automatically
train:
	@echo "Training is automatic when new photos are added; no manual step is required."
	@echo "If you want to assess model quality, run 'make evaluate' or 'make report'."

# Run evaluation and generate metrics
evaluate:
	@echo "Running evaluation and generating metrics..."
	python manage.py eval --split-csv reports/splits.csv
	@echo "Evaluation complete."

# Run ablation experiments
ablation:
	@echo "Running ablation experiments..."
	python manage.py ablation
	@echo "Ablation experiments complete."

# Generate all reports
report: evaluate ablation
	@echo "Generating comprehensive reports..."
	python manage.py export_reports
	@echo "Reports generated in reports/ directory."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov .coverage .pytest_cache
	@echo "Cleanup complete."

# Full reproducibility workflow: run the synthetic dataset evaluation
reproduce:
	@echo "=== Running sample-data reproducibility workflow ==="
	python scripts/reproduce_sample_results.py
	@echo "=== Reproducibility workflow complete ==="
	@echo "Artifacts available in reports/sample_repro/."

# Capture documentation screenshots (requires running server)
docs-screenshots:
	@echo "=== Capturing documentation screenshots ==="
	@echo "Make sure the dev server is running: python manage.py runserver"
	python scripts/capture_screenshots.py
	@echo "=== Screenshots saved to docs/screenshots/ ==="
