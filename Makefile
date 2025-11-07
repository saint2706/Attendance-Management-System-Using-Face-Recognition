# Makefile for Attendance-Management-System-Using-Face-Recognition

.PHONY: all setup run clean lint test migrate train evaluate report reproduce install-hooks

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

# Train target retained for compatibility but training happens automatically
train:
	@echo "Training is automatic when new photos are added; no manual step is required."
	@echo "If you want to assess model quality, run 'make evaluate' or 'make report'."

# Run evaluation and generate metrics
evaluate:
	@echo "Running evaluation and generating metrics..."
	python manage.py eval
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

# Full reproducibility workflow: seed, prepare data, run evaluation, produce artifacts
reproduce: setup
	@echo "=== Running reproducibility workflow ==="
	@echo "Step 1: Preparing sample dataset and splits..."
	python manage.py prepare_splits
	@echo "Step 2: Running evaluation with fixed seed..."
	python manage.py eval
	@echo "Step 3: Generating reports..."
	python manage.py export_reports
	@echo "=== Reproducibility workflow complete ==="
	@echo "Artifacts available in reports/ directory."
	@echo "You can now run the server with 'make run'."
