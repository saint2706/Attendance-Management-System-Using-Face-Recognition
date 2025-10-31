# Makefile for Attendance-Management-System-Using-Face-Recognition

.PHONY: all setup run clean

# Default target
all: setup run

# Project setup: install dependencies and run migrations
setup:
	@echo "Setting up the project..."
	pip install -r requirements.txt
	python manage.py migrate
	@echo "Setup complete."

# Run the development server
run:
	@echo "Starting the development server at http://127.0.0.1:8000/"
	python manage.py runserver

# Clean up generated files
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleanup complete."

# Target to reproduce the setup as mentioned in the review
reproduce: setup
	@echo "Reproducibility steps completed. You can now run the server with 'make run'."
