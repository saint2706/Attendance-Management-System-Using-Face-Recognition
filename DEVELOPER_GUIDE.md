# Developer Guide

This guide provides a comprehensive overview of the project's structure, development setup, coding conventions, and testing procedures. It is intended for developers who want to contribute to the project or understand its inner workings.

## 1. Project Structure

The project is organized into the following directories:

-   `attendance_system_facial_recognition`: The main Django project directory.
    -   `settings.py`: The project's settings.
    -   `urls.py`: The project's URL patterns.
-   `recognition`: The Django app that handles face recognition and attendance tracking.
    -   `views.py`: The views for the recognition app.
    -   `models.py`: The models for the recognition app.
    -   `forms.py`: The forms for the recognition app.
    -   `tests.py`: The tests for the recognition app.
    -   `static/`: The static files for the recognition app.
    -   `templates/`: The templates for the recognition app.
-   `users`: The Django app that handles user management.
    -   `views.py`: The views for the users app.
    -   `models.py`: The models for the users app.
    -   `tests.py`: The tests for the users app.
    -   `templates/`: The templates for the users app.
-   `face_recognition_data`: The directory where the face recognition data is stored.
    -   `training_dataset`: The directory where the training dataset is stored. Each subdirectory is named after a user and contains their face images.

## 2. Development Setup

To set up the project for development, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smart-attendance-system.git
    cd smart-attendance-system
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```

5.  **Create a superuser (admin account):**
    ```bash
    python manage.py createsuperuser
    ```
    Follow the prompts to create your admin username, email, and password.

6.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8000/`.

## 3. Coding Conventions

The project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. Please ensure that your code adheres to these conventions.

In addition, the project uses the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings. Please ensure that all functions and methods have a comprehensive docstring that follows this style.

## 4. Testing

The project uses Django's built-in test framework for testing. To run the tests, use the following command:

```bash
python manage.py test
```

Please ensure that all new features are accompanied by a comprehensive set of tests.

## 5. Makefile Targets

The project includes a comprehensive `Makefile` for common development tasks:

### Setup and Installation
- `make setup`: Install dependencies and run migrations
- `make install-hooks`: Install pre-commit hooks for code quality

### Development
- `make run`: Start the Django development server
- `make migrate`: Run database migrations

### Code Quality
- `make lint`: Check code quality with black, isort, and flake8
- `make format`: Auto-format code with black and isort

### Testing and Evaluation
- `make test`: Run all Django tests
- `make evaluate`: Run performance evaluation with metrics
- `make ablation`: Run ablation experiments
- `make report`: Generate comprehensive reports (evaluation + ablation)

### Reproducibility
- `make reproduce`: Complete reproducibility workflow (setup → splits → evaluation → reports)

### Cleanup
- `make clean`: Remove generated files and caches

## 6. Management Commands

The project includes several custom Django management commands for evaluation and analysis:

### Data Preparation
```bash
python manage.py prepare_splits --seed 42
```
Prepares stratified train/validation/test splits with identity-level grouping to prevent leakage. Options:
- `--seed`: Random seed for reproducibility (default: 42)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)

### Evaluation
```bash
python manage.py eval --seed 42 --n-bootstrap 1000
```
Runs comprehensive evaluation with verification-style metrics. Options:
- `--seed`: Random seed for reproducibility
- `--n-bootstrap`: Number of bootstrap samples for confidence intervals (default: 1000)
- `--output-dir`: Directory for reports (default: reports/)

Generates:
- ROC, PR, DET, and calibration curves
- Metrics with bootstrap confidence intervals
- Detailed results in `reports/metrics_with_ci.md`

### Threshold Selection
```bash
python manage.py threshold_select --method eer --seed 42
```
Selects optimal recognition threshold based on validation set. Options:
- `--method`: Selection method (eer, f1, or fixed, default: eer)
- `--seed`: Random seed for reproducibility
- `--threshold`: Fixed threshold value if method=fixed

Methods:
- `eer`: Threshold at Equal Error Rate (FAR = FRR)
- `f1`: Threshold that maximizes F1 score
- `fixed`: Use a manually specified threshold

### Ablation Experiments
```bash
python manage.py ablation --seed 42
```
Runs ablation studies to test different component configurations. Options:
- `--seed`: Random seed for reproducibility
- `--output-dir`: Directory for results (default: reports/)

Tests combinations of:
- Detectors: SSD, OpenCV, MTCNN
- Alignment: on/off
- Distance metrics: cosine, euclidean, L2
- Rebalancing: on/off

Results saved to `reports/ABLATIONS.md` and `reports/ablation_results.csv`

### Export Reports
```bash
python manage.py export_reports
```
Exports all generated reports and figures to a consolidated directory structure.

## 7. Pre-commit Hooks

The project uses pre-commit hooks to maintain code quality:

### Installation
```bash
make install-hooks
# or
pre-commit install
```

### Tools Used
- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting (compatible with Black)
- **Flake8**: Linting and style checking

### Configuration
- Line length: 100 characters
- Ignores: E203, W503, E501
- Excludes: migrations, __pycache__, .venv, venv

### Running Manually
```bash
make lint     # Check without modifying
make format   # Auto-fix formatting
```

## 8. Testing Strategy

The project uses a comprehensive testing approach:

### Unit Tests
Located in:
- `recognition/tests.py`: Main recognition logic tests
- `recognition/test_metrics.py`: Metrics calculation tests
- `recognition/test_data_splits.py`: Data splitting tests
- `recognition/test_ablation.py`: Ablation framework tests
- `recognition/test_failures.py`: Failure analysis tests
- `recognition/test_integration.py`: Integration tests
- `users/tests.py`: User management tests

### Running Tests
```bash
# All tests
python manage.py test

# Specific app
python manage.py test recognition

# Specific test file
python manage.py test recognition.test_metrics

# With verbose output
python manage.py test --verbosity=2
```

### CI/CD
The project includes GitHub Actions workflows (`.github/workflows/ci.yml`) that:
- Run on all PRs and pushes to main
- Test on multiple Python versions (3.10, 3.11, 3.12)
- Run linting checks
- Execute full test suite
- Generate coverage reports

### Test Data
- Uses stratified splits for realistic evaluation
- Fixed random seed (42) for reproducibility
- Identity-level grouping prevents data leakage
- Session-based grouping for enrollment batches

## 9. Evaluation Architecture

### Metrics Module (`recognition/evaluation/metrics.py`)
Implements verification-style metrics appropriate for face recognition:

**Core Metrics:**
- ROC AUC: Area under the ROC curve
- EER: Equal Error Rate (FAR = FRR)
- FAR@TPR: False Accept Rate at target True Positive Rate
- TPR@FAR: True Positive Rate at target False Accept Rate
- Brier Score: Calibration quality metric
- F1 Score: At optimal threshold

**Confidence Intervals:**
- Nonparametric bootstrap (default: 1000 resamples)
- 95% confidence intervals for all key metrics
- Stratified sampling preserves class distribution

**Visualizations:**
- ROC Curve: `reports/figures/roc.png`
- Precision-Recall Curve: `reports/figures/pr.png`
- DET Curve: `reports/figures/det.png`
- Calibration Plot: `reports/figures/calibration.png`

### Failure Analysis Module (`recognition/analysis/failures.py`)
Automatic detection and analysis of prediction failures:

**Features:**
- Identifies false accepts and false rejects
- Ranks failures by confidence/severity
- Analyzes metadata for patterns (lighting, pose, occlusion)
- Performs subgroup analysis for bias detection
- Generates actionable recommendations

**Outputs:**
- `reports/FAILURES.md`: Detailed failure report
- `reports/failure_cases.csv`: All failure cases with metadata
- `reports/subgroup_metrics.csv`: Performance by subgroup

### Ablation Module (`recognition/ablation.py`)
Systematic testing of component contributions:

**Tested Components:**
- Face detectors (SSD, OpenCV, MTCNN)
- Face alignment (on/off)
- Distance metrics (cosine, euclidean, L2)
- Class rebalancing (on/off)

**Analysis:**
- Measures impact of each component
- Statistical significance testing
- Identifies optimal configurations
- Documents performance trade-offs

### Data Splits Module (`recognition/data_splits.py`)
Ensures proper train/validation/test separation:

**Features:**
- Identity-level stratification
- Session-based grouping
- Automatic leakage prevention
- Configurable split ratios
- Reproducible with fixed seed

**Leakage Prevention:**
- All images of same person in same split
- Enrollment session images stay together
- Metadata fields filtered (username, paths)
- Temporal ordering preserved where applicable

## 10. Business Actions and Policy

### Policy Configuration (`configs/policy.yaml`)
Defines score-based action bands for business decisions:

**Score Bands:**
- Confident Accept: ≥ 0.80 (automatic approval)
- Uncertain: 0.50-0.80 (secondary verification)
- Reject: < 0.50 (deny, suggest re-enrollment)

**Actions:**
- Immediate approval (< 2 seconds)
- PIN/OTP verification (10-30 seconds)
- Contact HR (manual resolution)

### CLI Prediction Tool (`predict_cli.py`)
Standalone tool for testing policy-based predictions:

```bash
# Basic usage
python predict_cli.py --image path/to/image.jpg

# With custom threshold
python predict_cli.py --image path/to/image.jpg --threshold 0.75

# JSON output
python predict_cli.py --image path/to/image.jpg --json
```

**Output:**
- Predicted identity
- Confidence score
- Action band (accept/uncertain/reject)
- Recommended action
- Expected user experience

### Integration in Views
See `recognition/views.py` for how the policy is applied during attendance marking:
- Score calculation using DeepFace
- Band determination based on policy
- Conditional actions (immediate vs. provisional marking)
- User feedback and guidance

## 11. Configuration and Environment Variables

The system supports several environment variables for configuration:

### Core Settings
- `DJANGO_DEBUG`: Enable debug mode (default: False)
- `DJANGO_SECRET_KEY`: Secret key for cryptographic signing (required in production)
- `DJANGO_ALLOWED_HOSTS`: Comma-separated list of allowed hostnames

### Recognition Settings
- `RECOGNITION_DISTANCE_THRESHOLD`: Maximum distance for face match (default: 0.4)
  - Lower values = stricter matching (fewer false accepts)
  - Higher values = more permissive (fewer false rejects)
  - Tune based on evaluation metrics

### Example Configuration
```bash
# Development
export DJANGO_DEBUG=True
export RECOGNITION_DISTANCE_THRESHOLD=0.4

# Production
export DJANGO_DEBUG=False
export DJANGO_SECRET_KEY='your-secret-key-here'
export DJANGO_ALLOWED_HOSTS='example.com,www.example.com'
export RECOGNITION_DISTANCE_THRESHOLD=0.35
```

## 12. Reproducibility Guidelines

To ensure reproducible results:

1. **Use Fixed Seeds**: Always use `--seed 42` for commands that involve randomness
2. **Pin Dependencies**: Use `requirements.frozen.txt` for exact versions
3. **Document Environment**: Note Python version, OS, and hardware specs
4. **Version Control**: Commit all config files and random seeds
5. **Run Full Workflow**: Use `make reproduce` for complete reproducibility
6. **Archive Artifacts**: Save `reports/` directory with timestamps
7. **Document Changes**: Log any manual interventions or adjustments

## 13. Contributing Guidelines

When contributing to the project:

1. **Code Style**: Follow PEP 8 and Google Python Style Guide
2. **Pre-commit Hooks**: Always run `make install-hooks` and ensure they pass
3. **Testing**: Add tests for new features, maintain >80% coverage
4. **Documentation**: Update relevant docs (README, DEVELOPER_GUIDE, API.md)
5. **Commits**: Write clear, descriptive commit messages
6. **Pull Requests**: Include description, rationale, and test results
7. **Reviews**: Address all reviewer comments before merging

### Code Review Checklist
- [ ] Code follows style guidelines (black, isort, flake8)
- [ ] All tests pass (`make test`)
- [ ] New features have tests
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

