# CI Workflows - Local Execution Commands

## Prerequisites
- Docker installed and running
- PostgreSQL service available at localhost:5432
- Node.js 20+ installed
- Python 3.12 installed
- Environment variables set (see below)

## Environment Variables
```bash
export DJANGO_DEBUG="1"
export DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1"
export DJANGO_SECRET_KEY="dev-secret-long-value-with-more-than-fifty-characters-12345"
export DATA_ENCRYPTION_KEY="j7iSLd8SZ80sbA-jm0AbOonybFEq9XAAgo82TBnws6g="
export FACE_DATA_ENCRYPTION_KEY="j7iSLd8SZ80sbA-jm0AbOonybFEq9XAAgo82TBnws6g="
export RECOGNITION_HEADLESS="True"
export DATABASE_URL="postgresql://ams:ams@localhost:5432/ams"
export DJANGO_SETTINGS_MODULE="attendance_system_facial_recognition.settings"
```

## Frontend CI - Local Execution

### Manual Steps (Recommended)
Since `act` has compatibility issues with the setup-node action in Docker, the most reliable way to verify Frontend CI locally is to run the steps manually:

```bash
cd frontend

# Install dependencies
npm ci

# Lint
npm run lint

# Build
npm run build
```

**Expected Result:** All commands exit with code 0, no errors, no warnings.

### Act Command (Alternative - May have Docker issues)
```bash
act -W .github/workflows/frontend-ci.yml pull_request
```

**Note:** This may fail with certificate errors when setup-node tries to download Node.js in Docker. The manual approach above is more reliable.

## Django CI - Local Execution

### Manual Steps (Recommended)
```bash
# Set environment variables (see above)

# Ensure PostgreSQL is running
docker run -d --name postgres-test \
  -e POSTGRES_USER=ams \
  -e POSTGRES_PASSWORD=ams \
  -e POSTGRES_DB=ams \
  -p 5432:5432 \
  postgres:16

# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

# Install Python dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt

# Run Django checks
python manage.py check

# Check for missing migrations
python manage.py makemigrations --check --dry-run

# Apply migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

# Deployment checks
export DJANGO_DEBUG="0"
python manage.py check --deploy
export DJANGO_DEBUG="1"

# Lint with flake8
flake8 --max-line-length=100 --ignore=E203,W503,E501 --exclude=migrations,__pycache__,.venv,venv,node_modules,frontend .

# Check code formatting
black --check --line-length=100 .
isort --check-only --profile=black --line-length=100 .

# Run fast tests
pytest -m "not slow and not ui and not e2e" -n auto --maxfail=5 --durations=10 -v \
  --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing
```

**Expected Result:** All commands exit with code 0, 253 tests pass, 0 warnings become errors.

### Act Command (Alternative - Requires services setup)
```bash
# Start PostgreSQL service first (see above)

# Run with act (requires complex service configuration)
act -W .github/workflows/ci.yml pull_request -j tests_fast
```

**Note:** Act requires additional configuration for services. The manual approach is more straightforward.

## Summary

Both workflows pass all checks when run manually:

### ✅ Frontend CI
- Linting: 0 errors, 0 warnings
- Build: Success

### ✅ Django CI  
- Django checks: No issues
- Flake8: 0 violations
- Black/Isort: All files compliant
- Tests: 253 passed, 0 failed

## Changes Made

See `CI_FIX_LOG.md` for detailed list of all fixes applied.
