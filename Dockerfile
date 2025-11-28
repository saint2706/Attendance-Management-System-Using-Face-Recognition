# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libraries required by OpenCV/DeepFace
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

FROM python-base AS build

# Build tools required for Python packages with native extensions
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Leverage Docker layer caching for dependency installation
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the full project for asset compilation
COPY . /app

# Collect static files using production configuration during the build stage
RUN DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production \
    DJANGO_DEBUG=0 \
    DJANGO_SECRET_KEY=dummy-secret-key-for-build \
    DATA_ENCRYPTION_KEY=ufkljjgdbIMsc4N4-cVeRTtBk8sM6rDl6q-FMpepe8g= \
    FACE_DATA_ENCRYPTION_KEY=ufkljjgdbIMsc4N4-cVeRTtBk8sM6rDl6q-FMpepe8g= \
    python manage.py collectstatic --noinput

FROM python-base AS runtime

ENV PATH="/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production \
    DJANGO_DEBUG=0

WORKDIR /app

# Copy virtual environment with installed dependencies
COPY --from=build /venv /venv

# Copy application code and collected static files
COPY --from=build /app /app

# Default command runs the production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "attendance_system_facial_recognition.wsgi:application"]
