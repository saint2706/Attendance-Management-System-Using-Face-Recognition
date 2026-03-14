# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

# =============================================================================
# Stage 1: Build Frontend
# =============================================================================
FROM node:20.11.1-alpine AS frontend-build

RUN corepack enable pnpm

RUN mkdir -p /app/frontend && chown -R node:node /app/frontend

WORKDIR /app/frontend

USER node

COPY --chown=node:node frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY --chown=node:node frontend/ .
# Ensure vite config base is correct (should be set in code, but we can enforce if needed)
RUN pnpm run build

# =============================================================================
# Stage 2: Base Python Runtime
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libraries required by OpenCV/DeepFace
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM python-base AS build

# Build tools required for Python packages with native extensions
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Leverage Docker layer caching for dependency installation
COPY requirements.txt ./
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the full project for asset compilation
COPY . /app

# Remove raw frontend source to prevent bloat in the final image
RUN rm -rf /app/frontend

# Copy built frontend from Stage 1
COPY --from=frontend-build --chown=root:root /app/frontend/dist /app/frontend/dist

# Collect static files using production configuration during the build stage
RUN DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production \
    DJANGO_DEBUG=0 \
    DJANGO_SECRET_KEY=dummy-secret-key-for-build \
    DJANGO_ALLOWED_HOSTS=localhost \
    DATA_ENCRYPTION_KEY=ZHVtbXktZW5jcnlwdGlvbi1rZXktZm9yLWJ1aWxkISE= \
    FACE_DATA_ENCRYPTION_KEY=ZHVtbXktZW5jcnlwdGlvbi1rZXktZm9yLWJ1aWxkISE= \
    python manage.py collectstatic --noinput

FROM python-base AS runtime

ENV PATH="/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production \
    DJANGO_DEBUG=0

WORKDIR /app

# Create a non-root user for running the application
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Create directories for runtime data and set ownership
RUN mkdir -p /app/media /app/face_recognition_data /app/staticfiles \
    && chown -R appuser:appgroup /app

# Copy virtual environment with installed dependencies
COPY --from=build --chown=appuser:appgroup /venv /venv

# Copy application code and collected static files
COPY --from=build --chown=appuser:appgroup /app /app

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check to verify the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/ || exit 1

# Default command runs the production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--worker-class", "gthread", "attendance_system_facial_recognition.wsgi:application"]
