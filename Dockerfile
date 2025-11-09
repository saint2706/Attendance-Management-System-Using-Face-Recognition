FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV/DeepFace and build tools
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . /app

# Collect static assets using the production settings module during build
RUN DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production \
    DJANGO_DEBUG=1 \
    python manage.py collectstatic --noinput

# Ensure the production settings module is loaded by default at runtime
ENV DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.production

# Launch gunicorn by default
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "attendance_system_facial_recognition.wsgi:application"]
