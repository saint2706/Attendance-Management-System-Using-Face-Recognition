#!/bin/bash
DJANGO_SECRET_KEY='any-string' DJANGO_DEBUG=1 DJANGO_SETTINGS_MODULE=attendance_system_facial_recognition.settings.base python -m pytest tests/recognition/test_threshold_profiles.py
