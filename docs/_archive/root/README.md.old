# Attendance Management System Using Face Recognition

[![Django CI](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pkgs/container/attendance-management-system-using-face-recognition)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **A production-ready, privacy-first attendance solution that just works.**

**Attendance-Management-System-Using-Face-Recognition** is a modern, full-stack solution for automated time-tracking. It combines state-of-the-art deep learning (FaceNet) with a robust Django backend to deliver a seamless experience for employees and admins alike.

![Home Page Light Theme](docs/screenshots/home-light.png)

## Why this project?

- **🚀 Zero-Touch Attendance:** Employees just walk up to the camera. No badges, no PINs.
- **🔒 Privacy First:** Face data is encrypted at rest. Liveness detection prevents spoofing.
- **📱 Any Device:** Responsive PWA design works on tablets, mobiles, and desktops.
- **⚡ Production Ready:** Dockerized, tested, and scalable with Redis + Celery.

## Documentation

> 📚 **[Full Documentation Index](docs/DOCS_INDEX.md)** — organized by audience

- **[Quick Start](docs/QUICKSTART.md)**: Get running with synthetic data in 5 minutes
- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive guide for non-programmers
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Architecture, testing, and management commands
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Docker/Kubernetes setup and production hardening
- **[Security Policy](docs/security.md)**: Hardening defaults and compliance checks
- **[API Reference](docs/API_REFERENCE.md)**: Endpoints and command-line tools
- **[Architecture](docs/ARCHITECTURE.md)**: System design and data flow
- **[Changelog](CHANGELOG.md)**: Version history and release notes

## Key Features

- **Real-time Recognition:** Instant identification using the efficient "Facenet" model.
- **Anti-Spoofing:** Two-stage liveness detection (motion + texture) rejects photos and screens.
- **Smart Dashboard:** Comprehensive admin panel for reports, user management, and system health.
- **Offline Capable:** PWA architecture ensures basic functionality even with spotty internet.
- **Automated Training:** The model retrains itself in the background when new users are added.

## Who is this for?

✅ **Good fit:**

- Small-to-medium offices wanting touchless attendance
- Organizations with fixed camera locations for check-in/out
- Teams comfortable with self-hosting or Docker deployments
- Environments with consistent, well-lit camera positions

⚠️ **Consider alternatives if:**

- You need certified GDPR/HIPAA compliance out-of-the-box (this is a toolkit, not a certified product)
- Your environment has highly variable lighting or camera angles
- You require 99.99% recognition accuracy for critical access control
- You need to process 1000+ simultaneous recognitions per second
- You don't have technical staff to deploy and maintain the system

> **Note:** This system is designed for attendance tracking, not high-security access control. See [Fairness & Limitations](docs/FAIRNESS_AND_LIMITATIONS.md) for known constraints.

## Technical Stack

- **Backend:** Django 5+ with Django Rest Framework (DRF) and Celery workers
- **Face Recognition:** DeepFace (Facenet) + SSD detector with a motion-based liveness gate
- **Frontend:** React 18+ (Vite), TypeScript, Tailwind CSS, shadcn/ui (Installable PWA)
- **Database & cache:** Configurable via `DATABASE_URL` (PostgreSQL recommended) and Redis
- **Observability:** Sentry integration plus Silk for request profiling
- **Testing:** Pytest, Playwright, and GitHub Actions CI/CD

## Getting Started

For a complete walkthrough, see the **[Quick Start Guide](docs/QUICKSTART.md)**.

### Docker Quick Run

```bash
docker pull ghcr.io/saint2706/attendance-management-system-using-face-recognition:latest
docker run -d -p 8000:8000 ghcr.io/saint2706/attendance-management-system-using-face-recognition:latest
```

*(See [Deployment Guide](docs/DEPLOYMENT.md) for required environment variables)*

## Releases & Versioning

This project follows [Semantic Versioning](https://semver.org/).

- **Stable:** The `main` branch is always stable and deployable.
- **Tags:** Releases are tagged (e.g., `v1.0.0`).
- **Docker:** Images are automatically published to GHCR.

## Contributing

We welcome contributions! See the **[Contributing Guide](CONTRIBUTING.md)** and **[Code of Conduct](CODE_OF_CONDUCT.md)**.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
