# Attendance Management System Using Face Recognition

> **A production-ready, privacy-first attendance solution that just works.**

**Attendance-Management-System-Using-Face-Recognition** is a modern, full-stack solution for automated time-tracking. It combines state-of-the-art deep learning (FaceNet) with a robust Django backend to deliver a seamless experience for employees and admins alike.

![Home Page Light Theme](docs/screenshots/home-light.png)

## Why this project?

- **🚀 Zero-Touch Attendance:** Employees just walk up to the camera. No badges, no PINs.
- **🔒 Privacy First:** Face data is encrypted at rest. Liveness detection prevents spoofing.
- **📱 Any Device:** Responsive PWA design works on tablets, mobiles, and desktops.
- **⚡ Production Ready:** Dockerized, tested, and scalable with Redis + Celery.

## Documentation

- **[Quick Start](docs/QUICKSTART.md)**: The fastest way to run a local demo with synthetic data.
- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive guide for non-programmers.
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Architecture, pipeline details, and management commands.
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Docker/Kubernetes setup, environment variables, and production hardening.
- **[Security Policy](docs/security.md)**: Hardening defaults, secret management, and compliance checks.
- **[API Reference](docs/API_REFERENCE.md)**: Endpoints and command-line tools.
- **[Architecture](docs/ARCHITECTURE.md)**: System design and data flow.
- **[Evaluation](docs/EVALUATION.md)**: Benchmarking and accuracy reports.

## Key Features

- **Real-time Recognition:** Instant identification using the efficient "Facenet" model.
- **Anti-Spoofing:** Two-stage liveness detection (motion + texture) rejects photos and screens.
- **Smart Dashboard:** Comprehensive admin panel for reports, user management, and system health.
- **Offline Capable:** PWA architecture ensures basic functionality even with spotty internet.
- **Automated Training:** The model retrains itself in the background when new users are added.

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
