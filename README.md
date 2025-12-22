# Attendance Management System Using Face Recognition

[![Django CI](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pkgs/container/attendance-management-system-using-face-recognition)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-ready, privacy-first attendance tracking powered by face recognition.**

A modern, full-stack attendance management system that combines deep learning (FaceNet + DeepFace) with a robust Django backend and React frontend. Built for reliability, security, and ease of deployment.

![Home Page Light Theme](docs/screenshots/home-light.png)

## ✨ Key Features

- **🚀 Zero-Touch Check-In**: Employees authenticate via camera—no badges, no PINs, no physical contact
- **🔒 Privacy & Security**: Military-grade encryption (Fernet), liveness detection, anti-spoofing protection
- **📱 Universal Access**: Progressive Web App (PWA) works seamlessly on desktop, tablet, and mobile
- **⚡ Production-Ready**: Dockerized deployment, Redis caching, Celery async processing, PostgreSQL backend
- **🎯 Real-Time Recognition**: Sub-second face matching using optimized FaceNet embeddings
- **📊 Smart Dashboard**: Comprehensive admin panel with analytics, reports, and system health monitoring
- **🔧 Flexible Configuration**: Feature flags, threshold profiles, and hardware acceleration support

## 📖 Documentation

**Start Here:** [📚 Complete Documentation Index](docs/README.md)

### Quick Links by Audience

| I want to... | Read this |
|-------------|-----------|
| **Get a quick demo running** | [Quick Start Guide](docs/QUICKSTART.md) |
| **Learn how to use the system** | [User Guide](docs/USER_GUIDE.md) |
| **Deploy to production** | [Deployment Guide](docs/DEPLOYMENT.md) |
| **Develop or contribute** | [Developer Guide](docs/DEVELOPER_GUIDE.md) |
| **Understand the architecture** | [Architecture Overview](docs/ARCHITECTURE.md) |
| **Configure security settings** | [Security Guide](docs/SECURITY.md) |
| **Use the API or CLI** | [API Reference](docs/API_REFERENCE.md) |
| **Troubleshoot issues** | [Troubleshooting Guide](docs/TROUBLESHOOTING.md) |

### Essential Documents

- **[Contributing Guidelines](CONTRIBUTING.md)**: How to contribute, coding standards, PR process
- **[Code of Conduct](CODE_OF_CONDUCT.md)**: Community guidelines and expectations
- **[Security Policy](SECURITY.md)**: Vulnerability reporting and security practices
- **[Changelog](CHANGELOG.md)**: Version history and release notes

## 🛠️ Technology Stack

### Backend
- **Django 6.0**: Modern Python web framework with async support
- **DeepFace 0.0.93**: Face recognition with FaceNet model
- **PostgreSQL 16**: Primary relational database
- **Redis 7**: Caching and Celery message broker
- **Celery 5.4**: Distributed task queue for async processing
- **Sentry**: Error tracking and performance monitoring

### Frontend
- **React 19**: Modern UI with TypeScript
- **Vite**: Lightning-fast build tool and dev server
- **Tailwind CSS + shadcn/ui**: Utility-first styling with accessible components
- **PWA**: Installable, offline-capable progressive web app

### Machine Learning
- **FaceNet**: State-of-the-art face embeddings (128-D vectors)
- **SSD Detector**: Fast and accurate face detection
- **FAISS**: Vector similarity search for fast matching
- **Motion-based Liveness**: Anti-spoofing with motion analysis

### DevOps & Infrastructure
- **Docker**: Containerized deployment
- **Docker Compose**: Multi-container orchestration
- **Kubernetes**: Production-grade orchestration (optional)
- **GitHub Actions**: CI/CD pipelines with automated testing
- **Nginx**: Reverse proxy and static file serving

## 📊 Use Cases

### ✅ Ideal For

- **Small to medium offices** (10-500 employees) needing touchless attendance
- **Fixed check-in locations** with consistent camera positioning
- **Privacy-conscious organizations** requiring on-premise face data storage
- **Self-hosted environments** where you control the infrastructure
- **Consistent lighting conditions** (offices, reception areas, controlled environments)

### ⚠️ Not Recommended For

- **Critical access control** requiring 99.99% accuracy (use this for time-tracking, not security)
- **GDPR/HIPAA certified deployments** out-of-the-box (this is a toolkit, not a certified product)
- **Highly variable environments** (outdoor, poor lighting, extreme angles)
- **High-concurrency scenarios** (1000+ simultaneous recognitions per second)
- **Organizations without technical staff** to deploy and maintain the system

> **Important**: This system is designed for **attendance tracking**, not high-security access control. See [Fairness & Limitations](docs/FAIRNESS_AND_LIMITATIONS.md) for detailed constraints and bias mitigation strategies.

## 🎯 Core Capabilities

### Face Recognition Pipeline
- **Detection**: SSD-based face detector with configurable confidence threshold
- **Embedding**: FaceNet model generates 128-dimensional face embeddings
- **Matching**: Cosine similarity with distance threshold of 0.4 (configurable)
- **Liveness**: Motion-based anti-spoofing to prevent photo/video attacks

### Admin Features
- **User Management**: Create/edit employees, manage permissions, bulk operations
- **Dashboard**: Real-time attendance overview, system health, recent activity
- **Reports**: Daily/weekly/monthly reports, export to CSV/Excel/PDF
- **Analytics**: Attendance trends, punctuality metrics, department summaries
- **System Monitoring**: Performance metrics, Sentry error tracking, Silk profiling

### Employee Features
- **Self-Service Check-In/Out**: Face-based time-in and time-out
- **Personal Dashboard**: View own attendance history and statistics
- **Mobile Responsive**: Works on smartphones and tablets
- **Offline Support**: PWA caching for basic functionality without internet

## 🔐 Security & Privacy

- **Encryption at Rest**: Face data encrypted using Fernet (symmetric encryption)
- **Liveness Detection**: Motion-based validation prevents spoofing
- **Rate Limiting**: Protects against brute-force attacks
- **Audit Logging**: Complete activity trail for compliance
- **Secure Defaults**: HTTPS required in production, secure cookie flags enabled
- **No Third-Party Storage**: All face data stored on-premise, never sent to external services

See [Security Guide](docs/SECURITY.md) for complete hardening instructions.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

Pull and run the latest stable release:

```bash
docker pull ghcr.io/saint2706/attendance-management-system-using-face-recognition:latest
docker run -d -p 8000:8000 \
  -e DJANGO_SECRET_KEY="your-secret-key-here" \
  -e DATA_ENCRYPTION_KEY="your-encryption-key-here" \
  -e FACE_DATA_ENCRYPTION_KEY="your-face-encryption-key-here" \
  ghcr.io/saint2706/attendance-management-system-using-face-recognition:latest
```

Visit [http://localhost:8000](http://localhost:8000) to access the system.

> **Note**: For production deployment, see the [Deployment Guide](docs/DEPLOYMENT.md) for complete configuration.

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition.git
cd Attendance-Management-System-Using-Face-Recognition

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
python manage.py migrate

# Create admin user
python manage.py createsuperuser

# Start the development server
python manage.py runserver
```

For a complete walkthrough with sample data, see the [Quick Start Guide](docs/QUICKSTART.md).

## 🧪 Testing & Quality

The project maintains comprehensive test coverage with multiple test categories:

```bash
# Fast tests (unit + integration, excludes slow/UI tests)
make test-fast
# or: pytest -m "not slow and not ui and not e2e"

# Slow tests (heavy computation, model loading)
make test-slow
# or: pytest -m "slow or integration"

# UI/E2E tests (Playwright browser automation)
make test-ui
# or: pytest -m "ui or e2e" tests/ui

# Full test suite with coverage
make test-all
# or: pytest --cov=. --cov-report=html
```

**Test Coverage**: 60%+ overall, with higher coverage in critical paths (recognition pipeline, attendance logic).

**CI/CD**: GitHub Actions runs:
- Fast tests on every PR
- Slow/UI tests on main branch pushes
- Code formatting (Black, isort)
- Linting (flake8)
- Security scanning (CodeQL)
- Docker image builds

See [Test Documentation](docs/TEST_DOCUMENTATION.md) for details on test strategy and markers.

## 📦 Releases & Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **Stable**: The `main` branch is always stable and deployable
- **Tags**: Releases are tagged (e.g., `v1.7.0`)
- **Docker Images**: Automatically published to GitHub Container Registry
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for detailed release notes

**Current Version**: `1.7.0` (December 2025)

**Upgrade Path**: See [docs/UPGRADE_GUIDE.md](docs/UPGRADE_GUIDE.md) for migration instructions between major versions.

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Read** [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
2. **Check** [Good First Issues](docs/GOOD_FIRST_ISSUES.md) for beginner-friendly tasks
3. **Review** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
4. **Fork** the repository and create a feature branch
5. **Test** your changes locally
6. **Submit** a pull request with clear description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests locally
make test-fast

# Format code
make format
```

### Areas We Need Help

- 🌍 **Internationalization**: Translations for UI strings
- 🎨 **UI/UX**: Design improvements and accessibility enhancements
- 📖 **Documentation**: Tutorials, examples, and guides
- 🧪 **Testing**: Increase coverage in evaluation and API modules
- 🔧 **Features**: See [TODO.md](docs/TODO.md) for roadmap items

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Acknowledgments

- **DeepFace**: Face recognition library (MIT License)
- **TensorFlow**: Machine learning framework (Apache 2.0)
- **Django**: Web framework (BSD-3-Clause)
- **React**: UI library (MIT License)

See [docs/ATTRIBUTIONS.md](docs/ATTRIBUTIONS.md) for complete dependency list.

## 📞 Support & Community

### Getting Help

- **Documentation**: Start with [docs/README.md](docs/README.md)
- **Troubleshooting**: See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/discussions)

### Security Issues

For security vulnerabilities, please email the maintainers directly. See [SECURITY.md](SECURITY.md) for details.

### Commercial Support

For commercial support, custom features, or deployment assistance, contact the maintainers.

## 🎓 Educational Use

This project is ideal for learning:

- **Full-stack web development** (Django + React)
- **Face recognition** and computer vision
- **Docker deployment** and DevOps practices
- **Modern Python** best practices
- **CI/CD pipelines** with GitHub Actions

Feel free to use this as a reference or starting point for your own projects!

## 🗺️ Roadmap

See [docs/TODO.md](docs/TODO.md) for the complete roadmap. Highlights include:

### Planned for 2025
- [ ] Multi-factor authentication (face + PIN)
- [ ] Advanced analytics dashboard
- [ ] Mobile app (iOS/Android)
- [ ] Kubernetes Helm charts
- [ ] OpenID Connect (OIDC) integration

### Under Consideration
- [ ] Active Directory / LDAP integration
- [ ] Hardware acceleration (GPU/NPU)
- [ ] Multi-camera support
- [ ] Visitor management module
- [ ] Shift management integration

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project and motivates continued development.

---

**Built with ❤️ by the open-source community**

*Last Updated: December 2025 | Version 1.7.0*
