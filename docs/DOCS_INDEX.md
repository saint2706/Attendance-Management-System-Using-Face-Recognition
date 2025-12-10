# Documentation Index

This document provides a map of all documentation in this project, organized by audience and purpose.

## Quick Navigation

| Document | Audience | Description |
|----------|----------|-------------|
| [README](../README.md) | Everyone | Project overview, quick start, key features |
| [Quick Start](QUICKSTART.md) | New users | Get running with synthetic data in 5 minutes |
| [User Guide](USER_GUIDE.md) | Non-programmers | Step-by-step usage instructions |
| [Developer Guide](DEVELOPER_GUIDE.md) | Contributors | Setup, architecture, testing, commands |
| [Deployment Guide](DEPLOYMENT.md) | DevOps/Admins | Production deployment with Docker/K8s |

---

## By Audience

### For End Users (Non-Technical)

- **[User Guide](USER_GUIDE.md)** – Complete walkthrough of all features
- **[Quick Start](QUICKSTART.md)** – Get a demo running in minutes
- **[Troubleshooting](troubleshooting.md)** – Common issues and solutions
- **[Support](SUPPORT.md)** – How to get help

### For Developers

- **[Developer Guide](DEVELOPER_GUIDE.md)** – Local setup, testing, management commands
- **[Architecture](ARCHITECTURE.md)** – System design and data flow diagrams
- **[API Reference](API_REFERENCE.md)** – REST endpoints and CLI tools
- **[Feature Flags](FEATURE_FLAGS.md)** – Configurable features and toggles
- **[Test Documentation](TEST_DOCUMENTATION.md)** – Test strategy and coverage

### For DevOps / Administrators

- **[Deployment Guide](DEPLOYMENT.md)** – Docker, Kubernetes, environment variables
- **[Security Guide](security.md)** – Hardening, secrets, compliance
- **[Monitoring](monitoring.md)** – Observability, Sentry, Silk profiling
- **[Performance Tuning](performance-tuning.md)** – Optimization strategies

### For Contributors

- **[Contributing Guide](../CONTRIBUTING.md)** – How to contribute, coding standards
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** – Community guidelines
- **[AGENTS.md](../AGENTS.md)** – Instructions for AI coding assistants

### For Compliance / Security Reviewers

- **[Security Policy](../SECURITY.md)** – Vulnerability reporting
- **[Security Guide](security.md)** – Technical hardening details
- **[Data Card](DATA_CARD.md)** – Data handling, privacy, retention
- **[Fairness & Limitations](FAIRNESS_AND_LIMITATIONS.md)** – Bias mitigation, known limitations

---

## Special Topics

### Face Recognition Pipeline

- **[Training Protocol](TRAINING_PROTOCOL.md)** – How face data is collected and trained
- **[Multi-Face Guide](MULTI_FACE_GUIDE.md)** – Handling multiple faces in frame
- **[Liveness Evaluation](liveness_evaluation.md)** – Anti-spoofing documentation
- **[Evaluation](EVALUATION.md)** – Benchmarking methodology

### Project Management

- **[TODO / Roadmap](TODO.md)** – Feature roadmap for 2025-2026
- **[Changelog](../CHANGELOG.md)** – Version history and release notes
- **[Business Actions](BUSINESS_ACTIONS.md)** – Policy-based action mapping

---

## Document Maintenance

When updating documentation:

1. Check this index to ensure new docs are linked
2. Update the relevant section when adding new files
3. Keep cross-references consistent
4. Run `make docs-screenshots` after UI changes

> **Tip:** Use the [Documentation Improvement](../.github/ISSUE_TEMPLATE/docs_improvement.yml) issue template to suggest changes.
