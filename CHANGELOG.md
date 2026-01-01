# Changelog

All notable changes to the **Attendance Management System Using Face Recognition** project are documented in this file.

This changelog was initially reconstructed from the git history on 2025-11-29, and has been updated with actual GitHub releases starting from v1.0.0 (2025-11-30). For versions before v1.0.0, virtual version numbers were assigned based on major milestones and the nature of changes. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) principles, and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) conventions.

---

## [Unreleased]

### Added (Unreleased)

- Advanced admin commands and documentation updates (`9d1f2fb`)
- Domain adaptation, sample quality, and group threshold support (`ad9e8bf`)
- Enhanced liveness detection modules (`7bc47c3`)

### Fixed (Unreleased)

- Django CI and PostgreSQL test suite linting failures (`7dd47bf`)
- Flaky concurrent test due to uint8 overflow (`1c518e0`)

### Changed (Unreleased)

- Optimized GitHub workflow efficiency (`15022a1`)
- Removed Codecov badge and unused face recognition test workflow (`bbc1a1d`, `285365f`)

### Security (Unreleased)

- Bump urllib3 from 2.5.0 to 2.6.0 in the pip group (`6b55304`)

### Documentation (Unreleased)

- Comprehensive documentation refresh for 2026: updated React SPA frontend references, fixed broken links, updated agent instructions

---

## [1.6.0] - 2025-12-08

This release focuses on bug fixes, workflow optimizations, and dependency updates.

### Fixed in 1.6.0

- Fix flaky concurrent test due to uint8 overflow ([#138](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/138))
- Fix Django CI and PostgreSQL test suite linting failures ([#141](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/141))

### Changed in 1.6.0

- Optimize GitHub workflow efficiency ([#139](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/139))

### Security in 1.6.0

- Bump urllib3 from 2.5.0 to 2.6.0 in the pip group ([#140](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/140))

---

## [1.5.0] - 2025-12-05

This major release adds advanced features including encryption key rotation, hardware profiling, multi-face detection, and significant CI/CD improvements.

### Added in 1.5.0

- Encryption key rotation command and guidance ([#130](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/130))
- Hardware profiling and performance utilities (`5713047`)
- Multi-face detection support for group check-ins (`0473992`)
- Feature flag system and modular frontend JavaScript (`74bed54`)
- Deterministic dev encryption key handling ([#132](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/132))
- Tests for multi-face recognition logic (`abff156`)
- Authentication and rate limiting updates for face API ([#128](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/128))

### Fixed in 1.5.0

- Fix malformed migration and update Time model to use direction field ([#133](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/133))
- Fix flake8 linting errors and test failures in CI workflows ([#134](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/134))
- Fix Django CI and PostgreSQL test suite workflow failures ([#135](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/135))
- Fix flaky concurrent test, add coverage config, and ensure proper database shutdown ([#136](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/136))
- Fix test flakiness by ensuring older evaluation timestamps (`767ac7c`)
- Fix exception handling loop in Silk integration (`f3cc6c9`)

### Changed in 1.5.0

- Optimize CI test workflow: split fast/slow jobs, add markers, reduce timeout from 45min to 15min ([#126](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/126))
- Update dependencies and fix version mismatches (`6fcad81`)
- Update docs and screenshots for improved onboarding (`276ee34`)
- Refactor Silk integration (`f3cc6c9`)
- Validate liveness evaluation input ([#131](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/131))

### Security in 1.5.0

- Bump werkzeug from 3.1.3 to 3.1.4 in the pip group ([#129](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/129))
- Bump the pip group with 2 updates ([#127](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/pull/127))

---

## [1.4.0] - 2025-11-29

This version represents the state of the codebase when the original CHANGELOG.md was reconstructed from git history. All changes listed below were part of the unreleased section in the previous changelog.

### Added in 1.4.0

- Scheduled evaluation tasks with Celery Beat for automated model health monitoring
- Model health widget displaying accuracy trends on admin dashboard
- Single-node production deployment guide with architecture diagram
- Threshold profiles for recognition sensitivity tuning with UI and CLI support
- Enhanced liveness detection with motion-based validation
- Guided Setup Wizard for streamlined admin onboarding
- Model Fairness & Limitations admin page with privacy/consent UI hints
- Attendance dashboard with date filters, CSV export, and interactive charts
- Smoke test suite for CI health verification
- First-run checklist with inline help and tooltips for recognition errors
- Fairness audit threshold recommendations with `--recommend-thresholds` flag for per-group tuning
- `ThresholdProfile` model supporting per-group (site, lighting, camera, role) threshold configurations
- Hardware profiling command (`profile_hardware`) for NPU/GPU/CPU detection and benchmarking
- Camera calibration command (`calibrate_camera`) for multi-camera domain adaptation
- Threshold profile management command (`threshold_profile`) with CRUD and bulk-apply support
- `LivenessResult` model for persistent liveness check audit trail
- Camera bucket grouping in fairness audit reports
- Community scaffolding for open-source contributors (CODE_OF_CONDUCT, CONTRIBUTING, SUPPORT)
- Production-ready Docker setup with GHCR publishing workflow

### Fixed in 1.4.0

- Flaky test in `test_attendance_flows.py` with correct monkeypatch lambda return values
- Docstring order in `_build_onboarding_state` function
- Code review issues in liveness and template files
- Linting errors and epsilon for trend comparison in model health
- Docker workflow: add DJANGO_ALLOWED_HOSTS and remove DockerHub steps

### Changed in 1.4.0

- Improved demo experience and SQLite migration workflow
- Enhanced system health reporting and liveness logging
- Added attendance safeguards and coverage enforcement
- Updated documentation with UI screenshots and beginner setup

---

## [1.3.0] - 2025-11-18

### Added in 1.3.0

- Production-grade evaluation harness for face recognition performance testing (`5e23372`)
- Motion-based liveness gate to prevent spoofing attacks (`0a0d53f`)
- Fairness audit tooling and documentation for bias analysis (`2783455`)
- Synthetic dataset and reproducibility workflow (`f554158`)
- GitHub Copilot instructions file (`.github/copilot-instructions.md`) (`0e4ce9d`)
- Lighthouse CI workflow for frontend performance monitoring (`cb8bcca`)
- Playwright admin navigation tests (`48aab2c`)
- Refactored face recognition pipeline with comprehensive unit tests (`98c0eeb`)
- AGENTS.md with agent instructions for Attendance Management System (`d020135`)

### Changed in 1.3.0

- Aligned documentation with repository structure and cosine metric (`7d5451f`)
- Documentation alignment pass across all documentation files (`026c853`)
- Offloaded recognition workflows to Celery for async processing (`f4be1c6`)
- Improved attendance telemetry and CLI logging (`14fa932`)
- Added type hints and docstrings to admin monitoring views (`40d5da8`)

### Fixed in 1.3.0

- Django CI workflow regressions (`f966719`)
- Migration 0005 to handle PostgreSQL time to timestamp conversion (`30a81cb`)
- PostgreSQL CI with non-blocking Codecov upload (`55f675c`)
- Flaky `test_concurrent_attendance_requests` with thread-safe lock (`ed738d4`)
- Formatting issues with Black and isort (`47bfd4e`)
- Django.setup() conflicts in test files (`2c1e466`)
- Path filters for CI workflows to skip on docs-only changes (`4a782ec`)

### Internal

- Added `STATIC_ROOT` setting and updated `.gitignore` (`4fe1c63`)
- Added `collectstatic` step to CI and Lighthouse workflows (`dd0b5e9`)
- Stabilized Django and PostgreSQL workflows (`f9a2aca`)
- Updated CI workflow to install `libgl1` package (`ea4ba52`)

---

## [1.2.0] - 2025-11-11

### Added in 1.2.0

- Recognition outcome logging and analytics dashboard (`d4d3d95`)
- Health monitoring instrumentation and admin dashboard (`ade2dce`)
- Sentry SDK integration for production telemetry (`14a3ccf`)
- Silk performance monitoring configuration (`c47a87b`)
- Ops troubleshooting guide for Celery/Redis asynchronous pipeline (`e6681d4`)
- Security hardening guide linked from deployment documentation (`e2a2186`)
- Performance tuning guide for recognition pipeline (`2a9c034`)
- Deployment guide for Docker-based production operations (`5811400`)
- API Reference documentation for admin dashboards and attendance batch API (`9b0c2af`)
- Attendance analytics module (`AttendanceAnalytics`) for insights (`adf5f62`)
- Workflow tests for face recognition pipeline (`5d0a01b`)
- PostgreSQL coverage workflow and Codecov upload (`4a578df`)

### Changed in 1.2.0

- Extended data model for recognition logging (`6ca4fac`)
- Designed model to persist recognition outcomes (`fed7b77`)
- Configured DeepFace tuning via shared settings (`7c68836`)

### Fixed in 1.2.0

- Django CI linting and formatting errors (`9665a73`)
- Coverage files added to `.gitignore` (`44bbc34`)
- Test failures by updating function signatures for `enforce_detection` parameter (`7bb0e2b`)

### Internal in 1.2.0

- Added `django-ratelimit` dependency (`402b514`)
- Deleted Suggestions.md file (`b92b7da`)
- Image optimization via ImgBot (`b022f4d`)

---

## [1.1.0] - 2025-11-09

### Added in 1.1.0

- Production Docker setup with container documentation and `docker-compose.yml` (`4e11a87`)
- Celery and Redis integration for async attendance processing (`73aea89`)
- Django-RQ integration with incremental face training pipeline (`d4ddc98`)
- Camera manager utility with preview modal (`eb85796`)
- Face recognition REST API endpoint with tests (`9e730f4`)
- Encrypted dataset cache helper (`FaceDataEncryption`) with tests (`9ebd483`)
- Caching helper for DeepFace embeddings (`c2efaf9`)
- Composite indexes for attendance queries (`cc88555`)
- Database URL configuration support (`a9fc7a5`)
- Cached dataset embeddings with invalidation (`2392e63`)
- Shared webcam manager with refactored attendance streaming (`58c233d`)
- Encrypted storage helpers for secure dataset handling (`9334ae6`)
- Rate limiting for attendance endpoints (`1a79324`)
- Session security configuration via environment variables (`e74b5ab`)
- Liveness gating for attendance recognition (anti-spoofing) (`dd29f2f`)
- Repository analysis and improvement suggestions (`5b745a7`)

### Changed in 1.1.0

- Refactored settings module into a package structure (`2d3583d`)
- Styled camera preview modal for responsive layout (`ea645d7`)

### Fixed in 1.1.0

- All Django CI linting and formatting issues (`9c60cfb`, `a9a6ac9`)
- System package installation for OpenCV in CI (`5a3161b`)
- UI tests to use `live_server` fixture (`f17ee9b`)
- Missing migrations for Django models (`9c60cfb`)

### Security in 1.1.0

- Added `cryptography` package for Fernet encryption (`a572838`)
- Encrypted storage for sensitive face data (`9334ae6`)
- Rate limiting protection against abuse (`1a79324`)

### Internal in 1.1.0

- Removed coverage report and Lighthouse CI steps from workflow (`6e96460`, `ede333c`)
- Updated copyright year and owner in LICENSE (`44f7921`)
- Deleted obsolete documentation files (`c21a065`, `0a917b0`, `3066e6f`)

---

## [1.0.0] - 2025-11-08

### Added in 1.0.0

- Comprehensive design system with theme customization (`10bf77d`)
- Playwright test suite for UI testing (`f74bf88`)
- Lighthouse CI configuration and GitHub Actions integration (`156a8b9`)
- User guide, developer guide, and theme customization documentation (`db7c84a`)
- Actual screenshots for README documentation (`7306095`)
- Improved CI workflows with security scanning (`eae8728`)

### Changed in 1.0.0

- Replaced `SelectDateWidget` with HTML5 date input (`27f009f`)
- Refactored base template and updated main templates with modern design (`10bf77d`)
- Consolidated and refactored all documentation (`b60613a`)
- Clarified training workflow in Makefile (`127d13e`)
- Used media storage for attendance charts (`d534c8d`)

### Fixed in 1.0.0

- Flake8 and Black formatting issues (`a9a6ac9`, `5c913f6`)
- CI splits and subgroup analysis (`aac031b`)
- Pytest detection to avoid IndexError (`cecee6f`)
- Test failures by upgrading `deepface` to 0.0.93 (`8f53cf7`)
- `force_text` compatibility by upgrading `django-pandas` to >=0.6.7 (`4e2fa1e`)

### Documentation in 1.0.0

- Added comprehensive documentation including USER_GUIDE.md, DEVELOPER_GUIDE.md (`7dcabe4`)
- Updated non-programmer guide with recent features (`edd0bcb`)
- Image optimization via ImgBot (`ea6b90c`, `d58ce56`)

### Internal in 1.0.0

- Changed Dependabot update interval to daily (`76769ac`)
- Auto-enabled DEBUG mode when running tests (`1d5d2d4`)
- Reverted generated graphs and added to `.gitignore` (`02106c0`)
- Updated tf-keras version constraint (`1e308d4`)
- Removed version pinning for `crispy-bootstrap5` (`98dd2d4`)
- Bumped pip dependencies (`a3a89c2`, `16a29da`)

---

## [0.3.0] - 2025-10-29

### Added in 0.3.0

- Reproducibility features, DATA_CARD.md, and model training capabilities (`ab7bece`)
- Comprehensive documentation (docs: prefix) (`7dcabe4`)
- Reproducibility infrastructure and core modules (`6eda3af`)
- Management commands for data operations (`30ba686`)
- Admin UI views and tests for ablation and failure analysis (`aed0e15`)
- Integration tests and rigor pass summary (`da27d2b`)

### Changed in 0.3.0

- Handle headless camera sessions for server deployments (`d031c44`)
- Hardened Django configuration defaults for production (`58337ff`)
- Handle missing timestamps in Time model string representation (`c8ab464`)

### Documentation in 0.3.0

- Added extensive documentation and code comments (`099422a`)

---

## [0.2.0] - 2025-10-03

### Added in 0.2.0

- Configurable DeepFace match threshold (`d29301e`)
- Comprehensive non-programmer guide to README (`dc0afba`)

### Changed in 0.2.0

- Required login for attendance marking views (`24cbfd9`)
- Restricted register view to staff users with tests (`249de25`)
- Filtered staff users from employee count (`d68d7e3`)
- Updated admin access checks to use staff flags (`ffebfb9`)

### Fixed in 0.2.0

- Handle missing users in attendance updates (`6108496`)
- Ensure matplotlib Agg backend is set before pyplot import (`62220f3`)

### Security in 0.2.0

- Access control added to attendance views (`24cbfd9`)
- Staff-only restriction for registration (`249de25`)

---

## [0.1.0] - 2025-09-28

### Added in 0.1.0

- Major modernization and refactoring of the smart attendance system (`4164e14`)
- Django 5 compatibility updates
- Modern code standards and best practices

### Changed in 0.1.0

- Cleaned up README by removing redundant links and author info (`4d6c763`)
- Fixed predict return values and adjusted attendance views (`6b9dbf8`)
- Handled scalar predictions in attendance flow (`b16fb52`)
- Removed legacy merge remnants from recognition views (`a92c8cd`)
- Fixed decorator duplication (`ea22c77`)

---

## [0.0.2] - 2023-11-19

### Changed in 0.0.2

- Extensive HTML updates and improvements (`f118b09`)
- CSS styling updates (`d6b925e`)
- Removed training_dataset from git tracking (`9b8a792`)
- Folder structure reworking (`a1b412f`)

---

## [0.0.1] - 2021-04-27

### Added in 0.0.1

- Initial project release with full source code and documentation (`050ff69`)
- Django-based web application for attendance management
- Face recognition using OpenCV and machine learning
- User registration and attendance tracking
- Basic admin interface

### Security in 0.0.1

- Bumped Pillow for security patches (`8dc6cd2`)
- Bumped Django for security patches (`8c75234`)

### Documentation in 0.0.1

- Initial README with project documentation (`e9b313e`)
- Multiple README updates for clarity (`49c4c33`, `686a6cf`, `58576ed`)

---

## Breaking Changes Summary

### v1.5.0 (2025-12-05)

- **Time model direction field**: The Time model now uses a `direction` field for attendance tracking. Database migration required for PostgreSQL users. Run migrations before upgrading.
- **Face API authentication**: The face recognition API now requires authentication and has rate limiting enabled. Update API clients to include authentication credentials.

### v1.0.0 (2025-11-08)

- **Settings module restructured**: The settings are now organized as a package (`attendance_system_facial_recognition/settings/`). Update any direct imports from the old `settings.py` location.
- **Database configuration**: Now uses `DATABASE_URL` environment variable instead of direct SQLite configuration. Set `DATABASE_URL` for your database connection.

### v0.2.0 (2025-10-03)

- **Authentication required**: Attendance marking views now require user login. Anonymous users will be redirected to the login page.
- **Staff-only registration**: The `/register/` endpoint is now restricted to staff users only.

### v0.1.0 (2025-09-28)

- **Django 5 upgrade**: Requires Django 5.x. Previous Django 3.x code is not compatible.
- **Prediction return values**: The `predict` function now returns scalar values instead of arrays. Update any code that expects array returns.

---

[Unreleased]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/compare/v1.0.0...v1.5.0
[1.4.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2025-11-18&until=2025-11-29
[1.3.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2025-11-12&until=2025-11-18
[1.2.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2025-11-09&until=2025-11-11
[1.1.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2025-11-08&until=2025-11-09
[1.0.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2025-10-03&until=2025-10-29
[0.2.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2025-09-28&until=2025-10-03
[0.1.0]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2023-11-19&until=2025-09-28
[0.0.2]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?since=2021-04-29&until=2023-11-19
[0.0.1]: https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/commits/main?until=2021-04-29
