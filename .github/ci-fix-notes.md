# CI Fix Notes

## 2025-11-12
- Added missing runtime dependencies (`dj-database-url`, `cryptography`, and NumPy 1.26.x) to `pyproject.toml` and aligned requirement pins to prevent import errors during Django checks.
- Pinned Playwright tooling in both development extras and `requirements-dev.txt`, then installed Chromium browsers so UI tests execute locally and in CI.
- Hardened `_RecognitionAttemptLogger` against patched `time.monotonic()` returning mocks and guarded API logging to tolerate anonymous requests, fixing `pytest` failures.
- Cleared the rate-limit cache between face-recognition workflow tests to avoid leakage from prior API rate-limit tests.
