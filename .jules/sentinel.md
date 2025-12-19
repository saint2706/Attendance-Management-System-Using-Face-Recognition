## 2025-12-19 - [CRITICAL] Missing Authentication on Face Recognition API
**Vulnerability:** The `FaceRecognitionAPI` endpoint was accessible without any authentication (API key or session checks were missing in `views_legacy.py` which is the active module).
**Learning:** Duplicate code files (`views.py` vs `views_legacy.py`) caused confusion. The newer secure code in `views.py` was not being used. Always verify which module is actually imported and used by the application.
**Prevention:** Remove unused duplicate code. Ensure tests target the actual code used in production. Verify `__init__.py` exports.

## 2025-12-19 - [MEDIUM] Timing Attack on API Key Verification
**Vulnerability:** API key verification used `key in api_keys` and early return, allowing for timing attacks to enumerate valid API keys.
**Learning:** List membership checks are not constant time.
**Prevention:** Iterate through all allowed keys and use `secrets.compare_digest` for each comparison, maintaining constant time flow.

## 2025-12-19 - [MEDIUM] Unbounded Username Length
**Vulnerability:** The API accepted usernames of arbitrary length, potentially causing database errors or Denial of Service.
**Learning:** Always validate and sanitize input length at the boundary (view layer) before passing to the data layer.
**Prevention:** Truncate or reject inputs exceeding expected length limits.
