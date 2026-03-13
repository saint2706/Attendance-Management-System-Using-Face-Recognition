## 2025-05-24 - DOM XSS in Setup Wizard
**Vulnerability:** The camera test page in the setup wizard (`users/templates/users/setup_wizard/step2_camera_test.html`) used `innerHTML` to display error messages. While the current implementation mostly used trusted strings, it created a dangerous pattern where future changes or unexpected error objects could introduce XSS vulnerabilities.
**Learning:** Even "internal" tools like setup wizards must adhere to strict security standards. Relying on "trusted input" is fragile; secure coding patterns (like `textContent`) should be the default.
**Prevention:** Replaced `innerHTML` assignment with `replaceChildren()` and `document.createTextNode()` to safely render text content. This pattern should be used for all dynamic text insertion in the project.

## 2025-05-24 - Path Disclosure in Recognition API
**Vulnerability:** The FaceRecognitionAPI exposed the absolute server path of the matched identity image in its JSON response. This information disclosure could reveal directory structure and usernames.
**Learning:** API responses should be sanitized to only expose necessary data. Internal file paths are implementation details that should not be leaked to clients.
**Prevention:** Modified the API to only return the filename (stripped of the directory path) in the 'identity' field.

## 2025-05-24 - Vulnerable Dependencies
**Vulnerability:** The `django` package version `6.0` was vulnerable to several high severity CVEs including CVE-2025-13473. The `rollup` dependency in the frontend had an arbitrary file write via path traversal vulnerability (CVE-2024-36517) in versions `<4.59.0`.
**Learning:** Outdated dependencies are a common source of critical vulnerabilities. Routine scanning using tools like `pip-audit` and `pnpm audit` is essential to identify and patch vulnerable libraries.
**Prevention:** Updated `django` to version `6.0.3` (or newer) in `requirements.txt` and `rollup` to `4.59.0` (or newer) in the frontend. Both dependencies have been updated to non-vulnerable versions.
## 2026-03-09 - DOM XSS Remediation across Frontend Core\n\n**Vulnerability:** Widespread usage of `innerHTML` in core JavaScript modules (`attendance-session.js`, `ui.js`, `tables.js`) and Django template inline scripts (`attendance_dashboard.html`) opened the application to DOM XSS vectors, particularly when handling real-time WebSocket/API events containing error messages.\n\n**Prevention:** Completely refactored `innerHTML` usage to native DOM APIs. Specifically:\n- Replaced string template rows with `document.createElement` and `replaceChildren`.\n- Converted dynamically generated status badges and sort icons to explicit DOM element construction (`appendChild`).\n- Maintained critical performance optimizations in `attendance-session.js` by replacing HTML string equality checks with `JSON.stringify(events)` equality checks.\n- Updated testing infrastructure to mock `replaceChildren` instead of the `innerHTML` setter to guarantee continuous coverage of DOM update optimizations.

## 2026-03-12 - MD5 Hashing Replacement

**Vulnerability:** The `compute_dataset_hash` function in `recognition/embedding_cache.py` used the weak `hashlib.md5()` hashing algorithm to compute cache invalidation hashes. MD5 is vulnerable to collision attacks.

**Prevention:** Replaced `hashlib.md5()` with `hashlib.sha256()` to ensure strong cryptographic hashing is used consistently across the application.
## 2026-03-13 - Rate Limiting on Expensive Admin Views

**Vulnerability:** The `add_photos` and `train` views in `recognition/views.py` were not protected by rate limits. These views trigger expensive background tasks (dataset capture and model training). Without rate limits, a compromised admin account or an intentional abuse could flood the task queue, causing a Denial of Service (DoS) and degrading performance for all users.
**Learning:** Resource-intensive administrative endpoints must be rate-limited, even if they require authentication, to prevent abuse or compromised accounts from taking down the system. Defense in depth means assuming authenticated users can still perform malicious actions.
**Prevention:** Added `@ratelimit` decorators to `add_photos` (10/min) and `train` (3/min) views in `recognition/views.py`, mirroring the protections already established in `recognition/views_legacy.py`. The view logic now explicitly checks `getattr(request, 'limited', False)` and returns an error message instead of triggering the background tasks.
