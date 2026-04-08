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

# Sentinel Learnings

## CORS Misconfiguration
- **Vulnerability**: The application was setting `CORS_ALLOW_ALL_ORIGINS = True` when `DEBUG = True`.
- **Severity**: High (in development environments potentially exposed, or if DEBUG is mistakenly left on in production).
- **Fix**: Removed the conditional `CORS_ALLOW_ALL_ORIGINS` setting entirely to enforce strict CORS origins (`CORS_ALLOWED_ORIGINS`) regardless of the `DEBUG` flag. It is best practice never to allow open CORS unless deliberately intended for a public API, which is not the case here.

## 2026-03-23 - Rate Limiting on API Attendance Endpoint

**Severity:** High
**Vulnerability:** The `/api/attendance/mark/` endpoint lacked rate limiting, making it vulnerable to brute-force attacks and resource exhaustion (DoS) through rapid, repeated submissions of facial data.
**Fix:** Implemented Django REST Framework's native `UserRateThrottle` by applying an `AttendanceRateThrottle` custom class to the endpoint. It dynamically restricts requests according to the `RECOGNITION_ATTENDANCE_RATE_LIMIT` setting (e.g., 5/minute).

## 2026-03-16 - Timing Attack Vulnerability in API Key Validation

**Vulnerability:** The `_authenticate_request` method in `recognition/views.py` and `recognition/views_legacy.py` used a standard Python `in` operator (`if api_key in api_keys:`) to validate API keys against a tuple of allowed keys. This non-constant-time comparison can theoretically be exploited in a timing attack to guess API keys character by character.
**Learning:** Sensitive strings like API keys, tokens, or passwords must always be compared using a constant-time comparison function to mitigate timing side-channels.
**Prevention:** Refactored the key matching logic to iterate over all allowed keys and use `secrets.compare_digest(api_key, allowed_key)` from the standard `secrets` library, ensuring constant-time validation.

### Input Validation (Zod)
- Use `zod` for parsing and validating inputs on the frontend before interacting with APIs to ensure input constraints like minimum and maximum length are satisfied and fast fail.
- Added `zod` schema to validate `LoginCredentials` before executing login requests.
## 2025-05-24 - Input Validation in Auth API\n**Vulnerability:** The `registerEmployee` function in `frontend/src/api/auth.ts` lacked client-side input validation, which could allow malformed or missing data to be sent to the server.\n**Learning:** Client-side validation using libraries like Zod provides a crucial first layer of defense and improves user experience by failing fast on invalid inputs.\n**Prevention:** Added a Zod schema (`RegisterDataSchema`) to validate user registration data before making API requests, ensuring all required fields are present and meet complexity requirements.
- **Vulnerability:** Added rate limiting to the setup wizard views to prevent abuse.

## 2026-04-03 - Vulnerable Dependency: picomatch
**Vulnerability:** The `picomatch` package in the frontend dependencies had a high severity ReDoS vulnerability via extglob quantifiers (CVE-2024-XXXX) and a moderate Method Injection vulnerability (CVE-2024-XXXX).
**Learning:** Outdated dependencies in the Node.js ecosystem, particularly deep in the tree or in build tools, can expose the application or build environment to denial-of-service and code injection risks. Regular audits (`pnpm audit`) are essential.
**Prevention:** Updated `picomatch` to a patched version (4.0.4) using `pnpm install picomatch@4.0.4`.


## 2026-05-24 - JSON-LD XSS Remediation in React Components

**Vulnerability:** The React frontend was using `dangerouslySetInnerHTML={{ __html: JSON.stringify(...) }}` to inject JSON-LD `<script>` tags in `Home.tsx`, `Login.tsx`, `Dashboard.tsx`, and `MarkAttendance.tsx`. `JSON.stringify` does not escape `<` or `>` characters. If user input were ever included in these JSON objects, a malicious user could inject `</script><script>alert(1)</script>`, which the browser would parse and execute, leading to a DOM XSS vulnerability. Note that simply changing `dangerouslySetInnerHTML` to use React's native child string rendering for `<script>` tags does not solve this issue (as React does not HTML-escape `<script>` children either) and actually makes the sink less visible to static analysis tools like ESLint.
**Learning:** When embedding JSON inside a `<script>` tag, the `<` characters must be explicitly escaped to prevent closing the script tag prematurely. Furthermore, it is better to retain `dangerouslySetInnerHTML` for this specific use case because it flags the data sink for security linting (`react/no-danger`), ensuring developers remain aware of the need for manual sanitization.
**Prevention:** Maintained the use of `dangerouslySetInnerHTML` to keep the sink visible to static analysis, but sanitized the JSON output by replacing all instances of `<` with `\u003c` (i.e., `JSON.stringify(...).replace(/</g, '\\u003c')`).
