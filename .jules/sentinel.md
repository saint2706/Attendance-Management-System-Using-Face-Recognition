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
