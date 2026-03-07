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

## 2025-05-24 - CSV Injection in Frontend Export
**Vulnerability:** The table export logic in the frontend (`recognition/static/js/ui/tables.js` and `recognition/static/js/ui.js`) did not sanitize string fields starting with formula characters (`=`, `+`, `-`, `@`). This could allow attackers to inject malicious formulas that execute when the exported CSV is opened in spreadsheet software like Excel.
**Learning:** Data exported to CSV must be sanitized not just on the backend but also on the frontend to prevent Formula Injection (CSV Injection) attacks.
**Prevention:** Updated the `_escapeCSV` and `escapeCSV` functions to prepend a single quote (`'`) to any string that begins with a dangerous character (`=`, `+`, `-`, `@`), effectively neutralizing the formula execution.

## 2025-05-24 - DOM XSS via innerHTML
**Vulnerability:** The UI logic in `recognition/static/js/ui.js`, `recognition/static/js/ui/tables.js`, and `recognition/templates/recognition/admin/attendance_dashboard.html` assigned HTML strings to elements using `.innerHTML`. While currently using safe strings, this pattern is inherently dangerous and can lead to DOM-based XSS if the inputs become dynamic.
**Learning:** `innerHTML` should be strictly avoided in favor of safe DOM manipulation methods.
**Prevention:** Refactored the code to use `document.createElement()`, `textContent`, `append()`, and `replaceChildren()` to build and insert DOM elements safely without parsing HTML strings.
