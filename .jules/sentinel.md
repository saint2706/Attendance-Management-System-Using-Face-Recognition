## 2025-05-24 - DOM XSS in Setup Wizard
**Vulnerability:** The camera test page in the setup wizard (`users/templates/users/setup_wizard/step2_camera_test.html`) used `innerHTML` to display error messages. While the current implementation mostly used trusted strings, it created a dangerous pattern where future changes or unexpected error objects could introduce XSS vulnerabilities.
**Learning:** Even "internal" tools like setup wizards must adhere to strict security standards. Relying on "trusted input" is fragile; secure coding patterns (like `textContent`) should be the default.
**Prevention:** Replaced `innerHTML` assignment with `replaceChildren()` and `document.createTextNode()` to safely render text content. This pattern should be used for all dynamic text insertion in the project.
