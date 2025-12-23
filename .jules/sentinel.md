## 2025-05-23 - Stored XSS in Attendance Session Monitor
**Vulnerability:** The attendance session monitor (`recognition/static/js/attendance-session.js`) inserted unescaped API data (username, direction, error messages) directly into the DOM using `innerHTML`. This allowed Stored XSS if a malicious username was registered or spoofed in a recognition attempt.
**Learning:** Frontend code using `innerHTML` to render API responses bypasses Django's automatic template escaping. Even "internal" APIs should be treated as untrusted sources for frontend rendering.
**Prevention:** Use `textContent` for text updates, or explicitly escape HTML entities when building HTML strings in JavaScript. Added `_escapeHtml` helper to sanitise all user-controlled inputs before rendering.

## 2025-05-24 - DOM XSS in AlertManager
**Vulnerability:** The `AlertManager` utility (`recognition/static/js/ui/alerts.js`) used `innerHTML` to render alert messages, allowing DOM-based XSS if user-controlled input was passed to the `show()` method.
**Learning:** Generic UI utilities must be secure by default. Using `innerHTML` for convenience creates a "sink" that can be exploited by any caller passing unsanitized data.
**Prevention:** Replaced `innerHTML` with `textContent` for rendering the message body, and used `createElement`/`appendChild` for the close button. This ensures all message content is treated as text.

## 2025-05-24 - Unprotected Admin Actions
**Vulnerability:** Sensitive administrative actions like employee registration and setup wizard steps were missing rate limiting, relying solely on authentication.
**Learning:** Even authenticated admin endpoints are vulnerable to compromised account abuse (DoS, data spamming). "Internal" does not mean "Safe".
**Prevention:** Applied `django-ratelimit` to `register` and `setup_wizard` views to limit request frequency per user.
