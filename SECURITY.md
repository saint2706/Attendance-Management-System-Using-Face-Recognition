# Security Policy

This document outlines the security procedures and policies for the Attendance Management System Using Face Recognition.

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Instead, please report security issues by opening a [private security advisory](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/security/advisories/new) on GitHub.
3. Alternatively, contact the maintainers directly through the repository.

### What to Include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes (optional)

### Response Timeline

- We aim to acknowledge security reports within 48 hours.
- We will provide an initial assessment within 7 days.
- We will work with you to understand and address the issue promptly.

### Disclosure Policy

- We request that you give us reasonable time to address the vulnerability before public disclosure.
- We will credit researchers who report valid vulnerabilities (unless you prefer to remain anonymous).

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Encryption

The application uses Fernet symmetric encryption to protect sensitive data at rest.

-   **`DATA_ENCRYPTION_KEY`:** Used to encrypt general-purpose sensitive data.
-   **`FACE_DATA_ENCRYPTION_KEY`:** Used specifically to encrypt the face embeddings stored in the database.

These keys are generated using the Fernet algorithm and must be kept secret. They should be stored securely and never committed to the repository.

## Sensitive Data Handling

-   **Face Embeddings:** The raw face embeddings are encrypted before being stored in the database.
-   **User Metadata:** Personally Identifiable Information (PII) is handled with care and is not logged.
-   **Logging:** The application is configured to avoid logging sensitive information. Raw images, embeddings, and decrypted data should never be logged.

## Production Security Settings

For production deployments, the following security settings are recommended:

-   **HTTPS:** The application should be served over HTTPS to protect data in transit.
-   **Secure Cookies:** The following Django settings should be set to `True` in a production environment:
    -   `SESSION_COOKIE_SECURE`
    -   `CSRF_COOKIE_SECURE`
-   **HSTS:** HTTP Strict Transport Security (HSTS) should be enabled to prevent downgrade attacks.

## Sentry Error Tracking

The application uses Sentry for error tracking. Sentry is configured to not send sensitive payloads.

## Encryption Key Rotation Procedure

To keep encrypted artifacts fresh without downtime, use a dual-key rotation with backups:

1. **Back up encrypted assets:** snapshot `face_recognition_data/` (including `training_dataset/` and `encodings/`) before any mutation. The rotation command supports `--backup-dir` for this.
2. **Stage dual keys:** set `DATA_ENCRYPTION_KEY` and `FACE_DATA_ENCRYPTION_KEY` to comma-separated values of `"<new>,<current>"`. The first key is used for new writes; both are accepted for reads during the cutover window.
3. **Re-encrypt artifacts:** run `python manage.py rotate_encryption_keys --new-data-key <new_data_key> --new-face-key <new_face_key>` from the Django root. The command decrypts using the active key set and re-encrypts datasets, models, and facial encodings with the new keys atomically.
4. **Restart services:** update environment variables to the **new key only**, then restart web, worker, and scheduler processes to drop the old key while continuing to serve traffic.
5. **Validate:** confirm attendance marking and recognition flows still operate and that new records decrypt with the new key only.

If anything fails, restore the pre-rotation backups and revert the environment variables to the previous keys before retrying.
