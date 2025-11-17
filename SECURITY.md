# Security Policy

This document outlines the security procedures and policies for the Smart Attendance System.

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
