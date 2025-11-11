# Security & Compliance Guide

This document consolidates the security-critical configuration that must be in place when deploying the Attendance Management System. It cross-references the hardening defaults defined in [`attendance_system_facial_recognition/settings/base.py`](../attendance_system_facial_recognition/settings/base.py) and [`attendance_system_facial_recognition/settings/production.py`](../attendance_system_facial_recognition/settings/production.py).

## Secrets & Environment Variables

| Variable | Scope | Purpose | Enforcement |
| --- | --- | --- | --- |
| `DJANGO_SECRET_KEY` | Django | Cryptographically signs sessions and CSRF tokens. | Required whenever `DJANGO_DEBUG` is not enabled. The app exits if omitted. |
| `DATA_ENCRYPTION_KEY` | Application | 32-byte Fernet key used for encrypting sensitive artifacts (attendance exports, biometric caches). | Must be set outside of test/dev; invalid keys raise `ImproperlyConfigured`. |
| `FACE_DATA_ENCRYPTION_KEY` | Application | Separate Fernet key for cached facial encodings. | Required in production; validated for proper length and encoding. |
| `DJANGO_ALLOWED_HOSTS` | Django | Host header allow-list to prevent Host header attacks. | Mandatory when `DJANGO_DEBUG` is disabled. |
| `DJANGO_SESSION_COOKIE_SECURE` | Django | Forces the session cookie to HTTPS-only. | Defaults to `True` in production (`not DEBUG`). |
| `DJANGO_SESSION_COOKIE_HTTPONLY` | Django | Blocks JavaScript from reading the session cookie. | Defaults to `True`. |
| `DJANGO_SESSION_COOKIE_SAMESITE` | Django | Mitigates CSRF by limiting cross-site cookie sends. | Defaults to `Lax`; strict validation rejects unsupported values. |
| `DJANGO_SESSION_COOKIE_AGE` | Django | Limits session lifetime. | Must be a positive integer (defaults to 1800 seconds). |
| `DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE` | Django | Optional additional session restriction. | Parsed as boolean; defaults to `False`. |
| `DJANGO_CSRF_COOKIE_SECURE` | Django | Forces the CSRF cookie to HTTPS-only. | Defaults to `True` in production (`not DEBUG`). |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Worker | Broker/result transport URIs. | Default to local Redis for development; override in production. |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | Database | PostgreSQL connection settings. | Mandatory when using production settings; missing values raise `ImproperlyConfigured`. |
| `DB_CONN_MAX_AGE` | Database | Persistent connection lifetime. | Must be integer ≥ 0; defaults to 600 seconds in production. |
| `DB_SSL_REQUIRE` | Database | Enables `sslmode=require` on the default database connection. | When `true`, production settings inject the SSL flag. |

### Secret generation & storage guidance

* Generate Fernet keys with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` and store them in a dedicated secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault).
* Follow Django's [secret key management guidance](https://docs.djangoproject.com/en/stable/ref/settings/#std-setting-SECRET_KEY) and never commit secrets to source control.

## Cookie & CSRF Protections

* Session and CSRF cookies default to `Secure` and `HttpOnly` flags when `DJANGO_DEBUG` is false, aligning with Django's [session security recommendations](https://docs.djangoproject.com/en/stable/topics/security/#session-security).
* `DJANGO_SESSION_COOKIE_SAMESITE` defaults to `Lax`, providing baseline CSRF mitigation for same-site navigations. Acceptable overrides (`Lax`, `Strict`, `None`) are validated; invalid options raise configuration errors.
* The project enables `django.middleware.csrf.CsrfViewMiddleware`, enforcing token validation on state-changing requests in accordance with Django's [CSRF protection docs](https://docs.djangoproject.com/en/stable/ref/csrf/).

## SSL & Transport Security

* Set `DJANGO_SESSION_COOKIE_SECURE` and `DJANGO_CSRF_COOKIE_SECURE` to `true` in all non-local environments to ensure cookies are transmitted only over HTTPS.
* Require HTTPS termination at the load balancer or ingress and forward requests to Django with `X-Forwarded-Proto` headers so `SecurityMiddleware` can redirect insecure requests. Consult Django's [SSL/HTTPS deployment checklist](https://docs.djangoproject.com/en/stable/topics/security/#ssl-https).
* For managed PostgreSQL, enable `DB_SSL_REQUIRE=true` to append `sslmode=require`, aligning with Django's [database SSL guidance](https://docs.djangoproject.com/en/stable/ref/databases/#postgresql-notes).

## Operational Checklists

### Rotate `DATA_ENCRYPTION_KEY`

1. Generate a replacement Fernet key and stage it in your secrets manager.
2. Update the deployment environment (Kubernetes Secret, `.env`, etc.) with the new value.
3. Restart web and worker processes to load the new key.
4. Re-encrypt or rotate any persisted data that relies on the old key if applicable.

### Enable database SSL

1. Provision server certificates/CA bundles as required by your database provider.
2. Set `DB_SSL_REQUIRE=true` and restart the application to enforce TLS.
3. Verify connections use SSL by inspecting the PostgreSQL `ssl` column in `pg_stat_ssl` or similar diagnostics.

### Enforce HTTPS end-to-end

1. Terminate TLS at the edge (reverse proxy, load balancer) with strong ciphers.
2. Configure HTTP->HTTPS redirects (via the edge or `SecurityMiddleware`).
3. Set `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, and `SECURE_PROXY_SSL_HEADER` (if behind a proxy) per Django's [SECURE_* settings](https://docs.djangoproject.com/en/stable/ref/settings/#secure-proxy-ssl-header).
4. Enable HTTP Strict Transport Security (HSTS) at the proxy or by setting `SECURE_HSTS_SECONDS` in Django.

### Harden admin access

1. Restrict the `/admin/` route using network ACLs or VPN access.
2. Require staff users to configure strong, unique passwords and enable MFA via your SSO/IdP when possible.
3. Monitor Django admin logins and audit trails regularly.
4. Configure `ADMINS` and email backend to receive error notifications in line with Django's [deployment checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/#django-admin-check-deploy).

## Additional Recommendations

* Review Django's [deployment checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/) before promoting builds.
* Apply security patches promptly and track upstream Django security advisories.
* Keep dependency lockfiles up to date and run `pip install --require-hashes` or similar controls in CI/CD.

