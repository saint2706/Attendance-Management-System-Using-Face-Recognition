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
| `DJANGO_SECURE_PROXY_SSL_HEADER` | Django | Enables Django to trust the `X-Forwarded-Proto` header from a trusted proxy to determine if a request is secure (HTTPS). **CRITICAL**: Only enable when behind a trusted reverse proxy. | Defaults to `False` (disabled). See [Proxy SSL Configuration](#proxy-ssl-configuration) section below for detailed guidance. |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | Worker | Broker/result transport URIs. | Default to local Redis for development; override in production. |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | Database | PostgreSQL connection settings. | Mandatory when using production settings; missing values raise `ImproperlyConfigured`. |
| `DB_CONN_MAX_AGE` | Database | Persistent connection lifetime. | Must be integer ≥ 0; defaults to 600 seconds in production. |
| `DB_SSL_REQUIRE` | Database | Enables `sslmode=require` on the default database connection. | When `true`, production settings inject the SSL flag. |
| `RECOGNITION_MAX_UPLOAD_SIZE` | Application | Maximum allowed size in bytes for image uploads/payloads in recognition requests. Prevents memory exhaustion DoS attacks. | Defaults to 5242880 bytes (5 MiB). Enforced for file uploads, bytes/bytearray payloads, and base64-encoded strings before decoding. |

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

### Proxy SSL Configuration

The `DJANGO_SECURE_PROXY_SSL_HEADER` environment variable controls whether Django trusts the `X-Forwarded-Proto` header from a reverse proxy to determine if a request is secure (HTTPS). This setting is **critical** for correct HTTPS detection when running behind a proxy in production environments.

#### When to enable

**Only** enable `DJANGO_SECURE_PROXY_SSL_HEADER=true` when **all** of the following conditions are met:

1. The application is deployed behind a **trusted** reverse proxy or load balancer that correctly sets the `X-Forwarded-Proto` header
2. The proxy is configured to **always** strip any `X-Forwarded-Proto` headers from incoming client requests before adding its own
3. You control and trust the proxy layer completely

**Examples of trusted proxy environments:**
- Heroku Router
- AWS Elastic Load Balancer (ELB) / Application Load Balancer (ALB)
- Google Cloud Platform HTTP(S) Load Balancer
- Kubernetes Ingress controllers (NGINX Ingress, Traefik, etc.) with proper configuration
- Cloudflare with SSL/TLS settings configured

#### What this setting does

When `DJANGO_SECURE_PROXY_SSL_HEADER=true` is set:

- Django configures `SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')`
- Django treats a request as secure (`request.is_secure()` returns `True`) when the `X-Forwarded-Proto` header value is `https`
- This affects security-sensitive behavior including:
  - CSRF protection checks that depend on `request.is_secure()`
  - Secure cookie enforcement (`SESSION_COOKIE_SECURE` / `CSRF_COOKIE_SECURE`)
  - Generation of absolute URLs with the correct scheme in some contexts
  - Security middleware redirect logic

#### Security implications

**⚠️ SECURITY WARNING**: Enabling this setting in an untrusted environment can lead to serious security vulnerabilities.

**If you enable `DJANGO_SECURE_PROXY_SSL_HEADER` when your traffic is NOT always terminated by a trusted proxy:**
- A malicious client could spoof the `X-Forwarded-Proto: https` header
- Django would incorrectly treat a plain HTTP request as secure
- This can weaken protections around cookies and other security checks
- Session and CSRF cookies marked as `Secure` could be transmitted over unencrypted HTTP connections

**DO NOT enable this setting:**
- In local development environments (unless testing proxy behavior)
- In any environment where you do not fully control and trust the proxy layer
- When the application is directly exposed to the internet without a trusted proxy
- When the proxy does not strip client-supplied `X-Forwarded-Proto` headers

#### Relation to deployment documentation

The [DEPLOYMENT.md](DEPLOYMENT.md) guide describes how to configure the `X-Forwarded-Proto` header at the proxy/load balancer level. The `DJANGO_SECURE_PROXY_SSL_HEADER` environment variable is the corresponding Django-side setting that enables trust in that header.

**Configuration workflow:**
1. Configure your proxy/load balancer to set the `X-Forwarded-Proto` header (see [DEPLOYMENT.md](DEPLOYMENT.md))
2. Verify the proxy strips any client-supplied `X-Forwarded-Proto` headers
3. Set `DJANGO_SECURE_PROXY_SSL_HEADER=true` in your Django environment
4. Test that `request.is_secure()` correctly returns `True` for HTTPS requests

#### Testing

To verify correct configuration:

```python
# In Django shell or view
from django.http import HttpRequest
request = HttpRequest()
request.META['HTTP_X_FORWARDED_PROTO'] = 'https'
print(request.is_secure())  # Should print True if configured correctly
```

Or test via the browser by checking that secure cookies are being set correctly and HTTPS redirects work as expected.

## Operational Checklists

### Rotate `DATA_ENCRYPTION_KEY`

1. Generate a replacement Fernet key and stage it in your secrets manager.
2. Update the deployment environment (Kubernetes Secret, `.env`, etc.) with the new value.
3. Use the `rotate_encryption_keys` command to re-encrypt existing datasets:

   ```bash
   python manage.py rotate_encryption_keys \
     --new-data-key "new-base64-fernet-key" \
     --new-face-key "new-base64-fernet-key" \
     --backup-dir /path/to/backup
   ```

4. Restart web and worker processes to load the new key.
5. Verify the application can read existing data before removing backups.

See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md#encryption-key-rotation) for detailed command options.

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
