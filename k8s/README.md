# Kubernetes Deployment Configurations

This directory contains the Kubernetes manifests for deploying the Attendance Management System.

## Important Note Regarding Secrets

**NEVER** store raw secrets in YAML files within this repository. The `attendance-secrets` Secret is required for the application to run but is purposefully omitted from version control.

You must provision the `attendance-secrets` using a secure external secret management solution (such as Bitnami Sealed Secrets, HashiCorp Vault, AWS Secrets Manager, etc.) or manually create the secret in your cluster before deploying the application.

### Required Secret Data

The `attendance-secrets` generic Secret must contain the following keys:

- `DJANGO_SECRET_KEY`: A secure, random Django secret key.
- `DATA_ENCRYPTION_KEY`: A base64 Fernet encryption key.
- `FACE_DATA_ENCRYPTION_KEY`: A base64 Fernet encryption key.
- `DB_USER`: The username for the PostgreSQL database.
- `DB_PASSWORD`: The password for the PostgreSQL database user.
- `POSTGRES_PASSWORD`: The password for the PostgreSQL superuser.

### Manual Provisioning Example

```bash
kubectl create secret generic attendance-secrets \
  --namespace attendance-system \
  --from-literal=DJANGO_SECRET_KEY='your-secure-secret-key' \
  --from-literal=DATA_ENCRYPTION_KEY='your-fernet-key' \
  --from-literal=FACE_DATA_ENCRYPTION_KEY='your-fernet-key' \
  --from-literal=DB_USER='attendance' \
  --from-literal=DB_PASSWORD='your-secure-password' \
  --from-literal=POSTGRES_PASSWORD='your-secure-password'
```