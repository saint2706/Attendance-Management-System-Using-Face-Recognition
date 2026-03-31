# Kubernetes Manifests Audit & Optimization Report

## Stability and Security Gains

- **Resource Limits & Requests:**
  - Verified that all containers across `web`, `celery-worker`, `celery-beat`, `postgres`, and `redis` have strictly defined `requests` and `limits` for CPU and Memory, preventing resource starvation and noisy neighbor issues.
- **Probes (Liveness & Readiness):**
  - Confirmed robust HTTP and exec-based probes are active on all critical components to ensure automated traffic routing and recovery.
- **Security Contexts:**
  - Validated that `securityContext` settings are properly implemented, utilizing `runAsNonRoot`, `readOnlyRootFilesystem`, and `allowPrivilegeEscalation: false` to enforce least privilege.
- **Secrets Management:**
  - Removed raw plaintext placeholders from `k8s/base/secret.yaml`. Transitioned to a clean Opaque Secret template with documentation on creating secrets externally (e.g., via `ExternalSecrets`, Vault, or SealedSecrets) to adhere to the "Never store raw secrets" rule.
- **Validation:**
  - Ran `kubeconform` to validate schema configurations.
  - Completed syntax checks with `yamllint`.

