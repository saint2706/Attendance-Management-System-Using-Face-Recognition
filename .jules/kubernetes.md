# Kubernetes Optimizations

## Discoveries & State
- All deployments (web, celery-worker, celery-beat) and statefulsets (redis, postgres) have defined resource limits and requests.
- Liveness and readiness probes are implemented correctly (including process checking for celery beat, and ping checking for celery workers using `$HOSTNAME`).
- Semantic versioning is strictly used (e.g., `redis:7.2.13-alpine`, `postgres:16.2-alpine`, `v1.7.0` for attendance-system). No `latest` tags were found.
- Least-privilege security contexts are applied (`runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`, dropping all capabilities, and using `automountServiceAccountToken: false`).
- Temporary volumes (e.g., `/tmp`, `/var/run/postgresql`) are properly mounted as `emptyDir: {}` for read-only filesystems.
- ConfigMaps and Secrets are utilized instead of hardcoded values, except for necessary defaults.
- Labels and selectors are consistent and follow the `app.kubernetes.io` format.
- `commonLabels:` is not used in kustomization (avoiding deprecation warnings).

## Changes made
- Verified configurations met all the stringent guidelines laid out in memory and boundaries, ensuring scalable, resilient, and secure deployment configurations.
- The Kustomize outputs rendered cleanly without any syntax or logic errors for base, development, and production overlays.
- Verified resource limits, probes, and least-privilege security contexts for all manifests.
