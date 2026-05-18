# Kubernetes Optimization Summary

## Overview
This document serves as a summary of the stability, performance, and security gains implemented in the Kubernetes deployment configurations. All core resource definitions have been verified against best practices and the Kubernetes Persona boundaries to ensure scalable, resilient, and secure clusters.

## Resource Limits
All containers define explicit resource `requests` and `limits` to ensure optimal scheduling on Kubernetes nodes and to prevent any single pod from monopolizing cluster resources or suffering resource starvation.

- **Django Web (`web-deployment.yaml`) & Celery Worker (`celery-deployment.yaml`)**:
  - `requests`: memory: "512Mi", cpu: "250m"
  - `limits`: memory: "2Gi", cpu: "1000m"
  - *Reasoning*: These are memory-intensive components executing backend APIs and facial recognition tasks asynchronously. Reserving 512Mi accommodates the base Python process and libraries, while scaling to 2Gi ensures sufficient memory overhead during computationally expensive tasks.
- **Celery Beat (`celery-beat-deployment.yaml`)**:
  - `requests`: memory: "128Mi", cpu: "50m"
  - `limits`: memory: "256Mi", cpu: "100m"
  - *Reasoning*: As a lightweight periodic task scheduler, it requires far fewer resources compared to workers.
- **PostgreSQL (`postgres-statefulset.yaml`)**:
  - `requests`: memory: "256Mi", cpu: "100m"
  - `limits`: memory: "1Gi", cpu: "500m"
  - *Reasoning*: Optimized for database caching and queries, preventing unexpected out-of-memory errors while serving traffic.
- **Redis (`redis-statefulset.yaml`)**:
  - `requests`: memory: "128Mi", cpu: "50m"
  - `limits`: memory: "512Mi", cpu: "200m"
  - *Reasoning*: Support for in-memory caching and message brokering requires moderate RAM requests with headroom for spikes in cache utilization.

## Health Probes
Liveness and Readiness probes are correctly implemented for all containers to ensure resilience against unexpected hangs and application readiness before traffic is routed.

- **Web Deployment**:
  - Uses HTTP GET probe on `/monitoring/metrics/` to confirm that the Django application and routing mechanism are fully operational. Wait delays of 30-60s were chosen to give Python backend ample startup time before health validations begin.
- **Worker & Beat**:
  - `exec` probes run `celery inspect ping` (or `ps aux` for beat) to determine if processes are still responsive, thus ensuring deadlocked tasks trigger automated restart operations.
- **Databases**:
  - Uses application-specific commands (`pg_isready` for Postgres, `redis-cli ping` for Redis) to verify correct functioning below the HTTP stack, mitigating potential network routing false positives.

## Least-Privilege Security Context
Security policies conform rigorously to best practice isolation strategies:

- `automountServiceAccountToken: false` applied consistently to ensure workloads don't unnecessarily ingest high-privilege access keys.
- **Pod-level**:
  - `runAsNonRoot: true` enforces all executions occur within the safe confines of a non-root UNIX user context.
  - User definitions like `runAsUser: 1000` / `fsGroup: 1000` restrict permissions effectively. PostgreSQL uses explicit `fsGroup: 70` aligning with Alpine Linux standards.
  - `seccompProfile: { type: RuntimeDefault }` is enforced universally to restrict anomalous kernel syscalls.
- **Container-level**:
  - `allowPrivilegeEscalation: false` prevents SUID execution.
  - `readOnlyRootFilesystem: true` guards the primary image against compromise and persistence attempts. EmptyDir volume mounts at `/tmp` satisfy the application's runtime and caching requirements without granting root mount mutability.
  - `capabilities: { drop: [ALL] }` universally revokes Linux kernel capabilities irrelevant to individual application runtime requirements.

These configurations reflect proactive mitigation against common cloud-native security vectors, delivering a resilient, highly available standard.
Kubernetes configurations have been verified and optimized. Resource limits, liveness/readiness probes, and least-privilege security contexts (including automountServiceAccountToken: false and runAsNonRoot: true) are fully implemented and verified across all deployments.

## YAML Formatting Improvements
- Resolved `yamllint` warnings across all Kubernetes manifests by explicitly adding the `---` sequence to signify the start of a YAML document. This enforces standardized YAML formatting across the `k8s/` directory and improves consistency when processed by automated deployment tools.
