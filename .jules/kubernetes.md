# Kubernetes Optimizations

- Updated `prometheus.io/path` in `k8s/base/web-deployment.yaml` from `/metrics` to `/monitoring/metrics/` to accurately reflect the Prometheus metrics endpoint in the Django application (`recog_views.monitoring_metrics`), improving observability without causing 404 errors during metric collection.
- Validated modifications using `kubeconform` to ensure robust schema compliance and deployment reliability.
- Ensured least-privilege principles by keeping `automountServiceAccountToken: false` and secure Container contexts (`allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`, and standard non-root executions) fully enforced.
