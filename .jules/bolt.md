## 2025-12-24 - [Database Sorting Optimization]
**Learning:** Offloading sorting to the database (`.order_by("time")`) is more efficient and scalable than fetching unsorted data and sorting it in Python, especially for large datasets. It also simplifies the codebase.
**Action:** Always prefer database-level sorting for QuerySets when the field is indexed or part of the retrieved data, rather than sorting in application logic.

## 2025-01-08 - [Filesystem State Caching]
**Learning:** For high-traffic endpoints that depend on filesystem state (like dataset availability), caching the file scan results (`glob` + `stat`) significantly reduces IO overhead.
**Action:** Implement a "Cache-Aside" pattern for expensive filesystem checks, with explicit invalidation in the write path (e.g., Celery tasks that modify the dataset).
