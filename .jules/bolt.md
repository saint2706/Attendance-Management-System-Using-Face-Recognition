## 2025-12-24 - [Database Sorting Optimization]
**Learning:** Offloading sorting to the database (`.order_by("time")`) is more efficient and scalable than fetching unsorted data and sorting it in Python, especially for large datasets. It also simplifies the codebase.
**Action:** Always prefer database-level sorting for QuerySets when the field is indexed or part of the retrieved data, rather than sorting in application logic.
