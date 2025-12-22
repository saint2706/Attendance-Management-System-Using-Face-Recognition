# Bolt's Journal

## 2025-05-22 - [Dataset Health Caching]
**Learning:** `glob` operations on the dataset directory were slowing down the admin dashboard. Caching `dataset_health` proved effective (reduction from ~100ms to ~12ms for 5000 files).
**Action:** Use Django cache for expensive filesystem stats, ensuring active invalidation when files change.
