# Bolt's Journal

## 2025-05-22 - [Dataset Health Caching]
**Learning:** `glob` operations on the dataset directory were slowing down the admin dashboard. Caching `dataset_health` proved effective (reduction from ~100ms to ~12ms for 5000 files).
**Action:** Use Django cache for expensive filesystem stats, ensuring active invalidation when files change.

## 2025-12-23 - [Dataset Index Rebuild Optimization]
**Learning:** Rebuilding the face recognition dataset index required reading and decrypting every image file, even unchanged ones, leading to O(N) IO/CPU cost where N is dataset size.
**Action:** Implemented metadata-based caching (path + mtime + size) to skip expensive decryption for unchanged files during index rebuilds.
