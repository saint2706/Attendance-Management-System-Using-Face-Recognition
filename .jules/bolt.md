## 2025-01-20 - [Redundant Data Normalization Loop]
**Learning:** Legacy defensive programming (validating data structure on every request) can become a significant performance bottleneck (O(N)) when the data source (a cache) already guarantees the correct format.
**Action:** When consuming data from a trusted internal cache, optimistically verify the first entry to confirm the data contract (e.g., correct types) and skip redundant O(N) normalization loops. Always ensure the fallback logic remains for robustness against cache corruption or invalid states.

## 2025-01-21 - [Zero-Copy Cache Access]
**Learning:** Defensive copying of large cached datasets (`[dict(x) for x in cache]`) adds unnecessary O(N) overhead if the downstream consumers are read-only.
**Action:** Verify downstream functions (like `find_closest_dataset_match`) do not mutate input. If safe, pass the cached reference directly to save CPU and memory allocation on every request.
