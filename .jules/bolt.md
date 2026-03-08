## 2025-01-20 - [Redundant Data Normalization Loop]
**Learning:** Legacy defensive programming (validating data structure on every request) can become a significant performance bottleneck (O(N)) when the data source (a cache) already guarantees the correct format.
**Action:** When consuming data from a trusted internal cache, optimistically verify the first entry to confirm the data contract (e.g., correct types) and skip redundant O(N) normalization loops. Always ensure the fallback logic remains for robustness against cache corruption or invalid states.

## 2026-03-08 - [N+1 query with ModelSerializer in DRF]
**Learning:** In Django Rest Framework, using a generic `ModelSerializer` without explicitly fetching related objects can cause a huge amount of N+1 database queries. Since serializers will implicitly access related objects to get data for serializer fields, this triggers a new database fetch.
**Action:** When creating a ModelViewSet in Django Rest Framework, always use `select_related()` (for foreign keys and one-to-one relations) or `prefetch_related()` (for many-to-many and reverse foreign key relations) in the `get_queryset` method to prevent N+1 queries during serialization.
