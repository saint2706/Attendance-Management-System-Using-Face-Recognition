## 2025-01-20 - [Redundant Data Normalization Loop]
**Learning:** Legacy defensive programming (validating data structure on every request) can become a significant performance bottleneck (O(N)) when the data source (a cache) already guarantees the correct format.
**Action:** When consuming data from a trusted internal cache, optimistically verify the first entry to confirm the data contract (e.g., correct types) and skip redundant O(N) normalization loops. Always ensure the fallback logic remains for robustness against cache corruption or invalid states.

## 2026-03-08 - [N+1 query with ModelSerializer in DRF]
**Learning:** In Django Rest Framework, using a generic `ModelSerializer` without explicitly fetching related objects can cause a huge amount of N+1 database queries. Since serializers will implicitly access related objects to get data for serializer fields, this triggers a new database fetch.
**Action:** When creating a ModelViewSet in Django Rest Framework, always use `select_related()` (for foreign keys and one-to-one relations) or `prefetch_related()` (for many-to-many and reverse foreign key relations) in the `get_queryset` method to prevent N+1 queries during serialization.

## Optimization: N+1 query issue in attendance stats endpoint
- The `attendance-stats` API endpoint (`AttendanceViewSet.stats` in `recognition/api/views.py`) previously executed 4 separate redundant queries against the `RecognitionAttempt` table for the same date.
- The queries checked for IN attempts, OUT attempts, and both again to calculate pending checkout values.
- **Optimization:** Refactored the view to fetch a single list of `user_id` and `direction` values from the `RecognitionAttempt` table. Used Python `set` operations to efficiently group the distinct active IN/OUT attempts to compute the metrics. This avoids redundant roundtrips to the DB.
- **Result:** Decreased queries on `attendance-stats` endpoint from 5 down to 2.

## Optimization: N+1 query issue in UserViewSet
- The `user-list` API endpoint (`UserViewSet.get_queryset` in `recognition/api/views.py`) previously executed an N+1 query issue when serializing the User model by implicitly fetching its groups and user permissions.
- **Optimization:** Added `.prefetch_related("groups", "user_permissions")` on `User.objects` queries to batch database hits.
- **Result:** Decreased queries on `user-list` endpoint significantly for a list of users.

## Optimization: N+1 query issue in Django Admin models
**Learning:** In Django Admin, if `__str__` accesses a related field (like `self.user.username`) or if the `list_display` renders methods that traverse relations, Django will trigger an N+1 query for each object in the list view, degrading performance significantly when dealing with many objects like `Time`, `Present`, or `RecognitionAttempt`.
**Action:** Always create custom `ModelAdmin` classes for models displayed in the admin interface and add `list_select_related = ("user",)` (or relevant foreign keys) to explicitly join the related tables in a single query.
- **Result:** Decreased N+1 queries in the Django Admin for `Time`, `Present`, and `RecognitionAttempt` models.
