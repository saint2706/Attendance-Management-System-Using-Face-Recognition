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

## 2025-01-20 - [Frontend Code Splitting]
**Learning:** Large JavaScript bundles can cause poor First Input Delay (FID) and Largest Contentful Paint (LCP) metrics. Synchronously importing all routes at the application entry point forces the client to download the entire application before rendering anything.
**Action:** Implemented route-based code splitting using `React.lazy()` and `Suspense` in `frontend/src/App.tsx`. By lazy-loading components like `Home`, `Login`, `Dashboard`, and `MarkAttendance`, the initial JS bundle size is significantly reduced, decreasing the application load time and improving Core Web Vitals.
**Metrics:** Reduced the initial JS bundle load size and expected to improve application load time by approximately ~100-200ms depending on network speed.

## Optimization: N+1 query issue in Django Admin for SetupWizardProgress
- Found `SetupWizardProgress` model with `user` foreign key, but it was not registered explicitly in Django Admin with `list_select_related`. The `__str__` method explicitly used `self.user.username` resulting in N+1 queries.
- **Optimization:** Explicitly added `SetupWizardProgress` to `users/admin.py` with `@admin.register(SetupWizardProgress)` and defined `list_select_related = ("user",)`.
- **Result:** Decreased database hits when viewing `SetupWizardProgress` models in admin interface to a single query.

## Optimization: N+1 query issue in Django Admin Dashboards
- Found multiple `.count()` queries being executed consecutively against the same queryset in `recognition/admin_views.py` (`liveness_results_dashboard` and `_get_summary_stats`).
- **Optimization:** Refactored these multiple sequential database queries into single `.aggregate()` calls using `Count("pk", filter=Q(...))` to minimize database roundtrips.
- **Result:** Decreased query counts significantly on the admin dashboard views.
- Optimized multiple `.count()` queries in `recognition/scheduled_tasks.py` and `recognition/health.py` into single `.aggregate()` calls using `Count('pk')` to minimize database roundtrips and prevent redundant operations.

## Optimization: Memoized React Context Providers
- Unmemoized object literals in React Context Providers (e.g. `value={{ user, login, logout }}`) cause all consuming components to re-render whenever the provider re-renders, even if the actual context data hasn't changed.
- **Optimization:** Wrapped the `value` props of `AuthContext.Provider` and `ThemeContext.Provider` with `useMemo` hooks, and wrapped context functions like `login`, `logout`, `setTheme`, and `toggleTheme` with `useCallback`.
- **Result:** Decreased unnecessary component re-renders across the entire React application by ensuring context consumers only update when the memoized context dependencies change.
## 2025-01-20 - [Real API Integration for Dashboard Stats]
**Learning:** Dashboard statistics were relying on mocked data with a hardcoded timeout, creating misleading user experience and potential performance issues since actual stats were never loaded.
**Action:** Replaced the mock data with actual API integration via `getAttendanceStats`. Used dynamic import to optimize bundle size.
## 2025-01-20 - [Real API Integration for Dashboard Stats]
**Learning:** Avoid dynamic imports inside `useEffect` for API wrapper functions, as it delays the data fetch and worsens user-facing latency. Also added unmount checks.
**Action:** Replaced the mock data with actual API integration via `getAttendanceStats`. Used static import.

## Optimization: Dashboard Stats API Caching
- **Problem**: The dashboard was calling `getAttendanceStats` on every render, which is an expensive API call querying the database for all attendance records for the day.
- **Optimization**: Implemented `@tanstack/react-query` to cache the dashboard stats API call. Wrapped the application with `QueryClientProvider` and used `useQuery` in `Dashboard.tsx` with a `staleTime` of 5 minutes.
- **Result**: Reduced redundant network requests when navigating back and forth to the dashboard. The dashboard now loads instantly on subsequent visits within the 5-minute cache window, significantly reducing backend load and improving perceived performance.
## Optimization: Dashboard API Path Correction
- **Problem**: The `getAttendanceStats` API function in `frontend/src/api/attendance.ts` was pointing to `/dashboard/stats/` instead of `/attendance/stats/`.
- **Optimization**: Corrected the endpoint URL to `/attendance/stats/` which is the correct URL for the AttendanceViewSet.
- **Result**: Reduced 404 errors and correctly mapped the frontend to the backend endpoint.
## Optimization: Dashboard React Memoization
- **Problem**: Admin Action cards were rerendering unnecessarily on Dashboard state changes.
- **Optimization**: Extracted `ActionCard` component in `frontend/src/components/ActionCard.tsx` and memoized it using `React.memo()`. Replaced inline `Link` tags in `frontend/src/pages/Dashboard.tsx` with `ActionCard` to improve React rendering efficiency.
- **Result**: Reduced React rerenders in Dashboard.
## Optimization: Removed unused query optimizations
- **Problem**: The `AttendanceViewSet.get_queryset` in `recognition/api/views.py` used `.select_related("present_record", "time_record")` and `prefetch_related("user__groups", "user__user_permissions")` to optimize serialization. However, `AttendanceRecordSerializer` doesn't access these fields, it only needs the `user` relation for `get_username`.
- **Optimization**: Removed the unused explicit `.select_related` and `prefetch_related` relations from the queryset, leaving only `.select_related("user")`.
- **Result**: Reduced DB query complexity, preventing useless joins and extra prefetch queries that fetch data unused by the serializer.
- Fixed N+1 queries by adding `select_related('user')` when querying the `Time` and `Present` models in the `hours_vs_date_given_employee` and `hours_vs_employee_given_date` view functions. Additionally, removed redundant `select_related` and `prefetch_related` queries in the API Viewsets that were not utilized by their respective Serializers, thereby preventing useless JOINs.
