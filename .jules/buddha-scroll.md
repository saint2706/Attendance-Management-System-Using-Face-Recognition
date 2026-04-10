# Buddha Progress Log 🧘‍♂️

## [SEO] `robots.txt` improvements
- Ensured the root `robots.txt` matches `frontend/public/robots.txt` and contains `Allow: /llms.txt`.

## [GEO] JSON-LD Schema updates
- Replaced instances of `dangerouslySetInnerHTML` for inline script schemas across frontend pages (`Home`, `Login`, `Dashboard`, and `MarkAttendance`) to natively loaded React 19 schemas that hoist properly.

## [PERF] LCP Loading
- Changed `Home.tsx` to statically import in `App.tsx` instead of lazy loading. The home component contains our hero element (LCP) which is critical for Core Web Vitals to load correctly and quickly.
