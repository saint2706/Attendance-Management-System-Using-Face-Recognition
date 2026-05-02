# Buddha Scroll - GEO/SEO Improvements
- [GEO] Updated llms.txt across static directories to include backend API/admin routes and SPA routes.
- [SEO] Validated Sitemap URL in robots.txt.
- [PERF] Verified Hero image optimization in Home.tsx.
- [SEO] Verified JSON-LD schema.
## SEO/GEO Improvements
- `[PERF]`: Replaced lazy loading with eager loading for the `Home` route in `frontend/src/App.tsx` to significantly improve Largest Contentful Paint (LCP) by preventing the hero section from being network-delayed.
- `[GEO] [SEO]`: Moved static JSON-LD from `frontend/index.html` to dynamic React component using `dangerouslySetInnerHTML` in `frontend/src/pages/Home.tsx`.
- `[GEO] [SEO]`: Implemented JSON-LD WebPage schema in Dashboard, Login, and MarkAttendance pages using `dangerouslySetInnerHTML`.
- `[PERF]`: Removed unused `icon-512.png` image preload tag from `frontend/index.html` to fix Lighthouse "Remove unused preloads" and improve LCP for actual critical resources.
- `[SEO]`: Added React 19 Document Metadata (`<title>` and `<meta name="description">`) to `Home`, `Dashboard`, `Login`, and `MarkAttendance` SPA components to enable dynamic SEO routing.
- [SEO] Added twitter:image to index.html
- [GEO] Enhanced llms.txt with descriptive site context
- [GEO] Updated frontend/public/robots.txt to explicitly allow /llms.txt for AI discoverability.
- [GEO] Improved semantic HTML structure by converting root `div` components to `<main>` tags in `Home.tsx`, `Dashboard.tsx`, `Login.tsx`, and `MarkAttendance.tsx` to enhance SEO and AI vector friendliness.
- [GEO] Verified that JSON-LD dangerouslySetInnerHTML usages are already properly sanitized with HTML escaping.
- [SEO] Updated `robots.txt` and `frontend/public/robots.txt` to Disallow indexing of all private routes. Removed sitemap.xml to avoid duplicate content indexing.
- [SEO] Added `noindex, nofollow` meta tags to private SPA pages (`Dashboard`, `Login`, `MarkAttendance`).
- [SEO] Added canonical URL link to the `Home` page, and JSON-LD schema with `window.location.origin` for dynamic domains instead of hardcoding, to solidify its SEO authority.
- [PERF] Preloaded Google Fonts in `frontend/index.html` to prevent Flash of Unstyled Text (FOUT) and improve LCP and CLS.
- [GEO] Categorized endpoints in llms.txt files to improve AI readability
## 2024-05 - SEO/GEO Optimizations

- [SEO] Verified `robots.txt` in root and `frontend/public/` allows `/llms.txt` and restricts admin paths.
- [SEO] Verified `llms.txt` exists and accurately details site architecture.
- [PERF] Added explicit dimensions to the `.hero-icon` in the frontend home page to fix potential CLS.
- [GEO] JSON-LD structured data is properly formatted and sanitized.
- [GEO] Fixed nested `<main>` elements in App.tsx by replacing the wrapper with a `<div>` and updated `llms.txt` files with detailed endpoint descriptions for better AI discoverability.
- Added meta description to index.html
- Updated API Endpoints in llms.txt
- Replaced width and height with size for lucide-react components
