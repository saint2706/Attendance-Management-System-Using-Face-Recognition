# Buddha Scroll - GEO/SEO Improvements
- [GEO] Updated llms.txt across static directories to include backend API/admin routes and SPA routes.
- [SEO] Validated Sitemap URL in robots.txt.
- [PERF] Verified Hero image optimization in Home.tsx.
- [SEO] Verified JSON-LD schema.
## SEO/GEO Improvements
- `[GEO] [SEO]`: Moved static JSON-LD from `frontend/index.html` to dynamic React component using `dangerouslySetInnerHTML` in `frontend/src/pages/Home.tsx`.
- `[GEO] [SEO]`: Implemented JSON-LD WebPage schema in Dashboard, Login, and MarkAttendance pages using `dangerouslySetInnerHTML`.
- `[PERF]`: Removed unused `icon-512.png` image preload tag from `frontend/index.html` to fix Lighthouse "Remove unused preloads" and improve LCP for actual critical resources.
- `[SEO]`: Added React 19 Document Metadata (`<title>` and `<meta name="description">`) to `Home`, `Dashboard`, `Login`, and `MarkAttendance` SPA components to enable dynamic SEO routing.
- [SEO] Added twitter:image to index.html
- [GEO] Enhanced llms.txt with descriptive site context
- [GEO] Updated frontend/public/robots.txt to explicitly allow /llms.txt for AI discoverability.
- [GEO] Improved semantic HTML structure by converting root `div` components to `<main>` tags in `Home.tsx`, `Dashboard.tsx`, `Login.tsx`, and `MarkAttendance.tsx` to enhance SEO and AI vector friendliness.
