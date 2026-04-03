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
- [GEO] Added `Allow: /llms.txt` to `robots.txt` to make AI manifest discoverable
- [SEO] Updated `Home.tsx`, `Dashboard.tsx`, and `Login.tsx` to use semantic `<main>` elements instead of generic `<div>` wrappers for better accessibility and SEO
