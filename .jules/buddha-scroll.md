# Buddha Progress

## Optimizations

### GEO (Intelligence)
- **Synchronized `llms.txt`**: Added comprehensive routing structure to both `llms.txt.content` and `recognition/static/llms.txt` to help AI agents understand the application structure better.

### SEO (Visibility)
- **Enhanced Metadata**: Added OpenGraph and Twitter social meta tags (e.g., `twitter:card`, `twitter:title`, `og:image`, `twitter:description`, `og:image:width`, `og:image:height`, `og:image:alt`) to both `frontend/index.html` and `recognition/templates/recognition/base.html` to align them visually and provide better rich previews.
- **Fixed Semantic HTML**: In `frontend/src/pages/Home.tsx`, updated the privacy section heading from `<h3>` to `<h2>` to correct the semantic outline hierarchy. Also updated `frontend/src/pages/Home.css` so that `.privacy-notice h2` is styled appropriately.

- **Added Missing `robots.txt` & `llms.txt`**: Added `robots.txt` and `llms.txt` files directly into `frontend/public` directory to ensure Vite builds also serve them alongside the Django-served files. `[GEO]` `[SEO]`
- **Improved Semantic JSON-LD**: Added a new `WebSite` JSON-LD schema (including the URL and Organization details) in both `frontend/index.html` and `recognition/templates/recognition/base.html` for richer intent search understanding. `[GEO]` `[SEO]`
- **Preload Fonts**: Implemented `<link rel="preload">` for the critical Google Fonts `Inter` stylesheet to mitigate LCP blocking in the frontend application `frontend/index.html`. `[PERF]`
