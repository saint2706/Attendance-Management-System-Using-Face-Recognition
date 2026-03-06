# Buddha's GEO & SEO Improvements Scroll 🧘

## [GEO] [SEO] AI Readability and Crawlability
- Added `/robots.txt` endpoint exposed directly from `attendance_system_facial_recognition/urls.py` instructing bots with standard directives.
- Added `/llms.txt` endpoint with a markdown representation of the site architecture to help direct AI agents like ChatGPT, Perplexity, and others when scanning the site.
- Injected JSON-LD Schema (`WebSite`) and OpenGraph (`og:title`, `og:description`, `og:type`) into `frontend/index.html` to improve structured data representation.

## [SEO] Semantic HTML
- Modified `frontend/src/pages/Home.tsx` to ensure `h2` headings are used logically instead of skipping from `h1` to `h3`. The "Your Privacy Matters" text is now an `h2` heading.
