# 🧘 Buddha Scroll - SEO/GEO Improvements

## Improvements
- `[GEO]` Added `llms.txt` file outlining site architecture for AI agents
- `[GEO]` Added `robots.txt` file to allow general crawling and block specific routes (`/admin/`, `/django-admin/`, `/api/`)
- `[SEO]` Registered `llms.txt` and `robots.txt` dynamic views in URL configurations (`attendance_system_facial_recognition/urls.py`)
- `[SEO]` Added OpenGraph (`og:title`, `og:description`, `og:type`) meta tags to `frontend/index.html` and `recognition/templates/recognition/base.html`
- `[GEO]` Added `application/ld+json` schema for `SoftwareApplication` to `frontend/index.html` and `recognition/templates/recognition/base.html`
- `[GEO]` Updated `llms.txt.content` and `recognition/static/llms.txt` to explicitly document all primary user paths for AI agent discoverability
