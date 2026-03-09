# Picasso Learnings

- Keyboard accessibility focus states should be implemented using the `:focus-visible` pseudo-class (e.g., `outline: 2px solid var(--color-border-focus); outline-offset: 2px;`) to ensure clear visibility for keyboard users while avoiding lingering outlines on mouse clicks.
- When using Playwright to visually verify local frontend changes, start the Vite development server (`cd frontend && pnpm run dev`) rather than a generic python `http.server` on the build directory to properly handle SPA routing and assets.
