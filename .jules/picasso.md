# Picasso UX Learnings

## Date: 2024-03-21
- Added `title` attributes as tooltips to navigation cards and buttons in the admin dashboard. While elements with well-structured visible text do not strictly require `aria-label` (and removing redundant ones helps avoid breaking the WCAG 2.5.3 'Label in Name' criterion), providing tooltips via the `title` attribute improves usability for mouse and keyboard users by giving additional context on hover or focus without relying solely on screen readers.
- Learned to remove redundant `aria-label` attributes from elements that already contain well-structured, visible text to satisfy WCAG 2.5.3 'Label in Name' criterion.
Added aria-live for loading states, improved aria-labelledby for sections, and added tooltips to icon buttons.
