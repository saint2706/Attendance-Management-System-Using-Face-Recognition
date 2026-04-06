# Picasso UX Learnings

## Date: 2024-03-21
- Added `title` attributes as tooltips to navigation cards and buttons in the admin dashboard. While elements with well-structured visible text do not strictly require `aria-label` (and removing redundant ones helps avoid breaking the WCAG 2.5.3 'Label in Name' criterion), providing tooltips via the `title` attribute improves usability for mouse and keyboard users by giving additional context on hover or focus without relying solely on screen readers.
- Learned to remove redundant `aria-label` attributes from elements that already contain well-structured, visible text to satisfy WCAG 2.5.3 'Label in Name' criterion.
Added aria-live for loading states, improved aria-labelledby for sections, and added tooltips to icon buttons.

### Login Accessibility
- Improved focus management on error state to immediately redirect focus to error text utilizing `useRef`, `useEffect` and `tabIndex={-1}`.
- Kept `aria-hidden="true"` for purely visual, adjacent text describing icons.
- Updated boolean logic for aria invalid checks to `Boolean(error)`.

## Date: $(date +%Y-%m-%d)
- Added `aria-expanded` and `aria-controls` to the mobile menu toggle button in `Navbar.tsx` for better screen reader and keyboard support.
- Implemented a simulated loading skeleton state for the Quick Stats section in `Dashboard.tsx` using the `animate-pulse` utility to provide better feedback to users while data loads.
- Noted that `aria-live` should not be used for attribute mutations like `aria-label` changing, and redundant `aria-label`s on elements that already contain visible text should be avoided.
## UI/UX Improvements

- Replaced broken raster hero image (`/icons/icon-512.png`) with an accessible SVG `ScanFace` icon from Lucide React to gracefully handle missing static assets and match application aesthetics.
- Replaced hardcoded Tailwind-like classes (`bg-gray-200`) in Dashboard statistics skeletons with a new reusable, theme-aware CSS `.skeleton` class so loading skeletons correctly adjust visibility during dark mode.
## UX Improvements\n- Added `Escape` key accessibility for closing the mobile menu in the Navbar.
\n- Confirmed `Space` and `Escape` keyboard accessibility is working in MarkAttendance.tsx for capture and reset actions.
# Picasso UX Improvements

## Dashboard Empty State
- Added an empty state to `frontend/src/pages/Dashboard.tsx` for when there are no employees registered.
- Uses the `Inbox` icon with existing design tokens like `.text-center`, `.py-12`, `.text-muted`.
- Helps provide clear next steps for the admin to register an employee.
- Improved accessibility of the .skip-link element by changing its state styling from `:focus` to `:focus-visible` to better support keyboard navigation
- Improved accessibility on icon-only toggle buttons in Login and Navbar components by using static `aria-label`s alongside stateful attributes like `aria-pressed` rather than dynamically mutating the `aria-label`. Screen readers handle this pattern much better.

## Navigation & Dashboard Error State UX Polish
- Converted `Link` components in main navigation to `NavLink` from `react-router-dom` to automatically apply `aria-current="page"` and an `active` class to the current route. Styled the active class for clear visual feedback.
- Separated the API error state from the empty state in the Dashboard. Previously, API failures defaulted to 0 employees, triggering the "No employees yet" state incorrectly. Now, failures display a distinct error card with an actionable "Try Again" button, improving progressive disclosure and recovery options.

- Added explicit `title` tooltips to the main call-to-action buttons in `Home.tsx` ("Mark Time-In", "Mark Time-Out", "Dashboard Login") to provide additional context.
- Added `title` tooltips to the "Return Home", "Mark Another", "Try Again", and "Retry" action buttons in the `MarkAttendance.tsx` page to improve intent clarity.
- Added descriptive `title` tooltips to the "Try Again" error state button and the "Register Employee" empty state link in the `Dashboard.tsx` view for better usability.
