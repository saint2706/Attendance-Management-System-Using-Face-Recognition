# Picasso Learnings
- When using ARIA roles like `alert`, adding `aria-live="assertive"` explicitly ensures that screen readers announce the message as soon as it appears on the screen (e.g., in a dynamic React component).
- Adding `focus-visible` states and `aria-label` properties on interactive anchor links used as block elements (`.action-card`) drastically improves semantic meaning and keyboard navigation tracking.
- To satisfy the WCAG 2.5.3 'Label in Name' criterion, any `aria-label` added to an element containing visible text must strictly include that visible text (e.g. `aria-label="Setup Wizard - start"` for a button displaying the text 'Setup Wizard').
- Implementing explicit `:focus-visible` states globally on `.btn`, `.nav-link`, and `.theme-toggle` elements greatly enhances keyboard accessibility by providing standard clear visual feedback.
- Added meaningful `alt` text (`alt="Smart Attendance Logo"`) and removed `aria-hidden="true"` on the `.hero-image` in `Home.tsx` to ensure screen readers read it.
- Removed redundant `aria-label` attributes from `Link` elements with `.action-card` class in `Dashboard.tsx` to allow screen readers to naturally read the text inside them instead of hiding child elements.
- Updated `aria-label` on the `.capture-button` in `MarkAttendance.tsx` to include the dynamic visible text inside it in all states (e.g., 'Processing...', 'Capturing in 3...', 'Capture & Recognize') to satisfy WCAG 2.5.3 (Label in Name).
