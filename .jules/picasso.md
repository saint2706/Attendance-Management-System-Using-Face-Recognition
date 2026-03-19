# Picasso Learnings
- When using ARIA roles like `alert`, adding `aria-live="assertive"` explicitly ensures that screen readers announce the message as soon as it appears on the screen (e.g., in a dynamic React component).
- Adding `focus-visible` states and `aria-label` properties on interactive anchor links used as block elements (`.action-card`) drastically improves semantic meaning and keyboard navigation tracking.
- To satisfy the WCAG 2.5.3 'Label in Name' criterion, any `aria-label` added to an element containing visible text must strictly include that visible text (e.g. `aria-label="Setup Wizard - start"` for a button displaying the text 'Setup Wizard').
- Implementing explicit `:focus-visible` states globally on `.btn`, `.nav-link`, and `.theme-toggle` elements greatly enhances keyboard accessibility by providing standard clear visual feedback.
- Use `aria-live="polite"` for loading states and `title` attributes for tooltips on icon-only buttons to enhance screen reader support and mouse user feedback.

### Accessibility & UX Improvements
- **Label in Name Criterion (WCAG 2.5.3)**: Ensured that all `aria-label` attributes on elements containing visible text (like the Action Cards in the Dashboard and the Logout button) strictly include that visible text. This prevents cognitive mismatch for speech input users.
- **Checked Non-critical Loading States**: Verified that non-critical loading states in `MarkAttendance.tsx` correctly use `aria-live="polite"`.
