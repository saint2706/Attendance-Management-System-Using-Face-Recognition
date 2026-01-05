# Palette's Journal

## 2024-05-22 - Accessibility in Dynamic UI

**Learning:** When using conditional rendering for major UI states (like loading vs. content), standard focus management is often insufficient.
**Action:** Ensure that when a major state change occurs, focus is programmatically managed to a logical starting point in the new state.

## 2024-05-23 - Focus Management in Kiosk Mode

**Learning:** In kiosk-style interfaces where the entire view context changes significantly (e.g., Camera -> Result), losing keyboard focus is a critical accessibility failure. Screen reader users can get lost when the element they triggered an action from is removed from the DOM.
**Action:** When swapping major UI states, explicitly manage focus by moving it to the new container's heading or primary element using `useRef`, `useEffect`, and `tabIndex="-1"`. This provides immediate context to assistive technologies.

## 2024-05-24 - Visualizing Keyboard Shortcuts with `<Kbd>`

**Learning:**
Users often overlook keyboard shortcuts when they are presented as plain text. Visualizing them as key-like elements (using the `<kbd>` tag or a styled component) significantly improves discoverability and makes the interface feel more "pro" and accessible.

**Action:**
When adding keyboard shortcuts to an interface, always pair them with a visual indicator using the `<Kbd>` component pattern. This not only documents the shortcut but invites the user to use it.
