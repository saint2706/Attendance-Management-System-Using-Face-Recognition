## 2024-05-24 - Countdown Timer & Keyboard Accessibility
**Learning:** Adding a countdown timer (3-2-1) before camera capture significantly improves UX for "photo booth" style interactions. It gives users time to compose themselves, reducing bad photos.
**Action:** When implementing camera capture features, always consider a countdown.
**Learning:** For kiosk-style applications, keyboard shortcuts (Space to capture, Escape to reset) are critical for accessibility and ease of use, especially when a mouse might not be present.
**Action:** Always map primary actions to keyboard shortcuts in kiosk interfaces and provide visible hints (e.g., "Press Space to capture").
**Learning:** Using `key={value}` on an element is a simple way to force CSS animations to restart when the content changes (like a countdown number).
**Action:** Use `key` prop for animation resets instead of complex class toggling logic.
