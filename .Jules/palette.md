## 2025-12-19 - [Reusable Form Loading State]
**Learning:** Standard Django forms often lack immediate feedback on submit, leading to uncertainty. Enhancing 'FormEnhancer' provides a systemic fix for all validatable forms.
**Action:** Always check if a centralized form handler exists before patching individual templates.

## 2025-12-19 - [Icon Consistency]
**Learning:** Inconsistent icon usage (e.g., exit icon for entry) causes significant user confusion. Standardizing icons across the application (matching `base.html` and standard metaphors) improves intuitiveness.
**Action:** Audit icon usage when touching templates to ensure semantic consistency.

## 2025-12-21 - [Camera Initialization Feedback]
**Learning:** Browser camera initialization (`getUserMedia`) is asynchronous and can be slow. Without a visual loader, users see a blank screen, leading to confusion about app state.
**Action:** Implement an explicit loading state that persists until the video stream is active and playing.

## 2025-12-21 - [Password Visibility & Error Accessibility]
**Learning:** Hidden password fields often cause user errors, and error messages without ARIA roles are missed by screen readers.
**Action:** Always include a password visibility toggle and ensure error containers use `role="alert"` for immediate feedback.
