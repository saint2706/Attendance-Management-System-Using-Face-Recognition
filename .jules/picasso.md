# Picasso: UX Learnings & Progress

## [Date]
* **Accessibility improvements**: Implemented the "Form Label with Required Indicator" pattern on the `Login.tsx` page. Added a visual `*` (with `aria-hidden="true"` and a new utility class `.text-danger` mapped to the existing `--color-danger` token) and `aria-required="true"` to input elements to improve screen reader compatibility and visual cues for mandatory fields.
* **Dependencies**: Encountered a minor React version mismatch (`react` vs `react-dom` versions not aligning perfectly in a Vite project) when running local dev server; resolved by reinstalling the specific matching versions. Keep an eye on dependency resolution when updating packages to ensure `react` and `react-dom` are strictly aligned.
