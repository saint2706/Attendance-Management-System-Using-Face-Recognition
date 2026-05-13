# CI/CD Pipeline Improvements

## 2025-05-13: GitHub Actions and Dependabot Optimizations

- **GitHub Actions YAML Syntax Fix:**
  - *Before:* The `.github/workflows/close-all-prs.yml` file used `on:` for workflow triggers, which could cause parsing ambiguities and fail yamllint checks.
  - *After:* Wrapped the `on` keyword in quotes (`"on":`) to ensure compliant YAML syntax.

- **Dependabot Configuration Update:**
  - *Before:* `.github/dependabot.yml` was tracking pip, npm, and docker package ecosystems, but missing action dependencies.
  - *After:* Added `github-actions` package-ecosystem tracking with a weekly interval to automate updates for pinned SHA versions of GitHub Actions.

These changes ensure long-term maintainability and consistent security for our CI/CD workflows.