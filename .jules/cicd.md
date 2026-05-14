# CI/CD Improvements

## Yaml syntax
* Wrapped `"on":` in double quotes in `.github/workflows/close-all-prs.yml` to satisfy `yamllint` checking rules.
  * Before: 1 warning/error in `yamllint`.
  * After: 0 warnings/errors in `yamllint`.

## Dependabot
* Configured `github-actions` package-ecosystem tracking in `.github/dependabot.yml` to track updates to pinned SHAs for GitHub Actions.
  * Before: GitHub actions updates not tracked.
  * After: GitHub actions updates tracked securely.
