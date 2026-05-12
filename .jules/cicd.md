# CI/CD Pipeline Improvements

## Fixed yamllint validation
Fixed a yamllint validation error in `.github/workflows/close-all-prs.yml` by quoting the `on` key. This resolves yamllint errors during pipeline configuration syntax checks.

## Enabled Dependabot for GitHub Actions
Added `github-actions` package-ecosystem configuration to `.github/dependabot.yml` to automatically track and update pinned SHA versions for GitHub Actions dependencies, ensuring that we receive automated security updates and version bumps for actions used in our workflows.

## Upgraded Node version
Upgraded node-version from v20 to v22 in github actions and Dockerfile base images (Frontend stage) since Node v20 is deprecated in github actions and throws a dynamic import callback error during `pnpm install` in some builds.
