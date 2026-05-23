# CI/CD Pipeline Optimizations

## Concurrency and Resource Optimization
- **Added `concurrency` blocks to multiple GitHub Actions workflows:**
  - `codeql.yml`
  - `frontend-ci.yml`
  - `lighthouse.yml`
- **Configuration Details:**
  - `group: ${{ github.workflow }}-${{ github.ref }}`: Groups runs by workflow name and branch reference.
  - `cancel-in-progress: true`: Automatically cancels any currently running jobs for the same branch when a new commit is pushed.
- **Impact:**
  - Saves significant runner minutes and queue time by preventing redundant pipeline executions.
  - Speeds up the feedback loop for developers by ensuring only the latest commit is validated.
  - Avoids race conditions on shared deploy/test environments.
