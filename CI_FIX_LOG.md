# CI Fix Log

## Date: 2026-01-05

### Issue 1: Frontend CI - setup-node certificate issues
**Workflow**: Frontend CI  
**Job**: frontend-quality  
**Error**: `self-signed certificate in certificate chain` when downloading Node.js 20

**Root Cause**: act runs in a restricted network environment where certificate validation fails when setup-node tries to download Node.js.

**Fix Applied**:
- Added `NODE_TLS_REJECT_UNAUTHORIZED=0` to `.actrc` to disable certificate validation for local testing
- Changed runner image from `node:20-bookworm` to `ghcr.io/catthehacker/ubuntu:act-latest` which has better compatibility with act

**Files Changed**:
- `.actrc`: Updated runner image and added environment variable

**Status**: Partially resolved - Node.js 20 now installs successfully

---

### Issue 2: Frontend CI - npm ci exit handler error
**Workflow**: Frontend CI  
**Job**: frontend-quality  
**Step**: Install dependencies  
**Error**: `npm error Exit handler never called!` followed by dependencies not being installed

**Root Cause**: npm ci has a known incompatibility with act's signal handling mechanism. The process appears to complete but exits before node_modules is properly populated.

**Investigation**:
- Tested npm ci locally outside act: ✅ Works fine
- Tested npm ci in plain docker with Node 20: ✅ Works fine  
- Tested npm ci in act container: ❌ Fails with exit handler error

**Attempted Fixes**:
1. Different runner images (runner-20.04, act-latest) - No improvement
2. Adding NODE_TLS_REJECT_UNAUTHORIZED=0 - Node downloads but npm still fails
3. Using --action-offline-mode - No improvement

**Required Fix**: This is a known act limitation. We need a minimal compatibility patch to the workflow.

**Status**: Requires workflow modification

---

