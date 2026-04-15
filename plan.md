1. **Update `Dockerfile` build arguments for dummy keys**:
   - Replace the `ZHVtbXktZW5jcnlwdGlvbi1rZXktZm9yLWJ1aWxkISE=` value with a valid base64-encoded Fernet key (e.g., `OHEbvKQxZVrLosdwJ7Oh7RN3za6aAfIWrD5yS5WPlQI=`) for both `DATA_ENCRYPTION_KEY` and `FACE_DATA_ENCRYPTION_KEY` in the `RUN collectstatic` block to prevent `binascii.Error: Incorrect padding` errors during build.
   - I will do this in both `Dockerfile` and `Dockerfile.gpu`.

2. **Pre-commit**:
   - Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.

3. **Submit**:
   - Submit the PR.
