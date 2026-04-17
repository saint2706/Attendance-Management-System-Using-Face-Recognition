1. **Fix `exceptions.py` to hide raw exception class names**:
   - In `recognition/api/exceptions.py`, modify `custom_exception_handler` to properly set the `title` to "Internal Server Error" for unhandled exceptions instead of leaking the raw exception class name (`exc.__class__.__name__`).
   - Specifically, if `response` was initially `None` (meaning it's an unhandled exception), use a flag `is_unhandled = True`. Later when assigning the `title`, if `is_unhandled` is true, use "Internal Server Error" instead of `exc.__class__.__name__`.
   - Update `tests/recognition/test_api_exceptions.py` to expect `"title": "Internal Server Error"` instead of `"title": "Valueerror"` for unhandled exceptions.

2. **Fix `recognition/views.py` and `recognition/views_legacy.py` for raw error leaks**:
   - Both of these files have a catch-all block that returns `str(exc)` in the `detail` field of the JsonResponse (e.g., `{"detail": str(exc)}`).
   - I will replace `str(exc)` with `"An unexpected error occurred."` or similar generic error message to prevent leaking raw database errors or stack traces to the client.
   - Wait, `ValueError` is also caught and its `str(exc)` is returned. Since `ValueError`s are raised intentionally in `_extract_image_bytes`, I should ensure they are not raw database errors. But to be safe with the "Never leak raw database errors/stack traces (`str(exc)`)" rule, I will check all instances of `str(exc)` being returned in a response and sanitize them. Let me check `views.py`.

3. **Pre-commit**:
   - Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.
