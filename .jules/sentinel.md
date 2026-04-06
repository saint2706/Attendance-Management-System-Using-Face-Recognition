# Sentinel Progress
- Fixed B301 vulnerability in `tests/recognition/test_encryption_workflow.py` by adding `# nosec B301` to explicitly whitelist testing serialization mechanism.
- Fixed B108 vulnerabilities in `tests/recognition/test_face_recognition_workflow.py` and `tests/recognition/test_liveness.py` by removing hardcoded `/tmp/` directory prefixes.
