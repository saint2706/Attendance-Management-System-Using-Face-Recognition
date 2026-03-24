# AIML Optimization Log

## Issue: Optimization of data serialization specifically removing pickle due to security risk.
- **Action**: Modified `recognition/views.py` and `recognition/views_legacy.py` to use `json.loads` and `json.dumps` for dataset state serialization instead of the insecure `pickle.loads` and `pickle.dumps`. Also coerced lists to tuples appropriately when loading dataset_state from JSON.
- **Action**: Improved inference latency globally by implementing `get_loaded_model()` in both `recognition/views.py` and `recognition/views_legacy.py`. This caches the trained model to memory instead of reading and decrypting the `svc.sav` pickle file from disk on every `mark_attendance_view` API request.
- **Metrics/Impact**: Increased speed of API requests (Inference Latency) by caching the Scikit-Learn model to memory, reducing expensive IO and decryption. Removed potential RCE vulnerabilities by replacing `pickle` caching with safer `json` serialization.
