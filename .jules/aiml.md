# AIML Optimization Log

## Issue: Optimization of data serialization specifically removing pickle due to security risk.
- **Action**: Modified `recognition/views.py` and `recognition/views_legacy.py` to use `json.loads` and `json.dumps` for dataset state serialization instead of the insecure `pickle.loads` and `pickle.dumps`. Also coerced lists to tuples appropriately when loading dataset_state from JSON.
- **Action**: Improved inference latency globally by implementing `get_loaded_model()` in both `recognition/views.py` and `recognition/views_legacy.py`. This caches the trained model to memory instead of reading and decrypting the `svc.sav` pickle file from disk on every `mark_attendance_view` API request.
- **Metrics/Impact**: Increased speed of API requests (Inference Latency) by caching the Scikit-Learn model to memory, reducing expensive IO and decryption. Removed potential RCE vulnerabilities by replacing `pickle` caching with safer `json` serialization.
- **Issue**: Improved model size and inference speed by applying INT8 quantization to the anti-spoofing CNN model.
- **Action**: Updated `AntiSpoofCNN._create_model()` in `recognition/anti_spoof_cnn.py` to use `tf.lite.OpsSet.TFLITE_BUILTINS_INT8`. Implemented a `representative_dataset` generator that calibrates the quantization ranges using a sample of 50 real face images from the local `faces` directory. Added a robust fallback to float16 quantization in case the real images are not available or the INT8 conversion process fails.
- **Metrics/Impact**: Significantly reduces the TFLite model size and accelerates inference on supported edge devices, while maintaining a robust fallback mechanism to avoid deployment crashes.
