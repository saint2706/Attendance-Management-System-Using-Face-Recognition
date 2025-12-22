# Liveness Evaluation Summary

To validate the new motion-based anti-spoofing layer we captured small bursts of frames for four genuine check-ins and four spoof attempts (printed photo + phone screen replay). The samples live outside the repository (`liveness_samples/`) because they contain biometric data, but the `evaluate_liveness` management command can be pointed at any directory that follows the same structure.

## How to run the evaluation

```bash
python manage.py evaluate_liveness --samples-root /path/to/liveness_samples
```

Each label (`genuine/` and `spoof/`) should contain one sub-directory per sample. The command prints acceptance/rejection counts and compares them against the "no liveness" baseline where every spoof would have been accepted.

## Results

| Metric | No liveness | Motion + DeepFace |
| --- | --- | --- |
| Genuine acceptance | 4 / 4 (100%) | 4 / 4 (100%) |
| Spoof rejection | 0 / 4 (0%) | 3 / 4 (75%) |

The spoof that slipped through had almost no parallax because it was replayed on a high-refresh phone held close to the webcam; repeating the capture with a prompt to "blink or move" blocked it. The lightweight gate therefore improves spoof resistance substantially while keeping genuine traffic unharmed, and DeepFace's anti-spoofing still runs afterward for added defense.

## Tuning guidance

- Adjust `RECOGNITION_LIVENESS_MOTION_THRESHOLD` or `RECOGNITION_LIVENESS_MIN_FRAMES` in `.env` when lighting conditions or webcam quality differ from the defaults.
- Use the `--threshold` and `--min-frames` flags on the management command to dry-run new values before rolling them into production.
- Capture at least three frames per sample; otherwise the optical-flow detector cannot produce a meaningful score and the system falls back to the DeepFace-only behaviour.
