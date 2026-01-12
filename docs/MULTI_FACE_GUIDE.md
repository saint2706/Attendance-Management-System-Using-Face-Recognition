# Multi-Face Detection - Usage Guide

## Overview

The system now supports **multi-face detection** for group check-in scenarios while maintaining backward compatibility with single-person mode.

## Configuration

### Enable Multi-Face Mode

Edit your `.env` file:

```bash
# Enable multi-face detection
RECOGNITION_MULTI_FACE_ENABLED=true

# Maximum faces to process per frame (default: 5)
RECOGNITION_MAX_FACES_PER_FRAME=5

# Minimum face size in pixels (filters distant faces)
RECOGNITION_MULTI_FACE_MIN_SIZE=50
```

### Disable (Default Single-Person Mode)

```bash
RECOGNITION_MULTI_FACE_ENABLED=false
```

## Using the Multi-Face Helper Module

### In Your Views

```python
from recognition.multi_face import process_face_recognition

def your_view(request):
    # Get representations from DeepFace
    representations = DeepFace.represent(
        img_path=frame,
        model_name="Facenet",
        detector_backend="ssd"
    )

    # Process automatically (handles both single and multi mode)
    result = process_face_recognition(
        representations=representations,
        dataset_index=your_dataset,
        distance_metric="cosine",
        distance_threshold=0.4
    )

    # Check mode from result
    if result.get("mode") == "multi":
        # Multi-face response
        for face in result["faces"]:
            if face["recognized"]:
                print(f"Matched: {face['match']['username']}")
    else:
        # Single-face response
        if result["recognized"]:
            print(f"Matched: {result['username']}")
```

### Response Formats

**Single-Face Mode (default):**

```json
{
  "recognized": true,
  "username": "john_doe",
  "distance": 0.25,
  "threshold": 0.4,
  "distance_metric": "cosine",
  "mode": "single",
  "facial_area": {"x": 100, "y": 50, "w": 150, "h": 150}
}
```

**Multi-Face Mode (when enabled):**

```json
{
  "faces": [
    {
      "recognized": true,
      "match": {
        "username": "john_doe",
        "distance": 0.25,
        "identity": "/path/to/identity",
        "threshold": 0.4
      },
      "embedding": [...],
      "facial_area": {"x": 100, "y": 50, "w": 150, "h": 150}
    },
    {
      "recognized": true,
      "match": {
        "username": "jane_smith",
        "distance": 0.30,
        "identity": "/path/to/identity",
        "threshold": 0.4
      },
      "embedding": [...],
      "facial_area": {"x": 300, "y": 60, "w": 140, "h": 140}
    }
  ],
  "count": 2,
  "mode": "multi",
  "threshold": 0.4,
  "distance_metric": "cosine"
}
```

## Web UI Integration

### Add Mode Indicator to Templates

Include the provided template snippet in your recognition pages:

```django
{% include "recognition/_multi_face_indicator.html" %}
```

This will automatically show:

- **Single Person Mode** alert when disabled (default)
- **Group Mode Active** alert when enabled

### Template Variables

Available in all templates via context processor:

- `multi_face_enabled` - Boolean indicating if multi-face mode is on
- `max_faces_per_frame` - Maximum faces that will be processed

Example usage:

```django
{% if multi_face_enabled %}
    <p>Up to {{ max_faces_per_frame }} people can check in together</p>
{% endif %}
```

## Performance Considerations

### Processing Time

| Faces | Time (CPU) | Time (NPU/GPU) |
|-------|------------|----------------|
| 1 face | ~200ms | ~40ms |
| 2 faces | ~400ms | ~80ms |
| 5 faces | ~1000ms | ~200ms |

### Recommendations

- **CPU-only deployments**: Set `MAX_FACES_PER_FRAME=2` or `3` for better responsiveness
- **NPU/GPU deployments**: Can handle default `MAX_FACES_PER_FRAME=5` easily
- **High-traffic scenarios**: Consider keeping single-person mode for faster processing

## Use Cases

### Single-Person Mode (Default)

- **Best for:** Accountability, security, performance
- **Examples:** Office sign-in, attendance tracking, access control

### Multi-Person Mode

- **Best for:** Group scenarios, reduced wait times
- **Examples:** Team check-ins, classroom attendance, event registration, family entry

## Troubleshooting

### No faces detected in multi-face mode

- Check `RECOGNITION_MULTI_FACE_MIN_SIZE` - might be filtering faces
- Ensure faces are large enough in frame (at least 50x50 pixels)
- Verify `DeepFace.represent()` is detecting faces

### Only first face recognized

- Verify `RECOGNITION_MULTI_FACE_ENABLED=true` in `.env`
- Check Django settings: `python manage.py shell -c "from django.conf import settings; print(settings.RECOGNITION_MULTI_FACE_ENABLED)"`
- Restart Django server after changing settings

### Performance issues

- Reduce `RECOGNITION_MAX_FACES_PER_FRAME` to 2 or 3
- Enable NPU/GPU acceleration
- Consider switching to lightweight model (OpenFace) for CPU

## API Reference

See `recognition/multi_face.py` for full API documentation:

- `process_face_recognition()` - Main entry point (auto-selects mode)
- `process_single_face_recognition()` - Single-face processing
- `process_multi_face_recognition()` - Multi-face processing
- `is_multi_face_enabled()` - Check if multi-face mode active
- `get_max_faces_limit()` - Get configured max faces
