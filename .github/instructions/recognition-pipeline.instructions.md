---
applyTo: "recognition/**/*.py"
---

# Face Recognition Pipeline Guidelines

When modifying the face recognition pipeline under `recognition/`, follow these critical guidelines:

## Core Principles

1. **Keep embeddings reproducible** - Ensure deterministic output for the same input
2. **Never degrade accuracy silently** - Document any changes that could affect recognition accuracy
3. **Don't block Django request thread** - Use Celery for heavy processing tasks
4. **Maintain encryption** - Always use project's Fernet encryption helpers for sensitive data

## Architecture

- **DeepFace** with Facenet model for embedding generation
- **SSD detector** for face detection
- **Cosine similarity** for matching (threshold: 0.4 by default)
- **Two-stage liveness detection** (motion gate + anti-spoofing)

## Embedding Generation

```python
# Always use the project's pipeline for consistency
from recognition.pipeline import get_embedding

# Embeddings should be deterministic
embedding = get_embedding(face_image)
```

## Matching Logic

- Use cosine distance: `d(A, B) = 1 - sim(A, B)`
- Match accepted when distance ≤ `RECOGNITION_DISTANCE_THRESHOLD` (default: 0.4)
- Document any threshold changes with justification

## Async Processing

```python
# Heavy tasks should use Celery
from recognition.tasks import process_face_async

# Don't do this in views:
# embedding = compute_embedding(image)  # Blocks request

# Do this instead:
task = process_face_async.delay(image_path)
```

## Encryption

```python
from recognition.encryption import encrypt_face_data, decrypt_face_data

# Always encrypt before storage
encrypted = encrypt_face_data(embedding_bytes)

# Decrypt when needed
embedding = decrypt_face_data(encrypted)
```

## Testing Requirements

1. **Unit tests** - Test individual functions with mocked DeepFace
2. **Integration tests** - Test full pipeline with synthetic data
3. **Threshold validation** - Include tests that verify threshold behavior
4. **Liveness tests** - Validate both stages of liveness detection

## Environment Variables

- `FACE_DATA_ENCRYPTION_KEY` - Fernet key for face data encryption
- `RECOGNITION_DISTANCE_THRESHOLD` - Matching threshold (default: 0.4)
- `RECOGNITION_HEADLESS` - Disable GUI for CI environments

## Changes Requiring Documentation

- Threshold adjustments
- Model changes
- New preprocessing steps
- Liveness detection modifications
- Embedding format changes

Update `docs/liveness_evaluation.md` and `DEVELOPER_GUIDE.md` when making significant changes.
