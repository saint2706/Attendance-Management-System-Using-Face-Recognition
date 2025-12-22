# Test Documentation

Comprehensive documentation for the Attendance Management System test suite.

## Test Structure

```text
tests/
├── conftest.py              # Shared fixtures, TensorFlow/DeepFace mocking
├── evaluation/              # Model evaluation pipeline tests
│   ├── test_face_recognition_eval.py
│   └── test_fairness.py
├── recognition/             # Core face recognition tests
│   ├── test_edge_cases.py   # Edge case coverage
│   ├── test_faiss_index.py  # FAISS vector database tests
│   ├── test_multi_face.py   # Multi-face detection tests
│   ├── test_performance.py  # Load and benchmark tests
│   ├── test_liveness.py     # Liveness detection tests
│   └── ...                  # Additional test modules
├── settings/                # Django settings tests
├── ui/                      # Playwright UI tests
└── users/                   # User model tests
```

## Running Tests

### Quick Test Run (Default)

```bash
# Fast tests only (excludes slow, ui, e2e)
pytest
```

### Full Test Suite

```bash
# All tests including slow and integration
pytest -m "" --ignore=venv
```

### Specific Categories

```bash
# Edge case tests
pytest tests/recognition/test_edge_cases.py -v

# Performance benchmarks (slow)
pytest tests/recognition/test_performance.py -v -m slow

# UI tests (requires Playwright)
pytest -m ui --headed

# Integration tests
pytest -m integration

# Django database tests
pytest -m django_db
```

### Coverage Report

```bash
pytest --cov=recognition --cov-report=html
```

---

## Pytest Markers

Markers defined in `pytest.ini`:

| Marker | Description | Usage |
|--------|-------------|-------|
| `slow` | Long-running tests (>1s) | Model loading, heavy file I/O, benchmarks |
| `ui` | UI/browser tests | Playwright-based tests |
| `e2e` | End-to-end tests | Full stack with live server |
| `integration` | Cross-app boundary tests | Celery/Redis integration |
| `django_db` | Database access tests | Tests needing database |
| `accessibility` | Accessibility features | ARIA labels, keyboard navigation |
| `mobile` | Mobile responsiveness | Viewport, touch events |
| `theme` | Theme/dark mode tests | CSS variables, toggle |
| `attendance_flows` | Core workflows | Check-in/check-out flows |

---

## Edge Case Test Coverage

Comprehensive edge cases in `test_edge_cases.py`:

### Multiple Faces

| Test | Description |
|------|-------------|
| All faces too small | All detected faces filtered out |
| Mixed size faces | Partial filtering behavior |
| Exceeds max limit | Only process up to limit |
| Empty representations | Error handling |
| No faces detected | Multi-mode error response |

### Empty Frames

| Test | Description |
|------|-------------|
| Black frame depth | All-zero frame handling |
| White frame depth | Uniform intensity handling |
| Empty array depth | Zero-size input |
| Empty frame hash | Hash computation edge case |
| Single frame consistency | Insufficient frames |
| Empty frames liveness | Liveness verification failure |

### Invalid Images

| Test | Description |
|------|-------------|
| Wrong dtype | Float32 instead of uint8 |
| Grayscale frame | 2D array handling |
| NaN embedding | Invalid embedding values |
| Zero vector cosine | Zero-magnitude vectors |
| Inf in embedding | Infinite values |
| Negative distance | Invalid threshold check |

### FAISS Index

| Test | Description |
|------|-------------|
| k=0 search | Empty result set |
| k > index size | Return available items |
| Duplicate labels | Same user multiple embeddings |
| Unnormalized embeddings | Large magnitude vectors |
| High dimensionality | 512-dim embeddings (realistic) |

---

## Performance Thresholds

Expected thresholds defined in `test_performance.py`:

### FAISS Index Operations

| Operation | Threshold | Description |
|-----------|-----------|-------------|
| Build 100 embeddings | <10ms | Small dataset |
| Build 1,000 embeddings | <100ms | Medium dataset |
| Build 10,000 embeddings | <2,000ms | Large dataset |
| Single search | <1ms | k=1 nearest neighbor |
| k=10 search | <5ms | Top-10 neighbors |
| 100 concurrent searches | <50ms | Thread pool execution |

### Load Testing

| Test | Metric | Expected |
|------|--------|----------|
| 1,000 sequential searches | Average latency | <1ms/request |
| 200 concurrent searches | Errors | 0 |
| 5 build/search/clear cycles | Memory stability | No leaks |

### Scaling Behavior

Search latency should scale **sublinearly** with index size. For a 10x data increase, latency increase should be less than 5x.

---

## Evaluation Pipeline Thresholds

Thresholds used in model evaluation (`src/evaluation/face_recognition_eval.py`):

### Recognition Thresholds

| Threshold | Default | Env Variable | Description |
|-----------|---------|--------------|-------------|
| Distance threshold | 0.4 | `RECOGNITION_DISTANCE_THRESHOLD` | Cosine distance for match acceptance |
| Model load alert | 4.0s | `RECOGNITION_MODEL_LOAD_ALERT_SECONDS` | Maximum model load time |
| Warmup alert | 3.0s | `RECOGNITION_WARMUP_ALERT_SECONDS` | First inference threshold |
| Loop alert | 1.5s | `RECOGNITION_LOOP_ALERT_SECONDS` | Per-iteration bound |

### Liveness Thresholds

| Threshold | Default | Env Variable | Description |
|-----------|---------|--------------|-------------|
| CNN anti-spoof | 0.75 | `RECOGNITION_CNN_ANTISPOOF_THRESHOLD` | CNN confidence for real face |
| Depth variance | 0.1 | `RECOGNITION_DEPTH_VARIANCE_THRESHOLD` | Depth cue analysis |
| Motion threshold | Varies | `RECOGNITION_LIVENESS_MOTION_THRESHOLD` | Optical flow detection |
| Min frames | 5 | `RECOGNITION_FRAME_CONSISTENCY_MIN_FRAMES` | Minimum for consistency |

### Multi-Face Thresholds

| Threshold | Default | Env Variable | Description |
|-----------|---------|--------------|-------------|
| Min face size | 50px | `RECOGNITION_MULTI_FACE_MIN_SIZE` | Minimum face dimension |
| Max faces/frame | 5 | `RECOGNITION_MAX_FACES_PER_FRAME` | Processing limit |

---

## Expected Metrics

Target values for evaluation reports:

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | >95% | Overall classification accuracy |
| FAR | <1% | False Accept Rate (impostors accepted) |
| FRR | <5% | False Reject Rate (genuine users rejected) |
| Genuine acceptance | 100% | Liveness: real faces pass |
| Spoof rejection | >75% | Liveness: attacks blocked |

---

## Running Evaluation Commands

```bash
# Prepare test splits
python manage.py prepare_splits --seed 42

# Run full evaluation
python manage.py eval --split-csv reports/splits.csv

# Liveness evaluation
python manage.py evaluate_liveness --samples-root /path/to/samples

# Fairness audit
python manage.py fairness_audit --reports-dir reports/fairness

# Hardware profiling
python manage.py profile_hardware --iterations 10
```

---

## Test Dependencies

Required packages for testing:

```text
pytest>=7.0
pytest-django
pytest-cov
pytest-playwright (for UI tests)
numpy
```

Optional for full test suite:

```text
faiss-cpu
celery (for integration tests)
redis (for integration tests)
```
