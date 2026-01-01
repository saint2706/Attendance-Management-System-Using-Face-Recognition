# Architecture Overview

This document provides a high-level overview of the application's architecture, explaining how the different components work together to provide a seamless and automated way to track employee attendance.

## 1. Core Components

The application is built on the following core components:

- **Django**: The web framework that provides the application's structure, including the models, views, and templates.
- **DeepFace**: A lightweight face recognition and facial attribute analysis (age, gender, emotion, and race) library for Python. It is used to power the face recognition functionality of the application.
- **PostgreSQL (via `DATABASE_URL`)**: Primary relational database for production deployments. The application falls back to SQLite when no external database URL is provided, which is useful for quick prototyping.
- **Bootstrap**: The front-end framework that is used to create the application's responsive web interface.

## 2. Architecture Diagram

The following diagram provides a high-level overview of the application's architecture:

```text
+-----------------+      +-----------------+      +-----------------+
|   Web Browser   | <--> |     Django      | <--> |   PostgreSQL    |
+-----------------+      +-----------------+      +-----------------+
                           |
                           v
+-----------------+      +-----------------+
|   DeepFace      | <--> | Face Reco Data  |
+-----------------+      +-----------------+
```

## 3. Workflow

The application's workflow is as follows:

1. **User Registration**: An admin registers a new employee by creating a new user account for them.
2. **Add Photos**: The admin adds photos for the employee by using the "Add Photos" feature. The application uses the webcam to capture a set of photos for face recognition and stores them in the `face_recognition_data` directory.
3. **Mark Attendance**: The employee marks their attendance by using the "Mark Time-In" or "Mark Time-Out" feature. The application uses the webcam to capture a frame and sends it to DeepFace for face recognition.
4. **Face Recognition**: DeepFace compares the captured frame to the stored face images and returns the name of the recognized employee.
5. **Store Attendance**: The application stores the attendance record in the database.
6. **View Attendance**: The admin can view the attendance records for all employees, and the employee can view their own attendance records.

## 4. Evaluation and Validation Architecture

The system includes a comprehensive evaluation pipeline for measuring and improving performance.

### 4.1 Data Splitting Pipeline

```text
Face Recognition Data
        |
        v
+-------------------+
| Data Splits       |
| - Identity-level  |
| - Session-based   |
| - Stratified      |
+-------------------+
        |
        v
  +-----------+-----------+
  |           |           |
  v           v           v
Train (70%) Val (15%)  Test (15%)
```

**Key Features:**

- Identity-level splitting (all images of a person stay together)
- Session-based grouping (enrollment batches stay together)
- Stratified sampling (maintains class balance)
- Fixed random seed for reproducibility

### 4.2 Metrics Calculation Pipeline

```text
Test Data
    |
    v
+-----------------+
| Face Detection  |
| & Embedding     |
+-----------------+
    |
    v
+-----------------+
| Similarity      |
| Computation     |
+-----------------+
    |
    v
+-----------------+
| Metrics Engine  |
| - ROC AUC       |
| - EER           |
| - FAR/TPR       |
| - Brier Score   |
+-----------------+
    |
    v
+-----------------+
| Bootstrap CI    |
| (1000 samples)  |
+-----------------+
    |
    v
+-----------------+
| Reports &       |
| Visualizations  |
+-----------------+
```

**Metrics Computed:**

- Verification metrics (ROC AUC, EER, FAR@TPR, TPR@FAR)
- Calibration metrics (Brier Score, calibration curves)
- Classification metrics (F1, Precision, Recall at optimal threshold)
- Confidence intervals via nonparametric bootstrap

### 4.3 Failure Analysis Architecture

```text
Predictions
    |
    v
+-------------------+
| Error Detection   |
| - False Accepts   |
| - False Rejects   |
+-------------------+
    |
    v
+-------------------+
| Metadata Analysis |
| - Lighting        |
| - Pose            |
| - Occlusion       |
+-------------------+
    |
    v
+-------------------+
| Subgroup Analysis |
| - By camera       |
| - By time         |
| - By conditions   |
+-------------------+
    |
    v
+-------------------+
| Recommendations   |
| - Add photos      |
| - Adjust threshold|
| - Re-enroll       |
+-------------------+
```

**Outputs:**

- Ranked failure cases
- Pattern identification
- Bias detection
- Actionable recommendations

### 4.4 Ablation Framework

```text
Component Configurations
    |
    v
+-------------------+
| Ablation Runner   |
+-------------------+
    |
    +---> Detector Variants (SSD, OpenCV, MTCNN)
    +---> Alignment (On/Off)
    +---> Distance Metric (Cosine, Euclidean, L2)
    +---> Rebalancing (On/Off)
    |
    v
+-------------------+
| Performance       |
| Comparison        |
+-------------------+
    |
    v
+-------------------+
| Statistical       |
| Significance      |
+-------------------+
    |
    v
+-------------------+
| Recommendations   |
+-------------------+
```

**Purpose:**

- Understand component contributions
- Identify optimal configurations
- Measure performance trade-offs
- Guide optimization decisions

## 5. Business Logic Architecture

### 5.1 Policy-Based Action Mapping

```text
Face Recognition
    |
    v
+-------------------+
| Cosine Distance   |
| d = 1 - sim(A, B) |
+-------------------+
    |
    v
+-------------------+
| Policy Engine     |
| (policy.yaml)     |
+-------------------+
    |
    +--> d ≤ 0.30 ---------> Confident Accept --> Mark Attendance Immediately
    |
    +--> 0.30 < d ≤ 0.45 --> Secondary Check --> Request PIN/OTP
    |
    +--> d > 0.45 --------> Reject -----------> Notify User, Suggest Re-enrollment
```

**Benefits:**

- Reduces false accepts through secondary verification while keeping the cosine-distance math transparent (`sim(A, B) = (A · B) / (||A|| ||B||)` and `d(A, B) = 1 − sim(A, B)`).
- Improves user experience for edge cases by treating distance bands consistently.
- Provides clear guidance to users.
- Configurable thresholds per deployment via `RECOGNITION_DISTANCE_THRESHOLD` and `configs/policy.yaml`.

### 5.2 CLI Prediction Tool

```text
Image File
    |
    v
+-------------------+
| predict_cli.py    |
+-------------------+
    |
    v
+-------------------+
| Face Detection    |
| & Recognition     |
+-------------------+
    |
    v
+-------------------+
| Policy Lookup     |
+-------------------+
    |
    v
+-------------------+
| Output:           |
| - Identity        |
| - Score           |
| - Action Band     |
| - Recommendation  |
+-------------------+
```

**Use Cases:**

- Testing before enrollment
- Debugging recognition issues
- Batch processing
- Integration testing

## 6. Data Flow Diagrams

### 6.1 Enrollment Flow

```text
Employee --> Register (Admin) --> Add Photos (Webcam) --> Store Images
                                                              |
                                                              v
                                      face_recognition_data/<username>/
                                                              |
                                                              v
                                              Ready for Recognition
```

### 6.2 Attendance Marking Flow

```text
Employee --> Click Time-In/Out --> Webcam Capture --> DeepFace Recognition
                                                              |
                                                              v
                                                        Similarity Score
                                                              |
                                                              v
                                                        Policy Engine
                                                              |
                                  +-----------------------+---+----------------------+
                                  |                       |                          |
                                  v                       v                          v
                         Confident Accept          Uncertain                      Reject
                                  |                       |                          |
                                  v                       v                          v
                          Mark Attendance      Request PIN/OTP                Display Error
                                                       |                     Suggest Re-enrollment
                                                       v
                                              Verify --> Mark Attendance
```

### 6.3 Evaluation Flow

```text
Face Recognition Data --> prepare_splits --> Train/Val/Test Splits
                                                      |
                                                      v
                                                    eval
                                                      |
                          +-----------------------+---+----------------------+
                          |                       |                          |
                          v                       v                          v
                    Metrics Calculation    Visualizations            Failure Analysis
                          |                       |                          |
                          v                       v                          v
                    metrics_summary.json    threshold_sweep.csv       sample_predictions.csv
                                             confusion_matrix.png     fairness/summary.md
```

**Execution path:**

1. `python manage.py prepare_splits --seed 42` writes `reports/splits.csv` so every run evaluates the same hold-out identities.
2. `python manage.py eval --split-csv reports/splits.csv` (or `make evaluate`) reuses the live recognition stack to measure accuracy, precision/recall, macro F1, FAR, FRR, and cosine-distance sweeps.
3. Artifacts land in `reports/evaluation/` (`metrics_summary.json`, `sample_predictions.csv`, `confusion_matrix.csv/.png`, `threshold_sweep.csv/.png`).
4. Optional: `python manage.py fairness_audit` and `python manage.py evaluate_liveness` append subgroup stats and spoof metrics to `reports/fairness/` and `reports/liveness/` respectively so QA reviewers can trace every claim back to a script.

## 7. Extended Architecture Diagram

```text
                                    +------------------+
                                    |   Web Browser    |
                                    +--------+---------+
                                             |
                                             | HTTP
                                             |
                    +------------------------v------------------------+
                    |                     Django                      |
                    |                                                 |
                    | +-------------------+    +-------------------+  |
                    | |   Recognition     |    |   Evaluation      |  |
                    | |   Views           |    |   Views           |  |
                    | +-------------------+    +-------------------+  |
                    |          |                        |             |
                    | +--------v---------+    +---------v----------+  |
                    | | Policy Engine    |    | Metrics Engine     |  |
                    | +--------+---------+    +---------+----------+  |
                    |          |                        |             |
                    +----------+------------------------+-------------+
                               |                        |
              +----------------v----------------+       |
              |           DeepFace              |       |
              |  +---------------------------+  |       |
              |  | Detector (SSD/OpenCV/    |  |       |
              |  | MTCNN)                    |  |       |
              |  +---------------------------+  |       |
              |               |                 |       |
              |  +------------v--------------+  |       |
              |  | Facenet Embedding Model  |  |       |
              |  +---------------------------+  |       |
              |               |                 |       |
              |  +------------v--------------+  |       |
              |  | Distance Calculation     |  |       |
              |  | (Cosine/Euclidean/L2)   |  |       |
              |  +---------------------------+  |       |
              +----------------+----------------+       |
                               |                        |
              +----------------v----------------+       |
              |    face_recognition_data/       |       |
              |    - user1/                     |       |
              |    - user2/                     |       |
              |    - ...                        |       |
              +----------------+----------------+       |
                               |                        |
              +----------------v------------------------v-----+
              |                PostgreSQL / SQLite            |
              |  +------------------+  +------------------+   |
              |  | User Records     |  | Attendance       |   |
              |  |                  |  | Records          |   |
              |  +------------------+  +------------------+   |
              +----------------------------------------------+
                                       |
              +------------------------v---------------------+
              |              reports/                        |
              |  - evaluation/metrics_summary.json           |
              |  - evaluation/sample_predictions.csv         |
              |  - evaluation/confusion_matrix.{csv,png}     |
              |  - evaluation/threshold_sweep.{csv,png}      |
              |  - fairness/summary.md                       |
              |  - liveness/summary.md                       |
              +----------------------------------------------+
```

## 8. Module Dependencies

```text
attendance_system_facial_recognition/  (Project Root)
│
├── recognition/                       (Main App)
│   ├── views.py                      → DeepFace, Policy
│   ├── admin_views.py                → Metrics, Evaluation
│   ├── evaluation/
│   │   └── metrics.py                → sklearn, bootstrap
│   ├── analysis/
│   │   └── failures.py               → metrics.py, pandas
│   ├── ablation.py                   → DeepFace, metrics.py
│   └── data_splits.py                → sklearn, numpy
│
├── users/                            (User Management)
│   ├── models.py
│   └── views.py
│
├── configs/                          (Configuration)
│   └── policy.yaml                   (Action bands)
│
└── reports/                          (Generated Artifacts)
    ├── evaluation/
    │   ├── metrics_summary.json
    │   ├── sample_predictions.csv
    │   ├── confusion_matrix.{csv,png}
    │   └── threshold_sweep.{csv,png}
    ├── fairness/summary.md
    └── liveness/summary.md
```

## 9. Technology Stack Update

The complete technology stack now includes:

### Core Framework

- **Django 5+**: Web framework
- **Python 3.12+**: Programming language

### Face Recognition

- **DeepFace**: Face recognition library
- **Facenet**: Embedding model (default)
- **SSD/OpenCV/MTCNN**: Face detectors

### Data Science & Evaluation

- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning metrics
- **Matplotlib**: Visualization
- **Pandas**: Data analysis (for failure analysis)

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **pre-commit**: Git hooks

### Configuration & CLI

- **PyYAML**: Configuration parsing
- **argparse**: CLI argument parsing

### Database & Frontend

- **PostgreSQL**: Recommended production database (falls back to SQLite when `DATABASE_URL` is unset)
- **React 18+**: Frontend framework (SPA)
- **TypeScript**: Type-safe frontend development
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Component library

## 10. Security and Privacy Architecture

### Data Protection Measures

- Face images stored locally (not in database)
- Access control via Django authentication
- Admin-only access to evaluation tools
- Audit logging for data access
- Configurable data retention policies

### Privacy by Design

- Purpose limitation (attendance only)
- Consent required for enrollment
- Opt-out available (manual attendance)
- Data minimization (only necessary images)
- User rights supported (access, delete, correct, export)

### Compliance Support

- GDPR-ready architecture
- CCPA-compliant
- Data card documentation
- Privacy policy framework
- User consent tracking

See [DATA_CARD.md](DATA_CARD.md) for comprehensive dataset documentation and privacy details.
