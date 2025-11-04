# Architecture Overview

This document provides a high-level overview of the application's architecture, explaining how the different components work together to provide a seamless and automated way to track employee attendance.

## 1. Core Components

The application is built on the following core components:

-   **Django**: The web framework that provides the application's structure, including the models, views, and templates.
-   **DeepFace**: A lightweight face recognition and facial attribute analysis (age, gender, emotion, and race) library for Python. It is used to power the face recognition functionality of the application.
-   **SQLite**: The default database for the application. It is used to store the application's data, including the user accounts and attendance records.
-   **Bootstrap**: The front-end framework that is used to create the application's responsive web interface.

## 2. Architecture Diagram

The following diagram provides a high-level overview of the application's architecture:

```
+-----------------+      +-----------------+      +-----------------+
|   Web Browser   | <--> |     Django      | <--> |     SQLite      |
+-----------------+      +-----------------+      +-----------------+
                           |
                           v
+-----------------+      +-----------------+
|   DeepFace      | <--> | Face Reco Data  |
+-----------------+      +-----------------+
```

## 3. Workflow

The application's workflow is as follows:

1.  **User Registration**: An admin registers a new employee by creating a new user account for them.
2.  **Add Photos**: The admin adds photos for the employee by using the "Add Photos" feature. The application uses the webcam to capture a set of photos for face recognition and stores them in the `face_recognition_data` directory.
3.  **Mark Attendance**: The employee marks their attendance by using the "Mark Time-In" or "Mark Time-Out" feature. The application uses the webcam to capture a frame and sends it to DeepFace for face recognition.
4.  **Face Recognition**: DeepFace compares the captured frame to the stored face images and returns the name of the recognized employee.
5.  **Store Attendance**: The application stores the attendance record in the database.
6.  **View Attendance**: The admin can view the attendance records for all employees, and the employee can view their own attendance records.

## 4. Evaluation and Validation Architecture

The system includes a comprehensive evaluation pipeline for measuring and improving performance.

### 4.1 Data Splitting Pipeline

```
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

```
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

```
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

```
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

```
Face Recognition
    |
    v
+-------------------+
| Similarity Score  |
| (0.0 - 1.0)       |
+-------------------+
    |
    v
+-------------------+
| Policy Engine     |
| (policy.yaml)     |
+-------------------+
    |
    +--> Score >= 0.80 --> Confident Accept --> Mark Attendance Immediately
    |
    +--> 0.50 <= Score < 0.80 --> Uncertain --> Request PIN/OTP
    |
    +--> Score < 0.50 --> Reject --> Notify User, Suggest Re-enrollment
```

**Benefits:**
- Reduces false accepts through secondary verification
- Improves user experience for edge cases
- Provides clear guidance to users
- Configurable thresholds per deployment

### 5.2 CLI Prediction Tool

```
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

```
Employee --> Register (Admin) --> Add Photos (Webcam) --> Store Images
                                                              |
                                                              v
                                      face_recognition_data/<username>/
                                                              |
                                                              v
                                              Ready for Recognition
```

### 6.2 Attendance Marking Flow

```
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

```
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
                    metrics_with_ci.md      figures/*.png              FAILURES.md
```

## 7. Extended Architecture Diagram

```
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
              |                    SQLite                     |
              |  +------------------+  +------------------+   |
              |  | User Records     |  | Attendance       |   |
              |  |                  |  | Records          |   |
              |  +------------------+  +------------------+   |
              +----------------------------------------------+
                                       |
              +------------------------v---------------------+
              |              reports/                        |
              |  - metrics_with_ci.md                        |
              |  - FAILURES.md                               |
              |  - ABLATIONS.md                              |
              |  - figures/ (ROC, PR, DET, Calibration)     |
              |  - *.csv (failure cases, subgroup metrics)   |
              +----------------------------------------------+
```

## 8. Module Dependencies

```
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
    ├── metrics_with_ci.md
    ├── FAILURES.md
    ├── ABLATIONS.md
    └── figures/
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
- **SQLite**: Default database
- **Bootstrap 5**: Frontend framework
- **HTML5/CSS3**: Web technologies

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
