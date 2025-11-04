# Data Card: Attendance Management Face Recognition Dataset

## Overview

This data card describes the face image dataset used in the Attendance Management System for employee recognition and attendance tracking.

**Version**: 1.0  
**Last Updated**: 2024-11-04  
**Maintained By**: System Engineering Team

## Dataset Description

### Purpose

The dataset is used to train and evaluate a face recognition model for automated attendance tracking. Images are collected during employee enrollment and used to verify identity during check-in/check-out.

### Provenance

- **Source**: Images captured directly from employees via webcam during enrollment
- **Collection Method**: Automated capture using OpenCV VideoStream
- **Collection Period**: Ongoing (system operational since 2023)
- **Geographic Location**: Organization premises
- **Collection Conditions**: Indoor office environment with controlled lighting

### Data Characteristics

| Attribute | Description |
|-----------|-------------|
| **Total Images** | ~1000-5000 (varies by deployment) |
| **Number of Identities** | Typically 50-200 employees |
| **Images per Identity** | 50 (default capture count) |
| **Image Format** | JPEG |
| **Resolution** | Variable (webcam-dependent, typically 640x480 to 1920x1080) |
| **Color Space** | RGB |
| **File Size** | 20-100 KB per image |

## Variables

### Primary Data Fields

| Variable | Type | Description | Missing % | Leakage Risk |
|----------|------|-------------|-----------|--------------|
| `image` | image | RGB face image | 0% | Low (primary input) |
| `identity` | string | Employee username/ID | 0% | **HIGH** - Must not be used as model input |
| `capture_timestamp` | datetime | When image was captured | 0% | Medium - Could leak enrollment session |
| `image_path` | string | File system path | 0% | **HIGH** - Contains username |
| `session_id` | string | Enrollment session identifier | ~5% | Medium - Groups same-session captures |

### Derived Fields (Not Stored)

| Variable | Type | Description | Computed From |
|----------|------|-------------|---------------|
| `face_embedding` | vector | 128 or 512-dim embedding | Image (via DeepFace/Facenet) |
| `face_bbox` | coordinates | Bounding box coordinates | Image (via detector) |
| `confidence_score` | float | Recognition confidence | Distance between embeddings |

## Data Splits

### Split Strategy

- **Train**: 70% of identities (not images)
- **Validation**: 15% of identities
- **Test**: 15% of identities

**Critical**: All images from the same identity are kept in the same split to prevent identity leakage.

### Split Statistics

| Split | Identities | Images | Date Range (Example) |
|-------|------------|--------|----------------------|
| Train | ~105 (70%) | ~5,250 | Jan 2023 - Jun 2023 |
| Val | ~23 (15%) | ~1,150 | Jul 2023 - Aug 2023 |
| Test | ~22 (15%) | ~1,100 | Sep 2023 - Oct 2023 |

**Note**: Actual counts depend on deployment size and are documented in `reports/split_summary.json`.

### Stratification

- Splits maintain class (identity) balance proportionally
- Random state fixed at **42** for reproducibility

### Session-Based Leakage Prevention

All images captured in the same enrollment session (typically within 2-3 minutes) are kept in the same split to prevent:
- Recognition of background/lighting conditions instead of faces
- Overfitting to specific camera angles or poses

## Class Distribution

### Identity Balance

- **Balanced**: Each enrolled employee has approximately the same number of images (50)
- **Imbalance Sources**:
  - New employees added after initial enrollment
  - Re-enrollment due to poor initial photos (adds 50 more images)
  - Manual additions/deletions by admins

### Handling Class Imbalance

- **Threshold Selection**: Optional class rebalancing during validation-based threshold tuning
- **Training**: SVM classifier naturally handles moderate imbalance
- **Recommendation**: Re-balance enrollment if any identity has <20 or >100 images

## Collection Conditions

### Enrollment Setup

- **Camera**: Standard webcam (various models, typically 720p-1080p)
- **Lighting**: Office fluorescent/LED lighting (500-1000 lux)
- **Background**: Office setting (variable backgrounds)
- **Pose**: Frontal face requested, but may include slight variations
- **Distance**: Approximately 50-70 cm from camera
- **Duration**: 2-3 minutes to capture 50 frames
- **Operator**: Self-service (employee operates system) or HR-assisted

### Operational Conditions (Inference)

- **Same camera hardware** as enrollment
- **Variable lighting**: Morning/afternoon/evening differences
- **Variable pose**: Users may not be perfectly frontal
- **Occlusions**: Possible (glasses, masks, accessories)

## Known Biases and Limitations

### 1. Lighting Conditions

**Bias**: Model may perform worse in:
- Very bright conditions (windows, sunlight)
- Very dark conditions (early morning, late evening)
- Mixed lighting (one side lit, one shadow)

**Mitigation**: Collect enrollment photos at different times of day

### 2. Pose Variations

**Bias**: Enrollment photos are mostly frontal. Model may struggle with:
- Profile views
- Looking up/down
- Head tilted

**Mitigation**: Request diverse poses during enrollment (not yet implemented)

### 3. Accessories and Occlusions

**Bias**: Performance degradation with:
- Face masks (not in enrollment data)
- Sunglasses (rare in enrollment)
- Hats, scarves (context-dependent)

**Mitigation**: Consider re-enrollment if appearance changes significantly

### 4. Demographic Representation

**Limitations**: Dataset reflects organization demographics, which may not represent general population
- Gender distribution: Depends on organization
- Age distribution: Typically working-age adults (20-65)
- Ethnicity distribution: Varies by organization location

**Impact**: Model may perform differently on underrepresented groups
**Fairness Assessment**: Recommended to analyze performance across demographic subgroups (see `reports/subgroup_metrics.csv`)

### 5. Camera Quality

**Bias**: Performance depends on camera quality
- High-quality cameras: Better recognition
- Low-quality cameras: Higher error rates
- Camera aging/degradation: Gradual performance decline

**Mitigation**: Regular camera maintenance and periodic re-enrollment

## Data Leakage Risks

### High-Risk Fields (Never use as model input)

1. **`identity` / `username`**: Direct label leakage
2. **`image_path`**: Contains identity in path
3. **`employee_id`**: Direct identifier
4. **`full_name`**: Personal identifier

### Medium-Risk Fields

1. **`capture_timestamp`**: Could be used to group same-session images
2. **`session_id`**: Explicitly links same-session captures

### Mitigation

- Use `recognition/data_splits.filter_leakage_fields()` to automatically remove risky fields
- Manual review of any metadata before training
- Only use pixel data and derived embeddings for model training

## Privacy and Consent

### Data Collection

- **Consent**: Employees must consent to face data collection for attendance tracking
- **Purpose Limitation**: Data used only for attendance verification
- **Opt-Out**: Manual attendance entry available upon request

### Data Retention

- **Active Employees**: Images retained while employed
- **Departed Employees**: Images deleted within 30 days of departure (configurable)
- **Provisional Records**: Kept for 7 days before automatic deletion

### Data Security

- **Storage**: Local file system with restricted access
- **Access Control**: Only admin users can view/manage face images
- **Encryption**: At rest (file system encryption recommended)
- **Logging**: All access to face data is logged

### Compliance

- **GDPR**: Right to access, rectify, delete personal data
- **CCPA**: Disclosure of data collection and purpose
- **Local Regulations**: Compliance with jurisdiction-specific laws

### User Rights

1. **Access**: Employees can request their stored face images
2. **Deletion**: Employees can request deletion (switches to manual attendance)
3. **Correction**: Employees can request re-enrollment with new photos
4. **Portability**: Face images can be exported (implementation pending)

## Usage Guidelines

### Recommended Uses

✓ Employee attendance tracking in controlled office environments  
✓ Building access control (with appropriate security measures)  
✓ Time tracking for payroll purposes

### Discouraged Uses

✗ Surveillance or monitoring beyond attendance  
✗ Behavioral analysis or emotion detection  
✗ Reidentification in public spaces  
✗ Any use without informed consent

## Model Performance

See separate reports for detailed metrics:
- `reports/metrics_with_ci.md`: Verification metrics with confidence intervals
- `reports/ABLATIONS.md`: Component ablation analysis
- `reports/FAILURES.md`: Failure case analysis

**Expected Performance** (typical deployment):
- ROC AUC: 0.95-0.98
- Equal Error Rate (EER): 2-5%
- False Accept Rate (FAR@TPR=0.95): 1-3%

## Maintenance and Updates

### Re-enrollment Triggers

1. **Appearance Change**: Facial hair, glasses, significant weight change
2. **Age**: Recommended every 2-3 years
3. **Performance Degradation**: 5+ consecutive recognition failures
4. **Camera Change**: When hardware is upgraded

### Dataset Refresh

- **Frequency**: Quarterly review of class balance
- **New Employees**: Immediate enrollment
- **Departed Employees**: Remove within 30 days
- **Quality Check**: Annual audit of image quality

## Contact

For questions about this dataset or to report issues:
- **Technical**: System Engineering Team
- **Privacy**: Data Protection Officer
- **Access Requests**: HR Department

## References

1. DeepFace Library: https://github.com/serengil/deepface
2. Facenet Model: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
3. Project Documentation: `README.md`, `ARCHITECTURE.md`, `DEVELOPER_GUIDE.md`

---

**Document Version**: 1.0  
**Last Review Date**: 2024-11-04  
**Next Review Date**: 2025-02-04 (quarterly)
