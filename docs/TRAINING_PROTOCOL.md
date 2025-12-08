# Training Protocol for Fair Face Recognition

This document provides standardized guidelines for collecting diverse, high-quality training data to ensure fair and accurate face recognition across all users.

## Overview

A diverse training dataset is essential for minimizing bias and ensuring equitable recognition performance. Each employee should have **8-12 face images** captured under varied conditions.

---

## Image Collection Guidelines

### Minimum Requirements

| Requirement | Target | Purpose |
|------------|--------|---------|
| Images per user | 8-12 | Capture natural variation |
| Lighting conditions | 3+ | Handle environmental changes |
| Pose angles | 3+ | Improve recognition from different views |
| Expressions | 2+ | Account for natural expressions |

### Lighting Variations

Capture images under at least three lighting conditions:

| Condition | Description | Tips |
|-----------|-------------|------|
| **Bright** | Well-lit environment, daylight | Near windows, overhead lights on |
| **Moderate** | Standard indoor lighting | Normal office conditions |
| **Low light** | Dim environment | Early morning, evening |

> [!TIP]
> Avoid extreme backlighting (bright light behind the subject) which obscures facial features.

### Pose Variations

Include at least three head positions:

| Pose | Angle | Description |
|------|-------|-------------|
| **Frontal** | 0° | Looking directly at camera |
| **Slight left** | 15-30° | Head turned slightly left |
| **Slight right** | 15-30° | Head turned slightly right |
| **Tilted** | 10-15° | Head tilted up or down |

> [!IMPORTANT]
> Ensure the face remains fully visible in all poses. Extreme angles reduce recognition accuracy.

### Accessory Variations

Capture with and without common accessories:

- **Glasses** – Include images with and without if user wears glasses regularly
- **Masks** – Optional, for mask-required environments only
- **Hats/headwear** – Include if commonly worn at workplace
- **Facial hair changes** – Re-capture if significant changes occur

### Expression Variations

Include at least two expressions:

- **Neutral** – Relaxed, natural expression
- **Smiling** – Common everyday expression

---

## Sample Quality Checklist

Before accepting an image, verify:

- [ ] Face is **clearly visible** and not obscured
- [ ] Image is **in focus** (not blurry)
- [ ] **Adequate lighting** on face (no deep shadows)
- [ ] Face takes up **at least 20%** of the frame
- [ ] **Eyes are visible** (not closed or covered)
- [ ] Camera is at **approximately eye level**

---

## Capture Session Workflow

### Recommended Capture Order

```text
1. Frontal, neutral expression (bright light)
2. Frontal, smiling (bright light)
3. Slight left turn (bright light)
4. Slight right turn (moderate light)
5. Frontal, neutral (moderate light)
6. Frontal with glasses if applicable (moderate light)
7. Frontal, neutral (low light)
8. Slight tilt up/down (low light)
```

### Session Duration

A complete capture session typically takes **2-3 minutes** per employee.

---

## Re-Enrollment Guidelines

Re-capture face images when:

| Event | Action |
|-------|--------|
| Significant appearance change | Add 4-6 new images |
| Poor recognition performance | Review and re-capture all images |
| New camera/location added | Capture 2-3 images at new location |
| 12+ months since last capture | Consider refresh capture |

---

## Multi-Camera Deployments

For environments with multiple capture devices:

1. **Calibrate each camera** using `python manage.py calibrate_camera`
2. **Capture enrollment images** on each camera type if possible
3. **Run fairness audits** per camera to detect domain gaps
4. **Adjust thresholds** per camera if needed

See [FAIRNESS_AND_LIMITATIONS.md](FAIRNESS_AND_LIMITATIONS.md) for domain adaptation details.

---

## Fairness Verification

After enrollment, verify fair performance by running:

```bash
python manage.py fairness_audit --reports-dir reports/fairness
```

Review the generated `summary.md` for any groups with elevated False Reject Rates (FRR), which may indicate insufficient training diversity.

---

## Quick Reference Card

```text
┌─────────────────────────────────────────────────────────┐
│           FACE ENROLLMENT QUICK CHECKLIST               │
├─────────────────────────────────────────────────────────┤
│ ☐ 8-12 images captured                                  │
│ ☐ 3+ lighting conditions (bright, moderate, low)        │
│ ☐ 3+ poses (frontal, left, right)                       │
│ ☐ 2+ expressions (neutral, smiling)                     │
│ ☐ With/without glasses (if applicable)                  │
│ ☐ All images in focus and well-lit                      │
│ ☐ Face clearly visible in all images                    │
└─────────────────────────────────────────────────────────┘
```
