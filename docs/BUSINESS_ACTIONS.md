# Business Actions Mapping

## Overview

This document describes how cosine distance scores from the recognition pipeline are mapped to business actions in the Attendance Management System.

The service stores cosine similarity and distance for every attempt:

- `sim(A, B) = (A · B) / (||A|| ||B||)`
- `d(A, B) = 1 − sim(A, B)`

Lower distances indicate closer matches. The default global threshold is `d ≤ 0.40`, but policy bands provide additional guardrails for operators.

## Score Bands and Actions

### 1. Confident Accept (Distance ≤ 0.30)
- Automatic approval.
- Expected frequency: ~85% of valid attempts when cameras and lighting match enrollment conditions.

### 2. Uncertain (0.30 < Distance ≤ 0.45)
- Secondary verification required (PIN/OTP).
- Marked as provisional until the human challenge succeeds.
- Expected frequency: ~10-12% of attempts.

### 3. Reject (Distance > 0.45)
- Not marked, user notified.
- Suggest re-enrollment after 5 failures or trigger liveness retraining.
- Expected frequency: ~3-5% of attempts depending on the population and camera quality.

Periodically rerun `python manage.py eval --split-csv reports/splits.csv` to confirm these bands keep FAR/FRR within policy before modifying `RECOGNITION_DISTANCE_THRESHOLD` or `configs/policy.yaml`.
