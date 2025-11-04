# Business Actions Mapping

## Overview

This document describes how face recognition scores are mapped to business actions in the Attendance Management System.

## Score Bands and Actions

### 1. Confident Accept (Score ≥ 0.80)
- Automatic approval
- Expected frequency: ~85% of valid attempts

### 2. Uncertain (Score 0.50-0.80)
- Secondary verification required (PIN/OTP)
- Marked as provisional
- Expected frequency: ~10-12% of attempts

### 3. Reject (Score < 0.50)
- Not marked, user notified
- Suggest re-enrollment after 5 failures
- Expected frequency: ~3-5% of attempts

See full documentation in the file for details.
