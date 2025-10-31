# Data Card for Attendance Management System

This document provides a summary of the data used in the Attendance Management System using Face Recognition.

## Dataset

The dataset consists of images of employees used for training the face recognition model.

### Variables

| Variable      | Type  | Description                                       | Missing % | Leakage Risk |
|---------------|-------|---------------------------------------------------|-----------|--------------|
| `employee_id` | int   | Unique identifier for each employee.              | 0%        | Low          |
| `image`       | image | Image of the employee's face.                     | 0%        | Low          |
| `timestamp`   | datetime | Timestamp of when the attendance was marked.   | 0%        | Low          |
| `status`      | string | Status of attendance (e.g., present, absent).     | 0%        | Low          |

### Data Splits

The dataset is split into training, validation, and test sets.

| Split       | Count | Date Range                |
|-------------|-------|---------------------------|
| **Training**  | 800   | January 2023 - June 2023  |
| **Validation**| 100   | July 2023 - August 2023   |
| **Test**      | 100   | September 2023 - October 2023 |

**Note:** The data split is performed with a fixed seed to ensure reproducibility.
