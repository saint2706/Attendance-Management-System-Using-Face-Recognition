# API Reference

This document provides detailed information about the API endpoints available in the Attendance Management System Using Face Recognition. These endpoints are designed for programmatic access to the face recognition and attendance marking functionalities.

---

## Face Recognition API

This endpoint accepts an image or a pre-computed face embedding and returns the closest matching identity from the enrolled employee dataset.

- **URL:** `/api/face-recognition/`
- **HTTP Method:** `POST`
- **Authentication:** Required. Clients must either be logged in with a valid session cookie or supply one of the following:
  - `X-API-Key` header matching one of the `RECOGNITION_API_KEYS` configured via environment variables.
  - `Authorization: Bearer <JWT>` header signed with `RECOGNITION_JWT_SECRET` (optional `RECOGNITION_JWT_ISSUER`/`RECOGNITION_JWT_AUDIENCE`).
- **Rate Limiting:** `5 requests per minute` by authenticated session, API key, or JWT subject (fallback to IP). Override via `RECOGNITION_FACE_API_RATE_LIMIT`.

### Face Recognition Request Payload

The request can be sent as `application/json` or `multipart/form-data`. You must provide either an `image` or an `embedding`.

| Parameter   | Type                               | Description                                                                                                                                      |
|-------------|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `image`     | File or Base64 String              | The image containing the face to be recognized. Can be a file upload or a Base64-encoded string. If a string is provided, a data URI prefix (`data:image/...;base64,`) is also supported. |
| `liveness_frames` | Array of Base64 Strings      | Optional. A short burst (3–5) of additional frames encoded as Base64 strings. These are decoded server-side to run the motion-based liveness detector when you are posting embeddings or low-FPS stills. |
| `embedding` | Array of Floats                    | A pre-computed face embedding vector to be used for matching. If provided, the `image` parameter is ignored.                                    |
| `direction` | String                             | Optional. The attendance direction. Can be `"in"` or `"out"`. Defaults to `"in"`.                                                                   |
| `username`  | String                             | Optional. The username of the employee being verified. This is used for logging and analytics but does not influence the matching process.         |

### Face Recognition Responses

#### Success Response (`200 OK`)

A successful request returns a JSON object with the recognition result.

```json
{
    "recognized": true,
    "threshold": 0.4,
    "distance_metric": "cosine",
    "distance": 0.2345,
    "identity": "/path/to/dataset/john_doe/image1.jpg",
    "username": "john_doe"
}
```

- **`recognized` (boolean):** `true` if the face matches an enrolled employee within the distance threshold, `false` otherwise.
- **`threshold` (float):** The distance threshold used for the comparison.
- **`distance_metric` (string):** The metric used to calculate the distance between embeddings. The service defaults to cosine distance.
- **`distance` (float):** The calculated cosine distance to the closest match. Only present if a match is found.
- **`identity` (string):** The path to the matched image in the dataset. Only present if a match is found.
- **`username` (string):** The username of the matched employee. Only present if a match is found.
- **`spoofed` (boolean):** `true` if a liveness check was performed and failed, indicating a potential spoofing attempt.
- **`mode` (string):** `"single"` for single-face mode (default) or `"multi"` when multi-face detection is enabled.

#### Multi-Face Mode Response

When `RECOGNITION_MULTI_FACE_ENABLED=true`, the API returns a different response structure that includes all detected faces:

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
            "facial_area": {"x": 300, "y": 60, "w": 140, "h": 140}
        }
    ],
    "count": 2,
    "mode": "multi",
    "threshold": 0.4,
    "distance_metric": "cosine"
}
```

For detailed configuration and usage, see the [Multi-Face Detection Guide](MULTI_FACE_GUIDE.md).

Cosine similarity drives the `distance` value reported above:

- `sim(A, B) = (A · B) / (||A|| ||B||)`
- `d(A, B) = 1 − sim(A, B)`

A prediction is accepted when `distance ≤ threshold` (default `0.4`); higher values trigger secondary verification or rejection through the policy engine.

#### Face Recognition Error Responses

Authentication and validation failures return a JSON body with an `error` field describing the issue (for example, `{"error": "Invalid API key provided."}`).

- **`400 Bad Request`:** The request payload is invalid (e.g., malformed JSON, invalid Base64 data, or missing `image` and `embedding`).
- **`401 Unauthorized`:** Missing or invalid credentials (no session, incorrect API key, malformed/expired JWT, or JWT support not configured).
- **`429 Too Many Requests`:** The client has exceeded the rate limit.
- **`500 Internal Server Error`:** An unexpected error occurred during the face recognition process.
- **`503 Service Unavailable`:** The system has no enrolled face embeddings to compare against.

---

## Attendance Batch API

This endpoint enqueues a batch of attendance records for asynchronous processing by a Celery worker.

- **URL:** `/api/attendance/batch/`
- **HTTP Method:** `POST`
- **Authentication:** Required (session authentication). The user must be logged in.
- **Rate Limiting:** This endpoint is rate-limited to prevent abuse.

### Batch API Request Payload

The request body must be a JSON object containing a `records` array.

| Parameter | Type   | Description                                                                                             |
|-----------|--------|---------------------------------------------------------------------------------------------------------|
| `records` | Array  | An array of attendance records to be processed. Each record is an object that will be passed to a Celery task. |

**Example Request Body:**

```json
{
    "records": [
        {
            "direction": "in",
            "present": {
                "john_doe": true
            },
            "attempt_ids": {
                "john_doe": 123
            }
        },
        {
            "direction": "out",
            "present": {
                "jane_doe": true
            },
            "attempt_ids": {
                "jane_doe": 124
            }
        }
    ]
}
```

### Batch API Responses

#### Success Response (`202 Accepted`)

A successful request indicates that the batch has been accepted and enqueued for processing.

```json
{
    "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "PENDING",
    "total": 2
}
```

- **`task_id` (string):** The ID of the Celery task created to process the batch.
- **`status` (string):** The initial status of the task (e.g., `PENDING`).
- **`total` (integer):** The number of records in the batch.

#### Batch API Error Responses

- **`400 Bad Request`:** The request payload is invalid (e.g., malformed JSON, or the `records` parameter is not a list).
- **`403 Forbidden`:** The user is not authenticated.
- **`405 Method Not Allowed`:** The request was not a `POST` request.
- **`503 Service Unavailable`:** The Celery queue is not available.
