from django.core.exceptions import PermissionDenied
from django.http import Http404
from rest_framework.exceptions import APIException
from rest_framework.views import exception_handler


def custom_exception_handler(exc, context):
    """
    Custom exception handler that returns RFC 7807 (Problem Details) format.
    """
    # Call REST framework's default exception handler first,
    # to get the standard error response.
    response = exception_handler(exc, context)

    # Now add the HTTP status code to the response.
    if response is not None:
        if isinstance(exc, Http404):
            title = "Not Found"
            status_code = 404
            detail = "Not found."
        elif isinstance(exc, PermissionDenied):
            title = "Forbidden"
            status_code = 403
            detail = "You do not have permission to perform this action."
        else:
            title = getattr(exc, "default_code", exc.__class__.__name__)
            status_code = response.status_code

            # DRF validation errors often return a dict or list for detail
            if isinstance(response.data, dict) and "detail" in response.data:
                detail = response.data["detail"]
            elif isinstance(response.data, dict):
                # Flatten the dictionary if it's a validation error with field errors
                fields = []
                for k, v in response.data.items():
                    error_msg = v[0] if isinstance(v, list) else v
                    fields.append(f"{k}: {error_msg}")
                detail = " ".join(fields)
            elif isinstance(response.data, list):
                detail = " ".join([str(v) for v in response.data])
            else:
                detail = str(exc)

            # Use default detail if available and detail is empty
            if not detail and hasattr(exc, "default_detail"):
                detail = exc.default_detail

        request = context.get("request")
        instance = request.path if request else "unknown"

        custom_data = {
            "type": "about:blank",
            "title": str(title).replace("_", " ").title(),
            "status": status_code,
            "detail": detail,
            "instance": instance,
        }

        # In case we have extra fields we want to preserve from DRF's default payload,
        # we can add them here under an "errors" key if it was a validation error.
        if isinstance(exc, APIException) and isinstance(exc.detail, dict):
            custom_data["errors"] = exc.detail

        response.data = custom_data
        response["Content-Type"] = "application/problem+json"

    return response
