from django.core.exceptions import PermissionDenied
from django.http import Http404

import pytest
from rest_framework.exceptions import APIException, NotAuthenticated, ValidationError
from rest_framework.response import Response

from recognition.api.exceptions import custom_exception_handler


@pytest.fixture
def mock_context():
    class MockRequest:
        path = "/api/test/"

    return {"request": MockRequest()}


class TestCustomExceptionHandler:
    def test_http404_handled(self, mock_context):
        exc = Http404("Item not found")
        # DRF's exception_handler returns a 404 response for Http404
        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 404
        assert response["Content-Type"] == "application/problem+json"

        data = response.data
        assert data["title"] == "Not Found"
        assert data["status"] == 404
        assert data["detail"] == "Not found."
        assert data["instance"] == "/api/test/"

    def test_permission_denied_handled(self, mock_context):
        exc = PermissionDenied("Access denied")
        # DRF's exception_handler returns a 403 response for PermissionDenied
        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 403
        assert response["Content-Type"] == "application/problem+json"

        data = response.data
        assert data["title"] == "Forbidden"
        assert data["status"] == 403
        assert data["detail"] == "You do not have permission to perform this action."
        assert data["instance"] == "/api/test/"

    def test_drf_validation_error_dict(self, mock_context):
        exc = ValidationError({"field1": ["error 1"], "field2": "error 2"})
        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 400
        assert response["Content-Type"] == "application/problem+json"

        data = response.data
        assert data["title"] == "Invalid"
        assert data["status"] == 400
        # Detail will be flattened
        assert "field1: error 1" in data["detail"]
        assert "field2: error 2" in data["detail"]
        assert data["instance"] == "/api/test/"
        assert "errors" in data
        assert data["errors"] == exc.detail

    def test_drf_validation_error_list(self, mock_context):
        exc = ValidationError(["error 1", "error 2"])
        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 400
        assert response["Content-Type"] == "application/problem+json"

        data = response.data
        assert data["title"] == "Invalid"
        assert data["status"] == 400
        assert data["detail"] == "error 1 error 2"
        assert data["instance"] == "/api/test/"

    def test_drf_api_exception_with_detail(self, mock_context):
        exc = NotAuthenticated(detail="Login required")
        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 401
        assert response["Content-Type"] == "application/problem+json"

        data = response.data
        assert data["title"] == "Not Authenticated"
        assert data["status"] == 401
        assert data["detail"] == "Login required"
        assert data["instance"] == "/api/test/"

    def test_unhandled_exception_returns_none(self, mock_context):
        # ValueError is not handled by DRF's default exception_handler
        exc = ValueError("Something bad")
        response = custom_exception_handler(exc, mock_context)
        assert response is None

    def test_no_request_in_context(self):
        exc = NotAuthenticated(detail="Login required")
        response = custom_exception_handler(exc, {})

        assert response is not None
        data = response.data
        assert data["instance"] == "unknown"

    def test_drf_exception_with_detail_dict(self, mock_context):
        # Test case where response.data is a dict containing "detail"
        exc = APIException(detail="Some error message")
        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 500
        data = response.data
        assert data["detail"] == "Some error message"

    def test_drf_exception_no_detail_string(self, monkeypatch, mock_context):
        # Test case where detail is empty and hasattr default_detail
        class CustomException(APIException):
            status_code = 400
            default_code = "custom_error"
            default_detail = "Custom default error"

        exc = CustomException()
        exc.detail = ""  # explicitly empty out detail so it falls back

        # Monkeypatch exception handler to return an empty response string
        def mock_exception_handler(exc, context):
            return Response(data="", status=400)

        import recognition.api.exceptions

        monkeypatch.setattr(recognition.api.exceptions, "exception_handler", mock_exception_handler)

        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        data = response.data
        assert data["detail"] == "Custom default error"

    def test_drf_exception_bare_data(self, monkeypatch, mock_context):
        # Test case where response.data is neither a dict nor a list
        class CustomException(APIException):
            status_code = 400
            default_code = "custom_error"
            default_detail = "Custom error"

        exc = CustomException()

        # We need to monkeypatch the exception handler to return a string response
        def mock_exception_handler(exc, context):
            return Response(data="Just a string error", status=400)

        import recognition.api.exceptions

        monkeypatch.setattr(recognition.api.exceptions, "exception_handler", mock_exception_handler)

        response = custom_exception_handler(exc, mock_context)

        assert response is not None
        assert response.status_code == 400

        data = response.data
        assert data["title"] == "Custom Error"
        # It stringifies exc if it can't use anything else, and custom exception doesn't have str so it falls back to APIException str
        # which returns default_detail string or empty
        # wait, we can just check what it falls back to
        assert str(exc) in data["detail"]
