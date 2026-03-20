from unittest import mock

from django.core.exceptions import PermissionDenied
from django.http import Http404

from rest_framework.exceptions import APIException, NotFound, ValidationError

from recognition.api.exceptions import custom_exception_handler


class TestCustomExceptionHandler:
    def test_http404_handled(self):
        exc = Http404()
        request = mock.MagicMock()
        request.path = "/test/path"
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 404
        assert response.data["status"] == 404
        assert response.data["title"] == "Not Found"
        assert response.data["detail"] == "Not found."
        assert response.data["instance"] == "/test/path"
        assert response["Content-Type"] == "application/problem+json"

    def test_permission_denied_handled(self):
        exc = PermissionDenied()
        request = mock.MagicMock()
        request.path = "/test/path"
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 403
        assert response.data["status"] == 403
        assert response.data["title"] == "Forbidden"
        assert response.data["detail"] == "You do not have permission to perform this action."
        assert response.data["instance"] == "/test/path"
        assert response["Content-Type"] == "application/problem+json"

    def test_drf_validation_error_dict(self):
        exc = ValidationError({"field_name": ["This field is required."]})
        request = mock.MagicMock()
        request.path = "/api/test"
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 400
        assert response.data["status"] == 400
        assert "field_name: This field is required." in response.data["detail"]
        assert response.data["errors"] == {"field_name": ["This field is required."]}

    def test_drf_validation_error_list(self):
        exc = ValidationError(["Invalid value."])
        request = mock.MagicMock()
        request.path = "/api/test"
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 400
        assert response.data["status"] == 400
        assert "Invalid value." in response.data["detail"]

    def test_drf_api_exception_with_detail(self):
        exc = APIException(detail="A server error occurred.")
        request = mock.MagicMock()
        request.path = "/api/test"
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 500
        assert response.data["status"] == 500
        assert response.data["detail"] == "A server error occurred."

    def test_unhandled_exception_returns_none(self):
        exc = Exception("Something broke")
        request = mock.MagicMock()
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response is None

    def test_no_request_in_context(self):
        exc = NotFound()
        context = {}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 404
        assert response.data["instance"] == "unknown"

    def test_drf_exception_with_detail_dict(self):
        exc = ValidationError(detail={"detail": "Detailed validation message"})
        request = mock.MagicMock()
        request.path = "/api/test"
        context = {"request": request}

        response = custom_exception_handler(exc, context)

        assert response.status_code == 400
        assert response.data["detail"] == "Detailed validation message"

    def test_drf_exception_no_detail_string(self):
        class CustomError(APIException):
            status_code = 400
            default_detail = "Custom default error"

        with mock.patch("recognition.api.exceptions.exception_handler") as mock_handler:
            mock_response = mock.MagicMock()
            mock_response.status_code = 400
            mock_response.data = {}
            mock_handler.return_value = mock_response

            exc = CustomError()

            request = mock.MagicMock()
            request.path = "/api/test"
            context = {"request": request}

            response = custom_exception_handler(exc, context)

            assert response.data["detail"] == "Custom default error"

    def test_drf_exception_bare_data(self):
        with mock.patch("recognition.api.exceptions.exception_handler") as mock_handler:
            mock_response = mock.MagicMock()
            mock_response.status_code = 400
            mock_response.data = "This is a raw string error"
            mock_handler.return_value = mock_response

            exc = APIException("Something")
            # We must trick detail to evaluate as false so we fall through to exc default
            exc.default_detail = "Something"

            request = mock.MagicMock()
            request.path = "/api/test"
            context = {"request": request}

            response = custom_exception_handler(exc, context)

            # Since exception is string, string conversion of APIException happens
            assert "Something" in response.data["detail"]
