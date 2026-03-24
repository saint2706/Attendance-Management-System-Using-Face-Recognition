from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.test import RequestFactory

from rest_framework import exceptions, status

from recognition.api.exceptions import custom_exception_handler


class TestCustomExceptionHandler:
    def setup_method(self):
        self.factory = RequestFactory()
        self.request = self.factory.get("/api/v1/test/")
        self.context = {"request": self.request}

    def test_http404_handled_correctly(self):
        exc = Http404()
        response = custom_exception_handler(exc, self.context)

        assert response is not None
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.data["type"] == "about:blank"
        assert response.data["title"] == "Not Found"
        assert response.data["status"] == status.HTTP_404_NOT_FOUND
        assert response.data["detail"] == "Not found."
        assert response.data["instance"] == "/api/v1/test/"
        assert response["Content-Type"] == "application/problem+json"

    def test_permission_denied_handled_correctly(self):
        exc = PermissionDenied()
        response = custom_exception_handler(exc, self.context)

        assert response is not None
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.data["type"] == "about:blank"
        assert response.data["title"] == "Forbidden"
        assert response.data["status"] == status.HTTP_403_FORBIDDEN
        assert response.data["detail"] == "You do not have permission to perform this action."
        assert response.data["instance"] == "/api/v1/test/"
        assert response["Content-Type"] == "application/problem+json"

    def test_api_exception_with_string_detail(self):
        exc = exceptions.APIException("A server error occurred.")
        response = custom_exception_handler(exc, self.context)

        assert response is not None
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.data["type"] == "about:blank"
        assert response.data["title"] == "Error"
        assert response.data["status"] == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.data["detail"] == "A server error occurred."
        assert response.data["instance"] == "/api/v1/test/"

    def test_api_exception_with_dict_detail_with_detail_key(self):
        exc = exceptions.NotAuthenticated(
            {"detail": "Authentication credentials were not provided."}
        )
        response = custom_exception_handler(exc, self.context)

        assert response is not None
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.data["title"] == "Not Authenticated"
        assert response.data["detail"] == "Authentication credentials were not provided."

    def test_api_exception_with_dict_detail_field_errors(self):
        exc = exceptions.ValidationError(
            {"username": ["This field is required."], "email": "Invalid email."}
        )
        response = custom_exception_handler(exc, self.context)

        assert response is not None
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data["title"] == "Invalid"
        assert "username: This field is required." in response.data["detail"]
        assert "email: Invalid email." in response.data["detail"]
        assert "errors" in response.data
        assert response.data["errors"] == {
            "username": ["This field is required."],
            "email": "Invalid email.",
        }

    def test_api_exception_with_list_detail(self):
        exc = exceptions.ValidationError(["Error 1", "Error 2"])
        response = custom_exception_handler(exc, self.context)

        assert response is not None
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data["detail"] == "Error 1 Error 2"

    def test_exception_without_request_context(self):
        exc = Http404()
        response = custom_exception_handler(exc, {})

        assert response is not None
        assert response.data["instance"] == "unknown"

    def test_unhandled_exception(self):
        # DRF's default exception handler returns None for non-API exceptions
        # So custom_exception_handler should return None as well
        exc = ValueError("Some standard python error")
        response = custom_exception_handler(exc, self.context)
        assert response is None
