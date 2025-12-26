"""Tests for progressive web app assets."""

from __future__ import annotations

from django.http import Http404
from django.test import Client

import pytest

pytestmark = pytest.mark.django_db


def test_manifest_served_with_expected_content_type(client: Client) -> None:
    """The manifest should respond with the correct content type."""

    response = client.get("/manifest.json")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/manifest+json"


def test_service_worker_served_with_expected_headers(client: Client) -> None:
    """The service worker should be served with the expected headers."""

    response = client.get("/sw.js")

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/javascript"
    assert response.headers["Service-Worker-Allowed"] == "/"


def test_manifest_returns_404_when_asset_missing(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing manifest assets should return a 404 response."""

    from attendance_system_facial_recognition import urls

    def _raise_http_404(*_args: object, **_kwargs: object) -> None:
        raise Http404("Static asset not found")

    monkeypatch.setattr(urls, "_serve_static_asset", _raise_http_404)

    response = client.get("/manifest.json")

    assert response.status_code == 404
