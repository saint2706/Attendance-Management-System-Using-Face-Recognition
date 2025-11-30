"""Tests for the fairness dashboard admin view."""

from django.contrib.auth import get_user_model
from django.urls import reverse

import pytest


@pytest.mark.django_db
def test_fairness_dashboard_requires_staff(client):
    """Non-staff users should be redirected."""
    response = client.get(reverse("admin_fairness_dashboard"))
    # staff_member_required redirects to login
    assert response.status_code == 302


@pytest.mark.django_db
def test_fairness_dashboard_non_staff_user_redirected(client):
    """Non-staff authenticated users should be redirected."""
    user = get_user_model().objects.create_user(
        username="regular-user",
        password="StrongPass123!",
        is_staff=False,
    )
    client.force_login(user)

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 302


@pytest.mark.django_db
def test_fairness_dashboard_accessible_by_staff(client):
    """Staff users should be able to access the dashboard."""
    user = get_user_model().objects.create_user(
        username="fairness-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    assert b"Model Fairness" in response.content


@pytest.mark.django_db
def test_fairness_dashboard_shows_no_audit_message_when_no_reports(client, tmp_path, settings):
    """Dashboard should show message when no audit has been run."""
    user = get_user_model().objects.create_user(
        username="no-audit-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    # Ensure reports/fairness directory doesn't exist
    settings.BASE_DIR = tmp_path
    (tmp_path / "reports").mkdir(exist_ok=True)

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    assert b"No fairness audit has been run yet" in response.content


@pytest.mark.django_db
def test_fairness_dashboard_displays_audit_results(client, tmp_path, settings):
    """Dashboard should display audit results when reports exist."""
    user = get_user_model().objects.create_user(
        username="audit-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    settings.BASE_DIR = tmp_path
    fairness_dir = tmp_path / "reports" / "fairness"
    fairness_dir.mkdir(parents=True, exist_ok=True)

    # Create a mock summary.md
    summary_content = """# Fairness & Robustness Audit

Generated on 2024-01-15T10:30:00 UTC

## Overall evaluation

| Metric | Value |
| --- | --- |
| samples | 100 |
| accuracy | 0.9500 |
| precision | 0.9400 |
| recall | 0.9300 |
| f1 | 0.9350 |
| far | 0.0200 |
| frr | 0.0500 |

## Metrics By Role

| group | samples | accuracy |
| --- | --- | --- |
| employee | 80 | 0.9500 |
| staff_or_admin | 20 | 0.9600 |
"""
    (fairness_dir / "summary.md").write_text(summary_content, encoding="utf-8")

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    assert b"Completed" in response.content
    # Check overall metrics are displayed
    assert b"accuracy" in response.content or b"Accuracy" in response.content


@pytest.mark.django_db
def test_fairness_dashboard_loads_csv_metrics(client, tmp_path, settings):
    """Dashboard should load and display CSV metrics."""
    user = get_user_model().objects.create_user(
        username="csv-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    settings.BASE_DIR = tmp_path
    fairness_dir = tmp_path / "reports" / "fairness"
    fairness_dir.mkdir(parents=True, exist_ok=True)

    # Create summary.md
    (fairness_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")

    # Create metrics_by_role.csv
    csv_content = "group,samples,accuracy,precision,recall,f1,far,frr\n"
    csv_content += "employee,80,0.9500,0.9400,0.9300,0.9350,0.0200,0.0500\n"
    csv_content += "staff_or_admin,20,0.9600,0.9500,0.9400,0.9450,0.0100,0.0400\n"
    (fairness_dir / "metrics_by_role.csv").write_text(csv_content, encoding="utf-8")

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    assert response.context["group_metrics"]["by_role"] is not None
    assert len(response.context["group_metrics"]["by_role"]) == 2


@pytest.mark.django_db
def test_fairness_dashboard_flags_high_error_rates(client, tmp_path, settings):
    """Dashboard should flag groups with high FAR or FRR."""
    user = get_user_model().objects.create_user(
        username="flagged-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    settings.BASE_DIR = tmp_path
    fairness_dir = tmp_path / "reports" / "fairness"
    fairness_dir.mkdir(parents=True, exist_ok=True)

    # Create summary.md
    (fairness_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")

    # Create metrics_by_lighting.csv with high FRR
    csv_content = "group,samples,accuracy,precision,recall,f1,far,frr\n"
    csv_content += "low_light,50,0.7000,0.7500,0.6500,0.7000,0.0500,0.2000\n"  # FRR > 15%
    csv_content += "moderate_light,100,0.9500,0.9400,0.9300,0.9350,0.0200,0.0500\n"
    (fairness_dir / "metrics_by_lighting.csv").write_text(csv_content, encoding="utf-8")

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    assert len(response.context["flagged_groups"]) > 0
    assert response.context["flagged_groups"][0]["group"] == "low_light"


@pytest.mark.django_db
def test_fairness_dashboard_shows_known_limitations(client):
    """Dashboard should show known limitations section."""
    user = get_user_model().objects.create_user(
        username="limitations-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    assert b"Known Limitations" in response.content
    assert b"No demographic data" in response.content


@pytest.mark.django_db
def test_fairness_dashboard_links_to_related_pages(client):
    """Dashboard should have links to related admin pages."""
    user = get_user_model().objects.create_user(
        username="links-admin",
        password="StrongPass123!",
        is_staff=True,
    )
    client.force_login(user)

    response = client.get(reverse("admin_fairness_dashboard"))
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "admin_evaluation_dashboard" in content or "Evaluation Dashboard" in content
    assert "admin_system_health" in content or "System Health" in content
