"""Tests for the export_reports management command."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management import call_command


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test", encoding="utf-8")


def test_export_reports_warns_when_directory_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    """Should warn when the reports directory is missing."""
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)

    call_command("export_reports")

    captured = capsys.readouterr()
    assert "Reports directory not found" in captured.out


def test_export_reports_lists_reports_and_figures(tmp_path: Path, monkeypatch, capsys) -> None:
    """Should list present and missing report files plus figure counts."""
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    reports_dir = tmp_path / "reports"

    present_reports = ["splits.csv", "metrics_with_ci.json"]
    for report in present_reports:
        _write_file(reports_dir / report)

    _write_file(reports_dir / "figures" / "roc.png")

    call_command("export_reports")

    output = capsys.readouterr().out
    for report in present_reports:
        assert f"✓ {report}" in output

    assert "✗ split_summary.json" in output
    assert "✓ roc.png" in output

    found_count = len(present_reports) + 1
    assert f"{found_count} reports/figures found in {reports_dir}" in output
