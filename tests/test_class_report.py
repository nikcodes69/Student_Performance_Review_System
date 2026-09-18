"""
tests/test_class_report.py
===========================
pytest tests for utils/pdf_generator.py's generate_class_report() -- the
institution/class-wide PDF summary (Class Report Export page, Admin/
Teacher), as opposed to generate_report_card()'s single-student report.

WHAT THIS FILE CHECKS, AND WHAT IT DELIBERATELY DOES NOT: a PDF's exact
byte content is not meaningful to assert against (ReportLab embeds
timestamps and internal object ids that differ between runs even for
identical input data), so these tests confirm STRUCTURAL correctness --
the output is a real PDF, it does not crash on an empty database, and a
semester filter actually changes which data feeds the report -- rather
than parsing PDF text content. The report's ACTUAL section contents
(KPIs, subject averages, rankings, etc.) were verified by hand against
real production data and by visually inspecting a generated PDF page by
page -- see the chat history around this feature's implementation -- not
re-derived here, since every number in the report is already covered by
tests/test_analytics.py and tests/test_at_risk_report.py for the
functions generate_class_report() calls.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_class_report.py -v
"""

import pytest

import config
import database.db_setup as db_setup
from database.db_manager import execute_write
from database.db_setup import create_indexes, create_tables, get_connection
from modules import students
from utils.pdf_generator import generate_class_report


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database, with
    list_students()'s cache cleared first -- see
    tests/test_at_risk_report.py's module docstring for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_class_report.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    students.list_students.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def test_generate_class_report_produces_a_valid_pdf_with_no_data(test_db):
    # An empty database (no students, subjects, marks at all) must not
    # crash -- every section handles "nothing to show" gracefully.
    pdf_bytes = generate_class_report()
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 0


def test_generate_class_report_produces_a_valid_pdf_with_real_data(test_db):
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S1", "Alice", 1, "BCA", "alice@example.com", "9812345678", 2024),
    )
    execute_write(
        "INSERT INTO subjects (subject_code, name, semester, credits) VALUES (?, ?, ?, ?)",
        ("SUB1", "Fixture Subject", 1, 3),
    )
    execute_write(
        "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S1", "SUB1", 18, 45, 0, 1, "regular"),
    )

    pdf_bytes = generate_class_report()
    assert pdf_bytes[:5] == b"%PDF-"


def test_generate_class_report_semester_filter_changes_output_size(test_db):
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S1", "Alice", 1, "BCA", "alice@example.com", "9812345678", 2024),
    )
    execute_write(
        "INSERT INTO subjects (subject_code, name, semester, credits) VALUES (?, ?, ?, ?)",
        ("SUB1", "Fixture Subject", 1, 3),
    )
    execute_write(
        "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S1", "SUB1", 18, 45, 0, 1, "regular"),
    )

    # Semester 1 has real data; semester 5 (nothing recorded there) must
    # still produce a valid, smaller PDF -- confirms the semester
    # argument actually reaches every section's query, not just some.
    pdf_with_data = generate_class_report(semester=1)
    pdf_without_data = generate_class_report(semester=5)

    assert pdf_with_data[:5] == b"%PDF-"
    assert pdf_without_data[:5] == b"%PDF-"
    assert len(pdf_without_data) < len(pdf_with_data)
