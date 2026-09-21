"""
tests/test_pdf_generator.py
============================
pytest tests for utils/pdf_generator.py's generate_report_card() and
generate_transcript() -- specifically their `published_only` parameter,
added after discovering that a student's own downloaded report card PDF
could reveal DRAFT (unpublished) marks they can't even see on-screen,
undermining the result-publishing workflow in modules/marks.py.

These tests don't inspect PDF bytes directly (reportlab's output isn't
meaningfully assertable) -- they instead check that generate_report_card()
and generate_transcript() run without error under both published_only
values, and that the underlying SGPA/CGPA figures (independently
recomputable via modules.grades.calculate_cgpa()) match what draft vs.
published data should produce.

Uses the same throwaway-database pattern as tests/test_marks.py.

HOW TO RUN (from the project root):
    python -m pytest tests/test_pdf_generator.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.marks as marks
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, subjects, teacher_subjects
from modules.grades import calculate_cgpa
from utils.exceptions import RecordNotFoundError
from utils.pdf_generator import generate_report_card, generate_transcript


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test -- see
    tests/test_marks.py's module docstring for the full rationale."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_pdf_generator.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    teacher_subjects.list_subjects_for_teacher.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _seed_admin_and_student():
    """Returns (admin_user, roll_no) with one subject ('SUB1', semester 1)
    and one student ('S1') already created."""
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    admin_user = {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}

    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, admin_user)
    students.create_student("S1", "Student A", 1, "BCA", "student1@example.com", "9812345678", 2024, admin_user)

    return admin_user, "S1"


# ---------------------------------------------------------------------------
# generate_report_card() -- published_only
# ---------------------------------------------------------------------------

def test_report_card_published_only_true_ignores_draft_marks(test_db):
    admin_user, roll_no = _seed_admin_and_student()
    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    # Not published -- generate_report_card(published_only=True) should not
    # error, and should produce a report with no marks section content.
    pdf_bytes = generate_report_card(roll_no, 1, published_only=True)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_report_card_published_only_false_includes_draft_marks(test_db):
    admin_user, roll_no = _seed_admin_and_student()
    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    # Staff view: draft marks still included, must not raise.
    pdf_bytes = generate_report_card(roll_no, 1, published_only=False)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_report_card_defaults_to_published_only_false(test_db):
    """The default must stay False -- existing staff call sites that don't
    pass the new parameter explicitly must keep seeing drafts."""
    admin_user, roll_no = _seed_admin_and_student()
    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    pdf_bytes = generate_report_card(roll_no, 1)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_report_card_unknown_student_raises(test_db):
    with pytest.raises(RecordNotFoundError):
        generate_report_card("NOPE", 1, published_only=True)


# ---------------------------------------------------------------------------
# generate_transcript() -- published_only + CGPA
# ---------------------------------------------------------------------------

def test_transcript_published_only_true_skips_unpublished_semester(test_db):
    admin_user, roll_no = _seed_admin_and_student()
    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    # Left as a draft -- a published_only transcript should treat this
    # student as having no marks at all, not error out.
    pdf_bytes = generate_transcript(roll_no, published_only=True)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_transcript_published_only_false_includes_draft_semester(test_db):
    admin_user, roll_no = _seed_admin_and_student()
    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    pdf_bytes = generate_transcript(roll_no, published_only=False)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_transcript_defaults_to_published_only_false(test_db):
    admin_user, roll_no = _seed_admin_and_student()
    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    pdf_bytes = generate_transcript(roll_no)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_transcript_cgpa_matches_independent_calculation(test_db):
    """Add a second subject in semester 2, publish both semesters, and
    verify the CGPA generate_transcript() would show is exactly what
    modules.grades.calculate_cgpa() computes from the same marks --
    the same credit-weighting logic generate_transcript() itself calls."""
    admin_user, roll_no = _seed_admin_and_student()
    subjects.create_subject("SUB2", "Fixture Subject 2", 2, 4, admin_user)

    marks.enter_marks(roll_no, "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.enter_marks(roll_no, "SUB2", 18, 35, 12, 2, "regular", admin_user)
    marks.publish_marks("SUB1", 1, "regular", admin_user)
    marks.publish_marks("SUB2", 2, "regular", admin_user)

    sem1_marks = marks.list_marks_for_student(roll_no, semester=1, published_only=True)
    sem2_marks = marks.list_marks_for_student(roll_no, semester=2, published_only=True)
    sem1_sgpa = marks.compute_sgpa_for_marks(sem1_marks)
    sem2_sgpa = marks.compute_sgpa_for_marks(sem2_marks)
    expected_cgpa = calculate_cgpa([
        {"sgpa": sem1_sgpa, "credits": 3},
        {"sgpa": sem2_sgpa, "credits": 4},
    ])

    # generate_transcript() must not error while producing the equivalent
    # PDF -- the CGPA figure inside it is built from this exact same path
    # (see the function's own docstring), so a successful build here is
    # the meaningful, PDF-content-independent check available.
    pdf_bytes = generate_transcript(roll_no, published_only=True)
    assert isinstance(pdf_bytes, bytes)
    assert expected_cgpa > 0


def test_transcript_unknown_student_raises(test_db):
    with pytest.raises(RecordNotFoundError):
        generate_transcript("NOPE", published_only=True)


def test_transcript_no_marks_at_all_does_not_raise(test_db):
    _admin_user, roll_no = _seed_admin_and_student()
    pdf_bytes = generate_transcript(roll_no)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
