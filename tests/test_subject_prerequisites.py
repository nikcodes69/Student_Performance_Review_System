"""
tests/test_subject_prerequisites.py
=====================================
pytest tests for subject prerequisites: modules/subjects.py's
set_subject_prerequisite() (setting/clearing/validating one), and
modules/marks.py's enter_marks() enforcing it (has_passed_subject()).

Uses the same throwaway-database pattern as tests/test_marks.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_subject_prerequisites.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.marks as marks
import modules.subjects as subjects
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, teacher_subjects
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test, with
    every @st.cache_data-decorated lookup this file touches cleared first
    -- see tests/test_teacher_subjects.py's module docstring for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_subject_prerequisites.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    subjects.list_subjects.clear()
    teacher_subjects.list_subjects_for_teacher.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _admin_user() -> dict:
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    return {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}


def _teacher_user() -> dict:
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    return {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}


def _seed_two_subjects_and_student(admin: dict) -> None:
    """SUB1 (semester 1) and SUB2 (semester 2), plus one student S1."""
    subjects.create_subject("SUB1", "Fixture Subject 1", 1, 3, admin)
    subjects.create_subject("SUB2", "Fixture Subject 2", 2, 3, admin)
    students.create_student("S1", "Student A", 2, "BCA", "student1@example.com", "9812345678", 2024, admin)


# ---------------------------------------------------------------------------
# set_subject_prerequisite()
# ---------------------------------------------------------------------------

def test_set_prerequisite_requires_admin(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    _seed_two_subjects_and_student(admin)

    with pytest.raises(AuthorizationError):
        subjects.set_subject_prerequisite("SUB2", "SUB1", teacher)


def test_admin_can_set_a_valid_prerequisite(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)

    subjects.set_subject_prerequisite("SUB2", "SUB1", admin)

    assert subjects.get_subject("SUB2")["prerequisite_subject_code"] == "SUB1"


def test_admin_can_clear_a_prerequisite(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)
    subjects.set_subject_prerequisite("SUB2", "SUB1", admin)

    subjects.set_subject_prerequisite("SUB2", None, admin)

    assert subjects.get_subject("SUB2")["prerequisite_subject_code"] is None


def test_prerequisite_rejects_same_subject(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)

    with pytest.raises(ValidationError):
        subjects.set_subject_prerequisite("SUB1", "SUB1", admin)


def test_prerequisite_rejects_same_or_later_semester(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)

    # SUB1 is semester 1, SUB2 is semester 2 -- SUB1 cannot require SUB2
    # (a later semester) as its prerequisite.
    with pytest.raises(ValidationError):
        subjects.set_subject_prerequisite("SUB1", "SUB2", admin)


def test_prerequisite_rejects_unknown_subject(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)

    with pytest.raises(RecordNotFoundError):
        subjects.set_subject_prerequisite("SUB2", "NOPE", admin)


def test_set_prerequisite_rejects_unknown_target_subject(test_db):
    admin = _admin_user()
    with pytest.raises(RecordNotFoundError):
        subjects.set_subject_prerequisite("NOPE", None, admin)


# ---------------------------------------------------------------------------
# enter_marks() enforcing the prerequisite gate
# ---------------------------------------------------------------------------

def test_enter_marks_blocked_when_prerequisite_not_passed(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)
    subjects.set_subject_prerequisite("SUB2", "SUB1", admin)

    with pytest.raises(ValidationError):
        marks.enter_marks("S1", "SUB2", 15, 30, 10, 2, "regular", admin)


def test_enter_marks_allowed_once_prerequisite_passed(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)
    subjects.set_subject_prerequisite("SUB2", "SUB1", admin)

    # A clearly passing mark in SUB1 (semester 1) first.
    marks.enter_marks("S1", "SUB1", 20, 40, 20, 1, "regular", admin)

    marks.enter_marks("S1", "SUB2", 15, 30, 10, 2, "regular", admin)  # must not raise
    assert marks.get_marks_entry("S1", "SUB2", 2, "regular") is not None


def test_enter_marks_still_blocked_after_a_failing_attempt(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)
    subjects.set_subject_prerequisite("SUB2", "SUB1", admin)

    # A clearly failing mark in SUB1.
    marks.enter_marks("S1", "SUB1", 2, 5, 2, 1, "regular", admin)

    with pytest.raises(ValidationError):
        marks.enter_marks("S1", "SUB2", 15, 30, 10, 2, "regular", admin)


def test_enter_marks_unaffected_when_subject_has_no_prerequisite(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)
    # No prerequisite set on SUB2 at all.

    marks.enter_marks("S1", "SUB2", 15, 30, 10, 2, "regular", admin)  # must not raise


def test_has_passed_subject_true_for_a_backlog_pass(test_db):
    admin = _admin_user()
    _seed_two_subjects_and_student(admin)

    marks.enter_marks("S1", "SUB1", 2, 5, 2, 1, "regular", admin)  # fails
    marks.enter_marks("S1", "SUB1", 20, 40, 20, 1, "backlog", admin)  # passes on retake

    assert marks.has_passed_subject("S1", "SUB1") is True
