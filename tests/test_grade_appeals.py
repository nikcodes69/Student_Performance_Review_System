"""
tests/test_grade_appeals.py
============================
pytest tests for modules/grade_appeals.py: submitting an appeal against
a published mark, and a Teacher/Admin reviewer resolving it.

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_grade_appeals.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.grade_appeals as grade_appeals
import modules.marks as marks
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, subjects, teacher_subjects
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test, with
    teacher_subjects' cached lookup cleared first -- see
    tests/test_teacher_subjects.py's module docstring for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_grade_appeals.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    teacher_subjects.list_subjects_for_teacher.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _seed_with_published_mark():
    """Admin, Teacher (assigned to SUB1), one student (BCA1) with a
    PUBLISHED mark in SUB1/semester 1/regular. Returns
    (admin_user, teacher_user, student_user)."""
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    admin_user = {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}
    teacher_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}

    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, admin_user)
    teacher_subjects.assign_teacher_to_subject(teacher_id, "SUB1", admin_user)
    students.create_student("BCA1", "Alice", 1, "BCA", "alice@example.com", "9812345678", 2024, admin_user)

    student_id = auth.create_user("BCA1", "StudentPass1!", config.ROLE_STUDENT, force_password_change=False)
    student_user = {"user_id": student_id, "role": config.ROLE_STUDENT, "username": "BCA1"}

    marks.enter_marks("BCA1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.publish_marks("SUB1", 1, "regular", admin_user)

    return admin_user, teacher_user, student_user


# ---------------------------------------------------------------------------
# submit_appeal()
# ---------------------------------------------------------------------------

def test_submit_appeal_succeeds_for_a_published_mark(test_db):
    _admin_user, _teacher_user, student_user = _seed_with_published_mark()

    appeal_id = grade_appeals.submit_appeal(
        "BCA1", "SUB1", 1, "regular", "I think my practical marks were miscounted.", student_user,
    )

    appeals = grade_appeals.list_appeals_for_student("BCA1")
    assert len(appeals) == 1
    assert appeals[0]["appeal_id"] == appeal_id
    assert appeals[0]["status"] == config.APPEAL_PENDING


def test_submit_appeal_rejects_unpublished_mark(test_db):
    admin_user, _teacher_user, student_user = _seed_with_published_mark()
    # A second exam_type entry that was never published.
    marks.enter_marks("BCA1", "SUB1", 20, 40, 15, 1, "backlog", admin_user)

    with pytest.raises(ValidationError):
        grade_appeals.submit_appeal("BCA1", "SUB1", 1, "backlog", "Reason", student_user)


def test_submit_appeal_rejects_another_students_roll_no(test_db):
    _admin_user, _teacher_user, student_user = _seed_with_published_mark()

    with pytest.raises(AuthorizationError):
        grade_appeals.submit_appeal("BCA2", "SUB1", 1, "regular", "Reason", student_user)


def test_submit_appeal_rejects_duplicate_pending_appeal(test_db):
    _admin_user, _teacher_user, student_user = _seed_with_published_mark()
    grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "First reason", student_user)

    with pytest.raises(ValidationError):
        grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Second reason", student_user)


def test_submit_appeal_requires_student_role(test_db):
    admin_user, _teacher_user, _student_user = _seed_with_published_mark()

    with pytest.raises(AuthorizationError):
        grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", admin_user)


# ---------------------------------------------------------------------------
# list_appeals_for_reviewer()
# ---------------------------------------------------------------------------

def test_teacher_sees_pending_appeals_for_their_own_subject(test_db):
    _admin_user, teacher_user, student_user = _seed_with_published_mark()
    grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    pending = grade_appeals.list_appeals_for_reviewer(teacher_user)
    assert len(pending) == 1
    assert pending[0]["subject_code"] == "SUB1"


def test_teacher_does_not_see_appeals_for_unassigned_subjects(test_db):
    admin_user, _teacher_user, student_user = _seed_with_published_mark()
    grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    other_teacher_id = auth.create_user("teach2", "TeachPass1!", config.ROLE_TEACHER)
    other_teacher = {"user_id": other_teacher_id, "role": config.ROLE_TEACHER, "username": "teach2"}

    assert grade_appeals.list_appeals_for_reviewer(other_teacher) == []


def test_admin_sees_every_pending_appeal(test_db):
    admin_user, _teacher_user, student_user = _seed_with_published_mark()
    grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    assert len(grade_appeals.list_appeals_for_reviewer(admin_user)) == 1


# ---------------------------------------------------------------------------
# respond_to_appeal()
# ---------------------------------------------------------------------------

def test_respond_to_appeal_approves_and_records_response(test_db):
    _admin_user, teacher_user, student_user = _seed_with_published_mark()
    appeal_id = grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    grade_appeals.respond_to_appeal(appeal_id, True, "You're right, fixing it.", teacher_user)

    appeal = grade_appeals.list_appeals_for_student("BCA1")[0]
    assert appeal["status"] == config.APPEAL_APPROVED
    assert appeal["response"] == "You're right, fixing it."
    assert appeal["reviewed_by"] == teacher_user["user_id"]


def test_respond_to_appeal_rejects_with_response(test_db):
    _admin_user, teacher_user, student_user = _seed_with_published_mark()
    appeal_id = grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    grade_appeals.respond_to_appeal(appeal_id, False, "Marks were correct as entered.", teacher_user)

    appeal = grade_appeals.list_appeals_for_student("BCA1")[0]
    assert appeal["status"] == config.APPEAL_REJECTED


def test_respond_to_appeal_does_not_change_marks(test_db):
    # Confirms this module's own documented boundary: approving an
    # appeal is a communication/tracking action only, never an implicit
    # marks correction.
    _admin_user, teacher_user, student_user = _seed_with_published_mark()
    appeal_id = grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    grade_appeals.respond_to_appeal(appeal_id, True, "Approved.", teacher_user)

    entry = marks.get_marks_entry("BCA1", "SUB1", 1, "regular")
    assert (entry["internal"], entry["external"], entry["practical"]) == (15, 30, 10)


def test_respond_to_appeal_rejects_unassigned_teacher(test_db):
    _admin_user, _teacher_user, student_user = _seed_with_published_mark()
    appeal_id = grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)

    other_teacher_id = auth.create_user("teach2", "TeachPass1!", config.ROLE_TEACHER)
    other_teacher = {"user_id": other_teacher_id, "role": config.ROLE_TEACHER, "username": "teach2"}

    with pytest.raises(AuthorizationError):
        grade_appeals.respond_to_appeal(appeal_id, True, "Response", other_teacher)


def test_respond_to_appeal_rejects_already_resolved(test_db):
    _admin_user, teacher_user, student_user = _seed_with_published_mark()
    appeal_id = grade_appeals.submit_appeal("BCA1", "SUB1", 1, "regular", "Reason", student_user)
    grade_appeals.respond_to_appeal(appeal_id, True, "Approved.", teacher_user)

    with pytest.raises(ValidationError):
        grade_appeals.respond_to_appeal(appeal_id, False, "Too late.", teacher_user)


def test_respond_to_appeal_rejects_unknown_appeal(test_db):
    _admin_user, teacher_user, _student_user = _seed_with_published_mark()
    with pytest.raises(RecordNotFoundError):
        grade_appeals.respond_to_appeal(9999, True, "Response", teacher_user)
