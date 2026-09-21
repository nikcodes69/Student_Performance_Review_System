"""
tests/test_teacher_remarks.py
==============================
pytest tests for modules/teacher_remarks.py: leaving a remark on a
student's record (general or tied to a specific subject), listing them,
and taking one down.

Uses the same throwaway-database pattern as tests/test_marks.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_teacher_remarks.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.teacher_remarks as teacher_remarks
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, subjects, teacher_subjects
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test, with
    teacher_subjects' cached lookup cleared first -- see
    tests/test_teacher_subjects.py's module docstring for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_teacher_remarks.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

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


def _teacher_user(username: str = "teach1") -> dict:
    teacher_id = auth.create_user(username, "TeachPass1!", config.ROLE_TEACHER)
    return {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": username}


def _student_user() -> dict:
    student_id = auth.create_user("student1", "StudentPass1!", config.ROLE_STUDENT)
    return {"user_id": student_id, "role": config.ROLE_STUDENT, "username": "student1"}


def _seed_student(admin: dict, roll_no: str = "S1") -> None:
    students.create_student(roll_no, "Student A", 1, "BCA", "student1@example.com", "9812345678", 2024, admin)


# ---------------------------------------------------------------------------
# create_remark()
# ---------------------------------------------------------------------------

def test_create_remark_requires_admin_or_teacher(test_db):
    admin = _admin_user()
    _seed_student(admin)
    student = _student_user()

    with pytest.raises(AuthorizationError):
        teacher_remarks.create_remark("S1", None, "Keep it up.", student)


def test_create_general_remark_needs_no_subject(test_db):
    admin = _admin_user()
    _seed_student(admin)

    remark_id = teacher_remarks.create_remark("S1", None, "Great improvement.", admin)

    entries = teacher_remarks.list_remarks_for_student("S1")
    assert len(entries) == 1
    assert entries[0]["remark_id"] == remark_id
    assert entries[0]["subject_name"] is None


def test_create_remark_rejects_empty_text(test_db):
    admin = _admin_user()
    _seed_student(admin)

    with pytest.raises(ValidationError):
        teacher_remarks.create_remark("S1", None, "", admin)


def test_create_remark_rejects_oversized_text(test_db):
    admin = _admin_user()
    _seed_student(admin)
    too_long = "x" * (config.TEACHER_REMARK_MAX_LENGTH + 1)

    with pytest.raises(ValidationError):
        teacher_remarks.create_remark("S1", None, too_long, admin)


def test_create_remark_rejects_unknown_student(test_db):
    admin = _admin_user()
    with pytest.raises(RecordNotFoundError):
        teacher_remarks.create_remark("NOPE", None, "Message.", admin)


def test_teacher_can_leave_remark_on_assigned_subject(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    _seed_student(admin)
    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, admin)
    teacher_subjects.assign_teacher_to_subject(teacher["user_id"], "SUB1", admin)

    remark_id = teacher_remarks.create_remark("S1", "SUB1", "Strong practical work.", teacher)

    entries = teacher_remarks.list_remarks_for_student("S1")
    assert entries[0]["remark_id"] == remark_id
    assert entries[0]["subject_name"] == "Fixture Subject"


def test_teacher_cannot_leave_remark_on_unassigned_subject(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    _seed_student(admin)
    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, admin)
    # Deliberately NOT assigned to teacher.

    with pytest.raises(AuthorizationError):
        teacher_remarks.create_remark("S1", "SUB1", "Message.", teacher)


def test_teacher_can_leave_general_remark_without_any_assignment(test_db):
    """No subject_code means no subject-scoping applies -- see the
    module's docstring for why a general remark is unrestricted, unlike
    a subject-tied one."""
    admin = _admin_user()
    teacher = _teacher_user()
    _seed_student(admin)

    teacher_remarks.create_remark("S1", None, "General note.", teacher)  # must not raise


# ---------------------------------------------------------------------------
# list_remarks_for_student()
# ---------------------------------------------------------------------------

def test_list_remarks_orders_newest_first(test_db):
    admin = _admin_user()
    _seed_student(admin)
    teacher_remarks.create_remark("S1", None, "First remark.", admin)
    teacher_remarks.create_remark("S1", None, "Second remark.", admin)

    results = teacher_remarks.list_remarks_for_student("S1")
    assert [entry["remark"] for entry in results] == ["Second remark.", "First remark."]


def test_list_remarks_excludes_inactive_by_default(test_db):
    admin = _admin_user()
    _seed_student(admin)
    remark_id = teacher_remarks.create_remark("S1", None, "Old remark.", admin)
    teacher_remarks.deactivate_remark(remark_id, admin)

    assert teacher_remarks.list_remarks_for_student("S1") == []
    assert len(teacher_remarks.list_remarks_for_student("S1", include_inactive=True)) == 1


def test_list_remarks_scoped_to_one_student(test_db):
    admin = _admin_user()
    _seed_student(admin, "S1")
    _seed_student(admin, "S2")
    teacher_remarks.create_remark("S1", None, "About S1.", admin)
    teacher_remarks.create_remark("S2", None, "About S2.", admin)

    assert len(teacher_remarks.list_remarks_for_student("S1")) == 1
    assert len(teacher_remarks.list_remarks_for_student("S2")) == 1


# ---------------------------------------------------------------------------
# deactivate_remark()
# ---------------------------------------------------------------------------

def test_admin_can_deactivate_any_remark(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    _seed_student(admin)
    remark_id = teacher_remarks.create_remark("S1", None, "Message.", teacher)

    teacher_remarks.deactivate_remark(remark_id, admin)  # must not raise

    row = teacher_remarks.list_remarks_for_student("S1", include_inactive=True)[0]
    assert row["is_active"] == 0


def test_teacher_can_deactivate_their_own_remark(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    _seed_student(admin)
    remark_id = teacher_remarks.create_remark("S1", None, "Message.", teacher)

    teacher_remarks.deactivate_remark(remark_id, teacher)  # must not raise


def test_teacher_cannot_deactivate_another_teachers_remark(test_db):
    admin = _admin_user()
    teacher1 = _teacher_user("teach1")
    teacher2 = _teacher_user("teach2")
    _seed_student(admin)
    remark_id = teacher_remarks.create_remark("S1", None, "Message.", teacher1)

    with pytest.raises(AuthorizationError):
        teacher_remarks.deactivate_remark(remark_id, teacher2)


def test_deactivate_rejects_unknown_remark(test_db):
    admin = _admin_user()
    with pytest.raises(RecordNotFoundError):
        teacher_remarks.deactivate_remark(9999, admin)


def test_deactivate_rejects_already_inactive_remark(test_db):
    admin = _admin_user()
    _seed_student(admin)
    remark_id = teacher_remarks.create_remark("S1", None, "Message.", admin)
    teacher_remarks.deactivate_remark(remark_id, admin)

    with pytest.raises(ValidationError):
        teacher_remarks.deactivate_remark(remark_id, admin)
