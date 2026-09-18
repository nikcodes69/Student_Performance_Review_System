"""
tests/test_teacher_subjects.py
===============================
pytest tests for modules/teacher_subjects.py: the teacher-to-subject
assignment mapping that closes the "any Teacher can enter data for any
subject" gap documented as a known limitation in earlier versions of
this project's README.

Uses the SAME throwaway-database pattern as tests/test_database.py and
tests/test_auth.py -- see those files' module docstrings for the full
rationale (never the real local database, never the real Turso
database).

WHY list_subjects_for_teacher() IS CLEARED IN THE FIXTURE: it is
@st.cache_data-decorated -- see tests/test_analytics.py's module
docstring for why every cached function touched by a test file must be
cleared before that test's own database is seeded, to avoid a stale
result leaking in from an earlier test's throwaway database.

HOW TO RUN (from the project root):
    python -m pytest tests/test_teacher_subjects.py -v
"""

import pytest

import config
import database.db_setup as db_setup
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, marks, students, subjects, teacher_subjects
from utils.exceptions import AuthorizationError, DuplicateRecordError, RecordNotFoundError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test, with
    teacher_subjects' cached lookup cleared first -- see module docstring."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_teacher_subjects.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    teacher_subjects.list_subjects_for_teacher.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _seed_admin_teacher_and_subject():
    """Returns (admin_user, teacher_user) dicts, with one subject
    ('SUB1') and one student ('S1') already created."""
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    admin_user = {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}
    teacher_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}

    students.create_student("S1", "Alice", 1, "BCA", "alice@example.com", "9812345678", 2024, admin_user)
    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, admin_user)

    return admin_user, teacher_user


# ---------------------------------------------------------------------------
# assign_teacher_to_subject() / unassign_teacher_from_subject()
# ---------------------------------------------------------------------------

def test_assign_and_list_subjects_for_teacher(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()

    assert teacher_subjects.list_subjects_for_teacher(teacher_user["user_id"]) == []

    teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", admin_user)

    assigned = teacher_subjects.list_subjects_for_teacher(teacher_user["user_id"])
    assert [s["subject_code"] for s in assigned] == ["SUB1"]


def test_assign_rejects_duplicate_assignment(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()
    teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", admin_user)
    with pytest.raises(DuplicateRecordError):
        teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", admin_user)


def test_assign_rejects_unknown_teacher(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_and_subject()
    with pytest.raises(RecordNotFoundError):
        teacher_subjects.assign_teacher_to_subject(9999, "SUB1", admin_user)


def test_assign_rejects_non_teacher_account(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_and_subject()
    with pytest.raises(RecordNotFoundError):
        teacher_subjects.assign_teacher_to_subject(admin_user["user_id"], "SUB1", admin_user)


def test_assign_requires_admin_role(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()
    with pytest.raises(AuthorizationError):
        teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", teacher_user)


def test_unassign_removes_access(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()
    teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", admin_user)
    teacher_subjects.unassign_teacher_from_subject(teacher_user["user_id"], "SUB1", admin_user)
    assert teacher_subjects.list_subjects_for_teacher(teacher_user["user_id"]) == []


def test_unassign_rejects_nonexistent_assignment(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()
    with pytest.raises(RecordNotFoundError):
        teacher_subjects.unassign_teacher_from_subject(teacher_user["user_id"], "SUB1", admin_user)


# ---------------------------------------------------------------------------
# check_teacher_subject_access() -- the business-logic enforcement layer
# ---------------------------------------------------------------------------

def test_check_access_is_noop_for_admin(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_and_subject()
    teacher_subjects.check_teacher_subject_access(admin_user, "SUB1")  # must not raise, even unassigned


def test_check_access_blocks_unassigned_teacher(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_and_subject()
    with pytest.raises(AuthorizationError):
        teacher_subjects.check_teacher_subject_access(teacher_user, "SUB1")


def test_check_access_allows_assigned_teacher(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()
    teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", admin_user)
    teacher_subjects.check_teacher_subject_access(teacher_user, "SUB1")  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: enforcement inside modules/marks.py's write functions
# ---------------------------------------------------------------------------

def test_enter_marks_blocks_unassigned_teacher(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_and_subject()
    with pytest.raises(AuthorizationError):
        marks.enter_marks("S1", "SUB1", 15, 45, 0, 1, "regular", teacher_user)


def test_enter_marks_allows_assigned_teacher(test_db):
    admin_user, teacher_user = _seed_admin_teacher_and_subject()
    teacher_subjects.assign_teacher_to_subject(teacher_user["user_id"], "SUB1", admin_user)
    mark_id = marks.enter_marks("S1", "SUB1", 15, 45, 0, 1, "regular", teacher_user)
    assert mark_id is not None


def test_enter_marks_never_restricted_for_admin(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_and_subject()
    mark_id = marks.enter_marks("S1", "SUB1", 15, 45, 0, 1, "regular", admin_user)
    assert mark_id is not None
