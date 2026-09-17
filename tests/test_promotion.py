"""
tests/test_promotion.py
========================
pytest tests for modules/students.py's promote_students() -- the
Semester Promotion tool on the Students page (Admin only): advances
every active student in one semester to the next, all in one atomic
transaction.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

WHY students.list_students.clear() IS CALLED IN THE FIXTURE:
list_students() is @st.cache_data-decorated -- see
tests/test_at_risk_report.py's module docstring for the cross-test
cache-leak bug this exact clearing step was added to prevent.

HOW TO RUN (from the project root):
    python -m pytest tests/test_promotion.py -v
"""

import pytest

import config
import database.db_setup as db_setup
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students
from utils.exceptions import AuthorizationError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database, with
    list_students()'s cache cleared first -- see module docstring."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_promotion.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    students.list_students.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _admin_user() -> dict:
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    return {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}


def test_promote_students_moves_matching_active_students(test_db):
    admin_user = _admin_user()
    students.create_student("S1", "Alice", 1, "BCA", "a@example.com", "9812345678", 2024, admin_user)
    students.create_student("S2", "Bob", 1, "BCA", "b@example.com", "9812345679", 2024, admin_user)

    promoted = students.promote_students(1, admin_user)

    assert sorted(promoted) == ["S1", "S2"]
    assert students.get_student("S1")["semester"] == 2
    assert students.get_student("S2")["semester"] == 2


def test_promote_students_does_not_touch_other_semesters(test_db):
    admin_user = _admin_user()
    students.create_student("S1", "Alice", 1, "BCA", "a@example.com", "9812345678", 2024, admin_user)
    students.create_student("S3", "Carol", 2, "BCA", "c@example.com", "9812345680", 2024, admin_user)

    students.promote_students(1, admin_user)

    assert students.get_student("S1")["semester"] == 2  # promoted
    assert students.get_student("S3")["semester"] == 2  # already was 2, untouched


def test_promote_students_does_not_touch_inactive_students(test_db):
    admin_user = _admin_user()
    students.create_student("S1", "Alice", 1, "BCA", "a@example.com", "9812345678", 2024, admin_user)
    students.deactivate_student("S1", admin_user)

    promoted = students.promote_students(1, admin_user)

    assert promoted == []
    assert students.get_student("S1", include_inactive=True)["semester"] == 1


def test_promote_students_returns_empty_list_when_nothing_to_promote(test_db):
    admin_user = _admin_user()
    assert students.promote_students(1, admin_user) == []


def test_promote_students_rejects_max_semester(test_db):
    admin_user = _admin_user()
    with pytest.raises(ValidationError):
        students.promote_students(config.MAX_SEMESTER, admin_user)


def test_promote_students_requires_admin_role(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    teacher_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}

    with pytest.raises(AuthorizationError):
        students.promote_students(1, teacher_user)
