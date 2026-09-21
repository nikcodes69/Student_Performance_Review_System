"""
tests/test_announcements.py
============================
pytest tests for modules/announcements.py: posting, listing (by
role/audience), and taking down announcements.

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_announcements.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.announcements as announcements
import modules.auth as auth
from database.db_setup import create_indexes, create_tables, get_connection
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_announcements.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

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


# ---------------------------------------------------------------------------
# create_announcement()
# ---------------------------------------------------------------------------

def test_create_announcement_requires_admin_or_teacher(test_db):
    student = _student_user()
    with pytest.raises(AuthorizationError):
        announcements.create_announcement("Title", "Message", None, student)


def test_create_announcement_rejects_empty_title(test_db):
    admin = _admin_user()
    with pytest.raises(ValidationError):
        announcements.create_announcement("", "Message", None, admin)


def test_create_announcement_rejects_oversized_message(test_db):
    admin = _admin_user()
    too_long = "x" * (config.ANNOUNCEMENT_MESSAGE_MAX_LENGTH + 1)
    with pytest.raises(ValidationError):
        announcements.create_announcement("Title", too_long, None, admin)


def test_create_announcement_rejects_invalid_target_role(test_db):
    admin = _admin_user()
    with pytest.raises(ValidationError):
        announcements.create_announcement("Title", "Message", "superadmin", admin)


# ---------------------------------------------------------------------------
# list_announcements_for_role() -- audience filtering
# ---------------------------------------------------------------------------

def test_untargeted_announcement_visible_to_every_role(test_db):
    admin = _admin_user()
    announcements.create_announcement("Holiday", "College closed Monday.", None, admin)

    assert len(announcements.list_announcements_for_role(config.ROLE_ADMIN)) == 1
    assert len(announcements.list_announcements_for_role(config.ROLE_TEACHER)) == 1
    assert len(announcements.list_announcements_for_role(config.ROLE_STUDENT)) == 1


def test_targeted_announcement_only_visible_to_its_role(test_db):
    admin = _admin_user()
    announcements.create_announcement("Grades due", "Submit by Friday.", config.ROLE_TEACHER, admin)

    assert len(announcements.list_announcements_for_role(config.ROLE_TEACHER)) == 1
    assert len(announcements.list_announcements_for_role(config.ROLE_STUDENT)) == 0
    assert len(announcements.list_announcements_for_role(config.ROLE_ADMIN)) == 0


def test_list_announcements_orders_newest_first(test_db):
    admin = _admin_user()
    announcements.create_announcement("First", "Message 1", None, admin)
    announcements.create_announcement("Second", "Message 2", None, admin)

    results = announcements.list_announcements_for_role(config.ROLE_STUDENT)
    assert [entry["title"] for entry in results] == ["Second", "First"]


def test_list_announcements_excludes_inactive_by_default(test_db):
    admin = _admin_user()
    announcement_id = announcements.create_announcement("Old news", "Message", None, admin)
    announcements.deactivate_announcement(announcement_id, admin)

    assert announcements.list_announcements_for_role(config.ROLE_STUDENT) == []
    assert len(announcements.list_announcements_for_role(config.ROLE_STUDENT, include_inactive=True)) == 1


# ---------------------------------------------------------------------------
# deactivate_announcement()
# ---------------------------------------------------------------------------

def test_admin_can_deactivate_any_announcement(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    announcement_id = announcements.create_announcement("Notice", "Message", None, teacher)

    announcements.deactivate_announcement(announcement_id, admin)  # must not raise

    row = announcements.list_announcements_for_role(config.ROLE_STUDENT, include_inactive=True)[0]
    assert row["is_active"] == 0


def test_teacher_can_deactivate_their_own_announcement(test_db):
    teacher = _teacher_user()
    announcement_id = announcements.create_announcement("Notice", "Message", None, teacher)

    announcements.deactivate_announcement(announcement_id, teacher)  # must not raise


def test_teacher_cannot_deactivate_another_teachers_announcement(test_db):
    teacher1 = _teacher_user("teach1")
    teacher2 = _teacher_user("teach2")
    announcement_id = announcements.create_announcement("Notice", "Message", None, teacher1)

    with pytest.raises(AuthorizationError):
        announcements.deactivate_announcement(announcement_id, teacher2)


def test_deactivate_rejects_unknown_announcement(test_db):
    admin = _admin_user()
    with pytest.raises(RecordNotFoundError):
        announcements.deactivate_announcement(9999, admin)


def test_deactivate_rejects_already_inactive_announcement(test_db):
    admin = _admin_user()
    announcement_id = announcements.create_announcement("Notice", "Message", None, admin)
    announcements.deactivate_announcement(announcement_id, admin)

    with pytest.raises(ValidationError):
        announcements.deactivate_announcement(announcement_id, admin)


# ---------------------------------------------------------------------------
# mark_announcement_read() / get_unread_announcement_count()
# ---------------------------------------------------------------------------

def test_new_announcement_starts_unread(test_db):
    admin = _admin_user()
    student = _student_user()
    announcements.create_announcement("Notice", "Message", None, admin)

    assert announcements.get_unread_announcement_count(student) == 1
    entries = announcements.list_announcements_for_role(
        config.ROLE_STUDENT, viewer_user_id=student["user_id"],
    )
    assert entries[0]["is_read"] == 0


def test_mark_announcement_read_clears_unread_count(test_db):
    admin = _admin_user()
    student = _student_user()
    announcement_id = announcements.create_announcement("Notice", "Message", None, admin)

    announcements.mark_announcement_read(announcement_id, student)

    assert announcements.get_unread_announcement_count(student) == 0
    entries = announcements.list_announcements_for_role(
        config.ROLE_STUDENT, viewer_user_id=student["user_id"],
    )
    assert entries[0]["is_read"] == 1


def test_mark_announcement_read_twice_is_a_harmless_no_op(test_db):
    admin = _admin_user()
    student = _student_user()
    announcement_id = announcements.create_announcement("Notice", "Message", None, admin)

    announcements.mark_announcement_read(announcement_id, student)
    announcements.mark_announcement_read(announcement_id, student)  # must not raise

    assert announcements.get_unread_announcement_count(student) == 0


def test_read_status_is_per_user(test_db):
    admin = _admin_user()
    student1 = _student_user()
    student2 = auth.create_user("student2", "StudentPass1!", config.ROLE_STUDENT)
    student2_user = {"user_id": student2, "role": config.ROLE_STUDENT, "username": "student2"}
    announcement_id = announcements.create_announcement("Notice", "Message", None, admin)

    announcements.mark_announcement_read(announcement_id, student1)

    assert announcements.get_unread_announcement_count(student1) == 0
    assert announcements.get_unread_announcement_count(student2_user) == 1


def test_unread_count_ignores_announcements_outside_role_audience(test_db):
    admin = _admin_user()
    student = _student_user()
    announcements.create_announcement("Teacher notice", "Message", config.ROLE_TEACHER, admin)

    assert announcements.get_unread_announcement_count(student) == 0


def test_unread_count_ignores_inactive_announcements(test_db):
    admin = _admin_user()
    student = _student_user()
    announcement_id = announcements.create_announcement("Notice", "Message", None, admin)
    announcements.deactivate_announcement(announcement_id, admin)

    assert announcements.get_unread_announcement_count(student) == 0
