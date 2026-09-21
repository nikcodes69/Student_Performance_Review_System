"""
tests/test_academic_calendar.py
=================================
pytest tests for modules/academic_calendar.py: publishing, listing
(chronological order), and taking down academic calendar events.

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_academic_calendar.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.academic_calendar as academic_calendar
import modules.auth as auth
from database.db_setup import create_indexes, create_tables, get_connection
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_academic_calendar.db")
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


def _teacher_user() -> dict:
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    return {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}


def _student_user() -> dict:
    student_id = auth.create_user("student1", "StudentPass1!", config.ROLE_STUDENT)
    return {"user_id": student_id, "role": config.ROLE_STUDENT, "username": "student1"}


# ---------------------------------------------------------------------------
# create_calendar_event()
# ---------------------------------------------------------------------------

def test_create_calendar_event_requires_admin(test_db):
    teacher = _teacher_user()
    with pytest.raises(AuthorizationError):
        academic_calendar.create_calendar_event(
            "Mid-terms", None, config.CALENDAR_EVENT_EXAM, "2026-01-10", None, teacher,
        )


def test_admin_can_create_single_day_event(test_db):
    admin = _admin_user()
    event_id = academic_calendar.create_calendar_event(
        "Republic Day", None, config.CALENDAR_EVENT_HOLIDAY, "2026-01-15", None, admin,
    )

    events = academic_calendar.list_calendar_events()
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["end_date"] is None


def test_create_calendar_event_rejects_empty_title(test_db):
    admin = _admin_user()
    with pytest.raises(ValidationError):
        academic_calendar.create_calendar_event(
            "", None, config.CALENDAR_EVENT_HOLIDAY, "2026-01-15", None, admin,
        )


def test_create_calendar_event_rejects_invalid_event_type(test_db):
    admin = _admin_user()
    with pytest.raises(ValidationError):
        academic_calendar.create_calendar_event(
            "Title", None, "not-a-real-type", "2026-01-15", None, admin,
        )


def test_create_calendar_event_rejects_malformed_date(test_db):
    admin = _admin_user()
    with pytest.raises(ValidationError):
        academic_calendar.create_calendar_event(
            "Title", None, config.CALENDAR_EVENT_HOLIDAY, "15-01-2026", None, admin,
        )


def test_create_calendar_event_rejects_end_before_start(test_db):
    admin = _admin_user()
    with pytest.raises(ValidationError):
        academic_calendar.create_calendar_event(
            "Exams", None, config.CALENDAR_EVENT_EXAM, "2026-01-20", "2026-01-10", admin,
        )


def test_create_calendar_event_accepts_multi_day_range(test_db):
    admin = _admin_user()
    event_id = academic_calendar.create_calendar_event(
        "Mid-terms", "Written exams for all subjects.",
        config.CALENDAR_EVENT_EXAM, "2026-02-01", "2026-02-10", admin,
    )

    events = academic_calendar.list_calendar_events()
    assert events[0]["event_id"] == event_id
    assert events[0]["start_date"] == "2026-02-01"
    assert events[0]["end_date"] == "2026-02-10"


def test_create_calendar_event_rejects_oversized_description(test_db):
    admin = _admin_user()
    too_long = "x" * (config.CALENDAR_EVENT_DESCRIPTION_MAX_LENGTH + 1)
    with pytest.raises(ValidationError):
        academic_calendar.create_calendar_event(
            "Title", too_long, config.CALENDAR_EVENT_HOLIDAY, "2026-01-15", None, admin,
        )


# ---------------------------------------------------------------------------
# list_calendar_events() -- chronological ordering
# ---------------------------------------------------------------------------

def test_list_calendar_events_orders_soonest_first(test_db):
    admin = _admin_user()
    academic_calendar.create_calendar_event(
        "Later event", None, config.CALENDAR_EVENT_OTHER, "2026-06-01", None, admin,
    )
    academic_calendar.create_calendar_event(
        "Sooner event", None, config.CALENDAR_EVENT_OTHER, "2026-01-01", None, admin,
    )

    results = academic_calendar.list_calendar_events()
    assert [entry["title"] for entry in results] == ["Sooner event", "Later event"]


def test_list_calendar_events_excludes_inactive_by_default(test_db):
    admin = _admin_user()
    event_id = academic_calendar.create_calendar_event(
        "Old event", None, config.CALENDAR_EVENT_OTHER, "2026-01-01", None, admin,
    )
    academic_calendar.deactivate_calendar_event(event_id, admin)

    assert academic_calendar.list_calendar_events() == []
    assert len(academic_calendar.list_calendar_events(include_inactive=True)) == 1


# ---------------------------------------------------------------------------
# deactivate_calendar_event()
# ---------------------------------------------------------------------------

def test_deactivate_calendar_event_requires_admin(test_db):
    admin = _admin_user()
    teacher = _teacher_user()
    event_id = academic_calendar.create_calendar_event(
        "Title", None, config.CALENDAR_EVENT_OTHER, "2026-01-01", None, admin,
    )

    with pytest.raises(AuthorizationError):
        academic_calendar.deactivate_calendar_event(event_id, teacher)


def test_deactivate_calendar_event_requires_admin_not_student(test_db):
    admin = _admin_user()
    student = _student_user()
    event_id = academic_calendar.create_calendar_event(
        "Title", None, config.CALENDAR_EVENT_OTHER, "2026-01-01", None, admin,
    )

    with pytest.raises(AuthorizationError):
        academic_calendar.deactivate_calendar_event(event_id, student)


def test_deactivate_rejects_unknown_event(test_db):
    admin = _admin_user()
    with pytest.raises(RecordNotFoundError):
        academic_calendar.deactivate_calendar_event(9999, admin)


def test_deactivate_rejects_already_inactive_event(test_db):
    admin = _admin_user()
    event_id = academic_calendar.create_calendar_event(
        "Title", None, config.CALENDAR_EVENT_OTHER, "2026-01-01", None, admin,
    )
    academic_calendar.deactivate_calendar_event(event_id, admin)

    with pytest.raises(ValidationError):
        academic_calendar.deactivate_calendar_event(event_id, admin)
