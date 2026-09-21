"""
tests/test_audit.py
====================
pytest tests for modules/audit.py: build_audit_entry()/record_change()
(writing entries) and get_audit_logs() (reading them back, with its
table_name/record_id/user_id filters).

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_audit.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.audit as audit
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth
from utils.exceptions import ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_audit.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _admin_user_id() -> int:
    return auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)


# ---------------------------------------------------------------------------
# build_audit_entry() / record_change()
# ---------------------------------------------------------------------------

def test_build_audit_entry_rejects_unknown_action(test_db):
    with pytest.raises(ValidationError):
        audit.build_audit_entry(1, "DELETE", "students", "BCA001")


def test_build_audit_entry_rejects_unknown_table(test_db):
    with pytest.raises(ValidationError):
        audit.build_audit_entry(1, config.AUDIT_INSERT, "not_a_real_table", "BCA001")


def test_record_change_writes_a_retrievable_entry(test_db):
    admin_id = _admin_user_id()

    log_id = audit.record_change(
        admin_id, config.AUDIT_INSERT, "students", "BCA001", old_value=None, new_value={"name": "Alice"},
    )

    entries = audit.get_audit_logs(table_name="students", record_id="BCA001")
    assert len(entries) == 1
    assert entries[0]["log_id"] == log_id
    assert entries[0]["new_value"] == {"name": "Alice"}
    assert entries[0]["old_value"] is None


# ---------------------------------------------------------------------------
# get_audit_logs() filters
# ---------------------------------------------------------------------------

def test_get_audit_logs_filters_by_table_name(test_db):
    admin_id = _admin_user_id()
    audit.record_change(admin_id, config.AUDIT_INSERT, "students", "BCA001", new_value={"x": 1})
    audit.record_change(admin_id, config.AUDIT_INSERT, "subjects", "SUB1", new_value={"x": 1})

    student_entries = audit.get_audit_logs(table_name="students")
    assert len(student_entries) == 1
    assert student_entries[0]["table_name"] == "students"


def test_get_audit_logs_filters_by_record_id(test_db):
    admin_id = _admin_user_id()
    audit.record_change(admin_id, config.AUDIT_INSERT, "students", "BCA001", new_value={"x": 1})
    audit.record_change(admin_id, config.AUDIT_UPDATE, "students", "BCA001", new_value={"x": 2})
    audit.record_change(admin_id, config.AUDIT_INSERT, "students", "BCA002", new_value={"x": 1})

    entries = audit.get_audit_logs(table_name="students", record_id="BCA001")
    assert len(entries) == 2
    assert all(entry["record_id"] == "BCA001" for entry in entries)


def test_get_audit_logs_filters_by_user_id(test_db):
    admin_id = _admin_user_id()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    audit.record_change(admin_id, config.AUDIT_INSERT, "students", "BCA001", new_value={"x": 1})
    audit.record_change(teacher_id, config.AUDIT_UPDATE, "students", "BCA001", new_value={"x": 2})

    admin_entries = audit.get_audit_logs(user_id=admin_id)
    assert len(admin_entries) == 1
    assert admin_entries[0]["user_id"] == admin_id

    teacher_entries = audit.get_audit_logs(user_id=teacher_id)
    assert len(teacher_entries) == 1
    assert teacher_entries[0]["user_id"] == teacher_id


def test_get_audit_logs_orders_newest_first(test_db):
    admin_id = _admin_user_id()
    first_id = audit.record_change(admin_id, config.AUDIT_INSERT, "students", "BCA001", new_value={"x": 1})
    second_id = audit.record_change(admin_id, config.AUDIT_UPDATE, "students", "BCA001", new_value={"x": 2})

    entries = audit.get_audit_logs(table_name="students", record_id="BCA001")
    assert [entry["log_id"] for entry in entries] == [second_id, first_id]


def test_get_audit_logs_combines_filters_with_login_history_use_case(test_db):
    # This is exactly how render_user_management_page()'s "Login history"
    # section calls this function -- table_name="auth_events" + user_id.
    admin_id = _admin_user_id()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)

    auth.authenticate("teach1", "TeachPass1!")  # writes a login_success auth_events entry
    with pytest.raises(Exception):
        auth.authenticate("teach1", "WrongPassword!")  # writes a login_failed entry

    history = audit.get_audit_logs(table_name="auth_events", user_id=teacher_id)
    events = [entry["new_value"]["event"] for entry in history]
    assert "login_success" in events
    assert "login_failed" in events
