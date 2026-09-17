"""
tests/test_auth.py
===================
pytest tests for modules/auth.py's business logic: account creation
(including the "who gets a forced password change" default), the login
flow, and the change_password()/admin_reset_password() functions added
alongside the "forced change on first login" feature.

Uses the SAME throwaway-database pattern as tests/test_database.py: a
fresh, empty SQLite file per test, with the Turso cloud backend forced
off via monkeypatching -- so these tests never touch the real local
database or the real Turso database. See that file's module docstring
for the full explanation of why both patches are needed.

HOW THIS AVOIDS A STREAMLIT RUNTIME: modules/auth.py's pure-logic
functions (create_user, authenticate, change_password, etc.) never touch
st.session_state -- only the STREAMLIT SESSION MANAGEMENT half of that
file does (login(), logout(), require_login(), the render_*_page()
functions). This file only tests the pure-logic half, exactly like the
project's original verification approach described in that module's
docstring, so no running Streamlit server is needed to run it.

HOW TO RUN (from the project root):
    python -m pytest tests/test_auth.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.auth as auth
from database.db_setup import create_indexes, create_tables, get_connection
from utils.exceptions import AuthenticationError, DuplicateRecordError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database.
    See tests/test_database.py's test_db fixture for the full rationale;
    this is the same pattern, repeated here so this file can run alone."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_auth.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _insert_student(roll_no: str, email: str) -> None:
    from database.db_manager import execute_write
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (roll_no, "Fixture Student", 1, "BCA", email, "9812345678", 2024),
    )


# ---------------------------------------------------------------------------
# create_user() -- force_password_change default
# ---------------------------------------------------------------------------

def test_create_user_defaults_to_forced_password_change(test_db):
    from database.db_manager import fetch_one
    user_id = auth.create_user("teacher1", "TeacherPass1!", config.ROLE_TEACHER)
    row = fetch_one("SELECT must_change_password FROM users WHERE user_id = ?", (user_id,))
    assert row["must_change_password"] == 1


def test_create_user_can_opt_out_of_forced_change(test_db):
    from database.db_manager import fetch_one
    user_id = auth.create_user("teacher2", "TeacherPass1!", config.ROLE_TEACHER, force_password_change=False)
    row = fetch_one("SELECT must_change_password FROM users WHERE user_id = ?", (user_id,))
    assert row["must_change_password"] == 0


def test_create_user_rejects_duplicate_username(test_db):
    auth.create_user("dupuser", "SomePass1!", config.ROLE_TEACHER)
    with pytest.raises(DuplicateRecordError):
        auth.create_user("dupuser", "AnotherPass2!", config.ROLE_TEACHER)


# ---------------------------------------------------------------------------
# self_register_student() -- never forces a password change
# ---------------------------------------------------------------------------

def test_self_register_student_does_not_force_password_change(test_db):
    from database.db_manager import fetch_one
    _insert_student("BCA100", "student@example.com")
    user_id = auth.self_register_student("BCA100", "student@example.com", "StudentPass1!")
    row = fetch_one("SELECT must_change_password FROM users WHERE user_id = ?", (user_id,))
    assert row["must_change_password"] == 0


def test_self_register_student_rejects_mismatched_email(test_db):
    _insert_student("BCA101", "real@example.com")
    with pytest.raises(ValidationError):
        auth.self_register_student("BCA101", "wrong@example.com", "StudentPass1!")


def test_self_register_student_rejects_unknown_roll_no(test_db):
    with pytest.raises(RecordNotFoundError):
        auth.self_register_student("NOSUCHROLL", "anyone@example.com", "StudentPass1!")


# ---------------------------------------------------------------------------
# authenticate() -- returns must_change_password, and rejects bad credentials
# ---------------------------------------------------------------------------

def test_authenticate_returns_must_change_password_flag(test_db):
    auth.create_user("teacher3", "TeacherPass1!", config.ROLE_TEACHER)
    user = auth.authenticate("teacher3", "TeacherPass1!")
    assert user["must_change_password"] is True


def test_authenticate_rejects_wrong_password(test_db):
    auth.create_user("teacher4", "TeacherPass1!", config.ROLE_TEACHER)
    with pytest.raises(AuthenticationError):
        auth.authenticate("teacher4", "WrongPassword!")


def test_authenticate_rejects_unknown_username(test_db):
    with pytest.raises(AuthenticationError):
        auth.authenticate("nosuchuser", "AnyPassword1!")


# ---------------------------------------------------------------------------
# change_password()
# ---------------------------------------------------------------------------

def test_change_password_clears_forced_flag_and_updates_hash(test_db):
    from database.db_manager import fetch_one
    user_id = auth.create_user("teacher5", "OldPass1!", config.ROLE_TEACHER)

    auth.change_password(user_id, "OldPass1!", "NewPass2@")

    row = fetch_one("SELECT must_change_password FROM users WHERE user_id = ?", (user_id,))
    assert row["must_change_password"] == 0

    # Old password must no longer work; new one must.
    with pytest.raises(AuthenticationError):
        auth.authenticate("teacher5", "OldPass1!")
    user = auth.authenticate("teacher5", "NewPass2@")
    assert user["must_change_password"] is False


def test_change_password_rejects_wrong_old_password(test_db):
    user_id = auth.create_user("teacher6", "OldPass1!", config.ROLE_TEACHER)
    with pytest.raises(AuthenticationError):
        auth.change_password(user_id, "TotallyWrong!", "NewPass2@")


def test_change_password_rejects_invalid_new_password(test_db):
    user_id = auth.create_user("teacher7", "OldPass1!", config.ROLE_TEACHER)
    with pytest.raises(ValidationError):
        auth.change_password(user_id, "OldPass1!", "short")  # too short to pass validate_password


# ---------------------------------------------------------------------------
# admin_reset_password()
# ---------------------------------------------------------------------------

def test_admin_reset_password_sets_forced_change_flag(test_db):
    from database.db_manager import fetch_one
    admin_id = auth.create_user("admin1", "AdminPass1!", config.ROLE_ADMIN)
    target_id = auth.create_user("teacher8", "OldPass1!", config.ROLE_TEACHER, force_password_change=False)
    acting_user = {"user_id": admin_id, "username": "admin1", "role": config.ROLE_ADMIN}

    auth.admin_reset_password(target_id, "ResetPass3#", acting_user)

    row = fetch_one("SELECT must_change_password FROM users WHERE user_id = ?", (target_id,))
    assert row["must_change_password"] == 1
    user = auth.authenticate("teacher8", "ResetPass3#")
    assert user["must_change_password"] is True


def test_admin_reset_password_requires_admin_role(test_db):
    from utils.exceptions import AuthorizationError
    target_id = auth.create_user("teacher9", "OldPass1!", config.ROLE_TEACHER)
    non_admin_acting_user = {"user_id": target_id, "username": "teacher9", "role": config.ROLE_TEACHER}
    with pytest.raises(AuthorizationError):
        auth.admin_reset_password(target_id, "ResetPass3#", non_admin_acting_user)
