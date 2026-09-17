"""
tests/test_invited_signup.py
=============================
pytest tests for Teacher/Admin self-service signup: modules/auth.py's
invite_account(), revoke_invite(), list_pending_invites(),
signup_invited_account(), and self_link_google_account() -- see that
module's docstring's "WHO CAN CREATE AN ACCOUNT" section for the full
trust model (the same shape of rule as Student self-registration: prove
you're the one an Admin already vetted, via database/db_setup.py's
pending_accounts allowlist instead of an existing academic record).

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_invited_signup.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.auth as auth
from database.db_setup import create_indexes, create_tables, get_connection
from utils.exceptions import AuthenticationError, AuthorizationError, DuplicateRecordError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database -- see
    tests/test_database.py's test_db fixture for the full rationale."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_invited_signup.db")
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


# ---------------------------------------------------------------------------
# invite_account() / revoke_invite() / list_pending_invites()
# ---------------------------------------------------------------------------

def test_invite_account_requires_admin_role(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach0", "TeachPass1!", config.ROLE_TEACHER)
    teacher_acting_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach0"}

    with pytest.raises(AuthorizationError):
        auth.invite_account("x@gmail.com", config.ROLE_TEACHER, teacher_acting_user)


def test_invite_account_rejects_student_role(test_db):
    admin_user = _admin_user()
    with pytest.raises(ValidationError):
        auth.invite_account("x@gmail.com", config.ROLE_STUDENT, admin_user)


def test_invite_account_rejects_duplicate_email(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)
    with pytest.raises(DuplicateRecordError):
        auth.invite_account("teacher1@gmail.com", config.ROLE_ADMIN, admin_user)


def test_list_pending_invites_shows_created_invite(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)

    invites = auth.list_pending_invites()
    assert len(invites) == 1
    assert invites[0]["email"] == "teacher1@gmail.com"
    assert invites[0]["role"] == config.ROLE_TEACHER
    assert invites[0]["invited_by"] == "admin"


def test_revoke_invite_removes_it(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)
    auth.revoke_invite("teacher1@gmail.com", admin_user)
    assert auth.list_pending_invites() == []


def test_revoke_invite_requires_existing_invite(test_db):
    admin_user = _admin_user()
    with pytest.raises(RecordNotFoundError):
        auth.revoke_invite("nobody@gmail.com", admin_user)


def test_revoke_invite_requires_admin_role(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)
    teacher_id = auth.create_user("teach0", "TeachPass1!", config.ROLE_TEACHER)
    teacher_acting_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach0"}

    with pytest.raises(AuthorizationError):
        auth.revoke_invite("teacher1@gmail.com", teacher_acting_user)


# ---------------------------------------------------------------------------
# signup_invited_account()
# ---------------------------------------------------------------------------

def test_signup_invited_account_rejects_missing_invite(test_db):
    with pytest.raises(RecordNotFoundError):
        auth.signup_invited_account("nobody@gmail.com", "newuser1", "SomePass1!")


def test_signup_invited_account_creates_account_with_invited_role(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)

    user_id = auth.signup_invited_account("teacher1@gmail.com", "newteach1", "SomePass1!")

    login = auth.authenticate("newteach1", "SomePass1!")
    assert login["user_id"] == user_id
    assert login["role"] == config.ROLE_TEACHER
    assert login["must_change_password"] is False  # they chose their own password


def test_signup_invited_account_consumes_the_invite(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)
    auth.signup_invited_account("teacher1@gmail.com", "newteach1", "SomePass1!")

    assert auth.list_pending_invites() == []
    with pytest.raises(RecordNotFoundError):
        auth.signup_invited_account("teacher1@gmail.com", "newteach2", "SomePass1!")


def test_signup_invited_account_rejects_duplicate_username(test_db):
    admin_user = _admin_user()
    auth.create_user("existing", "SomePass1!", config.ROLE_TEACHER)
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)

    with pytest.raises(DuplicateRecordError):
        auth.signup_invited_account("teacher1@gmail.com", "existing", "SomePass1!")


# ---------------------------------------------------------------------------
# authenticate_with_google() interaction with pending_accounts
# ---------------------------------------------------------------------------

def test_google_sign_in_does_not_auto_create_invited_teacher_admin(test_db):
    """Regression guard: Google Sign-In must never auto-create a
    Teacher/Admin account directly, even when a pending invite exists --
    see modules/auth.py's module docstring for why (no well-defined
    username to derive from an arbitrary email, unlike a Student's
    roll_no)."""
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)

    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("teacher1@gmail.com")

    # The invite must still be there -- Google Sign-In didn't consume it.
    assert len(auth.list_pending_invites()) == 1


# ---------------------------------------------------------------------------
# self_link_google_account()
# ---------------------------------------------------------------------------

def test_self_link_google_account_lets_the_owner_sign_in_with_google(test_db):
    admin_user = _admin_user()
    auth.invite_account("teacher1@gmail.com", config.ROLE_TEACHER, admin_user)
    user_id = auth.signup_invited_account("teacher1@gmail.com", "newteach1", "SomePass1!")
    acting_user = {"user_id": user_id, "role": config.ROLE_TEACHER, "username": "newteach1"}

    auth.self_link_google_account(acting_user, "newteach1@gmail.com")

    logged_in = auth.authenticate_with_google("newteach1@gmail.com")
    assert logged_in["user_id"] == user_id


def test_self_link_google_account_rejects_email_linked_to_another_account(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "shared@gmail.com", admin_user)

    other_id = auth.create_user("teach2", "TeachPass2!", config.ROLE_TEACHER)
    other_acting_user = {"user_id": other_id, "role": config.ROLE_TEACHER, "username": "teach2"}

    with pytest.raises(DuplicateRecordError):
        auth.self_link_google_account(other_acting_user, "shared@gmail.com")
