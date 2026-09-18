"""
tests/test_google_auth.py
==========================
pytest tests for Google Sign-In: modules/auth.py's
authenticate_with_google(), link_google_account(), and
unlink_google_account() -- see that module's docstring's "GOOGLE
SIGN-IN" section for the full trust model these follow (the same
asymmetric rule as self_register_student()/create_user(): a Student can
self-associate via a matching email, Teacher/Admin never can).

WHY try_google_login() ITSELF IS NOT TESTED HERE: it reads st.user
(Streamlit's own OAuth session proxy) and drives st.session_state/
st.stop()/st.rerun() -- the same class of Streamlit-runtime-dependent
function as login()/render_change_password_page(), which this project
has never unit-tested directly either (see modules/auth.py's own module
docstring on why the file is split into a pure-logic half, tested here,
and a Streamlit-session half, verified manually). authenticate_with_google()
IS the pure-logic half for Google Sign-In, and is exactly what this file
covers.

REGRESSION COVERAGE FOR A REAL BUG FOUND WHILE BUILDING THIS FEATURE:
self_register_student() and authenticate_with_google() both derive a
Student's username directly from roll_no, but used to route it back
through create_user()'s own validate_username() (min 4 characters,
letters/digits/underscore only) -- rules that do not match
validate_roll_no()'s own, more permissive ones (no minimum length, no
character-set restriction). This silently broke both self-registration
and Google Sign-In for any roll_no shorter than 4 characters, including
this project's OWN real production data (roll_no "1" and "2"). Fixed by
extracting _insert_user_row() so the Student-specific paths skip the
mismatched generic username policy entirely -- see that function's
docstring. test_authenticate_with_google_auto_registers_short_roll_no
below is the regression test for this specific bug.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_google_auth.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.auth as auth
from database.db_setup import create_indexes, create_tables, get_connection
from utils.exceptions import AuthenticationError, AuthorizationError, DuplicateRecordError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database -- see
    tests/test_database.py's test_db fixture for the full rationale."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_google_auth.db")
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
# authenticate_with_google() -- Student auto-registration
# ---------------------------------------------------------------------------

def test_authenticate_with_google_rejects_unknown_email_without_a_role(test_db):
    # No expected_role given at all (e.g. a stale identity cookie from
    # before role-scoped login pages existed) -- nothing to register a
    # brand-new account as, so this must still fail, not silently pick a role.
    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("nobody@gmail.com")


def test_authenticate_with_google_auto_registers_matching_active_student(test_db):
    admin_user = _admin_user()
    from modules import students
    students.create_student("BCA0S1", "Alice", 1, "BCA", "alice@gmail.com", "9812345678", 2024, admin_user)

    user = auth.authenticate_with_google("alice@gmail.com", expected_role=config.ROLE_STUDENT)

    assert user["username"] == "BCA0S1"
    assert user["role"] == config.ROLE_STUDENT
    assert user["must_change_password"] is False  # they never need to -- there's no "given" password to replace


def test_authenticate_with_google_auto_registers_short_roll_no(test_db):
    """Regression test: see this file's module docstring for the bug
    this specifically guards against -- a 1-character roll_no (this
    project's own real data) used to be silently rejected."""
    admin_user = _admin_user()
    from modules import students
    students.create_student("1", "Nikhil Thapa", 1, "BCA", "nikhil@gmail.com", "9818182402", 2026, admin_user)

    user = auth.authenticate_with_google("nikhil@gmail.com", expected_role=config.ROLE_STUDENT)
    assert user["username"] == "1"


def test_authenticate_with_google_rejects_inactive_student(test_db):
    admin_user = _admin_user()
    from modules import students
    students.create_student("BCA0S2", "Bob", 1, "BCA", "bob@gmail.com", "9812345679", 2024, admin_user)
    students.deactivate_student("BCA0S2", admin_user)

    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("bob@gmail.com", expected_role=config.ROLE_STUDENT)


def test_authenticate_with_google_is_idempotent_on_repeated_sign_in(test_db):
    admin_user = _admin_user()
    from modules import students
    students.create_student("BCA0S1", "Alice", 1, "BCA", "alice@gmail.com", "9812345678", 2024, admin_user)

    first = auth.authenticate_with_google("alice@gmail.com", expected_role=config.ROLE_STUDENT)
    second = auth.authenticate_with_google("alice@gmail.com", expected_role=config.ROLE_STUDENT)

    assert first["user_id"] == second["user_id"]  # same account, not a duplicate


def test_authenticate_with_google_rejects_when_unlinked_login_already_exists(test_db):
    admin_user = _admin_user()
    from modules import students
    students.create_student("BCA0S3", "Carol", 1, "BCA", "carol@gmail.com", "9812345680", 2024, admin_user)
    auth.create_user("BCA0S3", "SomePass1!", config.ROLE_STUDENT, force_password_change=False)

    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("carol@gmail.com", expected_role=config.ROLE_STUDENT)


# ---------------------------------------------------------------------------
# link_google_account() / unlink_google_account() -- Teacher/Admin, Admin-driven
# ---------------------------------------------------------------------------

def test_link_google_account_lets_teacher_sign_in_with_google(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)

    auth.link_google_account(teacher_id, "teacher1@gmail.com", admin_user)
    logged_in = auth.authenticate_with_google("teacher1@gmail.com")

    assert logged_in["role"] == config.ROLE_TEACHER
    assert logged_in["user_id"] == teacher_id


def test_link_google_account_rejects_duplicate_email(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    other_teacher_id = auth.create_user("teach2", "TeachPass2!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "shared@gmail.com", admin_user)

    with pytest.raises(DuplicateRecordError):
        auth.link_google_account(other_teacher_id, "shared@gmail.com", admin_user)


def test_link_google_account_requires_admin_role(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    teacher_acting_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}

    with pytest.raises(AuthorizationError):
        auth.link_google_account(teacher_id, "x@gmail.com", teacher_acting_user)


def test_unlink_google_account_removes_google_sign_in_access(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "teacher1@gmail.com", admin_user)

    auth.unlink_google_account(teacher_id, admin_user)

    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("teacher1@gmail.com")


def test_unlink_google_account_rejects_when_nothing_linked(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)

    with pytest.raises(ValidationError):
        auth.unlink_google_account(teacher_id, admin_user)


# ---------------------------------------------------------------------------
# authenticate_with_google() -- expected_role (role-scoped Google Sign-In,
# see app.py's render_role_login_form()/auth.prepare_google_login())
# ---------------------------------------------------------------------------

def test_authenticate_with_google_accepts_matching_expected_role(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "teacher1@gmail.com", admin_user)

    user = auth.authenticate_with_google("teacher1@gmail.com", expected_role=config.ROLE_TEACHER)
    assert user["user_id"] == teacher_id


def test_authenticate_with_google_rejects_linked_account_of_a_different_role(test_db):
    # Clicking "Sign in with Google" on the Student portal, with a Google
    # account actually linked to a Teacher account, must fail -- not
    # silently sign them in as the Teacher.
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "teacher1@gmail.com", admin_user)

    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("teacher1@gmail.com", expected_role=config.ROLE_STUDENT)


def test_authenticate_with_google_self_registers_admin_ignoring_unrelated_student_record(test_db):
    # A Google email matching an active student's own email, but clicked
    # from the Admin login page -- self-registers a brand-new Admin
    # account instead, per the open self-registration design (the role
    # picked on the login page wins; the pre-existing student record for
    # this same email is unrelated and left untouched).
    admin_user = _admin_user()
    from modules import students
    students.create_student("BCA0S1", "Alice", 1, "BCA", "alice@gmail.com", "9812345678", 2024, admin_user)

    user = auth.authenticate_with_google("alice@gmail.com", expected_role=config.ROLE_ADMIN)

    assert user["role"] == config.ROLE_ADMIN
    assert user["username"] != "BCA0S1"  # a fresh username, not the student's roll_no
    # The student record's own roll_no login is untouched by this.
    assert auth.fetch_one("SELECT user_id FROM users WHERE username = 'BCA0S1'") is None


def test_authenticate_with_google_still_auto_registers_student_from_student_portal(test_db):
    admin_user = _admin_user()
    from modules import students
    students.create_student("BCA0S1", "Alice", 1, "BCA", "alice@gmail.com", "9812345678", 2024, admin_user)

    user = auth.authenticate_with_google("alice@gmail.com", expected_role=config.ROLE_STUDENT)
    assert user["role"] == config.ROLE_STUDENT


def test_authenticate_with_google_without_expected_role_ignores_role(test_db):
    # Backward compatibility: expected_role=None (its default) behaves
    # exactly as before role-scoped login pages existed.
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "teacher1@gmail.com", admin_user)

    user = auth.authenticate_with_google("teacher1@gmail.com")
    assert user["user_id"] == teacher_id


# ---------------------------------------------------------------------------
# authenticate_with_google() -- open self-registration for Admin/Teacher,
# see module docstring's "GOOGLE SIGN-IN" section for the explicit
# trade-off this accepts (no pre-approval needed for any role, by request)
# ---------------------------------------------------------------------------

def test_authenticate_with_google_self_registers_brand_new_admin(test_db):
    # No pre-existing admin invite, no pre-existing account of any kind --
    # signing in from the Admin login page creates one on the spot.
    user = auth.authenticate_with_google("newadmin@gmail.com", expected_role=config.ROLE_ADMIN)

    assert user["role"] == config.ROLE_ADMIN
    assert user["must_change_password"] is False
    row = auth.fetch_one("SELECT google_email FROM users WHERE user_id = ?", (user["user_id"],))
    assert row["google_email"] == "newadmin@gmail.com"


def test_authenticate_with_google_self_registers_brand_new_teacher(test_db):
    user = auth.authenticate_with_google("newteacher@gmail.com", expected_role=config.ROLE_TEACHER)
    assert user["role"] == config.ROLE_TEACHER


def test_authenticate_with_google_self_registration_username_avoids_collision(test_db):
    # Two different Gmail addresses that happen to share a local part
    # (different domains) must not collide on the same username.
    auth.create_user("samuel", "SomePass1!", config.ROLE_TEACHER)  # occupies "samuel" already

    user = auth.authenticate_with_google("samuel@gmail.com", expected_role=config.ROLE_ADMIN)

    assert user["username"] != "samuel"  # the pre-existing "samuel" account is untouched
    assert user["role"] == config.ROLE_ADMIN


def test_authenticate_with_google_second_self_registration_attempt_respects_first_role(test_db):
    # Once self-registered as Teacher, the SAME email trying to sign in
    # again from the Admin page must be rejected, not re-registered or
    # silently switched to Admin.
    auth.authenticate_with_google("person@gmail.com", expected_role=config.ROLE_TEACHER)

    with pytest.raises(AuthenticationError):
        auth.authenticate_with_google("person@gmail.com", expected_role=config.ROLE_ADMIN)


# ---------------------------------------------------------------------------
# list_users() -- google_email column
# ---------------------------------------------------------------------------

def test_list_users_includes_google_email(test_db):
    admin_user = _admin_user()
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    auth.link_google_account(teacher_id, "teacher1@gmail.com", admin_user)

    users = {u["username"]: u for u in auth.list_users()}
    assert users["admin"]["google_email"] is None
    assert users["teach1"]["google_email"] == "teacher1@gmail.com"
