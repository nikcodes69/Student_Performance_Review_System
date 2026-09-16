"""
modules/auth.py
================
Authentication (proving who you are) and authorization (checking what
you're allowed to do), plus Streamlit session management.

This file is split into two clearly separate halves:

  1. PURE-ISH FUNCTIONS (hash_password, verify_password, create_user,
     authenticate, check_permission) -- these only touch bcrypt and the
     database, never Streamlit. This is what let us verify this half of
     the file with a plain Python script instead of clicking through a
     browser (see the chat explanation after this file).

  2. STREAMLIT SESSION FUNCTIONS (login, logout, get_current_user,
     is_session_valid, require_login, require_role) -- these read and
     write st.session_state, which is Streamlit's built-in mechanism for
     remembering data (like "who is logged in") across reruns of the
     script. Streamlit re-runs app.py from top to bottom every time a
     user clicks anything, so without session_state, a user would be
     logged out by their own next click.

======================================================================
WHO CAN CREATE AN ACCOUNT, AND HOW -- A DELIBERATE, ASYMMETRIC DESIGN
======================================================================
This app does NOT offer one universal "sign up" form for every role.
That would mean any visitor to a publicly deployed instance of this app
could grant themselves an Admin account -- full access to every
student's records, the audit log, everything. Instead:

  - STUDENT: self-service, via self_register_student() below, but only
    to CLAIM A LOGIN for a roll_no that an Admin/Teacher already entered
    into the students table. A student cannot invent a new student record
    of themselves through signup -- academic records are only ever
    created by staff (modules/students.py). To prove the person signing
    up really is that student, self_register_student() also requires the
    email address already on file for that roll_no to match. This is
    deliberately lightweight (no emailed confirmation link, no OTP -- out
    of scope for a project this size) but it does mean a random visitor
    cannot claim an arbitrary roll_no just by guessing a number; they
    would also need to know the exact email the institution has on record.

  - TEACHER and ADMIN: never self-service. create_user() (below) is only
    ever called two ways: by an existing Admin, through
    render_user_management_page() at the bottom of this file, or by
    ensure_default_admin_exists()'s one-time bootstrap script
    (`python -m modules.auth`) when NO admin exists at all yet. There is
    no page anywhere in this app where a visitor can choose "Teacher" or
    "Admin" for themselves.
"""

from datetime import datetime, timedelta

import bcrypt
import pandas as pd
import streamlit as st

import config
from database.db_manager import execute_write, fetch_all, fetch_one
from utils.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DuplicateRecordError,
    RecordNotFoundError,
    ValidationError,
)
from utils.logger import get_logger
from utils.validators import validate_email, validate_password, validate_role, validate_roll_no, validate_username

logger = get_logger(__name__)

# Keys used inside Streamlit's st.session_state dictionary. Defined as
# constants so a typo (e.g. "auth_usr" instead of "auth_user") becomes an
# obvious bug instead of a silent one.
SESSION_KEY_USER = "auth_user"
SESSION_KEY_LAST_ACTIVITY = "auth_last_activity"


# ---------------------------------------------------------------------------
# PASSWORD HASHING
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash a plaintext password with bcrypt, ready to store in
    users.password_hash.

    HOW bcrypt WORKS (for the viva): bcrypt.gensalt() generates a random
    "salt" -- a short random value unique to this one password. The salt
    is combined with the password before hashing, which is what stops two
    users who happen to choose the same password from ending up with the
    same password_hash (which would otherwise leak that fact, and make
    precomputed "rainbow table" attacks possible). Critically, we do NOT
    need a separate salt column in the users table: bcrypt's output
    string already has the algorithm version, cost factor, salt, and hash
    all encoded together in one piece of text, e.g.:
        $2b$12$KIXQ4a5s7z...restofhash...
        ^^^ ^^ ^^^^^^^^^^^^
        alg cost   salt (the hash itself follows)
    bcrypt.checkpw() (used in verify_password below) reads the salt back
    out of that same string automatically.

    Args:
        password: The plaintext password (already validated by
            utils.validators.validate_password before this is called).

    Returns:
        The bcrypt hash, as a string, safe to store in the database.
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=config.BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Check a plaintext password against a stored bcrypt hash.

    Args:
        password: The plaintext password a user just typed in.
        password_hash: The hash stored in users.password_hash.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ---------------------------------------------------------------------------
# USER CREATION AND LOGIN (database-backed, no Streamlit here)
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, role: str) -> int:
    """
    Create a new user account.

    Args:
        username: The desired login username.
        password: The desired plaintext password (will be hashed, never
            stored in plaintext).
        role: One of config.VALID_ROLES.

    Returns:
        The new user's user_id.

    Raises:
        ValidationError: if username, password, or role fails validation.
        DuplicateRecordError: if the username is already taken.
        DatabaseError: if the insert fails for any other reason.
    """
    username = validate_username(username)
    password = validate_password(password)
    role = validate_role(role)

    existing = fetch_one("SELECT user_id FROM users WHERE username = ?", (username,))
    if existing is not None:
        raise DuplicateRecordError(f"Username '{username}' is already taken.")

    password_hash = hash_password(password)

    user_id = execute_write(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        (username, password_hash, role),
    )

    # We log that a user was created, and its role, but NEVER the password
    # or its hash -- logs are written to a plain text file
    # (logs/app.log) that should never contain secrets.
    logger.info("Created user '%s' with role '%s' (user_id=%s).", username, role, user_id)
    return user_id


def self_register_student(roll_no: str, email: str, password: str) -> int:
    """
    Let a student create their OWN login, for a student record an
    Admin/Teacher already created. See the module docstring's "WHO CAN
    CREATE AN ACCOUNT" section for the full reasoning -- this is
    deliberately the ONLY self-service signup path in the whole app.

    The student's username becomes their roll_no, matching the
    convention established in database/db_setup.py's students table
    (a Student-role user's username IS their roll_no) -- there is no
    separate "choose a username" step.

    WHY THIS IS A LAZY IMPORT (modules.students imported inside this
    function, not at the top of the file): modules/students.py imports
    `from modules import auth` (for check_permission()), so importing
    modules.students at the TOP of this file would create a circular
    import -- auth -> students -> auth, which Python cannot resolve.
    Importing it here instead, inside the function body, works because by
    the time this function is actually CALLED, both modules have already
    finished loading.

    Args:
        roll_no: The student's roll number -- must already exist as an
            active student record.
        email: The email address already on file for that student, used
            as a lightweight proof of identity.
        password: The desired plaintext password.

    Returns:
        The new user account's user_id.

    Raises:
        RecordNotFoundError: if no active student exists with this roll_no.
        ValidationError: if the email does not match the one on file, or
            the password fails validation.
        DuplicateRecordError: if a login already exists for this roll_no.
    """
    from modules.students import get_student  # see docstring above

    roll_no = validate_roll_no(roll_no)
    student = get_student(roll_no)  # raises RecordNotFoundError if missing/inactive

    email = validate_email(email)
    if email != student["email"]:
        raise ValidationError(
            "That email address does not match our record for this roll number. "
            "Contact an administrator if you believe this is an error."
        )

    user_id = create_user(username=roll_no, password=password, role=config.ROLE_STUDENT)
    logger.info("Student '%s' self-registered a login (user_id=%s).", roll_no, user_id)
    return user_id


def authenticate(username: str, password: str) -> dict:
    """
    Verify a username/password pair and return the matching user's
    public details.

    SECURITY NOTE: whether the USERNAME does not exist, or the username
    exists but the PASSWORD is wrong, this function raises the exact same
    AuthenticationError message ("Invalid username or password."). If we
    used two different messages ("no such user" vs "wrong password"), an
    attacker could use that difference to discover which usernames are
    real, one guess at a time -- this is a well-known attack called
    "username enumeration". Giving away nothing extra closes that door.

    Args:
        username: The submitted username.
        password: The submitted plaintext password.

    Returns:
        A dict with keys "user_id", "username", "role" for the
        authenticated user. Deliberately does NOT include password_hash --
        this dict is what gets stored in the Streamlit session and we
        never want the hash sitting in memory longer than it needs to.

    Raises:
        AuthenticationError: if the username doesn't exist, the account
            is deactivated, or the password is wrong.
    """
    user_row = fetch_one(
        "SELECT user_id, username, password_hash, role, is_active FROM users WHERE username = ?",
        (username,),
    )

    generic_error = "Invalid username or password."

    if user_row is None:
        logger.warning("Login failed: unknown username '%s'.", username)
        raise AuthenticationError(generic_error)

    if not user_row["is_active"]:
        logger.warning("Login failed: account '%s' is deactivated.", username)
        raise AuthenticationError("This account has been deactivated. Contact an administrator.")

    if not verify_password(password, user_row["password_hash"]):
        logger.warning("Login failed: wrong password for username '%s'.", username)
        raise AuthenticationError(generic_error)

    execute_write(
        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
        (user_row["user_id"],),
    )

    logger.info("User '%s' authenticated successfully.", username)

    return {
        "user_id": user_row["user_id"],
        "username": user_row["username"],
        "role": user_row["role"],
    }


def check_permission(user_role: str, allowed_roles: tuple[str, ...]) -> None:
    """
    Pure authorization check: raise AuthorizationError if a role is not
    permitted.

    This exists as a SEPARATE, lower-level check from require_role()
    below (which is Streamlit-specific and used at the top of a page).
    Business-logic functions in later modules (e.g. modules/students.py's
    delete_student()) can call THIS function directly, so that permission
    is enforced even if a function were ever called from somewhere other
    than a Streamlit page (a script, a test, a future API) -- the page-
    level check in require_role() is a convenience for the UI, not the
    only line of defence.

    Args:
        user_role: The role of the user attempting the action.
        allowed_roles: The roles permitted to perform it.

    Raises:
        AuthorizationError: if user_role is not in allowed_roles.
    """
    if user_role not in allowed_roles:
        raise AuthorizationError(
            f"Role '{user_role}' is not permitted to perform this action."
        )


def ensure_default_admin_exists() -> None:
    """
    Create the bootstrap admin account (config.DEFAULT_ADMIN_USERNAME /
    config.DEFAULT_ADMIN_PASSWORD) if, and only if, no admin account
    exists yet. Safe to call every time the app starts -- it does nothing
    once a real admin exists.

    See the SECURITY NOTE next to config.DEFAULT_ADMIN_USERNAME: this
    default password is publicly visible in the source code and is meant
    to be changed immediately after the first login, in a real deployment.
    """
    existing_admin = fetch_one(
        "SELECT user_id FROM users WHERE role = ?", (config.ROLE_ADMIN,)
    )
    if existing_admin is not None:
        return

    create_user(config.DEFAULT_ADMIN_USERNAME, config.DEFAULT_ADMIN_PASSWORD, config.ROLE_ADMIN)
    logger.warning(
        "No admin account existed, so a default admin was created (username='%s'). "
        "This password is publicly visible in config.py -- change it immediately.",
        config.DEFAULT_ADMIN_USERNAME,
    )


def list_users() -> list[dict]:
    """
    Fetch every user account (Admin, Teacher, and Student logins alike).

    Not permission-gated itself -- consistent with every other list_*
    function in this project (see modules/students.py's module docstring
    for the full reasoning): the real gate is
    render_user_management_page() below calling require_role(ROLE_ADMIN)
    before this is ever called. Deliberately excludes password_hash from
    the SELECT entirely, not just from what's displayed -- a hash that is
    never fetched can never accidentally be shown or logged.

    Returns:
        A list of dicts: user_id, username, role, is_active, created_at,
        last_login. Ordered by username.
    """
    return fetch_all(
        "SELECT user_id, username, role, is_active, created_at, last_login "
        "FROM users ORDER BY username"
    )


def deactivate_user(user_id: int, acting_user: dict) -> None:
    """
    Soft-delete a user account (disable login) -- never hard-deleted,
    consistent with every other table in this project.

    Args:
        user_id: The account to deactivate.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if user_id does not exist.
        ValidationError: if the account is already inactive, or
            acting_user is trying to deactivate their own account (an
            Admin locking themselves out would have no way back in short
            of another Admin existing -- simplest to just disallow it).
    """
    from modules.audit import build_audit_entry  # see self_register_student()'s docstring for why this is a lazy import
    from database.db_manager import execute_transaction

    check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    existing = fetch_one(
        "SELECT user_id, username, is_active FROM users WHERE user_id = ?", (user_id,)
    )
    if existing is None:
        raise RecordNotFoundError(f"No user found with user_id {user_id}.")
    if not existing["is_active"]:
        raise ValidationError(f"User '{existing['username']}' is already inactive.")
    if user_id == acting_user["user_id"]:
        raise ValidationError("You cannot deactivate your own account.")

    update_statement = (
        "UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (user_id,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_SOFT_DELETE, "users", str(user_id),
        old_value={"is_active": 1}, new_value={"is_active": 0},
    )
    execute_transaction([update_statement, audit_statement])
    logger.info("User '%s' (user_id=%s) deactivated by user_id=%s.", existing["username"], user_id, acting_user["user_id"])


def reactivate_user(user_id: int, acting_user: dict) -> None:
    """
    Reverse a soft-delete: sets is_active back to 1.

    Args:
        user_id: The account to reactivate.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if user_id does not exist.
        ValidationError: if the account is already active.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    existing = fetch_one(
        "SELECT user_id, username, is_active FROM users WHERE user_id = ?", (user_id,)
    )
    if existing is None:
        raise RecordNotFoundError(f"No user found with user_id {user_id}.")
    if existing["is_active"]:
        raise ValidationError(f"User '{existing['username']}' is already active.")

    update_statement = (
        "UPDATE users SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (user_id,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "users", str(user_id),
        old_value={"is_active": 0}, new_value={"is_active": 1},
    )
    execute_transaction([update_statement, audit_statement])
    logger.info("User '%s' (user_id=%s) reactivated by user_id=%s.", existing["username"], user_id, acting_user["user_id"])


# ---------------------------------------------------------------------------
# STREAMLIT SESSION MANAGEMENT
# ---------------------------------------------------------------------------

def login(username: str, password: str) -> dict:
    """
    Authenticate a user and start their Streamlit session.

    Args:
        username: The submitted username.
        password: The submitted plaintext password.

    Returns:
        The logged-in user's dict (user_id, username, role).

    Raises:
        AuthenticationError: if the credentials are invalid (see authenticate()).
    """
    user = authenticate(username, password)
    st.session_state[SESSION_KEY_USER] = user
    st.session_state[SESSION_KEY_LAST_ACTIVITY] = datetime.now()
    return user


def logout() -> None:
    """
    End the current Streamlit session, clearing the logged-in user.
    Safe to call even if nobody is currently logged in.
    """
    user = st.session_state.get(SESSION_KEY_USER)
    if user is not None:
        logger.info("User '%s' logged out.", user["username"])

    st.session_state.pop(SESSION_KEY_USER, None)
    st.session_state.pop(SESSION_KEY_LAST_ACTIVITY, None)


def get_current_user() -> dict | None:
    """
    Returns:
        The current user's dict (user_id, username, role) if someone is
        logged in this session, otherwise None. Does NOT check whether
        the session has timed out -- call is_session_valid() for that.
    """
    return st.session_state.get(SESSION_KEY_USER)


def is_session_valid() -> bool:
    """
    Check whether there is a logged-in user AND their session has not
    timed out from inactivity.

    THIS IS A "SLIDING" TIMEOUT: every time this function confirms the
    session is still valid, it also refreshes last_activity to right now.
    That means the timeout is measured from the user's LAST CLICK, not
    from when they first logged in -- an actively-used session never
    expires, only an abandoned one does. This is the common, user-friendly
    interpretation of "session timeout" (as opposed to a fixed expiry that
    would log someone out mid-task after, say, exactly 30 minutes no
    matter what they were doing).

    Returns:
        True if the session is valid and was just refreshed, False if
        nobody is logged in or the session expired.
    """
    user = st.session_state.get(SESSION_KEY_USER)
    last_activity = st.session_state.get(SESSION_KEY_LAST_ACTIVITY)

    if user is None or last_activity is None:
        return False

    elapsed = datetime.now() - last_activity
    if elapsed > timedelta(minutes=config.SESSION_TIMEOUT_MINUTES):
        return False

    st.session_state[SESSION_KEY_LAST_ACTIVITY] = datetime.now()
    return True


def require_login() -> dict:
    """
    Call this at the very top of any page that requires a logged-in user.

    If the session is missing or expired, this shows a message and calls
    st.stop() -- which halts the rest of THIS page's script immediately,
    the same way a return statement would exit a function, so no code
    below this call runs. It does not crash the app or raise a Python
    exception; it is Streamlit's intended way to stop a page early.

    Returns:
        The current user's dict, if the session is valid.
    """
    was_logged_in = SESSION_KEY_USER in st.session_state

    if not is_session_valid():
        logout()  # clear any stale/expired session data
        if was_logged_in:
            st.warning("Your session has expired due to inactivity. Please log in again.")
        else:
            st.info("Please log in to continue.")
        st.stop()

    return get_current_user()


def require_role(*allowed_roles: str) -> dict:
    """
    Call this at the top of a page restricted to specific roles. Combines
    require_login() with a role check -- this is the page-level half of
    Role-Based Access Control; check_permission() above is the business-
    logic half.

    Args:
        *allowed_roles: One or more roles permitted to view this page,
            e.g. require_role(config.ROLE_ADMIN, config.ROLE_TEACHER).

    Returns:
        The current user's dict, if they are logged in AND their role is allowed.
    """
    user = require_login()

    if user["role"] not in allowed_roles:
        logger.warning(
            "Access denied: user '%s' (role=%s) attempted a page requiring roles %s.",
            user["username"], user["role"], allowed_roles,
        )
        st.error("You do not have permission to view this page.")
        st.stop()

    return user


def render_user_management_page() -> None:
    """
    Streamlit page: Admin creates Teacher/Admin/Student login accounts by
    hand, and can deactivate/reactivate existing ones. Admin-only -- this
    is deliberately the ONLY way a Teacher or Admin account ever comes
    into existence (besides the one-time bootstrap script) -- see the
    module docstring's "WHO CAN CREATE AN ACCOUNT" section.
    """
    current_user = require_role(config.ROLE_ADMIN)

    st.title("User Management")

    st.subheader("Create a new account")
    st.caption(
        "Use this to create Teacher and Admin accounts. Students should "
        "normally use the Sign Up tab on the login page themselves -- see "
        "there first if you're creating a login for an existing student."
    )
    with st.form("create_user_form", clear_on_submit=True):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", options=config.VALID_ROLES)
        create_submitted = st.form_submit_button("Create Account")

    if create_submitted:
        try:
            create_user(new_username, new_password, new_role)
            st.success(f"Account '{new_username}' created with role '{new_role}'.")
            st.rerun()
        except (ValidationError, DuplicateRecordError) as error:
            st.error(str(error))

    st.divider()
    st.subheader("Existing accounts")

    users = list_users()
    if not users:
        st.info("No user accounts found.")
        return

    display_rows = [
        {
            "Username": u["username"],
            "Role": u["role"],
            "Active": "Yes" if u["is_active"] else "No",
            "Created": u["created_at"],
            "Last Login": u["last_login"] or "Never",
        }
        for u in users
    ]
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    st.subheader("Deactivate / reactivate an account")
    username_choice = st.selectbox("Select an account", options=[u["username"] for u in users])
    selected_user = next(u for u in users if u["username"] == username_choice)

    deactivate_col, reactivate_col = st.columns(2)
    with deactivate_col:
        is_self = selected_user["user_id"] == current_user["user_id"]
        if selected_user["is_active"] and st.button(
            "Deactivate this account", disabled=is_self,
            help="You cannot deactivate your own account." if is_self else None,
        ):
            try:
                deactivate_user(selected_user["user_id"], current_user)
                st.success(f"'{username_choice}' deactivated.")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))
    with reactivate_col:
        if not selected_user["is_active"] and st.button("Reactivate this account"):
            try:
                reactivate_user(selected_user["user_id"], current_user)
                st.success(f"'{username_choice}' reactivated.")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))


if __name__ == "__main__":
    # Run once, from the project root: python -m modules.auth
    # Creates the first admin account so there is a way to log in at all.
    ensure_default_admin_exists()
