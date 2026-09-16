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
"""

from datetime import datetime, timedelta

import bcrypt
import streamlit as st

import config
from database.db_manager import execute_write, fetch_one
from utils.exceptions import AuthenticationError, AuthorizationError, DuplicateRecordError
from utils.logger import get_logger
from utils.validators import validate_password, validate_role, validate_username

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


if __name__ == "__main__":
    # Run once, from the project root: python -m modules.auth
    # Creates the first admin account so there is a way to log in at all.
    ensure_default_admin_exists()
