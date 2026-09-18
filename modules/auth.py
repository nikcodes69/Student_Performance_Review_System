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
This app does NOT offer one universal "sign up" form where a visitor
just picks any role. That would mean any visitor to a publicly deployed
instance of this app could grant themselves an Admin account -- full
access to every student's records, the audit log, everything. Every
signup path below still requires proof tied back to something an Admin
(or, for Students, institutional staff) already put on record FIRST:

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

  - TEACHER and ADMIN: self-service, via signup_invited_account() below,
    but ONLY for an email an existing Admin has already pre-approved via
    invite_account() (see database/db_setup.py's pending_accounts table).
    A visitor cannot invite themselves -- invite_account() is Admin-only,
    from render_user_management_page(). This is the SAME shape of trust
    rule as Student signup (prove you're the one the institution already
    vetted), just using an Admin-maintained allowlist instead of an
    existing academic record, since there is no equivalent "roll_no" for
    staff. An Admin can also still create a Teacher/Admin account
    directly, by hand, via create_user() -- invite-based self-service is
    an additional path, not a replacement for that one.

======================================================================
GOOGLE SIGN-IN -- A DELIBERATELY MORE PERMISSIVE RULE, BY EXPLICIT REQUEST
======================================================================
authenticate_with_google() (below) is the Google-authenticated counterpart
to authenticate()/self_register_student(). Earlier versions of this
function followed the EXACT SAME pre-approval rule as the section above
(Teacher/Admin could never be auto-created, only Student could, via a
matching on-file email). That was DELIBERATELY CHANGED, on explicit
request, to this instead:

  - Every role -- Student, Teacher, AND Admin -- self-registers a brand
    new account the first time someone signs in with a Google account
    that isn't linked to anything yet, using whichever role's login page
    (Admin/Teacher/Student) they picked before clicking "Sign in with
    Google". No Admin pre-approval, no invite, no matching student
    record required for Teacher/Admin (Student still prefers matching an
    existing roll_no's on-file email when one exists, for a sensible
    username, but falls back to the same open self-registration if not).

  - THE TRADE-OFF, STATED PLAINLY: this means anyone who can reach this
    app's login page and control a Google account can grant themselves
    an Admin account, with full access to every student's records and
    the audit log -- there is no gate left to stop them. This is the
    OPPOSITE of the WHO CAN CREATE AN ACCOUNT section above, which exists
    specifically to prevent exactly that. Both cannot be true at once;
    this project accepted that trade-off for Google Sign-In specifically,
    in exchange for zero-setup onboarding, while keeping the invite/
    manual-creation paths above as the stricter alternative for
    username/password accounts. If this app is ever deployed somewhere
    the login page is reachable by people who should NOT be able to
    self-grant Admin, this trade-off needs revisiting first.

  - Once an account exists (via self-registration or any other path),
    its role is fixed: a later Google Sign-In with the SAME email is
    rejected, not re-created, if attempted from a DIFFERENT role's login
    page than the one it was first registered under (see
    authenticate_with_google()'s expected_role parameter).
"""

import re
import secrets
from datetime import datetime, timedelta

import bcrypt
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
from utils.backup import backup_filename, generate_full_backup
from utils.logger import get_logger
from utils.table_view import render_data_table
from utils.validators import validate_email, validate_password, validate_role, validate_roll_no, validate_username

logger = get_logger(__name__)

# Keys used inside Streamlit's st.session_state dictionary. Defined as
# constants so a typo (e.g. "auth_usr" instead of "auth_user") becomes an
# obvious bug instead of a silent one.
SESSION_KEY_USER = "auth_user"
SESSION_KEY_LAST_ACTIVITY = "auth_last_activity"
# Set by prepare_google_login() right before st.login("google") redirects
# to Google, read (and cleared) by try_google_login() after the round
# trip back -- see prepare_google_login()'s docstring for why
# st.session_state, specifically, is what survives that redirect.
SESSION_KEY_GOOGLE_LOGIN_ROLE = "auth_google_login_role"


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

def _insert_user_row(
    username: str, password: str, role: str, force_password_change: bool,
    google_email: str | None = None,
) -> int:
    """
    Shared INSERT logic behind create_user() and the two places that
    build a Student login directly from an already-validated roll_no
    (self_register_student(), authenticate_with_google() below) --
    duplicate-username check, password hashing, and the actual write.

    WHY THOSE TWO CALLERS DO NOT GO THROUGH create_user() ITSELF: this
    project's own convention (see this module's docstring) is that a
    Student's username IS their roll_no -- but validate_username() (which
    create_user() calls) enforces rules meant for a HUMAN-CHOSEN Admin/
    Teacher username (config.USERNAME_MIN_LENGTH=4, letters/digits/
    underscore only), which do not match validate_roll_no()'s own, more
    permissive rules (no minimum length, no character-set restriction --
    see that function). A real roll number can legitimately be shorter
    than 4 characters (this project's own real student data uses "1" and
    "2"), which validate_username() would silently reject even though
    validate_roll_no() already accepted it -- self-registration and
    Google Sign-In would both quietly break for exactly those students.
    roll_no has ALREADY been validated (and normalised) by
    validate_roll_no() by the time it reaches this function in both
    callers, so re-running the mismatched, stricter username policy on
    top of it would be actively wrong, not just redundant.

    Args:
        username: Already validated (either via validate_username(), by
            create_user() below, or via validate_roll_no(), by the two
            Student-specific callers).
        password: Already-validated plaintext password.
        role: Already-validated role.
        force_password_change: See create_user()'s docstring.
        google_email: If given, linked on this same INSERT (used only by
            authenticate_with_google()'s auto-registration path).

    Returns:
        The new user's user_id.

    Raises:
        DuplicateRecordError: if the username is already taken.
        DatabaseError: if the insert fails for any other reason.
    """
    existing = fetch_one("SELECT user_id FROM users WHERE username = ?", (username,))
    if existing is not None:
        raise DuplicateRecordError(f"Username '{username}' is already taken.")

    password_hash = hash_password(password)

    user_id = execute_write(
        "INSERT INTO users (username, password_hash, role, must_change_password, google_email) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, password_hash, role, 1 if force_password_change else 0, google_email),
    )

    # We log that a user was created, and its role, but NEVER the password
    # or its hash -- logs are written to a plain text file
    # (logs/app.log) that should never contain secrets.
    logger.info("Created user '%s' with role '%s' (user_id=%s).", username, role, user_id)
    return user_id


def create_user(username: str, password: str, role: str, force_password_change: bool = True) -> int:
    """
    Create a new user account with a HUMAN-CHOSEN username (an Admin
    typing a Teacher/Admin username, or the bootstrap admin account) --
    see _insert_user_row()'s docstring for why the two Student-specific
    callers (self_register_student(), authenticate_with_google()) build
    their own username from roll_no and call _insert_user_row() directly
    instead of this function.

    Args:
        username: The desired login username.
        password: The desired plaintext password (will be hashed, never
            stored in plaintext).
        role: One of config.VALID_ROLES.
        force_password_change: If True (the default), the account is
            created with must_change_password=1, so its first login is
            forced through render_change_password_page() before anything
            else in the app is reachable -- see require_login() below.
            This default fits the common case: whoever CALLS create_user()
            (an Admin, or the one-time bootstrap script) is choosing a
            password FOR someone else, so that person should pick their
            own the moment they first log in.

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

    return _insert_user_row(username, password, role, force_password_change)


def _random_unusable_password() -> str:
    """
    Generate a random plaintext password for a Google-only account -- one
    that satisfies the schema's password_hash NOT NULL constraint (see
    database/db_setup.py's users table) but is never shown to anyone and
    never meant to be typed in: an account created this way (see
    authenticate_with_google() below) can only ever be reached by signing
    in with the SAME Google account again, unless an Admin later resets
    its password by hand (admin_reset_password()).

    secrets.token_urlsafe(32), not random/uuid: this module already
    relies on bcrypt for genuine cryptographic randomness in
    hash_password() (via bcrypt.gensalt()) -- token_urlsafe uses Python's
    os.urandom() under the hood, the same class of cryptographically
    secure source, appropriate for a value that (however briefly, before
    being hashed and discarded) stands in as this account's password.
    """
    return secrets.token_urlsafe(32)


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

    password = validate_password(password)

    # _insert_user_row(), not create_user(): roll_no is already validated
    # (and normalised) by validate_roll_no() above -- see that function's
    # docstring for why re-running create_user()'s OWN validate_username()
    # on top of it would be wrong, not just redundant, for a roll number
    # shorter than config.USERNAME_MIN_LENGTH (e.g. this project's real
    # data: roll_no "1" and "2").
    #
    # force_password_change=False: unlike an Admin-created account, this
    # student already chose their own password just now -- there is no
    # "given" password to replace on first login.
    user_id = _insert_user_row(
        username=roll_no, password=password, role=config.ROLE_STUDENT, force_password_change=False,
    )
    logger.info("Student '%s' self-registered a login (user_id=%s).", roll_no, user_id)
    return user_id


def invite_account(email: str, role: str, acting_user: dict) -> None:
    """
    Pre-approve an email to self-register as a Teacher or Admin -- see
    database/db_setup.py's pending_accounts table for the full reasoning
    behind this mechanism. Admin-only, from render_user_management_page()
    below. Does not send an email itself (this project has no email/SMTP
    integration -- the same deliberate scope limit self_register_student()
    already documents for students); the Admin is expected to tell the
    invited person directly.

    Args:
        email: The email address being pre-approved.
        role: config.ROLE_TEACHER or config.ROLE_ADMIN (never
            config.ROLE_STUDENT -- see config.INVITABLE_ROLES).
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if email fails validation, or role is not one
            of config.INVITABLE_ROLES.
        DuplicateRecordError: if this email is already invited.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    email = validate_email(email)
    if role not in config.INVITABLE_ROLES:
        raise ValidationError(f"Role must be one of {config.INVITABLE_ROLES}.")

    existing = fetch_one("SELECT email FROM pending_accounts WHERE email = ?", (email,))
    if existing is not None:
        raise DuplicateRecordError(f"'{email}' is already invited.")

    insert_statement = (
        "INSERT INTO pending_accounts (email, role, invited_by) VALUES (?, ?, ?)",
        (email, role, acting_user["user_id"]),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "pending_accounts", email,
        old_value=None, new_value={"email": email, "role": role},
    )
    execute_transaction([insert_statement, audit_statement])
    logger.info("'%s' invited as '%s' by user_id=%s.", email, role, acting_user["user_id"])


def revoke_invite(email: str, acting_user: dict) -> None:
    """
    Cancel a pending invite before it's used. Admin-only.

    Args:
        email: The invited email to revoke.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if no pending invite exists for this email.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    email = validate_email(email)
    existing = fetch_one("SELECT email, role FROM pending_accounts WHERE email = ?", (email,))
    if existing is None:
        raise RecordNotFoundError(f"No pending invite found for '{email}'.")

    delete_statement = ("DELETE FROM pending_accounts WHERE email = ?", (email,))
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "pending_accounts", email,
        old_value={"email": email, "role": existing["role"]}, new_value=None,
    )
    execute_transaction([delete_statement, audit_statement])
    logger.info("Invite for '%s' revoked by user_id=%s.", email, acting_user["user_id"])


def list_pending_invites() -> list[dict]:
    """
    Every pending (unused) invite, most recent first. Not permission-
    gated itself -- consistent with every other list_* function in this
    project (see modules/students.py's module docstring); the real gate
    is render_user_management_page() calling require_role(ROLE_ADMIN).

    Returns:
        A list of dicts: email, role, invited_by (the inviting Admin's
        username), created_at.
    """
    return fetch_all(
        "SELECT p.email, p.role, u.username AS invited_by, p.created_at "
        "FROM pending_accounts p JOIN users u ON p.invited_by = u.user_id "
        "ORDER BY p.created_at DESC"
    )


def signup_invited_account(email: str, username: str, password: str) -> int:
    """
    Let a pre-approved Teacher/Admin claim their account -- the
    Teacher/Admin counterpart to self_register_student(), gated by
    invite_account() above rather than an existing student record. See
    this module's docstring's "WHO CAN CREATE AN ACCOUNT" section for how
    this keeps the same security guarantee (nobody can grant themselves
    Teacher/Admin access; only an Admin who already invited that exact
    email can).

    Args:
        email: The invited email address.
        username: The desired login username (freely chosen here, unlike
            a Student's -- there is no roll_no to derive it from).
        password: The desired plaintext password.

    Returns:
        The new user account's user_id.

    Raises:
        RecordNotFoundError: if no pending invite exists for this email.
        ValidationError: if username or password fails validation.
        DuplicateRecordError: if the username is already taken.
    """
    from database.db_manager import execute_transaction

    email = validate_email(email)
    invite = fetch_one("SELECT role FROM pending_accounts WHERE email = ?", (email,))
    if invite is None:
        raise RecordNotFoundError(
            f"No pending invite found for '{email}'. Ask an administrator to invite you first."
        )

    username = validate_username(username)
    password = validate_password(password)

    existing_user = fetch_one("SELECT user_id FROM users WHERE username = ?", (username,))
    if existing_user is not None:
        raise DuplicateRecordError(f"Username '{username}' is already taken.")

    # No audit_log entry for the account-creation event itself, for the
    # same reason self_register_student() has never written one: an
    # audit_log row's user_id is NOT NULL and foreign-keys to an
    # EXISTING account (see database/db_setup.py's audit_log table) --
    # but the actor here IS the account being created, whose user_id
    # does not exist until this very INSERT completes. There is no
    # earlier "acting user" to attribute the row to.
    password_hash = hash_password(password)
    insert_statement = (
        "INSERT INTO users (username, password_hash, role, must_change_password, google_email) "
        "VALUES (?, ?, ?, ?, ?)",
        # force_password_change=False (0): they just chose their own
        # password, same reasoning as self_register_student().
        (username, password_hash, invite["role"], 0, None),
    )
    delete_invite_statement = ("DELETE FROM pending_accounts WHERE email = ?", (email,))
    results = execute_transaction([insert_statement, delete_invite_statement])
    user_id = results[0]

    logger.info(
        "'%s' signed up as '%s' (user_id=%s) via invite from '%s'.",
        username, invite["role"], user_id, email,
    )
    return user_id


def authenticate(username: str, password: str, expected_role: str | None = None) -> dict:
    """
    Verify a username/password pair and return the matching user's
    public details.

    SECURITY NOTE ON expected_role -- WHY IT IS PART OF THE QUERY ITSELF,
    NOT A CHECK AFTERWARDS: app.py's landing page routes a visitor to one
    of three role-specific login views (Admin/Teacher/Student), and each
    one passes its own role here as expected_role. When given, the SQL
    itself becomes "WHERE username = ? AND role = ?" -- so a Teacher's
    username simply DOES NOT MATCH any row when submitted on the Admin
    login view, even with the exact right password. This is not the same
    as fetching the user by username alone and then comparing
    user_row["role"] == expected_role in Python afterwards: that
    alternative would still need a SEPARATE error message for "right
    password, wrong portal" versus "no such user", and any observable
    difference between those two outcomes (a different message, a
    different response time, a different code path) is exactly the kind
    of side channel that lets an attacker discover which roles specific
    usernames hold -- a variant of the same "username enumeration"
    problem explained below, just leaking ROLE instead of EXISTENCE. By
    building the role into the WHERE clause, "wrong role" and "no such
    user" become the literal same case from this function's point of
    view: user_row is None either way, and the exact same branch below
    handles both identically, with the exact same message, with no
    extra code path to accidentally diverge from it.

    SECURITY NOTE ON THE GENERIC ERROR MESSAGE: whether the username does
    not exist, exists under a different role than expected_role, or
    exists but the PASSWORD is wrong, this function raises the exact same
    AuthenticationError message ("Invalid username or password."). If we
    used different messages for each case, an attacker could use those
    differences to discover which usernames are real (and which role
    each one holds), one guess at a time -- this is a well-known attack
    called "username enumeration". Giving away nothing extra closes that
    door.

    Args:
        username: The submitted username.
        password: The submitted plaintext password.
        expected_role: If given, the username must belong to EXACTLY
            this role, or authentication fails with the same generic
            error as any other mismatch. If None (the default), any
            role is accepted -- used by paths that do not route through
            a role-specific login view (e.g. a future API, or tests).

    Returns:
        A dict with keys "user_id", "username", "role",
        "must_change_password" for the authenticated user. Deliberately
        does NOT include password_hash -- this dict is what gets stored in
        the Streamlit session and we never want the hash sitting in memory
        longer than it needs to.

    Raises:
        AuthenticationError: if the username (with the matching role, if
            expected_role was given) doesn't exist, the account is
            deactivated, or the password is wrong.
    """
    if expected_role is not None:
        user_row = fetch_one(
            "SELECT user_id, username, password_hash, role, is_active, must_change_password "
            "FROM users WHERE username = ? AND role = ?",
            (username, expected_role),
        )
    else:
        user_row = fetch_one(
            "SELECT user_id, username, password_hash, role, is_active, must_change_password "
            "FROM users WHERE username = ?",
            (username,),
        )

    generic_error = "Invalid username or password."

    if user_row is None:
        logger.warning(
            "Login failed: unknown username '%s' (expected_role=%s).", username, expected_role,
        )
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
        "must_change_password": bool(user_row["must_change_password"]),
    }


def _generate_username_from_google_email(google_email: str) -> str:
    """
    Build a fresh, available, valid username from a Google email address,
    for self-registering a brand-new Admin/Teacher account that has no
    pre-existing record to take a username from (unlike a Student's own
    roll_no -- see authenticate_with_google()'s Student branch, which
    prefers that instead of this whenever a matching roll_no exists).

    Takes the email's local part (before '@'), replaces every character
    validate_username() would reject with '_', pads it out to
    config.USERNAME_MIN_LENGTH with trailing zeros if it's too short (a
    local part can legally be a single character), and appends an
    incrementing numeric suffix if that username is already taken -- so
    two different people whose emails happen to share a local part (e.g.
    two different "admin" addresses at different providers) still each
    get their own distinct username instead of colliding.
    """
    local_part = google_email.split("@", 1)[0]
    base = re.sub(r"[^A-Za-z0-9_]", "_", local_part)[: config.USERNAME_MAX_LENGTH]
    if len(base) < config.USERNAME_MIN_LENGTH:
        base = base.ljust(config.USERNAME_MIN_LENGTH, "0")

    candidate = base
    suffix = 1
    while fetch_one("SELECT user_id FROM users WHERE username = ?", (candidate,)) is not None:
        suffix += 1
        candidate = f"{base}{suffix}"[: config.USERNAME_MAX_LENGTH]

    return candidate


def authenticate_with_google(google_email: str, expected_role: str | None = None) -> dict:
    """
    The Google-authenticated counterpart to authenticate() -- see this
    module's docstring's "GOOGLE SIGN-IN" section for the full trust
    model this follows (deliberately open self-registration for every
    role, not pre-approval-gated like the username/password paths).
    Called from try_google_login() below, once Streamlit's own OAuth
    round-trip (st.login("google")) has already confirmed google_email is
    a real, provider-verified identity -- this function never itself
    talks to Google or checks a password; its job is "map this already-
    verified email to an app account, creating one if none exists yet".

    Args:
        google_email: The verified email address from st.user.email.
        expected_role: If given, the role this Google sign-in is FOR --
            set when the "Sign in with Google" button was clicked from a
            specific role's login page (see prepare_google_login()). Used
            both to reject a mismatch against an EXISTING account's real
            role, and as the role assigned to a brand-new self-registered
            account. None means "no role context available" (e.g. a
            stale identity cookie from before role-scoped login pages
            existed) -- an existing account is still reached by email
            alone, but a brand-new account cannot be created without
            knowing which role to give it.

    Returns:
        A dict with keys "user_id", "username", "role",
        "must_change_password" -- identical shape to authenticate(), so
        try_google_login() can populate the Streamlit session exactly
        the way login() already does for a password login.

    Raises:
        AuthenticationError: if the matched account has been deactivated,
            if expected_role is given and an existing account's role
            doesn't equal it, if expected_role is given and matches a
            Student roll_no whose login already exists under a DIFFERENT
            (unlinked) account, or if no account exists yet and
            expected_role is None (nothing to register the new account as).

    WHY A SPECIFIC "wrong portal" MESSAGE IS SAFE HERE, UNLIKE
    authenticate()'s DELIBERATELY GENERIC "invalid credentials": that
    genericness exists to prevent USERNAME ENUMERATION -- an attacker
    guessing usernames/roles shouldn't get different feedback for "wrong
    role" vs "no such user". That threat doesn't apply here: Google's own
    OAuth step has ALREADY proven, before this function ever runs, that
    the caller controls google_email. Telling them "this Google account
    is registered as a Teacher" is telling them something about their
    OWN account, not leaking anything about anyone else's.
    """
    google_email = validate_email(google_email)

    existing = fetch_one(
        "SELECT user_id, username, role, is_active, must_change_password "
        "FROM users WHERE google_email = ?",
        (google_email,),
    )

    if existing is not None and expected_role is not None and existing["role"] != expected_role:
        raise AuthenticationError(
            f"This Google account is registered as {existing['role'].capitalize()}, not "
            f"{expected_role.capitalize()}. Use the {existing['role'].capitalize()} login page instead."
        )

    if existing is None:
        if expected_role is None:
            raise AuthenticationError(
                "No account is linked to this Google email. Please sign in from one of the "
                "role-specific login pages (Admin, Teacher, or Student) so a new account can "
                "be created for you automatically."
            )

        # A Student signing in from the Student page whose email matches
        # a student record inherits that record's own roll_no as their
        # username (the same match self_register_student() already
        # requires for a typed-password signup) -- a more meaningful
        # username than one minted from the email, when one is available.
        # Checked regardless of is_active, unlike the match itself further
        # down: a DEACTIVATED student's email should reject clearly with
        # "this student is deactivated", not fall through and get treated
        # as a total stranger who happens to self-register a brand-new,
        # unrelated account under the same email.
        student_row = None
        if expected_role == config.ROLE_STUDENT:
            student_row = fetch_one(
                "SELECT roll_no, is_active FROM students WHERE email = ?", (google_email,),
            )
            if student_row is not None and not student_row["is_active"]:
                raise AuthenticationError(
                    "This email is on file for a deactivated student record. Contact an administrator."
                )

        if student_row is not None:
            roll_no = student_row["roll_no"]
            already_has_login = fetch_one("SELECT user_id FROM users WHERE username = ?", (roll_no,))
            if already_has_login is not None:
                raise AuthenticationError(
                    f"A login already exists for roll number '{roll_no}', but it is not linked "
                    "to this Google email yet. Log in with your username and password instead, "
                    "then link your Google account from the Change Password page."
                )

            # _insert_user_row(), not create_user(): roll_no came straight
            # from our own students table, already validated (and
            # normalised) when that record was created -- see
            # _insert_user_row()'s docstring for why running it through
            # create_user()'s OWN validate_username() on top of that would
            # be wrong, not just redundant, for a short roll number (e.g.
            # this project's real data: roll_no "1" and "2").
            user_id = _insert_user_row(
                username=roll_no, password=_random_unusable_password(), role=config.ROLE_STUDENT,
                force_password_change=False, google_email=google_email,
            )
            logger.info("Student '%s' auto-registered via Google Sign-In (user_id=%s).", roll_no, user_id)
        else:
            # No matching record of any kind -- self-register a brand-new
            # account AS whichever role the visitor selected before
            # clicking "Sign in with Google". See module docstring's
            # "GOOGLE SIGN-IN" section for the explicit trade-off this
            # accepts (open self-registration, including for Admin).
            username = _generate_username_from_google_email(google_email)
            user_id = _insert_user_row(
                username=username, password=_random_unusable_password(), role=expected_role,
                force_password_change=False, google_email=google_email,
            )
            logger.info(
                "New %s account '%s' self-registered via Google Sign-In (user_id=%s).",
                expected_role, username, user_id,
            )

        existing = fetch_one(
            "SELECT user_id, username, role, is_active, must_change_password FROM users WHERE user_id = ?",
            (user_id,),
        )

    if not existing["is_active"]:
        logger.warning("Google Sign-In failed: account '%s' is deactivated.", existing["username"])
        raise AuthenticationError("This account has been deactivated. Contact an administrator.")

    execute_write("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (existing["user_id"],))
    logger.info("User '%s' authenticated via Google Sign-In.", existing["username"])

    return {
        "user_id": existing["user_id"],
        "username": existing["username"],
        "role": existing["role"],
        "must_change_password": bool(existing["must_change_password"]),
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
        last_login, google_email (None if no Google account is linked).
        Ordered by username.
    """
    return fetch_all(
        "SELECT user_id, username, role, is_active, created_at, last_login, google_email "
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


def change_password(user_id: int, old_password: str, new_password: str) -> None:
    """
    Self-service password change: a logged-in user replaces their OWN
    password, proving they know the current one first. This is the same
    function used for both a voluntary change (User Management-adjacent
    "Change Password" page) and a FORCED first-login change (see
    require_login() below, which redirects here whenever
    must_change_password is 1) -- there is only one code path for
    "a password gets changed", not two.

    Clears must_change_password back to 0 as part of the same UPDATE,
    since a password that was just deliberately chosen no longer needs
    forcing.

    WHY THE AUDIT ENTRY NEVER INCLUDES THE PASSWORD OR ITS HASH, EVEN IN
    old_value/new_value: audit_log is meant to be safely readable by an
    Admin on the Audit Log page without ever exposing secret material --
    a hashed password is still something we'd rather not have sitting in
    a log table at all, on principle. Recording only the fact that a
    change happened (and whether it cleared a forced-change flag) is
    enough for the audit trail's purpose ("who changed what, and when").

    Args:
        user_id: The account changing its own password.
        old_password: The current plaintext password, for verification.
        new_password: The desired new plaintext password.

    Raises:
        RecordNotFoundError: if user_id does not exist.
        AuthenticationError: if old_password does not match what's stored.
        ValidationError: if new_password fails validation.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    existing = fetch_one(
        "SELECT user_id, username, password_hash, must_change_password FROM users WHERE user_id = ?",
        (user_id,),
    )
    if existing is None:
        raise RecordNotFoundError(f"No user found with user_id {user_id}.")

    if not verify_password(old_password, existing["password_hash"]):
        raise AuthenticationError("Current password is incorrect.")

    new_password = validate_password(new_password)
    new_hash = hash_password(new_password)

    update_statement = (
        "UPDATE users SET password_hash = ?, must_change_password = 0, "
        "updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (new_hash, user_id),
    )
    audit_statement = build_audit_entry(
        user_id, config.AUDIT_UPDATE, "users", str(user_id),
        old_value={"must_change_password": bool(existing["must_change_password"])},
        new_value={"must_change_password": False, "password_changed": True},
    )
    execute_transaction([update_statement, audit_statement])
    logger.info("User '%s' (user_id=%s) changed their own password.", existing["username"], user_id)


def admin_reset_password(user_id: int, new_password: str, acting_user: dict) -> None:
    """
    Admin-driven "forgot password" support: an Admin sets a NEW password
    for someone else's account (e.g. a Teacher who is locked out), and the
    account is forced to change it again on next login --
    force_password_change semantics identical to a freshly created
    account, via the same must_change_password flag. This deliberately
    does NOT require knowing the old password (an Admin resetting a
    forgotten password never could), which is exactly why it is a
    separate, Admin-only function from change_password() above rather
    than a shared one.

    Args:
        user_id: The account being reset.
        new_password: The new plaintext password, chosen by the Admin.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if user_id does not exist.
        ValidationError: if new_password fails validation.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    existing = fetch_one("SELECT user_id, username FROM users WHERE user_id = ?", (user_id,))
    if existing is None:
        raise RecordNotFoundError(f"No user found with user_id {user_id}.")

    new_password = validate_password(new_password)
    new_hash = hash_password(new_password)

    update_statement = (
        "UPDATE users SET password_hash = ?, must_change_password = 1, "
        "updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (new_hash, user_id),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "users", str(user_id),
        old_value=None, new_value={"password_reset_by_admin": True},
    )
    execute_transaction([update_statement, audit_statement])
    logger.info(
        "Password for user '%s' (user_id=%s) was reset by admin user_id=%s.",
        existing["username"], user_id, acting_user["user_id"],
    )


def _link_google_account_row(user_id: int, google_email: str, actor_user_id: int) -> str:
    """
    Shared logic behind link_google_account() (Admin, any account) and
    self_link_google_account() (any logged-in user, their OWN account
    only) below -- both need the identical duplicate-check + UPDATE +
    audit entry, differing only in WHO is allowed to call it for WHICH
    user_id, which each caller enforces separately before this runs.

    Returns:
        The linked account's username (for the caller's own log message).

    Raises:
        RecordNotFoundError: if user_id does not exist.
        ValidationError: if google_email fails validation.
        DuplicateRecordError: if google_email is already linked to a
            DIFFERENT account.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    existing = fetch_one("SELECT user_id, username, google_email FROM users WHERE user_id = ?", (user_id,))
    if existing is None:
        raise RecordNotFoundError(f"No user found with user_id {user_id}.")

    google_email = validate_email(google_email)

    already_linked = fetch_one(
        "SELECT user_id FROM users WHERE google_email = ? AND user_id != ?", (google_email, user_id),
    )
    if already_linked is not None:
        raise DuplicateRecordError("This Google email is already linked to another account.")

    update_statement = (
        "UPDATE users SET google_email = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (google_email, user_id),
    )
    audit_statement = build_audit_entry(
        actor_user_id, config.AUDIT_UPDATE, "users", str(user_id),
        old_value={"google_email": existing["google_email"]}, new_value={"google_email": google_email},
    )
    execute_transaction([update_statement, audit_statement])
    return existing["username"]


def link_google_account(user_id: int, google_email: str, acting_user: dict) -> None:
    """
    Link an EXISTING account to a Google email, so it can be reached via
    "Sign in with Google" from then on. The Admin-driven counterpart to
    self_link_google_account() below (any logged-in user linking their
    OWN account) -- this is how an Admin links SOMEONE ELSE's account,
    e.g. right after inviting a Teacher/Admin, or if that person would
    rather not do it themselves.

    Args:
        user_id: The account to link.
        google_email: The Google account's email address.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if user_id does not exist.
        ValidationError: if google_email fails validation.
        DuplicateRecordError: if google_email is already linked to a
            DIFFERENT account.
    """
    check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    username = _link_google_account_row(user_id, google_email, acting_user["user_id"])
    logger.info(
        "Google account linked for user '%s' (user_id=%s) by user_id=%s.",
        username, user_id, acting_user["user_id"],
    )


def self_link_google_account(acting_user: dict, google_email: str) -> None:
    """
    Let ANY logged-in user link their OWN account to a Google email
    themselves, without needing an Admin -- the self-service counterpart
    to link_google_account() above. Being logged in already proves they
    own this account (the same standard change_password() uses for a
    self-service password change), so no separate Admin approval is
    needed to link one MORE way of reaching an account they can already
    reach -- this can never grant a new role or a new account to anyone,
    only expand how an already-authenticated one is signed into.

    Args:
        acting_user: The logged-in user linking their own account.
        google_email: The Google account's email address.

    Raises:
        ValidationError: if google_email fails validation.
        DuplicateRecordError: if google_email is already linked to a
            DIFFERENT account.
    """
    username = _link_google_account_row(acting_user["user_id"], google_email, acting_user["user_id"])
    logger.info("User '%s' (user_id=%s) linked their own Google account.", username, acting_user["user_id"])


def unlink_google_account(user_id: int, acting_user: dict) -> None:
    """
    Reverse link_google_account(): clears google_email back to NULL, so
    the account can only be reached by username/password again (e.g. if
    a Teacher's Google account changed, or they no longer want Google
    Sign-In available for this login).

    Args:
        user_id: The account to unlink.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if user_id does not exist.
        ValidationError: if the account has no Google account linked.
    """
    from modules.audit import build_audit_entry
    from database.db_manager import execute_transaction

    check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    existing = fetch_one("SELECT user_id, username, google_email FROM users WHERE user_id = ?", (user_id,))
    if existing is None:
        raise RecordNotFoundError(f"No user found with user_id {user_id}.")
    if existing["google_email"] is None:
        raise ValidationError(f"'{existing['username']}' has no Google account linked.")

    update_statement = (
        "UPDATE users SET google_email = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
        (user_id,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "users", str(user_id),
        old_value={"google_email": existing["google_email"]}, new_value={"google_email": None},
    )
    execute_transaction([update_statement, audit_statement])
    logger.info(
        "Google account unlinked for user '%s' (user_id=%s) by user_id=%s.",
        existing["username"], user_id, acting_user["user_id"],
    )


# ---------------------------------------------------------------------------
# STREAMLIT SESSION MANAGEMENT
# ---------------------------------------------------------------------------

def login(username: str, password: str, expected_role: str | None = None) -> dict:
    """
    Authenticate a user and start their Streamlit session.

    Args:
        username: The submitted username.
        password: The submitted plaintext password.
        expected_role: If given, login only succeeds for this exact role
            -- see authenticate()'s docstring for the full security
            reasoning (this is what makes app.py's role-specific login
            views -- Admin/Teacher/Student -- actually enforce the role
            they claim to be for, at the database query level).

    Returns:
        The logged-in user's dict (user_id, username, role).

    Raises:
        AuthenticationError: if the credentials are invalid (see authenticate()).
    """
    user = authenticate(username, password, expected_role=expected_role)
    st.session_state[SESSION_KEY_USER] = user
    st.session_state[SESSION_KEY_LAST_ACTIVITY] = datetime.now()
    return user


def prepare_google_login(role: str) -> None:
    """
    Call this immediately before st.login("google"), from a role-scoped
    login page (see app.py's render_role_login_form()), so
    try_google_login() below knows which role this sign-in attempt is
    FOR once the browser returns from Google.

    WHY st.session_state SURVIVES st.login()'s FULL-PAGE REDIRECT, UNLIKE
    A NORMAL PYTHON VARIABLE: st.login() sends the browser away to
    Google entirely (not an in-app rerun), then Google redirects it back
    to this app's base URL -- a brand new page load. Streamlit's own
    session-reconnection mechanism (the browser remembers its session id
    across that navigation and presents it again when reconnecting)
    is what makes st.session_state -- and only st.session_state, not a
    plain module-level variable, which would reset -- still hold this
    value afterward.

    Args:
        role: One of config.ROLE_ADMIN/ROLE_TEACHER/ROLE_STUDENT -- the
            portal this "Sign in with Google" button was clicked from.
    """
    st.session_state[SESSION_KEY_GOOGLE_LOGIN_ROLE] = role


def try_google_login() -> bool:
    """
    If this browser already has a verified Google identity (from a
    successful st.login("google") round-trip -- see app.py's
    render_login_form()) but our OWN custom session (st.session_state)
    doesn't know about it yet, map that identity to an app account via
    authenticate_with_google() and start a normal session for it -- the
    exact same session_state keys login() populates above, so every
    existing page's require_login()/require_role() check keeps working
    completely unchanged regardless of which login path was used.

    Meant to be called once, unconditionally, at the very top of
    app.py's main() on every single script rerun.

    SAFE TO CALL EVEN WHEN GOOGLE SIGN-IN ISN'T CONFIGURED AT ALL:
    st.user.is_logged_in raises AttributeError (confirmed directly
    against Streamlit's own user_info.py source) when no [auth] section
    exists anywhere in secrets.toml -- the normal state for local
    development without Google credentials set up. That specific,
    expected case is caught here and treated as "no Google identity to
    check", not an error -- the rest of the app (username/password login)
    is completely unaffected either way.

    Returns:
        True if a session was JUST started this call (the caller should
        st.rerun() so every already-drawn widget reflects the new
        session immediately). False if there was nothing to do --
        already logged in some other way, no Google identity present, or
        Google Sign-In isn't configured.
    """
    if get_current_user() is not None:
        return False  # already logged in via some path -- nothing to do

    try:
        if not st.user.is_logged_in:
            return False
        google_email = st.user.email
    except AttributeError:
        return False  # Google Sign-In not configured in secrets.toml

    # Popped (not just read): a one-shot value, consumed by the very
    # Google round-trip it was set for. None here just means "the
    # 'Sign in with Google' button wasn't clicked from a role-scoped
    # login page" (e.g. a stale identity cookie from before role-scoped
    # pages existed) -- authenticate_with_google() treats that the same
    # as before, no role restriction.
    expected_role = st.session_state.pop(SESSION_KEY_GOOGLE_LOGIN_ROLE, None)

    try:
        user = authenticate_with_google(google_email, expected_role=expected_role)
    except AuthenticationError as error:
        st.error(str(error))
        if st.button("Log Out of Google", key="google_auth_error_logout"):
            st.logout()
            st.rerun()
        st.stop()
        return False  # unreachable -- st.stop() halts the script above

    st.session_state[SESSION_KEY_USER] = user
    st.session_state[SESSION_KEY_LAST_ACTIVITY] = datetime.now()
    return True


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

    # Also clear Streamlit's own Google identity cookie, if any --
    # otherwise try_google_login() would silently log the same
    # Google-authenticated browser right back in on the very next
    # rerun, defeating the point of clicking "Log Out". st.logout()
    # itself never raises just because Google Sign-In was never used
    # this session (confirmed directly against Streamlit's own
    # user_info.py) -- only the is_logged_in READ can raise, when no
    # [auth] section is configured at all, which is what the
    # try/except below actually guards against.
    try:
        if st.user.is_logged_in:
            st.logout()
    except AttributeError:
        pass


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


def render_change_password_page(forced: bool = False) -> None:
    """
    Streamlit page/panel: the logged-in user changes their own password.

    Used two ways by app.py:
      - forced=True: shown INSTEAD of the normal sidebar/page routing,
        the moment a user with must_change_password=1 logs in (a freshly
        Admin-created account, or the bootstrap admin -- see
        create_user()'s force_password_change parameter). There is no way
        to reach any other page until this succeeds.
      - forced=False: a normal, voluntary "Change Password" page anyone
        can visit any time from the sidebar, same underlying form and
        change_password() call.

    Args:
        forced: Changes only the page's wording (why the user is seeing
            this) -- the actual logic is identical either way.
    """
    user = require_login()

    st.title("Change Password")
    if forced:
        st.warning(
            "You must set a new password before continuing. "
            "This account was created with a temporary password."
        )
    else:
        st.caption("Change your own account password.")

    with st.form("change_password_form", clear_on_submit=True):
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Change Password", type="primary")

    if submitted:
        if new_password != confirm_password:
            st.error("New passwords do not match.")
        else:
            try:
                change_password(user["user_id"], old_password, new_password)
                # Keep the in-session user dict consistent with the
                # database we just updated -- otherwise the forced-change
                # gate in app.py would keep re-showing this page every
                # rerun, since it reads must_change_password straight from
                # st.session_state, not a fresh database query.
                st.session_state[SESSION_KEY_USER]["must_change_password"] = False
                st.success("Password changed successfully.")
                st.rerun()
            except (AuthenticationError, ValidationError) as error:
                st.error(str(error))

    if not forced:
        st.divider()
        st.subheader("Google Sign-In")
        current_google_email = fetch_one(
            "SELECT google_email FROM users WHERE user_id = ?", (user["user_id"],),
        )["google_email"]

        if current_google_email:
            st.write(f"Your account is linked to: **{current_google_email}**")
            st.caption("Contact an administrator to unlink it.")
        else:
            st.caption(
                "Link your account to a Google email so you can sign in with Google "
                "instead of your username/password."
            )
            with st.form("self_link_google_form", clear_on_submit=True):
                link_email = st.text_input("Your Google email")
                link_submitted = st.form_submit_button("Link Google Account")
            if link_submitted:
                try:
                    self_link_google_account(user, link_email)
                    st.success("Google account linked.")
                    st.rerun()
                except (ValidationError, DuplicateRecordError) as error:
                    st.error(str(error))


def render_user_management_page() -> None:
    """
    Streamlit page: Admin creates Teacher/Admin/Student login accounts by
    hand, invites a Teacher/Admin to self-register instead, and can
    deactivate/reactivate existing ones. Admin-only -- see the module
    docstring's "WHO CAN CREATE AN ACCOUNT" section for how this page's
    two Teacher/Admin creation paths (hand-created here, vs. an invited
    self-signup on the login page) both trace back to an Admin action.
    """
    current_user = require_role(config.ROLE_ADMIN)

    st.title("User Management")

    st.subheader("Create a new account")
    st.caption(
        "Use this to create a Teacher/Admin account directly by hand. Students should "
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
    st.subheader("Invite a Teacher / Admin to self-register")
    st.caption(
        "Pre-approve an email so that person can create their own account (username, "
        "password, and optionally Google Sign-In) from the login page's \"Sign Up "
        "(Invited)\" tab, instead of you creating it for them above. Tell them directly -- "
        "this does not send an email."
    )
    with st.form("invite_account_form", clear_on_submit=True):
        invite_email = st.text_input("Email to invite")
        invite_role = st.selectbox("Role", options=config.INVITABLE_ROLES, key="invite_role")
        invite_submitted = st.form_submit_button("Send Invite")

    if invite_submitted:
        try:
            invite_account(invite_email, invite_role, current_user)
            st.success(f"'{invite_email}' invited as {invite_role}.")
            st.rerun()
        except (ValidationError, DuplicateRecordError) as error:
            st.error(str(error))

    pending_invites = list_pending_invites()
    if pending_invites:
        st.write("**Pending invites**")
        invite_display_rows = [
            {
                "Email": inv["email"], "Role": inv["role"],
                "Invited By": inv["invited_by"], "Invited On": inv["created_at"],
            }
            for inv in pending_invites
        ]
        st.dataframe(invite_display_rows, use_container_width=True, hide_index=True)

        revoke_email_choice = st.selectbox(
            "Revoke an invite", options=[inv["email"] for inv in pending_invites],
        )
        if st.button("Revoke Selected Invite"):
            try:
                revoke_invite(revoke_email_choice, current_user)
                st.success(f"Invite for '{revoke_email_choice}' revoked.")
                st.rerun()
            except RecordNotFoundError as error:
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
            "Google Account": u["google_email"] or "Not linked",
            "Created": u["created_at"],
            "Last Login": u["last_login"] or "Never",
        }
        for u in users
    ]
    render_data_table(display_rows, key_prefix="users_table", filename_prefix="users")

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

    st.subheader("Reset a forgotten password")
    st.caption(
        "Sets a new password for the selected account and forces them to "
        "change it again the next time they log in."
    )
    with st.form("reset_password_form", clear_on_submit=True):
        reset_new_password = st.text_input("New temporary password", type="password")
        reset_submitted = st.form_submit_button("Reset Password")

    if reset_submitted:
        try:
            admin_reset_password(selected_user["user_id"], reset_new_password, current_user)
            st.success(f"Password for '{username_choice}' has been reset.")
        except ValidationError as error:
            st.error(str(error))

    st.subheader("Link / unlink Google Sign-In")
    st.caption(
        "Students are linked automatically the first time they sign in with a Google "
        "account matching their student record's email. Teacher and Admin accounts must "
        "be linked here -- there is no self-service path for those roles."
    )
    if selected_user["google_email"]:
        st.write(f"Currently linked to: **{selected_user['google_email']}**")
        if st.button("Unlink Google Account"):
            try:
                unlink_google_account(selected_user["user_id"], current_user)
                st.success(f"Google account unlinked for '{username_choice}'.")
                st.rerun()
            except (ValidationError, RecordNotFoundError) as error:
                st.error(str(error))
    else:
        with st.form("link_google_form", clear_on_submit=True):
            new_google_email = st.text_input("Google email to link")
            link_submitted = st.form_submit_button("Link Google Account")
        if link_submitted:
            try:
                link_google_account(selected_user["user_id"], new_google_email, current_user)
                st.success(f"Google account linked for '{username_choice}'.")
                st.rerun()
            except (ValidationError, RecordNotFoundError, DuplicateRecordError) as error:
                st.error(str(error))

    st.divider()
    st.subheader("Full Database Backup")
    st.caption(
        "Downloads every table in the system (students, subjects, marks, attendance, "
        "assignments, semesters, teacher assignments, user accounts, and the audit log) "
        "as a single .zip of CSV files. Passwords are never included in this export."
    )
    if st.button("Prepare Backup"):
        backup_bytes = generate_full_backup()
        st.download_button(
            "Download Backup (.zip)", data=backup_bytes,
            file_name=backup_filename(), mime="application/zip",
            icon=":material/download:",
        )


if __name__ == "__main__":
    # Run once, from the project root: python -m modules.auth
    # Creates the first admin account so there is a way to log in at all.
    ensure_default_admin_exists()
