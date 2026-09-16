"""
utils/validators.py
====================
The validation layer: every function here checks ONE piece of input data
against a business rule and either returns a cleaned-up value (success) or
raises utils.exceptions.ValidationError with a human-readable message
(failure).

WHY THIS LAYER EXISTS, GIVEN THE DATABASE ALREADY HAS CHECK CONSTRAINTS:
database/db_setup.py already enforces things like "semester must be
between 1 and 8" at the database level. So why check it again here? Three
reasons:

  1. FRIENDLY ERRORS. If bad data reaches the database, SQLite raises a
     raw sqlite3.IntegrityError like "CHECK constraint failed: semester
     BETWEEN 1 AND 8". That is not something we want to show a Teacher
     typing into a Streamlit form. Catching the problem here first lets us
     raise ValidationError("Semester must be between 1 and 8.") instead --
     a message meant for a human.
  2. PRECISION THE DATABASE CANNOT EXPRESS. The real ceiling for a mark
     ("internal must not exceed THIS subject's max_internal") depends on
     looking up a value in a different table (subjects). SQLite's CHECK
     constraints cannot do that reliably. This file CAN, because it runs
     in Python and can query the database first. See validate_mark_value()
     below -- this is the function referenced in database/db_setup.py's
     comment on the marks table.
  3. FAIL BEFORE SPENDING A DATABASE CALL. Rejecting bad input in Python,
     before opening a transaction, is cheaper and keeps invalid data from
     ever touching the database in the first place -- "defense in depth":
     two independent layers (this file, and the database's own CHECK
     constraints) both have to agree data is valid before it's stored.

CONVENTION USED BY EVERY FUNCTION BELOW:
    cleaned_value = validate_something(raw_value)
Each function returns a CLEANED version of its input on success (e.g. with
leading/trailing whitespace removed, or case normalised) and raises
ValidationError on failure. It never returns False or None to mean
"invalid" -- if a function returns at all, the value is valid. This means
calling code never needs an if/else to check the result; it just calls the
function and keeps going, e.g.:

    name = validate_name(form_input)   # raises ValidationError, or continues
"""

import re
from datetime import datetime

import config
from utils.exceptions import ValidationError


def _require_non_empty(value: str, field_name: str) -> str:
    """
    Shared helper: confirm a string field was actually provided and is not
    just whitespace, then return it with whitespace stripped.

    This is used inside several validators below so the same three-line
    check (None? empty after stripping?) is not repeated in every one of
    them -- this is plain code reuse (a helper function), not the kind of
    "advanced OOP" you asked to avoid; there is no class or inheritance
    here.

    Args:
        value: The raw string to check.
        field_name: Human-readable field name, used in the error message.

    Returns:
        The value with leading/trailing whitespace removed.

    Raises:
        ValidationError: if value is None or empty/whitespace-only.
    """
    if value is None:
        raise ValidationError(f"{field_name} is required.")
    stripped = value.strip()
    if not stripped:
        raise ValidationError(f"{field_name} cannot be empty.")
    return stripped


# ---------------------------------------------------------------------------
# USER ACCOUNT FIELDS (used by modules/auth.py)
# ---------------------------------------------------------------------------

def validate_username(username: str) -> str:
    """
    Validate a login username.

    Rules: required, length between config.USERNAME_MIN_LENGTH and
    config.USERNAME_MAX_LENGTH, and only letters, digits, and underscores
    (this keeps usernames simple to type and safe to display anywhere in
    the UI without needing to escape special characters).

    Args:
        username: The raw username.

    Returns:
        The username with whitespace stripped.

    Raises:
        ValidationError: if any rule above is broken.
    """
    username = _require_non_empty(username, "Username")

    if not (config.USERNAME_MIN_LENGTH <= len(username) <= config.USERNAME_MAX_LENGTH):
        raise ValidationError(
            f"Username must be between {config.USERNAME_MIN_LENGTH} and "
            f"{config.USERNAME_MAX_LENGTH} characters long."
        )

    if not re.fullmatch(r"[A-Za-z0-9_]+", username):
        raise ValidationError(
            "Username can only contain letters, digits, and underscores."
        )

    return username


def validate_password(password: str) -> str:
    """
    Validate a plaintext password BEFORE it is hashed with bcrypt.

    Rule: must be at least config.PASSWORD_MIN_LENGTH characters. We do
    not strip whitespace here (unlike other text fields) because a
    trailing space could be a character the user genuinely intended to
    type as part of their password.

    Args:
        password: The raw plaintext password.

    Returns:
        The password, unchanged.

    Raises:
        ValidationError: if password is missing or too short.
    """
    if not password:
        raise ValidationError("Password is required.")

    if len(password) < config.PASSWORD_MIN_LENGTH:
        raise ValidationError(
            f"Password must be at least {config.PASSWORD_MIN_LENGTH} characters long."
        )

    return password


def validate_role(role: str) -> str:
    """
    Validate a user role against the fixed set of roles this system
    supports.

    Args:
        role: The raw role string.

    Returns:
        The role, unchanged.

    Raises:
        ValidationError: if role is not one of config.VALID_ROLES.
    """
    if role not in config.VALID_ROLES:
        allowed = ", ".join(config.VALID_ROLES)
        raise ValidationError(f"Role must be one of: {allowed}.")

    return role


# ---------------------------------------------------------------------------
# STUDENT FIELDS (used by modules/students.py)
# ---------------------------------------------------------------------------

def validate_roll_no(roll_no: str) -> str:
    """
    Validate a student roll number.

    Rules: required, at most config.ROLL_NO_MAX_LENGTH characters.
    Normalised to UPPERCASE -- roll_no is used as a primary key and as a
    foreign key in marks/attendance/semesters, and SQLite's default TEXT
    comparison is case-sensitive, so "bca001" and "BCA001" would otherwise
    be treated as two different students. Normalising once, here, means
    that can never happen.

    Args:
        roll_no: The raw roll number.

    Returns:
        The roll number, stripped and uppercased.

    Raises:
        ValidationError: if roll_no is empty or too long.
    """
    roll_no = _require_non_empty(roll_no, "Roll number")

    if len(roll_no) > config.ROLL_NO_MAX_LENGTH:
        raise ValidationError(
            f"Roll number cannot be longer than {config.ROLL_NO_MAX_LENGTH} characters."
        )

    return roll_no.upper()


def validate_name(name: str) -> str:
    """
    Validate a person's full name.

    Rules: required, at most config.NAME_MAX_LENGTH characters, and only
    letters, spaces, apostrophes, and hyphens (covers names like
    "Mary-Jane" or "O'Brien" without allowing digits or symbols).

    Args:
        name: The raw name.

    Returns:
        The name, stripped.

    Raises:
        ValidationError: if any rule above is broken.
    """
    name = _require_non_empty(name, "Name")

    if len(name) > config.NAME_MAX_LENGTH:
        raise ValidationError(f"Name cannot be longer than {config.NAME_MAX_LENGTH} characters.")

    if not re.fullmatch(r"[A-Za-z' \-]+", name):
        raise ValidationError("Name can only contain letters, spaces, hyphens, and apostrophes.")

    return name


def validate_semester(semester: int) -> int:
    """
    Validate a semester number.

    Args:
        semester: The semester as an integer (1-8).

    Returns:
        The semester, unchanged.

    Raises:
        ValidationError: if semester is outside config.MIN_SEMESTER..MAX_SEMESTER.
    """
    if not (config.MIN_SEMESTER <= semester <= config.MAX_SEMESTER):
        raise ValidationError(
            f"Semester must be between {config.MIN_SEMESTER} and {config.MAX_SEMESTER}."
        )

    return semester


def validate_branch(branch: str) -> str:
    """
    Validate a student's branch/program name (e.g. "BCA").

    Args:
        branch: The raw branch name.

    Returns:
        The branch name, stripped.

    Raises:
        ValidationError: if branch is empty or too long.
    """
    branch = _require_non_empty(branch, "Branch")

    if len(branch) > config.NAME_MAX_LENGTH:
        raise ValidationError(f"Branch cannot be longer than {config.NAME_MAX_LENGTH} characters.")

    return branch


def validate_email(email: str) -> str:
    """
    Validate an email address against config.EMAIL_REGEX.

    Args:
        email: The raw email address.

    Returns:
        The email, stripped and lower-cased (email addresses are
        conventionally treated as case-insensitive).

    Raises:
        ValidationError: if email is empty or badly formatted.
    """
    email = _require_non_empty(email, "Email")
    email = email.lower()

    if not re.fullmatch(config.EMAIL_REGEX, email):
        raise ValidationError("Email address is not a valid format (example: name@example.com).")

    return email


def validate_phone(phone: str) -> str:
    """
    Validate a phone number against config.PHONE_REGEX (a 10-digit Nepali
    mobile number starting with 96, 97, or 98).

    Args:
        phone: The raw phone number.

    Returns:
        The phone number, stripped.

    Raises:
        ValidationError: if phone is empty or badly formatted.
    """
    phone = _require_non_empty(phone, "Phone number")

    if not re.fullmatch(config.PHONE_REGEX, phone):
        raise ValidationError(
            "Phone number must be a valid 10-digit mobile number starting with 96, 97, or 98."
        )

    return phone


def validate_admission_year(admission_year: int) -> int:
    """
    Validate a student's admission year.

    The upper bound is computed as THIS YEAR, right now, using
    datetime.now() -- not a hard-coded number -- so this validator keeps
    working correctly next year without needing to be edited (see the
    comment on config.ADMISSION_YEAR_MIN for the same reasoning).

    Args:
        admission_year: The year the student was admitted.

    Returns:
        The admission year, unchanged.

    Raises:
        ValidationError: if the year is before config.ADMISSION_YEAR_MIN
            or after the current year.
    """
    current_year = datetime.now().year

    if not (config.ADMISSION_YEAR_MIN <= admission_year <= current_year):
        raise ValidationError(
            f"Admission year must be between {config.ADMISSION_YEAR_MIN} and {current_year}."
        )

    return admission_year


# ---------------------------------------------------------------------------
# SUBJECT FIELDS (used by modules/subjects.py)
# ---------------------------------------------------------------------------

def validate_subject_code(subject_code: str) -> str:
    """
    Validate a subject code (e.g. "CACS201").

    Normalised to UPPERCASE for the same reason as validate_roll_no: it is
    used as a primary/foreign key and must compare consistently.

    Args:
        subject_code: The raw subject code.

    Returns:
        The subject code, stripped and uppercased.

    Raises:
        ValidationError: if subject_code is empty or too long.
    """
    subject_code = _require_non_empty(subject_code, "Subject code")

    if len(subject_code) > config.ROLL_NO_MAX_LENGTH:
        raise ValidationError(
            f"Subject code cannot be longer than {config.ROLL_NO_MAX_LENGTH} characters."
        )

    return subject_code.upper()


def validate_credits(credits: int) -> int:
    """
    Validate a subject's credit value.

    Args:
        credits: The credit value.

    Returns:
        The credits, unchanged.

    Raises:
        ValidationError: if credits is outside config.MIN_CREDITS..MAX_CREDITS.
    """
    if not (config.MIN_CREDITS <= credits <= config.MAX_CREDITS):
        raise ValidationError(
            f"Credits must be between {config.MIN_CREDITS} and {config.MAX_CREDITS}."
        )

    return credits


def validate_subject_name(name: str) -> str:
    """
    Validate a subject/course name (e.g. "C++ Programming", "DBMS-II").

    Deliberately MORE PERMISSIVE than validate_name() above: a person's
    name and a course name are different kinds of data. validate_name()
    restricts to letters/spaces/hyphens/apostrophes, which would reject
    completely normal subject names containing "+", digits, or
    parentheses. This validator only checks that something was actually
    provided, and that it fits in the column -- it does not restrict
    which characters a course title may contain.

    Args:
        name: The raw subject name.

    Returns:
        The subject name, stripped.

    Raises:
        ValidationError: if empty or too long.
    """
    name = _require_non_empty(name, "Subject name")

    if len(name) > config.NAME_MAX_LENGTH:
        raise ValidationError(
            f"Subject name cannot be longer than {config.NAME_MAX_LENGTH} characters."
        )

    return name


def validate_max_marks_configuration(
    max_internal: int, max_external: int, max_practical: int
) -> None:
    """
    Validate the maximum-marks configuration for a subject.

    Mirrors the subjects table's CHECK constraints (none negative, not
    all three zero at once -- a subject needs at least one gradable
    component), PLUS one more rule the database itself cannot express:
    none of the three may exceed config.MAX_MARK_CEILING. That ceiling is
    also the upper bound the marks table's own CHECK constraints enforce
    on marks.internal/external/practical (see database/db_setup.py) -- so
    if a subject's max_internal were allowed to be, say, 150, no student
    could ever actually be given a matching mark, because the database
    would reject any internal mark above 100 regardless of what this
    subject claims its maximum is. Keeping both bounds equal here prevents
    that silent inconsistency.

    Args:
        max_internal: Maximum internal marks for the subject.
        max_external: Maximum external (theory exam) marks for the subject.
        max_practical: Maximum practical marks for the subject.

    Raises:
        ValidationError: if any value is negative, exceeds
            config.MAX_MARK_CEILING, or all three are zero.
    """
    for label, value in (
        ("max_internal", max_internal),
        ("max_external", max_external),
        ("max_practical", max_practical),
    ):
        if value < 0:
            raise ValidationError(f"{label} cannot be negative.")
        if value > config.MAX_MARK_CEILING:
            raise ValidationError(
                f"{label} cannot exceed {config.MAX_MARK_CEILING} "
                "(the maximum any single mark component can ever be)."
            )

    if max_internal + max_external + max_practical <= 0:
        raise ValidationError(
            "A subject must have at least one gradable component "
            "(max_internal, max_external, or max_practical greater than zero)."
        )


# ---------------------------------------------------------------------------
# MARKS FIELDS (used by modules/marks.py)
# ---------------------------------------------------------------------------

def validate_exam_type(exam_type: str) -> str:
    """
    Validate an exam type against config.EXAM_TYPES.

    Args:
        exam_type: The raw exam type.

    Returns:
        The exam type, unchanged.

    Raises:
        ValidationError: if exam_type is not one of config.EXAM_TYPES.
    """
    if exam_type not in config.EXAM_TYPES:
        allowed = ", ".join(config.EXAM_TYPES)
        raise ValidationError(f"Exam type must be one of: {allowed}.")

    return exam_type


def validate_mark_value(value: int, max_allowed: int, component_name: str) -> int:
    """
    Validate ONE mark component (internal, external, or practical) against
    the ACTUAL maximum for the specific subject it belongs to.

    This is the precise, per-subject check that database/db_setup.py's
    comment on the marks table refers to: the database itself can only
    enforce a generic 0-100 sanity ceiling (config.MAX_MARK_CEILING),
    because a CHECK constraint cannot look up another table's value. This
    function CAN, because the caller (modules/marks.py) first looks up the
    subject's real max_internal/max_external/max_practical from the
    subjects table and passes it in as max_allowed.

    Args:
        value: The mark obtained.
        max_allowed: The maximum possible mark for this component, as
            configured for this specific subject (e.g. subjects.max_internal).
        component_name: Which component this is ("internal", "external",
            or "practical"), used in the error message.

    Returns:
        The mark value, unchanged.

    Raises:
        ValidationError: if value is negative or greater than max_allowed.
    """
    if value < 0:
        raise ValidationError(f"{component_name.capitalize()} marks cannot be negative.")

    if value > max_allowed:
        raise ValidationError(
            f"{component_name.capitalize()} marks ({value}) cannot exceed "
            f"the maximum for this subject ({max_allowed})."
        )

    return value


# ---------------------------------------------------------------------------
# ATTENDANCE FIELDS (used by modules/attendance.py)
# ---------------------------------------------------------------------------

def validate_attendance_values(classes_held: int, classes_attended: int) -> None:
    """
    Validate a pair of attendance numbers together (they depend on each
    other, so they are validated together rather than as two separate
    single-field functions).

    Mirrors the attendance table's CHECK constraint exactly: both values
    must be non-negative, and a student cannot have attended more classes
    than were held.

    Args:
        classes_held: Total classes held for this subject/semester.
        classes_attended: Classes the student actually attended.

    Raises:
        ValidationError: if either value is negative, or attended > held.
    """
    if classes_held < 0:
        raise ValidationError("Classes held cannot be negative.")

    if classes_attended < 0:
        raise ValidationError("Classes attended cannot be negative.")

    if classes_attended > classes_held:
        raise ValidationError("Classes attended cannot be more than classes held.")


# ---------------------------------------------------------------------------
# AUDIT LOG FIELDS (used by modules/audit.py)
# ---------------------------------------------------------------------------

def validate_audit_action(action: str) -> str:
    """
    Validate an audit_log action against config.AUDIT_ACTIONS.

    Args:
        action: The raw action string.

    Returns:
        The action, unchanged.

    Raises:
        ValidationError: if action is not one of config.AUDIT_ACTIONS.
    """
    if action not in config.AUDIT_ACTIONS:
        allowed = ", ".join(config.AUDIT_ACTIONS)
        raise ValidationError(f"Audit action must be one of: {allowed}.")

    return action


def validate_table_name(table_name: str) -> str:
    """
    Validate a table name against config.AUDITED_TABLES.

    This catches a typo (e.g. "student" instead of "students") at the
    moment an audit entry is created, rather than letting it become a
    row in audit_log that silently never matches the real table it was
    supposed to describe.

    Args:
        table_name: The raw table name.

    Returns:
        The table name, unchanged.

    Raises:
        ValidationError: if table_name is not one of config.AUDITED_TABLES.
    """
    if table_name not in config.AUDITED_TABLES:
        allowed = ", ".join(config.AUDITED_TABLES)
        raise ValidationError(f"table_name must be one of: {allowed}.")

    return table_name
