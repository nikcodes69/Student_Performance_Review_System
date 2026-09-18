"""
utils/validators.py
====================
The validation layer: every function here checks ONE piece of input data
against a business rule and either returns a cleaned-up value (success) or
raises utils.exceptions.ValidationError with a human-readable message
(failure).

WHY THIS LAYER EXISTS, GIVEN THE DATABASE ALREADY HAS CHECK CONSTRAINTS:
database/db_setup.py already enforces things like "semester must be
between 1 and 8" at the database level. So why check it again here? Two
reasons:

  1. FRIENDLY ERRORS. If bad data reaches the database, SQLite raises a
     raw sqlite3.IntegrityError like "CHECK constraint failed: semester
     BETWEEN 1 AND 8". That is not something we want to show a Teacher
     typing into a Streamlit form. Catching the problem here first lets us
     raise ValidationError("Semester must be between 1 and 8.") instead --
     a message meant for a human. See validate_mark_value() below for the
     same reasoning applied to marks: every subject uses the exact same
     fixed marks breakdown (config.MAX_INTERNAL_MARKS/MAX_EXTERNAL_MARKS/
     MAX_PRACTICAL_MARKS -- see config.py), so the database's own CHECK
     constraints on the marks table now express the EXACT real rule
     directly (they no longer need a second, more precise layer above
     them the way an earlier, per-subject-configurable version of this
     schema did) -- this file still checks it first purely for the
     friendlier message.
  2. FAIL BEFORE SPENDING A DATABASE CALL. Rejecting bad input in Python,
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
    Validate the LOCAL 10-digit part of a Nepali mobile number, and
    return the full number with config.PHONE_COUNTRY_CODE ("+977")
    attached.

    THE COUNTRY CODE IS FIXED BY THE SYSTEM, NEVER TYPED BY A USER: every
    phone field in this app (modules/students.py's create/edit forms, and
    the bulk-import "phone" column) only ever collects the 10-digit local
    number -- the "+977" is shown as a separate, disabled box next to the
    input (see render_students_page()), not part of what a person types
    or what gets passed into this function. This function is what
    actually attaches the prefix, so the DATABASE always stores ONE
    consistent, unambiguous format ("+977XXXXXXXXXX") -- the same
    principle validate_email() already applies by always lower-casing,
    and validate_roll_no() by always upper-casing: the caller never needs
    to guess which format is sitting in a given column, because this
    layer normalises it once, here, on every write.

    THIS FUNCTION DOES NOT TRY TO STRIP A REDUNDANT PREFIX IF ONE WAS
    PASTED IN BY MISTAKE (e.g. "+9779812345678" or "9779812345678"): it
    is deliberately strict, not lenient -- if the input is not EXACTLY
    the 10-digit local part, it fails validation with a clear message,
    rather than silently guessing what the user meant. Predictable
    failure is easier to defend than silent "magic" reformatting.

    Rules: exactly 10 digits, starting with 97 or 98 (see
    config.PHONE_LOCAL_REGEX -- this project's Nepali mobile numbers
    starting with 96 are no longer accepted for NEW entries, a
    deliberate tightening of the previous 96/97/98 rule).

    Args:
        phone: The raw 10-digit local number (never includes "+977").

    Returns:
        "+977" followed by the validated 10-digit number, e.g.
        "+9779812345678".

    Raises:
        ValidationError: if phone is empty or not exactly a 10-digit
            number starting with 97 or 98.
    """
    phone = _require_non_empty(phone, "Phone number")

    if not re.fullmatch(config.PHONE_LOCAL_REGEX, phone):
        raise ValidationError(
            "Phone number must be exactly 10 digits, starting with 97 or 98 "
            "(do not include the +977 country code -- it's added automatically)."
        )

    return f"{config.PHONE_COUNTRY_CODE}{phone}"


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
    Validate ONE mark component (internal, external, or practical)
    against its ceiling.

    STILL TAKES max_allowed AS A PARAMETER, RATHER THAN READING
    config.MAX_INTERNAL_MARKS/etc. DIRECTLY: every subject now uses the
    exact same fixed marks breakdown (see config.py), so in practice the
    caller (modules/marks.py) always passes one of those three constants
    -- but keeping this function generic, rather than hard-coding which
    constant applies to which component, means it stays exactly as
    reusable and exactly as easy to unit-test with plain numbers as it
    already was (see tests/test_validators.py) -- the same reasoning
    modules/grades.py's calculate_percentage() already documents for
    taking its own maximums as parameters instead of importing config
    directly.

    Args:
        value: The mark obtained.
        max_allowed: The maximum possible mark for this component (one
            of config.MAX_INTERNAL_MARKS/MAX_EXTERNAL_MARKS/
            MAX_PRACTICAL_MARKS, passed in by the caller).
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
            f"{component_name.capitalize()} marks ({value}) cannot exceed the maximum "
            f"of {max_allowed}."
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
# ASSIGNMENT FIELDS (used by modules/assignments.py)
# ---------------------------------------------------------------------------

def validate_assignment_values(total_assigned: int, submitted: int) -> None:
    """
    Validate a pair of assignment numbers together -- mirrors
    validate_attendance_values() exactly, for the same reason: these two
    values depend on each other, so they are validated as a pair rather
    than as two independent single-field checks.

    Mirrors the assignments table's CHECK constraint exactly: both values
    must be non-negative, and a student cannot have submitted more
    assignments than were actually assigned.

    Args:
        total_assigned: Total assignments given for this subject/semester.
        submitted: Assignments the student actually submitted.

    Raises:
        ValidationError: if either value is negative, or submitted > total_assigned.
    """
    if total_assigned < 0:
        raise ValidationError("Total assigned cannot be negative.")

    if submitted < 0:
        raise ValidationError("Submitted cannot be negative.")

    if submitted > total_assigned:
        raise ValidationError("Submitted cannot be more than total assigned.")


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
