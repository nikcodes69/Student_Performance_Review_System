"""
tests/test_validators.py
=========================
pytest unit tests for utils/validators.py.

HOW TO RUN (from the project root, student_performance_system/):
    python -m pytest tests/test_validators.py -v

WHY "python -m pytest" AND NOT JUST "pytest":
Same reason explained in database/db_setup.py -- these tests do
`import config` and `from utils.validators import ...`, and those modules
live at the project root. Running "python -m pytest" adds the current
working directory to Python's import path before pytest starts, so the
project root is visible. Running the bare "pytest" command sometimes works
too depending on how pytest is installed, but "python -m pytest" is the
reliable version, so that's the one we standardise on.

TESTING STYLE:
Every validator follows one of two outcomes: it returns a cleaned value
(valid input) or raises ValidationError (invalid input). So every test
below does one of two things:
  - calls the function and asserts on the RETURN VALUE (valid case), or
  - wraps the call in `with pytest.raises(ValidationError):` and asserts
    it does raise (invalid case).
We do not test the private helper _require_non_empty() directly -- it is
an internal implementation detail of validators.py, not something other
files call. Testing it indirectly, through the public functions that use
it, is the correct level to test at.
"""

import pytest

from utils.exceptions import ValidationError
from utils.validators import (
    validate_admission_year,
    validate_assignment_values,
    validate_attendance_values,
    validate_audit_action,
    validate_branch,
    validate_credits,
    validate_email,
    validate_exam_type,
    validate_mark_value,
    validate_max_marks_configuration,
    validate_name,
    validate_password,
    validate_phone,
    validate_role,
    validate_roll_no,
    validate_semester,
    validate_subject_code,
    validate_subject_name,
    validate_table_name,
    validate_username,
)


# ---------------------------------------------------------------------------
# validate_username
# ---------------------------------------------------------------------------

def test_validate_username_accepts_valid_username():
    assert validate_username("teacher01") == "teacher01"


def test_validate_username_strips_whitespace():
    assert validate_username("  teacher01  ") == "teacher01"


def test_validate_username_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_username("")


def test_validate_username_rejects_too_short():
    with pytest.raises(ValidationError):
        validate_username("ab")  # below USERNAME_MIN_LENGTH


def test_validate_username_rejects_special_characters():
    with pytest.raises(ValidationError):
        validate_username("teacher@01")


# ---------------------------------------------------------------------------
# validate_password
# ---------------------------------------------------------------------------

def test_validate_password_accepts_valid_password():
    assert validate_password("supersecret123") == "supersecret123"


def test_validate_password_rejects_too_short():
    with pytest.raises(ValidationError):
        validate_password("short")


def test_validate_password_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_password("")


def test_validate_password_does_not_strip_whitespace():
    # A password's exact characters (including spaces) must be preserved --
    # stripping it would silently change what the user typed.
    password = "  spacedpass123  "
    assert validate_password(password) == password


# ---------------------------------------------------------------------------
# validate_role
# ---------------------------------------------------------------------------

def test_validate_role_accepts_each_valid_role():
    assert validate_role("admin") == "admin"
    assert validate_role("teacher") == "teacher"
    assert validate_role("student") == "student"


def test_validate_role_rejects_unknown_role():
    with pytest.raises(ValidationError):
        validate_role("superuser")


# ---------------------------------------------------------------------------
# validate_roll_no
# ---------------------------------------------------------------------------

def test_validate_roll_no_uppercases_result():
    assert validate_roll_no("bca001") == "BCA001"


def test_validate_roll_no_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_roll_no("   ")


def test_validate_roll_no_rejects_too_long():
    with pytest.raises(ValidationError):
        validate_roll_no("X" * 25)  # ROLL_NO_MAX_LENGTH is 20


# ---------------------------------------------------------------------------
# validate_name
# ---------------------------------------------------------------------------

def test_validate_name_accepts_valid_name():
    assert validate_name("Mary-Jane O'Connor") == "Mary-Jane O'Connor"


def test_validate_name_rejects_digits():
    with pytest.raises(ValidationError):
        validate_name("Student123")


def test_validate_name_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_name("")


# ---------------------------------------------------------------------------
# validate_semester
# ---------------------------------------------------------------------------

def test_validate_semester_accepts_boundary_values():
    assert validate_semester(1) == 1
    assert validate_semester(8) == 8


def test_validate_semester_rejects_zero():
    with pytest.raises(ValidationError):
        validate_semester(0)


def test_validate_semester_rejects_nine():
    with pytest.raises(ValidationError):
        validate_semester(9)


# ---------------------------------------------------------------------------
# validate_branch
# ---------------------------------------------------------------------------

def test_validate_branch_accepts_valid_branch():
    assert validate_branch("BCA") == "BCA"


def test_validate_branch_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_branch("")


# ---------------------------------------------------------------------------
# validate_email
# ---------------------------------------------------------------------------

def test_validate_email_accepts_valid_email():
    assert validate_email("Student@Example.com") == "student@example.com"


def test_validate_email_rejects_missing_at_symbol():
    with pytest.raises(ValidationError):
        validate_email("studentexample.com")


def test_validate_email_rejects_missing_domain():
    with pytest.raises(ValidationError):
        validate_email("student@")


# ---------------------------------------------------------------------------
# validate_phone
# ---------------------------------------------------------------------------
# The "+977" country code is fixed by the system, never typed by a user
# (see validate_phone()'s docstring) -- these tests pass in only the
# 10-digit LOCAL number, exactly what modules/students.py's phone input
# widget actually collects, and check that "+977" comes back attached.

def test_validate_phone_accepts_valid_number_starting_98():
    assert validate_phone("9812345678") == "+9779812345678"


def test_validate_phone_accepts_valid_number_starting_97():
    assert validate_phone("9741234567") == "+9779741234567"


def test_validate_phone_rejects_wrong_length_too_short():
    with pytest.raises(ValidationError):
        validate_phone("98123456")  # only 8 digits


def test_validate_phone_rejects_wrong_length_too_long():
    with pytest.raises(ValidationError):
        validate_phone("981234567890")  # 12 digits


def test_validate_phone_rejects_wrong_prefix():
    with pytest.raises(ValidationError):
        validate_phone("9512345678")  # must start with 97 or 98, not 95


def test_validate_phone_rejects_96_prefix():
    # A deliberate tightening from the previous 96/97/98 rule -- 96 is no
    # longer accepted for new entries (see config.PHONE_LOCAL_REGEX).
    with pytest.raises(ValidationError):
        validate_phone("9612345678")


def test_validate_phone_rejects_already_prefixed_input():
    # Strict, not lenient -- see validate_phone()'s docstring on why a
    # redundant "+977"/"977" pasted into the local-number field is
    # rejected outright rather than silently stripped.
    with pytest.raises(ValidationError):
        validate_phone("+9779812345678")


def test_validate_phone_rejects_non_digit_characters():
    with pytest.raises(ValidationError):
        validate_phone("98123 45678")  # space


# ---------------------------------------------------------------------------
# validate_admission_year
# ---------------------------------------------------------------------------

def test_validate_admission_year_accepts_current_year():
    from datetime import datetime
    current_year = datetime.now().year
    assert validate_admission_year(current_year) == current_year


def test_validate_admission_year_rejects_future_year():
    from datetime import datetime
    future_year = datetime.now().year + 1
    with pytest.raises(ValidationError):
        validate_admission_year(future_year)


def test_validate_admission_year_rejects_too_early():
    with pytest.raises(ValidationError):
        validate_admission_year(1999)  # before ADMISSION_YEAR_MIN (2000)


# ---------------------------------------------------------------------------
# validate_subject_code
# ---------------------------------------------------------------------------

def test_validate_subject_code_uppercases_result():
    assert validate_subject_code("cacs201") == "CACS201"


def test_validate_subject_code_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_subject_code("")


# ---------------------------------------------------------------------------
# validate_subject_name
# ---------------------------------------------------------------------------

def test_validate_subject_name_accepts_punctuation_and_digits():
    # A person's name (validate_name) would reject all of these -- subject
    # names legitimately contain "+", digits, and parentheses.
    assert validate_subject_name("C++ Programming") == "C++ Programming"
    assert validate_subject_name("DBMS-II") == "DBMS-II"
    assert validate_subject_name("E-Governance (Elective)") == "E-Governance (Elective)"


def test_validate_subject_name_rejects_empty_string():
    with pytest.raises(ValidationError):
        validate_subject_name("   ")


# ---------------------------------------------------------------------------
# validate_credits
# ---------------------------------------------------------------------------

def test_validate_credits_accepts_boundary_values():
    assert validate_credits(1) == 1
    assert validate_credits(6) == 6


def test_validate_credits_rejects_zero():
    with pytest.raises(ValidationError):
        validate_credits(0)


def test_validate_credits_rejects_too_high():
    with pytest.raises(ValidationError):
        validate_credits(7)


# ---------------------------------------------------------------------------
# validate_max_marks_configuration
# ---------------------------------------------------------------------------

def test_validate_max_marks_configuration_accepts_valid_split():
    # Should not raise.
    validate_max_marks_configuration(max_internal=20, max_external=80, max_practical=0)


def test_validate_max_marks_configuration_rejects_negative_value():
    with pytest.raises(ValidationError):
        validate_max_marks_configuration(max_internal=-5, max_external=80, max_practical=0)


def test_validate_max_marks_configuration_rejects_all_zero():
    with pytest.raises(ValidationError):
        validate_max_marks_configuration(max_internal=0, max_external=0, max_practical=0)


def test_validate_max_marks_configuration_rejects_above_ceiling():
    # 150 exceeds config.MAX_MARK_CEILING (100) -- no student could ever
    # legally be given a matching mark, since marks.internal is itself
    # capped at 100 by the database's own CHECK constraint.
    with pytest.raises(ValidationError):
        validate_max_marks_configuration(max_internal=150, max_external=80, max_practical=0)


# ---------------------------------------------------------------------------
# validate_exam_type
# ---------------------------------------------------------------------------

def test_validate_exam_type_accepts_valid_type():
    assert validate_exam_type("regular") == "regular"


def test_validate_exam_type_rejects_unknown_type():
    with pytest.raises(ValidationError):
        validate_exam_type("resit")


# ---------------------------------------------------------------------------
# validate_mark_value
# ---------------------------------------------------------------------------

def test_validate_mark_value_accepts_value_within_subject_maximum():
    # Subject's max_internal is 20 for this example; 18 is within range.
    assert validate_mark_value(18, max_allowed=20, component_name="internal") == 18


def test_validate_mark_value_rejects_negative_value():
    with pytest.raises(ValidationError):
        validate_mark_value(-1, max_allowed=20, component_name="internal")


def test_validate_mark_value_rejects_value_above_subject_maximum():
    # 25 exceeds this subject's own max_internal of 20, even though 25 is
    # well within the database's generic 0-100 sanity ceiling -- this is
    # exactly the per-subject precision the database CHECK constraint
    # cannot express on its own.
    with pytest.raises(ValidationError):
        validate_mark_value(25, max_allowed=20, component_name="internal")


# ---------------------------------------------------------------------------
# validate_attendance_values
# ---------------------------------------------------------------------------

def test_validate_attendance_values_accepts_valid_pair():
    # Should not raise.
    validate_attendance_values(classes_held=30, classes_attended=25)


def test_validate_attendance_values_rejects_attended_greater_than_held():
    with pytest.raises(ValidationError):
        validate_attendance_values(classes_held=30, classes_attended=35)


def test_validate_attendance_values_rejects_negative_held():
    with pytest.raises(ValidationError):
        validate_attendance_values(classes_held=-1, classes_attended=0)


# ---------------------------------------------------------------------------
# validate_assignment_values
# ---------------------------------------------------------------------------

def test_validate_assignment_values_accepts_valid_pair():
    # Should not raise.
    validate_assignment_values(total_assigned=8, submitted=6)


def test_validate_assignment_values_rejects_submitted_greater_than_total():
    with pytest.raises(ValidationError):
        validate_assignment_values(total_assigned=8, submitted=9)


def test_validate_assignment_values_rejects_negative_total():
    with pytest.raises(ValidationError):
        validate_assignment_values(total_assigned=-1, submitted=0)


def test_validate_assignment_values_rejects_negative_submitted():
    with pytest.raises(ValidationError):
        validate_assignment_values(total_assigned=5, submitted=-1)


# ---------------------------------------------------------------------------
# validate_audit_action
# ---------------------------------------------------------------------------

def test_validate_audit_action_accepts_each_valid_action():
    assert validate_audit_action("INSERT") == "INSERT"
    assert validate_audit_action("UPDATE") == "UPDATE"
    assert validate_audit_action("SOFT_DELETE") == "SOFT_DELETE"


def test_validate_audit_action_rejects_raw_delete():
    # A real SQL "DELETE" is deliberately not a valid audit action -- this
    # system only ever soft-deletes (see config.AUDIT_ACTIONS).
    with pytest.raises(ValidationError):
        validate_audit_action("DELETE")


# ---------------------------------------------------------------------------
# validate_table_name
# ---------------------------------------------------------------------------

def test_validate_table_name_accepts_known_table():
    assert validate_table_name("marks") == "marks"


def test_validate_table_name_rejects_unknown_table():
    with pytest.raises(ValidationError):
        validate_table_name("markss")  # typo
