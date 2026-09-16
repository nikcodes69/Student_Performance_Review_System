"""
modules/grades.py
==================
The grade engine: pure calculation functions that turn raw marks into
percentages, letter grades, grade points, SGPA, CGPA, and pass/fail
status.

WHY THIS FILE HAS NO DATABASE CODE AND NO STREAMLIT CODE:
Every function here takes plain numbers (and lists/dicts of plain
numbers) as input, and returns a plain number, string, or dict as output.
None of them open a database connection, read a file, or draw anything on
screen. This separation matters for two concrete reasons:

  1. TESTABILITY. Because these functions only depend on their arguments
     (not on the database being set up, or Streamlit being running), we
     can unit-test the entire grading logic in isolation, instantly, with
     plain numbers -- see tests/test_grades.py. That is precisely why the
     build order tackles this file before modules/marks.py: the marks
     module will call these functions, but these functions do not need
     the marks module (or the database) to exist first.
  2. REUSE. The exact same functions will be called from at least three
     places later: modules/marks.py (to show a grade right after entry),
     modules/analytics.py (to compute dashboard statistics), and
     utils/pdf_generator.py (to print a grade on the report card). Writing
     the calculation once here means all three automatically agree with
     each other -- there is only one place a grading bug could exist.
"""

import config
from utils.exceptions import ValidationError


# ---------------------------------------------------------------------------
# SUBJECT-LEVEL CALCULATIONS
# ---------------------------------------------------------------------------

def calculate_percentage(
    internal: int,
    external: int,
    practical: int,
    max_internal: int,
    max_external: int,
    max_practical: int,
) -> float:
    """
    Calculate the percentage a student scored in one subject.

    percentage = (marks obtained) / (marks possible) * 100

    Args:
        internal: Internal marks obtained.
        external: External marks obtained.
        practical: Practical marks obtained.
        max_internal: Maximum possible internal marks for this subject.
        max_external: Maximum possible external marks for this subject.
        max_practical: Maximum possible practical marks for this subject.

    Returns:
        The percentage, rounded to config.ROUND_DECIMALS places.

    Raises:
        ValidationError: if the total possible marks is zero (would cause
            division by zero) -- utils/validators.py's
            validate_max_marks_configuration() should already prevent this
            from ever happening, but this function guards independently in
            case it is ever called with data that skipped validation.
    """
    total_obtained = internal + external + practical
    total_max = max_internal + max_external + max_practical

    if total_max <= 0:
        raise ValidationError(
            "Cannot calculate percentage: this subject has no gradable marks configured."
        )

    percentage = (total_obtained / total_max) * 100
    return round(percentage, config.ROUND_DECIMALS)


def get_grade(percentage: float) -> tuple[str, float]:
    """
    Look up the letter grade and grade point for a percentage, using
    config.GRADE_SCALE.

    config.GRADE_SCALE is a list of (minimum_percentage, letter, grade_point)
    tuples sorted from HIGHEST threshold to LOWEST. This function walks
    down the list and returns the first entry whose threshold the
    percentage meets or exceeds -- e.g. a percentage of 85 is not >= 90
    (the 'O' threshold), but it IS >= 80 (the 'A+' threshold), so it stops
    there and returns 'A+'.

    Args:
        percentage: A percentage from 0 to 100.

    Returns:
        A (grade_letter, grade_point) tuple, e.g. ("A+", 9.0).

    Raises:
        ValidationError: if percentage does not match any threshold (only
            possible if percentage is negative, since GRADE_SCALE's lowest
            threshold is 0).
    """
    for threshold, letter, grade_point in config.GRADE_SCALE:
        if percentage >= threshold:
            return letter, grade_point

    raise ValidationError(f"Percentage {percentage} is not a valid value (cannot be negative).")


def is_pass(percentage: float) -> bool:
    """
    Determine whether a percentage counts as a pass for one subject.

    Args:
        percentage: A percentage from 0 to 100.

    Returns:
        True if percentage >= config.PASS_PERCENTAGE, else False.
    """
    return percentage >= config.PASS_PERCENTAGE


def evaluate_subject_marks(
    internal: int,
    external: int,
    practical: int,
    max_internal: int,
    max_external: int,
    max_practical: int,
) -> dict:
    """
    Convenience function that runs the full subject-level calculation in
    one call: percentage, grade letter, grade point, and pass/fail.

    This is just calculate_percentage() + get_grade() + is_pass() composed
    together -- it exists so callers (modules/marks.py, in a later step)
    do not have to repeat those three calls every time they need a
    complete picture of one subject's result.

    Args:
        internal: Internal marks obtained.
        external: External marks obtained.
        practical: Practical marks obtained.
        max_internal: Maximum possible internal marks for this subject.
        max_external: Maximum possible external marks for this subject.
        max_practical: Maximum possible practical marks for this subject.

    Returns:
        A dict with keys: "percentage", "grade_letter", "grade_point", "passed".
    """
    percentage = calculate_percentage(
        internal, external, practical, max_internal, max_external, max_practical
    )
    grade_letter, grade_point = get_grade(percentage)
    passed = is_pass(percentage)

    return {
        "percentage": percentage,
        "grade_letter": grade_letter,
        "grade_point": grade_point,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# AGGREGATE CALCULATIONS (SGPA, CGPA)
# ---------------------------------------------------------------------------

def _weighted_average(pairs: list[tuple[float, float]], error_message: str) -> float:
    """
    Shared helper: calculate a weight-adjusted average from a list of
    (value, weight) pairs.

    This one small function is the math behind BOTH calculate_sgpa() and
    calculate_cgpa() below -- SGPA is subjects weighted by their credits,
    and CGPA is semesters weighted by their credits. Rather than write the
    same weighted-average formula twice, both functions call this helper.
    That also guarantees they can never quietly drift apart in behaviour.

    Formula: weighted_average = sum(value_i * weight_i) / sum(weight_i)

    Args:
        pairs: A list of (value, weight) tuples.
        error_message: Message to raise if total weight is zero.

    Returns:
        The weighted average, rounded to config.ROUND_DECIMALS places.

    Raises:
        ValidationError: if the pairs list is empty or all weights are zero
            (both would cause division by zero).
    """
    total_weight = sum(weight for _, weight in pairs)

    if total_weight <= 0:
        raise ValidationError(error_message)

    weighted_sum = sum(value * weight for value, weight in pairs)
    return round(weighted_sum / total_weight, config.ROUND_DECIMALS)


def calculate_sgpa(subject_results: list[dict]) -> float:
    """
    Calculate SGPA (Semester Grade Point Average): the credit-weighted
    average of grade points across every subject in one semester.

    A 3-credit subject counts three times as much toward SGPA as a
    1-credit subject -- that is what "credit-weighted" means, and it is
    why this is not a plain average of grade points.

    Args:
        subject_results: A list of dicts, one per subject, each shaped
            like {"credits": <int>, "grade_point": <float>}. Typically
            built by calling evaluate_subject_marks() for every subject a
            student took that semester and reading its "grade_point".

    Returns:
        SGPA, rounded to config.ROUND_DECIMALS places.

    Raises:
        ValidationError: if subject_results is empty, or total credits is
            zero.
    """
    if not subject_results:
        raise ValidationError("Cannot calculate SGPA: no subject results were provided.")

    pairs = [(result["grade_point"], result["credits"]) for result in subject_results]
    return _weighted_average(pairs, "Cannot calculate SGPA: total credits is zero.")


def calculate_cgpa(semester_results: list[dict]) -> float:
    """
    Calculate CGPA (Cumulative Grade Point Average): the credit-weighted
    average of SGPA across every semester a student has completed so far.

    Uses the exact same weighted-average logic as calculate_sgpa() above,
    just one level up: instead of averaging subjects within a semester, it
    averages semesters within the whole program.

    Args:
        semester_results: A list of dicts, one per completed semester,
            each shaped like {"sgpa": <float>, "credits": <int>} where
            "credits" is the total credits taken that semester.

    Returns:
        CGPA, rounded to config.ROUND_DECIMALS places.

    Raises:
        ValidationError: if semester_results is empty, or total credits is
            zero.
    """
    if not semester_results:
        raise ValidationError("Cannot calculate CGPA: no semester results were provided.")

    pairs = [(result["sgpa"], result["credits"]) for result in semester_results]
    return _weighted_average(pairs, "Cannot calculate CGPA: total credits is zero.")


# ---------------------------------------------------------------------------
# RESULT STATUS
# ---------------------------------------------------------------------------

def calculate_result_status(subject_pass_flags: list[bool]) -> str:
    """
    Determine a semester's overall result status from each subject's
    pass/fail outcome.

    Rules:
      - No subjects recorded yet -> "pending" (marks entry is incomplete).
      - Every subject passed -> "pass".
      - At least one subject failed -> "fail" (most academic systems treat
        a single failed subject as failing the semester overall, even if
        every other subject was passed).

    Args:
        subject_pass_flags: A list of booleans, one per subject, typically
            the "passed" value from evaluate_subject_marks() for each
            subject in the semester.

    Returns:
        One of config.RESULT_PASS, config.RESULT_FAIL, config.RESULT_PENDING.
    """
    if not subject_pass_flags:
        return config.RESULT_PENDING

    return config.RESULT_PASS if all(subject_pass_flags) else config.RESULT_FAIL
