"""
tests/test_grades.py
=====================
pytest unit tests for modules/grades.py (the grade engine).

HOW TO RUN (from the project root, student_performance_system/):
    python -m pytest tests/test_grades.py -v

See tests/test_validators.py for why "python -m pytest" (not bare
"pytest") is the command to use.

A NOTE ON BOUNDARY TESTING:
Several tests below check values exactly AT a threshold (e.g. percentage
== 90, the cutoff for grade 'O') and one unit just below it (89.99...).
This is deliberate: grading thresholds are classic "off-by-one" bug
territory -- a developer might accidentally write "> 90" instead of
">= 90" and only a boundary test would catch that mistake. Testing only
"clearly high" and "clearly low" values would miss it entirely.
"""

import pytest

from utils.exceptions import ValidationError
from modules.grades import (
    calculate_cgpa,
    calculate_percentage,
    calculate_result_status,
    calculate_sgpa,
    evaluate_subject_marks,
    get_grade,
    is_pass,
)


# ---------------------------------------------------------------------------
# calculate_percentage
# ---------------------------------------------------------------------------

def test_calculate_percentage_computes_correct_value():
    # 18 + 65 + 0 = 83 obtained, out of 20 + 80 + 0 = 100 possible -> 83%.
    percentage = calculate_percentage(
        internal=18, external=65, practical=0,
        max_internal=20, max_external=80, max_practical=0,
    )
    assert percentage == 83.0


def test_calculate_percentage_handles_all_three_components():
    # 10 + 30 + 15 = 55 obtained, out of 20 + 50 + 30 = 100 possible -> 55%.
    percentage = calculate_percentage(
        internal=10, external=30, practical=15,
        max_internal=20, max_external=50, max_practical=30,
    )
    assert percentage == 55.0


def test_calculate_percentage_rejects_zero_total_maximum():
    with pytest.raises(ValidationError):
        calculate_percentage(
            internal=0, external=0, practical=0,
            max_internal=0, max_external=0, max_practical=0,
        )


# ---------------------------------------------------------------------------
# get_grade
# ---------------------------------------------------------------------------

def test_get_grade_at_exact_thresholds():
    assert get_grade(90) == ("O", 10.0)
    assert get_grade(80) == ("A+", 9.0)
    assert get_grade(70) == ("A", 8.0)
    assert get_grade(60) == ("B+", 7.0)
    assert get_grade(50) == ("B", 6.0)
    assert get_grade(40) == ("C", 5.0)
    assert get_grade(0) == ("F", 0.0)


def test_get_grade_just_below_o_threshold():
    # 89.99 must NOT qualify for 'O' (which requires >= 90) -- this is the
    # off-by-one boundary check described in the module docstring above.
    assert get_grade(89.99) == ("A+", 9.0)


def test_get_grade_just_below_pass_threshold():
    assert get_grade(39.99) == ("F", 0.0)


def test_get_grade_rejects_negative_percentage():
    with pytest.raises(ValidationError):
        get_grade(-5)


# ---------------------------------------------------------------------------
# is_pass
# ---------------------------------------------------------------------------

def test_is_pass_at_exact_pass_threshold():
    # config.PASS_PERCENTAGE is 40.0; exactly 40 must count as a pass
    # (the rule is ">=", not ">").
    assert is_pass(40.0) is True


def test_is_pass_just_below_threshold():
    assert is_pass(39.99) is False


def test_is_pass_well_above_threshold():
    assert is_pass(95.0) is True


# ---------------------------------------------------------------------------
# evaluate_subject_marks
# ---------------------------------------------------------------------------

def test_evaluate_subject_marks_matches_individual_calls():
    result = evaluate_subject_marks(
        internal=18, external=65, practical=0,
        max_internal=20, max_external=80, max_practical=0,
    )

    expected_percentage = calculate_percentage(18, 65, 0, 20, 80, 0)
    expected_letter, expected_point = get_grade(expected_percentage)

    assert result == {
        "percentage": expected_percentage,
        "grade_letter": expected_letter,
        "grade_point": expected_point,
        "passed": True,
    }


def test_evaluate_subject_marks_reports_failure_correctly():
    # 10 out of 100 is well below the 40% pass mark.
    result = evaluate_subject_marks(
        internal=5, external=5, practical=0,
        max_internal=20, max_external=80, max_practical=0,
    )
    assert result["passed"] is False
    assert result["grade_letter"] == "F"


# ---------------------------------------------------------------------------
# calculate_sgpa
# ---------------------------------------------------------------------------

def test_calculate_sgpa_is_credit_weighted_not_a_plain_average():
    # (3*10.0 + 4*8.0 + 3*6.0) / (3+4+3) = (30+32+18)/10 = 8.0
    # A plain, unweighted average of the three grade points would give
    # (10.0 + 8.0 + 6.0) / 3 = 8.0 too in this particular example, so also
    # check a case where the two approaches would clearly disagree, below.
    sgpa = calculate_sgpa([
        {"credits": 3, "grade_point": 10.0},
        {"credits": 4, "grade_point": 8.0},
        {"credits": 3, "grade_point": 6.0},
    ])
    assert sgpa == 8.0


def test_calculate_sgpa_weighting_changes_result():
    # Here a plain average of grade points would be (10.0 + 4.0) / 2 = 7.0,
    # but the high-credit subject (5 credits) should pull the weighted
    # SGPA up much closer to 10.0, proving the credit-weighting is real.
    sgpa = calculate_sgpa([
        {"credits": 5, "grade_point": 10.0},
        {"credits": 1, "grade_point": 4.0},
    ])
    # (5*10.0 + 1*4.0) / 6 = 54 / 6 = 9.0
    assert sgpa == 9.0


def test_calculate_sgpa_rejects_empty_list():
    with pytest.raises(ValidationError):
        calculate_sgpa([])


# ---------------------------------------------------------------------------
# calculate_cgpa
# ---------------------------------------------------------------------------

def test_calculate_cgpa_computes_correct_value():
    # (8.0*10 + 7.0*12) / (10+12) = (80+84)/22 = 164/22 = 7.4545... -> 7.45
    cgpa = calculate_cgpa([
        {"sgpa": 8.0, "credits": 10},
        {"sgpa": 7.0, "credits": 12},
    ])
    assert cgpa == 7.45


def test_calculate_cgpa_rejects_empty_list():
    with pytest.raises(ValidationError):
        calculate_cgpa([])


# ---------------------------------------------------------------------------
# calculate_result_status
# ---------------------------------------------------------------------------

def test_calculate_result_status_all_passed():
    assert calculate_result_status([True, True, True]) == "pass"


def test_calculate_result_status_one_failed():
    assert calculate_result_status([True, False, True]) == "fail"


def test_calculate_result_status_no_subjects_yet():
    assert calculate_result_status([]) == "pending"
