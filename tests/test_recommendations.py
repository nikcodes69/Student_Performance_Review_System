"""
tests/test_recommendations.py
==============================
pytest tests for modules/ml_predictions.py's generate_recommendations()
-- the rule-based, explainable improvement-tip engine for at-risk
students (see that function's docstring for why it's rule-based rather
than an external AI/LLM call).

NO DATABASE FIXTURE NEEDED: unlike every other test file in this
project, generate_recommendations() is a pure function -- it only reads
the features dict and semester it's given, never touches the database
or a trained model file. These tests construct feature dicts by hand.

HOW TO RUN (from the project root):
    python -m pytest tests/test_recommendations.py -v
"""

import config
from modules.ml_predictions import generate_recommendations

# A feature dict where every value is comfortably on the "healthy" side
# of every threshold generate_recommendations() checks -- the baseline
# every test below starts from and then deliberately worsens ONE field
# at a time, so each test isolates exactly one rule.
_HEALTHY_FEATURES = {
    "internal_pct": 80.0,
    "attendance_pct": 90.0,
    "practical_pct": 80.0,
    "average_marks": 80.0,
    "consistency": 5.0,
    "improvement_rate": 5.0,
    "previous_sgpa": 8.0,
    "backlog_count": 0,
    "assignments_submitted": 9.0,
}


def test_no_recommendations_when_every_feature_is_healthy():
    assert generate_recommendations(_HEALTHY_FEATURES, semester=3) == []


def test_low_attendance_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "attendance_pct": 60.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("Attendance" in tip for tip in recommendations)


def test_backlog_triggers_recommendation_with_correct_singular_plural():
    one_backlog = generate_recommendations({**_HEALTHY_FEATURES, "backlog_count": 1}, semester=3)
    assert any("1 subject not yet passed" in tip for tip in one_backlog)

    two_backlogs = generate_recommendations({**_HEALTHY_FEATURES, "backlog_count": 2}, semester=3)
    assert any("2 subjects not yet passed" in tip for tip in two_backlogs)


def test_low_average_marks_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "average_marks": 30.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("Overall average" in tip for tip in recommendations)


def test_low_internal_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "internal_pct": 25.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("Internal assessment" in tip for tip in recommendations)


def test_low_practical_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "practical_pct": 25.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("Practical average" in tip for tip in recommendations)


def test_low_assignment_engagement_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "assignments_submitted": 3.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("Assignment engagement" in tip for tip in recommendations)


def test_negative_improvement_rate_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "improvement_rate": -10.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("DECLINED" in tip for tip in recommendations)


def test_low_previous_sgpa_triggers_recommendation_from_semester_2_onward():
    features = {**_HEALTHY_FEATURES, "previous_sgpa": 3.0}
    recommendations = generate_recommendations(features, semester=2)
    assert any("Previous semester" in tip for tip in recommendations)


def test_low_previous_sgpa_is_ignored_in_semester_1():
    # semester 1 has no previous semester at all -- _compute_live_features()
    # defaults previous_sgpa to 0.0 in that case, which must NOT be
    # mistaken for "actually scored 0.0 last semester".
    features = {**_HEALTHY_FEATURES, "previous_sgpa": 0.0}
    recommendations = generate_recommendations(features, semester=config.MIN_SEMESTER)
    assert not any("Previous semester" in tip for tip in recommendations)


def test_high_inconsistency_triggers_its_own_recommendation():
    features = {**_HEALTHY_FEATURES, "consistency": 25.0}
    recommendations = generate_recommendations(features, semester=3)
    assert any("uneven" in tip for tip in recommendations)


def test_multiple_triggered_rules_all_appear():
    features = {
        **_HEALTHY_FEATURES,
        "attendance_pct": 50.0,
        "backlog_count": 2,
        "internal_pct": 20.0,
    }
    recommendations = generate_recommendations(features, semester=3)
    assert len(recommendations) == 3


def test_attendance_recommendation_comes_before_backlog_and_internal():
    # Ordering is deliberate (most urgent first) -- see this function's
    # docstring. Confirms the actual order, not just "all three present".
    features = {
        **_HEALTHY_FEATURES,
        "attendance_pct": 50.0,
        "backlog_count": 1,
        "internal_pct": 20.0,
    }
    recommendations = generate_recommendations(features, semester=3)
    attendance_index = next(i for i, tip in enumerate(recommendations) if "Attendance" in tip)
    backlog_index = next(i for i, tip in enumerate(recommendations) if "backlog" in tip.lower())
    internal_index = next(i for i, tip in enumerate(recommendations) if "Internal assessment" in tip)
    assert attendance_index < backlog_index < internal_index
