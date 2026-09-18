"""
tests/test_analytics.py
========================
pytest tests for the analytics functions added alongside the rank/
percentile, attendance shortage alert, and comparative-analytics
features: get_class_rankings(), get_student_rank(),
get_attendance_shortage_list(), and get_student_vs_class_trend().

Uses the SAME throwaway-database pattern as tests/test_database.py and
tests/test_auth.py -- see those files' module docstrings for the full
rationale (never the real local database, never the real Turso database).

WHY EVERY st.cache_data-DECORATED FUNCTION IS EXPLICITLY .clear()-ED IN
THE FIXTURE, BEFORE INSERTING THIS TEST'S DATA: Streamlit's cache_data
caches a function's return value keyed by its ARGUMENTS, not by which
database happens to be active underneath it. Without clearing the cache
first, a later test could receive a STALE result computed from an
EARLIER test's throwaway database (both might call, say,
get_student_averages(semester=None) with the exact same arguments), even
though config.DB_PATH now points somewhere else entirely. Clearing every
relevant cache at the start of each test removes that risk -- this is
not needed in modules/test_auth.py or test_database.py because neither
of those touch any @st.cache_data-decorated function.

HOW TO RUN (from the project root):
    python -m pytest tests/test_analytics.py -v
"""

import pytest

import config
import database.db_setup as db_setup
from database.db_manager import execute_write
from database.db_setup import create_indexes, create_tables, get_connection
from modules import analytics


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test, with
    every @st.cache_data-decorated analytics function cleared first -- see
    module docstring above for why that clearing is necessary here and not
    in the project's other test files."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_analytics.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    for cached_fn in (
        analytics.get_student_averages,
        analytics.get_student_attendance_averages,
        analytics.get_student_performance_trend,
        analytics.get_class_average_by_semester,
        analytics.get_dashboard_summary,
    ):
        cached_fn.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _seed_three_students_one_subject() -> None:
    """
    Three students, one subject, one semester -- with deliberately
    distinct marks and attendance percentages, chosen so rank order,
    percentile, and shortage status can all be hand-verified:

        roll_no  marks%   attendance%
        S1       90       90   (no shortage)
        S2       50       50   (shortage)
        S3       75       70   (shortage)

    Expected class rank (marks, descending): S1 (rank 1, 100th
    percentile), S3 (rank 2, 50th percentile), S2 (rank 3, 0th
    percentile). Expected class average = (90 + 50 + 75) / 3 = 71.67
    (rounded to config.ROUND_DECIMALS = 2).
    """
    execute_write(
        "INSERT INTO subjects (subject_code, name, semester, credits) VALUES (?, ?, ?, ?)",
        ("SUB1", "Fixture Subject", 1, 3),
    )
    for roll_no, name in (("S1", "Alice"), ("S2", "Bob"), ("S3", "Carol")):
        execute_write(
            "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (roll_no, name, 1, "BCA", f"{roll_no.lower()}@example.com", "9812345678", 2024),
        )

    # internal/external/practical chosen so their sum out of the fixed
    # 100-mark total gives exactly the target percentage from the table
    # above, while staying within each component's own ceiling (25/50/25).
    marks_by_roll = {"S1": (25, 50, 15), "S2": (25, 25, 0), "S3": (25, 50, 0)}
    for roll_no, (internal, external, practical) in marks_by_roll.items():
        execute_write(
            "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (roll_no, "SUB1", internal, external, practical, 1, "regular"),
        )

    attendance_by_roll = {"S1": (20, 18), "S2": (20, 10), "S3": (20, 14)}
    for roll_no, (held, attended) in attendance_by_roll.items():
        execute_write(
            "INSERT INTO attendance (roll_no, subject_code, classes_held, classes_attended, semester) "
            "VALUES (?, ?, ?, ?, ?)",
            (roll_no, "SUB1", held, attended, 1),
        )


# ---------------------------------------------------------------------------
# get_class_rankings() / get_student_rank()
# ---------------------------------------------------------------------------

def test_class_rankings_order_and_percentile(test_db):
    _seed_three_students_one_subject()

    rankings = analytics.get_class_rankings()
    ranking_by_roll = {r["roll_no"]: r for r in rankings}

    assert ranking_by_roll["S1"]["rank"] == 1
    assert ranking_by_roll["S1"]["percentile"] == 100.0
    assert ranking_by_roll["S3"]["rank"] == 2
    assert ranking_by_roll["S3"]["percentile"] == 50.0
    assert ranking_by_roll["S2"]["rank"] == 3
    assert ranking_by_roll["S2"]["percentile"] == 0.0
    assert all(r["total_students"] == 3 for r in rankings)


def test_class_rankings_ties_share_a_rank_and_skip_the_next_one(test_db):
    # Two students tied at the top must both get rank 1, and the next
    # distinct student must get rank 3 (not rank 2) -- "competition
    # ranking", see get_class_rankings()'s docstring.
    execute_write(
        "INSERT INTO subjects (subject_code, name, semester, credits) VALUES (?, ?, ?, ?)",
        ("SUB1", "Fixture Subject", 1, 3),
    )
    for roll_no, name in (("S1", "Alice"), ("S2", "Bob"), ("S3", "Carol")):
        execute_write(
            "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (roll_no, name, 1, "BCA", f"{roll_no.lower()}@example.com", "9812345678", 2024),
        )
    # S1 and S2 both score exactly 90%; S3 scores 50%. (25 + 50 + 15 = 90,
    # 25 + 25 + 0 = 50 -- within each component's own ceiling of 25/50/25.)
    for roll_no, (internal, external, practical) in {
        "S1": (25, 50, 15), "S2": (25, 50, 15), "S3": (25, 25, 0),
    }.items():
        execute_write(
            "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (roll_no, "SUB1", internal, external, practical, 1, "regular"),
        )

    rankings = analytics.get_class_rankings()
    ranking_by_roll = {r["roll_no"]: r for r in rankings}

    assert ranking_by_roll["S1"]["rank"] == 1
    assert ranking_by_roll["S2"]["rank"] == 1
    assert ranking_by_roll["S3"]["rank"] == 3


def test_student_rank_returns_none_for_student_with_no_marks(test_db):
    _seed_three_students_one_subject()
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S4", "Dave", 1, "BCA", "s4@example.com", "9812345678", 2024),
    )
    assert analytics.get_student_rank("S4") is None


# ---------------------------------------------------------------------------
# get_attendance_shortage_list()
# ---------------------------------------------------------------------------

def test_attendance_shortage_list_flags_only_students_below_threshold(test_db):
    _seed_three_students_one_subject()

    shortage = analytics.get_attendance_shortage_list()
    shortage_rolls = [row["roll_no"] for row in shortage]

    # S1 is at 90% (no shortage); S2 (50%) and S3 (70%) are both below
    # config.ATTENDANCE_SHORTAGE_THRESHOLD (75%), worst first.
    assert shortage_rolls == ["S2", "S3"]
    assert "S1" not in shortage_rolls


# ---------------------------------------------------------------------------
# get_class_average_by_semester() / get_student_vs_class_trend()
# ---------------------------------------------------------------------------

def test_class_average_by_semester(test_db):
    _seed_three_students_one_subject()
    class_average = analytics.get_class_average_by_semester()
    assert class_average[1] == round((90 + 50 + 75) / 3, config.ROUND_DECIMALS)


def test_student_vs_class_trend_pairs_student_and_class_percentage(test_db):
    _seed_three_students_one_subject()
    trend = analytics.get_student_vs_class_trend("S1")
    assert trend == [{
        "semester": 1,
        "student_percentage": 90.0,
        "class_percentage": round((90 + 50 + 75) / 3, config.ROUND_DECIMALS),
    }]


def test_student_vs_class_trend_empty_for_student_with_no_marks(test_db):
    _seed_three_students_one_subject()
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S4", "Dave", 1, "BCA", "s4@example.com", "9812345678", 2024),
    )
    assert analytics.get_student_vs_class_trend("S4") == []
