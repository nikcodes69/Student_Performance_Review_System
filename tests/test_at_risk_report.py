"""
tests/test_at_risk_report.py
=============================
pytest tests for modules/ml_predictions.py's get_at_risk_report() -- the
class-wide, per-student expansion of get_at_risk_count()'s single
number, shown on the At-Risk Prediction page's "Class-wide At-Risk
Report" button.

WHY predict_at_risk_for_student() IS MONKEYPATCHED HERE, RATHER THAN
LET THE REAL TRAINED MODEL RUN: get_at_risk_report()'s own logic (sort
At-Risk-first by descending probability, then On-Track, then "No marks
yet" students last) is independent of what the model actually predicts
for any given student -- it only needs SOME predict_at_risk_for_student()
that returns the expected shape or raises ValidationError. Patching it
with a small, fully deterministic fake means this test verifies the
SORTING AND LABELLING LOGIC precisely, without being a flaky test that
depends on a real classifier's coefficients agreeing with hand-picked
"obviously good/bad" synthetic student data (which is what the actual
pipeline was instead verified against manually, using real production
data -- see the chat history around this feature's implementation).

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

WHY students.list_students.clear() IS CALLED IN THE FIXTURE: get_at_risk_
report() calls modules.students.list_students(), which is
@st.cache_data-decorated -- see tests/test_analytics.py's module
docstring for why every cached function a test touches must be cleared
before that test's own (different, throwaway) database is seeded, to
avoid a stale student list leaking in from an earlier test's database.
This bug was caught for real while writing this file: a later "no
students" test was failing because an earlier test's cached
list_students() result was still being returned. Fixed by clearing it
here, not by getting lucky with test ordering.

HOW TO RUN (from the project root):
    python -m pytest tests/test_at_risk_report.py -v
"""

import pytest

import config
import database.db_setup as db_setup
from database.db_setup import create_indexes, create_tables, get_connection
from modules import ml_predictions, students
from utils.exceptions import ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database, with
    students.list_students()'s cache cleared first -- see module
    docstring above for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_at_risk_report.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    students.list_students.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _create_student(roll_no: str, name: str) -> None:
    from database.db_manager import execute_write
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (roll_no, name, 1, "BCA", f"{roll_no.lower()}@example.com", "9812345678", 2024),
    )


def test_at_risk_report_sorts_at_risk_first_by_descending_probability(test_db, monkeypatch):
    _create_student("S1", "Low Risk")
    _create_student("S2", "Highest Risk")
    _create_student("S3", "Medium Risk")

    fake_results = {
        "S1": {"at_risk": False, "risk_probability": 0.2},
        "S2": {"at_risk": True, "risk_probability": 0.95},
        "S3": {"at_risk": True, "risk_probability": 0.6},
    }
    monkeypatch.setattr(
        ml_predictions, "predict_at_risk_for_student",
        lambda roll_no, semester: fake_results[roll_no],
    )

    report = ml_predictions.get_at_risk_report()

    assert [row["roll_no"] for row in report] == ["S2", "S3", "S1"]
    assert [row["status"] for row in report] == ["At Risk", "At Risk", "On Track"]


def test_at_risk_report_labels_students_with_no_marks_yet(test_db, monkeypatch):
    _create_student("S1", "Has Marks")
    _create_student("S2", "No Marks Yet")

    def fake_predict(roll_no, semester):
        if roll_no == "S2":
            raise ValidationError("No marks recorded yet.")
        return {"at_risk": False, "risk_probability": 0.1}

    monkeypatch.setattr(ml_predictions, "predict_at_risk_for_student", fake_predict)

    report = ml_predictions.get_at_risk_report()
    report_by_roll = {row["roll_no"]: row for row in report}

    assert report_by_roll["S2"]["status"] == "No marks yet"
    assert report_by_roll["S2"]["at_risk"] is None
    assert report_by_roll["S2"]["risk_probability"] is None
    # Students with no marks sort LAST, after both At Risk and On Track.
    assert report[-1]["roll_no"] == "S2"


def test_at_risk_report_includes_every_active_student(test_db, monkeypatch):
    _create_student("S1", "Alice")
    _create_student("S2", "Bob")
    _create_student("S3", "Carol")
    monkeypatch.setattr(
        ml_predictions, "predict_at_risk_for_student",
        lambda roll_no, semester: {"at_risk": False, "risk_probability": 0.1},
    )

    report = ml_predictions.get_at_risk_report()
    assert {row["roll_no"] for row in report} == {"S1", "S2", "S3"}


def test_at_risk_report_empty_when_no_students(test_db):
    assert ml_predictions.get_at_risk_report() == []
