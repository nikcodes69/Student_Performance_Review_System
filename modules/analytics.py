"""
modules/analytics.py
=====================
The analytics dashboard: subject averages, student performance trends,
grade distribution, top/bottom performers, attendance-vs-marks
correlation, and a subject difficulty index.

READ-ONLY: this file never writes to the database, so unlike
students.py/subjects.py/marks.py/attendance.py, there is no audit trail
and no build_audit_entry()/execute_transaction() here -- every function
below is a plain SELECT.

WHERE THE MATH HAPPENS -- SQL vs. modules/grades.py -- IS A DELIBERATE LINE:
  - Functions that only need a NUMBER (an average percentage, a
    correlation coefficient) let SQL's AVG() do the arithmetic directly
    (see PERCENTAGE_EXPR below). There is no business RULE attached to
    "what is the average of these percentages" -- it is just arithmetic,
    so there is nothing that could drift out of sync by computing it in
    SQL.
  - Functions that need to CLASSIFY a percentage (into a letter grade, or
    into pass/fail) fetch the raw marks and call modules.grades.get_grade()
    / is_pass() in Python instead (see get_grade_distribution() and
    get_subject_difficulty_index()). Grading is a BUSINESS RULE -- it
    lives in exactly one place, modules/grades.py, built all the way back
    in step 4 -- so nothing here ever re-implements "what letter grade is
    83%" as a second, potentially-inconsistent copy of that logic in SQL.

PERCENTAGE_EXPR is built with an f-string, like database/db_setup.py's
schema-creation SQL: it only ever splices in a FIXED, developer-written
SQL fragment (never a value from a user or the database), so it carries
none of the SQL-injection risk that rule exists to prevent -- see
db_setup.py's module docstring for the fuller explanation of that
distinction.

WHY PERCENTAGE_EXPR MULTIPLIES BY 100.0, NOT 100: internal/external/
practical/max_* are all stored as INTEGER columns. SQLite performs
INTEGER division when both sides of a division are integers, silently
truncating (83 / 100 would give 0, not 0.83). Multiplying by the FLOAT
literal 100.0 first forces the whole expression into floating-point
arithmetic, so the result is a real percentage like 83.0, not 0. This is
a classic, easy-to-miss bug -- worth having a ready answer for if asked.

WHY PERCENTAGE_EXPR NEVER DIVIDES BY ZERO: the denominator is
max_internal + max_external + max_practical, which
utils.validators.validate_max_marks_configuration() (used by every write
in modules/subjects.py) guarantees is always greater than zero -- backed
by the database's own CHECK constraint as a second, independent
guarantee (see database/db_setup.py's subjects table). No subject can
exist in this database with all three maximums at zero, so this division
is always safe.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from database.db_manager import fetch_all
from modules import auth, students
from modules.grades import calculate_percentage, get_grade, is_pass
from utils.logger import get_logger
from utils.validators import validate_roll_no

logger = get_logger(__name__)

PERCENTAGE_EXPR = (
    "(m.internal + m.external + m.practical) * 100.0 / "
    "(s.max_internal + s.max_external + s.max_practical)"
)


# ---------------------------------------------------------------------------
# DASHBOARD SUMMARY (app.py's home page)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_dashboard_summary() -> dict:
    """
    A handful of top-level counts and rates for the home page dashboard.

    Deliberately kept to COUNT()/SUM()/AVG() aggregate queries only -- no
    per-row Python loop, unlike get_grade_distribution() and
    get_subject_difficulty_index() above, which need modules/grades.py's
    classification logic and so cannot avoid one. A dashboard that loads
    on every single login and page return trip should stay as cheap as
    possible; a shorter 30-second TTL (vs. 60s elsewhere in this file)
    reflects that this is the FIRST thing anyone sees, so it should lag
    behind a fresh mark/attendance entry a little less than the deeper
    analytics pages do.

    Returns:
        A dict: student_count, subject_count, marks_count, pass_rate
        (percentage, or None if no marks exist yet), avg_attendance
        (percentage, or None if no attendance exists yet).
    """
    student_count = fetch_all("SELECT COUNT(*) AS c FROM students WHERE is_active = 1")[0]["c"]
    subject_count = fetch_all("SELECT COUNT(*) AS c FROM subjects WHERE is_active = 1")[0]["c"]
    marks_count = fetch_all("SELECT COUNT(*) AS c FROM marks")[0]["c"]

    pass_row = fetch_all(
        f"SELECT "
        f"SUM(CASE WHEN {PERCENTAGE_EXPR} >= ? THEN 1 ELSE 0 END) AS passed, "
        f"COUNT(*) AS total "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code",
        (config.PASS_PERCENTAGE,),
    )[0]
    pass_rate = (
        round(pass_row["passed"] / pass_row["total"] * 100, 1) if pass_row["total"] else None
    )

    attendance_row = fetch_all(
        "SELECT AVG(classes_attended * 100.0 / classes_held) AS avg_pct "
        "FROM attendance WHERE classes_held > 0"
    )[0]
    avg_attendance = round(attendance_row["avg_pct"], 1) if attendance_row["avg_pct"] is not None else None

    return {
        "student_count": student_count,
        "subject_count": subject_count,
        "marks_count": marks_count,
        "pass_rate": pass_rate,
        "avg_attendance": avg_attendance,
    }


# ---------------------------------------------------------------------------
# SUBJECT AVERAGES
# ---------------------------------------------------------------------------
# Every DB-querying function in this file below is wrapped in
# @st.cache_data(ttl=60): each involves a JOIN across two or three tables,
# and several also loop over every row in Python to classify a grade (see
# the module docstring's SQL-vs-grades.py section) -- genuinely more
# expensive than a simple lookup. Unlike modules/students.py's
# list_students() (cached WITH explicit .clear() calls on every write, so
# a new student appears in pickers instantly), these analytics functions
# use TTL-only caching with no manual invalidation: this is a read-only
# dashboard, so a chart being up to 60 seconds behind the latest mark
# entered is a reasonable, explainable trade-off for not re-running these
# heavier queries on every single click anywhere in the app.

@st.cache_data(ttl=60)
def get_subject_averages(semester: int | None = None) -> list[dict]:
    """
    Average percentage and number of marks entries, per subject.

    Args:
        semester: If given, only marks recorded in this semester.

    Returns:
        A list of dicts: subject_code, subject_name, average_percentage,
        entry_count. Ordered by subject_code.
    """
    query = (
        f"SELECT s.subject_code, s.name AS subject_name, "
        f"AVG({PERCENTAGE_EXPR}) AS average_percentage, COUNT(*) AS entry_count "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code"
    )
    params: list = []
    if semester is not None:
        query += " WHERE m.semester = ?"
        params.append(semester)
    query += " GROUP BY s.subject_code, s.name ORDER BY s.subject_code"

    return [_round_field(dict(row), "average_percentage") for row in fetch_all(query, tuple(params))]


# ---------------------------------------------------------------------------
# STUDENT AVERAGES, TOP/BOTTOM PERFORMERS, PERFORMANCE TREND
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_student_averages(semester: int | None = None) -> list[dict]:
    """
    Average percentage per student, across every subject they have marks
    in (optionally restricted to one semester).

    Args:
        semester: If given, only marks recorded in this semester.

    Returns:
        A list of dicts: roll_no, student_name, average_percentage,
        subjects_count.
    """
    query = (
        f"SELECT m.roll_no, st.name AS student_name, "
        f"AVG({PERCENTAGE_EXPR}) AS average_percentage, COUNT(*) AS subjects_count "
        "FROM marks m "
        "JOIN subjects s ON m.subject_code = s.subject_code "
        "JOIN students st ON m.roll_no = st.roll_no"
    )
    params: list = []
    if semester is not None:
        query += " WHERE m.semester = ?"
        params.append(semester)
    query += " GROUP BY m.roll_no, st.name"

    return [_round_field(dict(row), "average_percentage") for row in fetch_all(query, tuple(params))]


def get_top_performers(n: int = 5, semester: int | None = None) -> list[dict]:
    """The n students with the highest average percentage (see get_student_averages)."""
    averages = get_student_averages(semester=semester)
    averages.sort(key=lambda entry: entry["average_percentage"], reverse=True)
    return averages[:n]


def get_bottom_performers(n: int = 5, semester: int | None = None) -> list[dict]:
    """The n students with the lowest average percentage (see get_student_averages)."""
    averages = get_student_averages(semester=semester)
    averages.sort(key=lambda entry: entry["average_percentage"])
    return averages[:n]


@st.cache_data(ttl=60)
def get_student_performance_trend(roll_no: str) -> list[dict]:
    """
    One student's average percentage in EACH semester they have marks in,
    ordered by semester -- the data behind a performance-over-time line
    chart.

    Args:
        roll_no: The student to trace.

    Returns:
        A list of dicts: semester, average_percentage.
    """
    roll_no = validate_roll_no(roll_no)

    query = (
        f"SELECT m.semester, AVG({PERCENTAGE_EXPR}) AS average_percentage "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code "
        "WHERE m.roll_no = ? GROUP BY m.semester ORDER BY m.semester"
    )
    return [_round_field(dict(row), "average_percentage") for row in fetch_all(query, (roll_no,))]


# ---------------------------------------------------------------------------
# GRADE DISTRIBUTION AND SUBJECT DIFFICULTY (classification -- routed through
# modules/grades.py, not SQL -- see module docstring)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_grade_distribution(semester: int | None = None) -> dict[str, int]:
    """
    Count how many marks entries fall into each letter grade.

    Args:
        semester: If given, only marks recorded in this semester.

    Returns:
        A dict of {grade_letter: count}, with every grade from
        config.GRADE_SCALE present (even at 0), in the SAME order as
        config.GRADE_SCALE (highest grade first) -- so a chart built from
        this dict always displays grades from best to worst, not
        alphabetically.
    """
    query = (
        "SELECT m.internal, m.external, m.practical, "
        "s.max_internal, s.max_external, s.max_practical "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code"
    )
    params: list = []
    if semester is not None:
        query += " WHERE m.semester = ?"
        params.append(semester)

    distribution = {letter: 0 for _, letter, _ in config.GRADE_SCALE}

    for row in fetch_all(query, tuple(params)):
        percentage = calculate_percentage(
            row["internal"], row["external"], row["practical"],
            row["max_internal"], row["max_external"], row["max_practical"],
        )
        letter, _ = get_grade(percentage)
        distribution[letter] += 1

    return distribution


@st.cache_data(ttl=60)
def get_subject_difficulty_index(semester: int | None = None) -> list[dict]:
    """
    Rank subjects by how many students failed them.

    The "difficulty index" here is simply the FAIL RATE: the percentage
    of marks entries in that subject that did not pass (see
    modules.grades.is_pass()). A subject where 40% of students failed is,
    by this measure, harder than one where 5% did -- a directly
    explainable definition, as opposed to something more opaque like an
    average-percentage-only ranking (which can be skewed by one very
    strong or very weak individual result).

    Args:
        semester: If given, only marks recorded in this semester.

    Returns:
        A list of dicts: subject_code, subject_name, average_percentage,
        student_count, difficulty_index (0-100). Sorted hardest first.
    """
    query = (
        "SELECT m.subject_code, s.name AS subject_name, "
        "m.internal, m.external, m.practical, "
        "s.max_internal, s.max_external, s.max_practical "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code"
    )
    params: list = []
    if semester is not None:
        query += " WHERE m.semester = ?"
        params.append(semester)

    per_subject = defaultdict(lambda: {"subject_name": None, "percentages": [], "fail_count": 0})

    for row in fetch_all(query, tuple(params)):
        percentage = calculate_percentage(
            row["internal"], row["external"], row["practical"],
            row["max_internal"], row["max_external"], row["max_practical"],
        )
        bucket = per_subject[row["subject_code"]]
        bucket["subject_name"] = row["subject_name"]
        bucket["percentages"].append(percentage)
        if not is_pass(percentage):
            bucket["fail_count"] += 1

    results = []
    for subject_code, data in per_subject.items():
        total = len(data["percentages"])
        results.append({
            "subject_code": subject_code,
            "subject_name": data["subject_name"],
            "average_percentage": round(sum(data["percentages"]) / total, config.ROUND_DECIMALS),
            "student_count": total,
            "difficulty_index": round((data["fail_count"] / total) * 100, config.ROUND_DECIMALS),
        })

    results.sort(key=lambda entry: entry["difficulty_index"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# ATTENDANCE-VS-MARKS CORRELATION
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_attendance_marks_correlation(semester: int | None = None) -> tuple[list[dict], float | None]:
    """
    Pair up each (student, subject, semester)'s attendance percentage with
    their marks percentage in that same subject/semester, and compute the
    Pearson correlation coefficient between the two.

    A Pearson correlation of +1 means attendance and marks rise together
    perfectly; 0 means no linear relationship; a value clearly above 0
    (e.g. 0.5+) supports "students who attend more tend to score higher",
    which is exactly the assumption ml/generate_data.py's synthetic data
    was built around (see that file's module docstring).

    Args:
        semester: If given, only this semester's records.

    Returns:
        A (data_points, correlation) tuple:
          - data_points: a list of dicts (roll_no, subject_code, semester,
            attendance_percentage, marks_percentage) -- the scatter plot's
            raw data.
          - correlation: the Pearson correlation coefficient, rounded to
            config.ROUND_DECIMALS, or None if there were fewer than 2 data
            points, or if attendance or marks values had zero variance
            (a correlation is mathematically undefined when one variable
            never changes -- numpy would return NaN, which we deliberately
            convert to a clear None rather than displaying "nan" to a user).
    """
    query = (
        "SELECT m.roll_no, m.subject_code, m.semester, "
        f"{PERCENTAGE_EXPR} AS marks_percentage, "
        "a.classes_attended, a.classes_held "
        "FROM marks m "
        "JOIN subjects s ON m.subject_code = s.subject_code "
        "JOIN attendance a ON m.roll_no = a.roll_no "
        "AND m.subject_code = a.subject_code AND m.semester = a.semester"
    )
    params: list = []
    if semester is not None:
        query += " WHERE m.semester = ?"
        params.append(semester)

    data_points = []
    for row in fetch_all(query, tuple(params)):
        if row["classes_held"] == 0:
            continue  # 0/0 attendance is undefined -- exclude, don't treat as 0%.
        attendance_percentage = round((row["classes_attended"] / row["classes_held"]) * 100, config.ROUND_DECIMALS)
        data_points.append({
            "roll_no": row["roll_no"],
            "subject_code": row["subject_code"],
            "semester": row["semester"],
            "attendance_percentage": attendance_percentage,
            "marks_percentage": round(row["marks_percentage"], config.ROUND_DECIMALS),
        })

    correlation = None
    if len(data_points) >= 2:
        attendance_values = [point["attendance_percentage"] for point in data_points]
        marks_values = [point["marks_percentage"] for point in data_points]
        if np.std(attendance_values) > 0 and np.std(marks_values) > 0:
            raw_correlation = np.corrcoef(attendance_values, marks_values)[0, 1]
            correlation = round(float(raw_correlation), config.ROUND_DECIMALS)

    return data_points, correlation


def _round_field(row: dict, field: str) -> dict:
    """Round one field of a dict to config.ROUND_DECIMALS, in place, then return it."""
    row[field] = round(row[field], config.ROUND_DECIMALS)
    return row


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_analytics_page() -> None:
    """
    Streamlit page: the full analytics dashboard, Admin and Teacher only
    (this page shows every student's name and marks side by side, which a
    Student should not see about their classmates -- there is no
    "my own analytics" view in this system yet; that would be a separate,
    Student-scoped page, out of scope here).
    """
    auth.require_role(config.ROLE_ADMIN, config.ROLE_TEACHER)

    st.title("Analytics Dashboard")

    semester_choice = st.selectbox(
        "Filter by semester",
        options=["(all)"] + list(range(config.MIN_SEMESTER, config.MAX_SEMESTER + 1)),
    )
    semester = None if semester_choice == "(all)" else int(semester_choice)

    st.subheader("Subject Averages")
    subject_averages = get_subject_averages(semester=semester)
    if subject_averages:
        fig = px.bar(
            pd.DataFrame(subject_averages), x="subject_code", y="average_percentage",
            hover_data=["subject_name", "entry_count"], title="Average Percentage by Subject",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No marks data available yet for this selection.")

    st.subheader("Grade Distribution")
    grade_distribution = get_grade_distribution(semester=semester)
    if sum(grade_distribution.values()) > 0:
        distribution_df = pd.DataFrame(
            {"grade": list(grade_distribution.keys()), "count": list(grade_distribution.values())}
        )
        fig = px.bar(distribution_df, x="grade", y="count", title="Grade Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No marks data available yet for this selection.")

    st.subheader("Top and Bottom Performers")
    performer_count = st.slider("Number of students to show", min_value=3, max_value=15, value=5)
    top_col, bottom_col = st.columns(2)
    with top_col:
        st.write("**Top Performers**")
        top_performers = get_top_performers(n=performer_count, semester=semester)
        if top_performers:
            st.dataframe(pd.DataFrame(top_performers), use_container_width=True, hide_index=True)
        else:
            st.info("No data.")
    with bottom_col:
        st.write("**Bottom Performers**")
        bottom_performers = get_bottom_performers(n=performer_count, semester=semester)
        if bottom_performers:
            st.dataframe(pd.DataFrame(bottom_performers), use_container_width=True, hide_index=True)
        else:
            st.info("No data.")

    st.subheader("Attendance vs. Marks Correlation")
    data_points, correlation = get_attendance_marks_correlation(semester=semester)
    if data_points:
        fig = px.scatter(
            pd.DataFrame(data_points), x="attendance_percentage", y="marks_percentage",
            hover_data=["roll_no", "subject_code"], title="Attendance % vs. Marks %",
        )
        st.plotly_chart(fig, use_container_width=True)
        if correlation is not None:
            st.metric("Pearson correlation coefficient", correlation)
        else:
            st.info("Not enough variation in the data yet to compute a correlation coefficient.")
    else:
        st.info("No attendance and marks data pairs available yet for this selection.")

    st.subheader("Subject Difficulty Index")
    st.caption("Difficulty index = percentage of students who failed this subject (higher = harder).")
    difficulty = get_subject_difficulty_index(semester=semester)
    if difficulty:
        fig = px.bar(
            pd.DataFrame(difficulty), x="subject_code", y="difficulty_index",
            hover_data=["subject_name", "average_percentage", "student_count"],
            title="Subject Difficulty Index (Fail Rate %)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No marks data available yet for this selection.")

    st.subheader("Individual Student Performance Trend")
    all_students = students.list_students()
    if all_students:
        student_labels = {f"{s['roll_no']} - {s['name']}": s["roll_no"] for s in all_students}
        picked_label = st.selectbox("Select a student", options=list(student_labels.keys()))
        trend = get_student_performance_trend(student_labels[picked_label])
        if trend:
            fig = px.line(
                pd.DataFrame(trend), x="semester", y="average_percentage", markers=True,
                title=f"Performance Trend: {student_labels[picked_label]}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No marks recorded yet for this student.")
    else:
        st.info("No students found.")
