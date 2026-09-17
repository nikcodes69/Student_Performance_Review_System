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
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        (percentage, or None if no attendance exists yet), class_average
        (overall average marks percentage across every mark on file, or
        None if no marks exist yet).
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

    class_average_row = fetch_all(
        f"SELECT AVG({PERCENTAGE_EXPR}) AS avg_pct "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code"
    )[0]
    class_average = round(class_average_row["avg_pct"], 1) if class_average_row["avg_pct"] is not None else None

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
        "class_average": class_average,
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


# ---------------------------------------------------------------------------
# CLASS RANK AND PERCENTILE
# ---------------------------------------------------------------------------

def get_class_rankings(semester: int | None = None) -> list[dict]:
    """
    Every student's average percentage (see get_student_averages), ranked
    best-first with a class rank and percentile attached.

    RANKING METHOD: "competition ranking" (also called "1224" ranking) --
    students with the EXACT SAME average percentage share the same rank,
    and the NEXT distinct rank then skips ahead by however many students
    tied, rather than assigning consecutive ranks to students who are, by
    this measure, equally placed. E.g. three students tied for the
    highest average all get rank 1, and whoever is next gets rank 4 (not
    rank 2) -- this is the same convention real academic rank lists use,
    and it is why the loop below tracks the PREVIOUS distinct percentage
    rather than just using each student's position in the sorted list as
    their rank.

    PERCENTILE FORMULA: percentile = (total_students - rank) / (total_students - 1) * 100
    -- the standard "percentile rank" definition. Rank 1 (the very top)
    lands at the 100th percentile; the lowest rank lands at the 0th
    percentile. With only one student, percentile is defined as 100
    (nothing to be better than, but not left undefined either).

    Args:
        semester: If given, only marks recorded in this semester (passed
            straight through to get_student_averages()).

    Returns:
        A list of dicts: roll_no, student_name, average_percentage, rank,
        percentile -- sorted by rank (best first).
    """
    averages = get_student_averages(semester=semester)
    averages.sort(key=lambda entry: entry["average_percentage"], reverse=True)

    total = len(averages)
    results = []
    previous_percentage = None
    previous_rank = 0

    for index, entry in enumerate(averages, start=1):
        if entry["average_percentage"] == previous_percentage:
            rank = previous_rank
        else:
            rank = index
        previous_percentage = entry["average_percentage"]
        previous_rank = rank

        percentile = 100.0 if total <= 1 else round((total - rank) / (total - 1) * 100, config.ROUND_DECIMALS)

        results.append({
            "roll_no": entry["roll_no"],
            "student_name": entry["student_name"],
            "average_percentage": entry["average_percentage"],
            "rank": rank,
            "total_students": total,
            "percentile": percentile,
        })

    return results


def get_student_rank(roll_no: str, semester: int | None = None) -> dict | None:
    """
    One student's own entry from get_class_rankings().

    Args:
        roll_no: The student to look up.
        semester: If given, only marks recorded in this semester.

    Returns:
        A dict (see get_class_rankings()) or None if this student has no
        marks yet for this selection -- there is nothing to rank.
    """
    roll_no = validate_roll_no(roll_no)
    for entry in get_class_rankings(semester=semester):
        if entry["roll_no"] == roll_no:
            return entry
    return None


# ---------------------------------------------------------------------------
# ATTENDANCE SHORTAGE ALERTS
# ---------------------------------------------------------------------------

def get_attendance_shortage_list(semester: int | None = None) -> list[dict]:
    """
    Every student whose AVERAGE attendance percentage, across every
    subject they have attendance recorded for (see
    get_student_attendance_averages()), falls below
    config.ATTENDANCE_SHORTAGE_THRESHOLD.

    WHY "AVERAGE ACROSS SUBJECTS", NOT "ANY SINGLE SUBJECT BELOW
    THRESHOLD": modules/attendance.py's per-row 'shortage' flag already
    answers "is this student short in THIS ONE subject", wherever a
    single subject's attendance is shown (e.g. render_attendance_page()'s
    table). This function answers a different, institution-wide question
    for the Dashboard's alert panel: "which students are short OVERALL"
    -- the number that actually determines whether a student meets a
    college's attendance requirement to sit an exam.

    Args:
        semester: If given, only attendance recorded in this semester.

    Returns:
        A list of dicts: roll_no, student_name, average_attendance --
        sorted worst (lowest attendance) first.
    """
    averages = get_student_attendance_averages(semester=semester)
    shortage = [row for row in averages if row["average_attendance"] < config.ATTENDANCE_SHORTAGE_THRESHOLD]
    shortage.sort(key=lambda entry: entry["average_attendance"])
    return shortage


# ---------------------------------------------------------------------------
# DASHBOARD-SPECIFIC ANALYTICS (spotlight cards, distributions -- see
# render_dashboard_page() at the bottom of this file)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_student_attendance_averages(semester: int | None = None) -> list[dict]:
    """
    Average attendance percentage per student -- the attendance
    equivalent of get_student_averages() above.

    Args:
        semester: If given, only attendance recorded in this semester.

    Returns:
        A list of dicts: roll_no, student_name, average_attendance.
        Students with zero classes_held anywhere are excluded (nothing
        meaningful to average -- the same "undefined, not 0%" reasoning
        as modules/attendance.py's _compute_attendance_percentage()).
    """
    query = (
        "SELECT a.roll_no, st.name AS student_name, "
        "AVG(a.classes_attended * 100.0 / a.classes_held) AS average_attendance "
        "FROM attendance a JOIN students st ON a.roll_no = st.roll_no "
        "WHERE a.classes_held > 0"
    )
    params: list = []
    if semester is not None:
        query += " AND a.semester = ?"
        params.append(semester)
    query += " GROUP BY a.roll_no, st.name"

    return [_round_field(dict(row), "average_attendance") for row in fetch_all(query, tuple(params))]


def get_top_attendance_performers(n: int = 1, semester: int | None = None) -> list[dict]:
    """The n students with the highest average attendance (see get_student_attendance_averages)."""
    averages = get_student_attendance_averages(semester=semester)
    averages.sort(key=lambda entry: entry["average_attendance"], reverse=True)
    return averages[:n]


def _latest_semester_with_marks() -> int | None:
    """The highest semester number that has any marks recorded at all --
    used as the default "current" semester for dashboard comparisons when
    the caller doesn't specify one. Returns None if there are no marks
    anywhere yet."""
    row = fetch_all("SELECT MAX(semester) AS latest FROM marks")[0]
    return row["latest"]


def get_most_improved_marks(n: int = 1, semester: int | None = None) -> list[dict]:
    """
    Students with the largest positive change in average marks percentage
    between the semester before `semester` and `semester` itself.

    Args:
        n: How many students to return.
        semester: The "current" semester to compare against the one
            immediately before it. Defaults to the latest semester with
            any marks recorded.

    Returns:
        A list of dicts: roll_no, student_name, improvement (percentage
        points, current minus previous -- can be negative, though this
        function only returns the TOP n, so a negative value here would
        mean even the "most improved" student actually declined). Empty
        if there's no semester with marks in both it and the one before.
    """
    if semester is None:
        semester = _latest_semester_with_marks()
    if semester is None or semester <= config.MIN_SEMESTER:
        return []

    current = {r["roll_no"]: r for r in get_student_averages(semester=semester)}
    previous = {r["roll_no"]: r["average_percentage"] for r in get_student_averages(semester=semester - 1)}

    improvements = [
        {
            "roll_no": roll_no,
            "student_name": row["student_name"],
            "improvement": round(row["average_percentage"] - previous[roll_no], config.ROUND_DECIMALS),
        }
        for roll_no, row in current.items() if roll_no in previous
    ]
    improvements.sort(key=lambda entry: entry["improvement"], reverse=True)
    return improvements[:n]


def get_most_improved_attendance(n: int = 1, semester: int | None = None) -> list[dict]:
    """Same as get_most_improved_marks(), but for attendance percentage instead of marks."""
    if semester is None:
        semester = _latest_semester_with_marks()
    if semester is None or semester <= config.MIN_SEMESTER:
        return []

    current = {r["roll_no"]: r for r in get_student_attendance_averages(semester=semester)}
    previous = {
        r["roll_no"]: r["average_attendance"] for r in get_student_attendance_averages(semester=semester - 1)
    }

    improvements = [
        {
            "roll_no": roll_no,
            "student_name": row["student_name"],
            "improvement": round(row["average_attendance"] - previous[roll_no], config.ROUND_DECIMALS),
        }
        for roll_no, row in current.items() if roll_no in previous
    ]
    improvements.sort(key=lambda entry: entry["improvement"], reverse=True)
    return improvements[:n]


@st.cache_data(ttl=60)
def get_students_by_semester_distribution() -> list[dict]:
    """Count of active students per semester -- feeds the dashboard's donut chart."""
    return fetch_all(
        "SELECT semester, COUNT(*) AS student_count FROM students "
        "WHERE is_active = 1 GROUP BY semester ORDER BY semester"
    )


@st.cache_data(ttl=60)
def get_subject_pass_fail_breakdown(semester: int | None = None) -> list[dict]:
    """
    Pass/fail counts per subject -- classification, so (per this file's
    module docstring) computed via modules.grades.is_pass() in Python,
    not as a second copy of the pass/fail rule in SQL.

    Args:
        semester: If given, only marks recorded in this semester.

    Returns:
        A list of dicts: subject_code, subject_name, pass_count, fail_count.
    """
    query = (
        "SELECT m.subject_code, s.name AS subject_name, "
        "m.internal, m.external, m.practical, s.max_internal, s.max_external, s.max_practical "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code"
    )
    params: list = []
    if semester is not None:
        query += " WHERE m.semester = ?"
        params.append(semester)

    per_subject = defaultdict(lambda: {"subject_name": None, "pass_count": 0, "fail_count": 0})
    for row in fetch_all(query, tuple(params)):
        percentage = calculate_percentage(
            row["internal"], row["external"], row["practical"],
            row["max_internal"], row["max_external"], row["max_practical"],
        )
        bucket = per_subject[row["subject_code"]]
        bucket["subject_name"] = row["subject_name"]
        if is_pass(percentage):
            bucket["pass_count"] += 1
        else:
            bucket["fail_count"] += 1

    return [{"subject_code": code, **data} for code, data in per_subject.items()]


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
# COMPARATIVE ANALYTICS -- one student vs. the whole class, same chart
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_class_average_by_semester() -> dict[int, float]:
    """
    Overall class average percentage (across every mark on file,
    regardless of subject or student), grouped by semester -- the
    "class" side of get_student_vs_class_trend()'s comparison below.

    Returns:
        A dict of {semester: average_percentage}, only for semesters that
        have at least one mark recorded anywhere.
    """
    query = (
        f"SELECT m.semester, AVG({PERCENTAGE_EXPR}) AS average_percentage "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code "
        "GROUP BY m.semester"
    )
    return {row["semester"]: round(row["average_percentage"], config.ROUND_DECIMALS) for row in fetch_all(query)}


def get_student_vs_class_trend(roll_no: str) -> list[dict]:
    """
    One student's performance trend (see get_student_performance_trend())
    paired, semester by semester, with the CLASS-WIDE average for that
    same semester -- the data behind a "you vs. the class" comparison
    line chart, used by both render_analytics_page() (Admin/Teacher,
    picking any student) and modules/student_portal.py (a Student, always
    pinned to their own roll_no -- see that module's docstring for why).

    Args:
        roll_no: The student to compare.

    Returns:
        A list of dicts: semester, student_percentage, class_percentage.
        Only includes semesters the STUDENT has marks in -- a semester
        they haven't reached yet has nothing of theirs to compare.
    """
    student_trend = get_student_performance_trend(roll_no)
    class_by_semester = get_class_average_by_semester()

    return [
        {
            "semester": row["semester"],
            "student_percentage": row["average_percentage"],
            "class_percentage": class_by_semester.get(row["semester"]),
        }
        for row in student_trend
    ]


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

    st.subheader("Individual Student Performance Trend vs. Class Average")
    all_students = students.list_students()
    if all_students:
        student_labels = {f"{s['roll_no']} - {s['name']}": s["roll_no"] for s in all_students}
        picked_label = st.selectbox("Select a student", options=list(student_labels.keys()))
        picked_roll_no = student_labels[picked_label]
        comparison = get_student_vs_class_trend(picked_roll_no)
        if comparison:
            comparison_df = pd.DataFrame(comparison)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison_df["semester"], y=comparison_df["student_percentage"],
                mode="lines+markers", name=picked_roll_no,
            ))
            fig.add_trace(go.Scatter(
                x=comparison_df["semester"], y=comparison_df["class_percentage"],
                mode="lines+markers", name="Class Average", line=dict(dash="dash"),
            ))
            fig.update_layout(
                title=f"{picked_label}: Own Average vs. Class Average",
                xaxis_title="Semester", yaxis_title="Percentage",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No marks recorded yet for this student.")

        rank_entry = get_student_rank(picked_roll_no)
        if rank_entry:
            rank_cols = st.columns(3)
            rank_cols[0].metric("Class Rank", f"#{rank_entry['rank']} of {rank_entry['total_students']}")
            rank_cols[1].metric("Percentile", f"{rank_entry['percentile']}th")
            rank_cols[2].metric("Average Percentage", f"{rank_entry['average_percentage']}%")
    else:
        st.info("No students found.")

    st.subheader("Class Rankings")
    rankings = get_class_rankings(semester=semester)
    if rankings:
        rankings_df = pd.DataFrame(rankings)[
            ["rank", "roll_no", "student_name", "average_percentage", "percentile"]
        ]
        rankings_df.columns = ["Rank", "Roll No", "Student", "Average %", "Percentile"]
        st.dataframe(rankings_df, use_container_width=True, hide_index=True)
    else:
        st.info("No marks data available yet for this selection.")


# ---------------------------------------------------------------------------
# DASHBOARD PAGE -- an at-a-glance overview, distinct from "Analytics"
# above (which is a deep-dive page with filters and correlation charts).
# This page is meant to be understood in a glance: KPI cards, spotlight
# cards for top/most-improved students, and a couple of summary charts.
# ---------------------------------------------------------------------------

# A small fixed palette for avatar background colors -- picked once here,
# reused deterministically per name (see _avatar_color()) so the SAME
# student always gets the SAME color across reruns, rather than a new
# random one every time the page refreshes.
_AVATAR_COLORS = ("#4F46E5", "#059669", "#DC2626", "#D97706", "#7C3AED", "#0891B2", "#DB2777", "#65A30D")


def _avatar_initials(name: str) -> str:
    """First letter of up to the first two words of a name, uppercased --
    e.g. 'Nikhil Thapa' -> 'NT'. This system has no photo uploads, so
    initials-in-a-colored-circle (the same fallback LinkedIn/Slack/etc.
    use) stands in for a real avatar."""
    words = name.split()
    return "".join(word[0] for word in words[:2]).upper() or "?"


def _avatar_color(seed_text: str) -> str:
    """Deterministic color from _AVATAR_COLORS, picked by summing the
    character codes of seed_text -- same input always gives the same
    color, without needing to store a color choice anywhere."""
    index = sum(ord(character) for character in seed_text) % len(_AVATAR_COLORS)
    return _AVATAR_COLORS[index]


def _render_spotlight_card(name: str, roll_no: str, stat_label: str, stat_value: str) -> None:
    """
    Render one spotlight card: a colored initials avatar, the student's
    name/roll_no, and one highlighted stat -- inside a bordered container
    so it stays visually consistent with the rest of this app's card
    styling and adapts automatically to light/dark mode (st.container
    itself is theme-aware; only the avatar circle's own fixed background
    color is hardcoded, which is intentional -- a solid color circle
    should look the same regardless of page theme).

    SECURITY NOTE: name and roll_no are STUDENT-ENTERED data (via
    modules/students.py's create_student()/update_student()), being
    interpolated into raw HTML below (unsafe_allow_html=True). Both are
    passed through html.escape() first -- the same injection-prevention
    principle as parameterized SQL queries elsewhere in this project,
    just for a different attack surface (HTML/script injection instead
    of SQL injection). Without this, a student name containing
    "<script>...</script>" would be rendered as live markup instead of
    literal text.
    """
    safe_name = escape(name)
    safe_roll_no = escape(roll_no)
    color = _avatar_color(name)
    initials = _avatar_initials(name)

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:44px; height:44px; border-radius:50%; background:{color};
                            display:flex; align-items:center; justify-content:center;
                            color:white; font-weight:600; font-size:15px; flex-shrink:0;">
                    {initials}
                </div>
                <div style="min-width:0;">
                    <div style="font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{safe_name}</div>
                    <div style="font-size:12px; opacity:0.65;">{safe_roll_no}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"{stat_label}: **{stat_value}**")


def render_dashboard_page() -> None:
    """
    Streamlit page: KPI cards, spotlight cards (best/most-improved
    students in marks and attendance, with avatar-style cards), a
    student-distribution donut chart, a per-subject pass/fail breakdown,
    and per-subject average-score gauges.
    """
    user = auth.require_role(config.ROLE_ADMIN, config.ROLE_TEACHER)

    st.title("Dashboard")
    st.caption(f"Signed in as **{user['username']}** ({user['role'].capitalize()})")

    # Lazy import: modules/ml_predictions.py is the ONLY function this
    # read-only analytics module reaches outside plain SQL for (see the
    # module docstring's "READ-ONLY" note) -- kept local to this one page
    # function, rather than a top-level import, so that fact stays visible
    # right where it's used instead of hiding in the file's import block.
    from modules.ml_predictions import get_at_risk_count

    summary = get_dashboard_summary()
    at_risk_count = get_at_risk_count()

    kpi_cols = st.columns(6)
    kpi_cols[0].metric(":material/group: Students", summary["student_count"])
    kpi_cols[1].metric(":material/menu_book: Subjects", summary["subject_count"])
    kpi_cols[2].metric(
        ":material/monitoring: Class Average",
        f"{summary['class_average']}%" if summary["class_average"] is not None else "N/A",
    )
    kpi_cols[3].metric(
        ":material/check_circle: Pass Rate",
        f"{summary['pass_rate']}%" if summary["pass_rate"] is not None else "N/A",
    )
    kpi_cols[4].metric(
        ":material/event_available: Avg. Attendance",
        f"{summary['avg_attendance']}%" if summary["avg_attendance"] is not None else "N/A",
    )
    kpi_cols[5].metric(
        ":material/warning: At-Risk Count",
        at_risk_count if at_risk_count is not None else "N/A",
        help="Active students the deployed at-risk model currently flags, evaluated at each student's own current semester.",
    )

    shortage_list = get_attendance_shortage_list()
    if shortage_list:
        st.divider()
        with st.container(border=True):
            st.markdown(
                f":material/warning: **Attendance Shortage Alert** -- "
                f"{len(shortage_list)} student(s) below {config.ATTENDANCE_SHORTAGE_THRESHOLD}% average attendance"
            )
            shortage_df = pd.DataFrame(shortage_list)[["roll_no", "student_name", "average_attendance"]]
            shortage_df.columns = ["Roll No", "Student", "Average Attendance %"]
            st.dataframe(shortage_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Spotlight")

    spotlight_cols = st.columns(4)

    with spotlight_cols[0]:
        top_marks = get_top_performers(n=1)
        if top_marks:
            entry = top_marks[0]
            _render_spotlight_card(entry["student_name"], entry["roll_no"], "Best in Marks", f"{entry['average_percentage']}%")
        else:
            st.info("No marks data yet.")

    with spotlight_cols[1]:
        top_attendance = get_top_attendance_performers(n=1)
        if top_attendance:
            entry = top_attendance[0]
            _render_spotlight_card(entry["student_name"], entry["roll_no"], "Best in Attendance", f"{entry['average_attendance']}%")
        else:
            st.info("No attendance data yet.")

    with spotlight_cols[2]:
        most_improved_marks = get_most_improved_marks(n=1)
        if most_improved_marks:
            entry = most_improved_marks[0]
            sign = "+" if entry["improvement"] >= 0 else ""
            _render_spotlight_card(entry["student_name"], entry["roll_no"], "Most Improved (Marks)", f"{sign}{entry['improvement']} pts")
        else:
            st.info("Need 2 semesters of marks to compute this.")

    with spotlight_cols[3]:
        most_improved_attendance = get_most_improved_attendance(n=1)
        if most_improved_attendance:
            entry = most_improved_attendance[0]
            sign = "+" if entry["improvement"] >= 0 else ""
            _render_spotlight_card(entry["student_name"], entry["roll_no"], "Most Improved (Attendance)", f"{sign}{entry['improvement']} pts")
        else:
            st.info("Need 2 semesters of attendance to compute this.")

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Students by Semester")
        distribution = get_students_by_semester_distribution()
        if distribution:
            distribution_df = pd.DataFrame(distribution)
            distribution_df["label"] = "Semester " + distribution_df["semester"].astype(str)
            fig = px.pie(distribution_df, names="label", values="student_count", hole=0.55)
            fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No students yet.")

    with chart_col2:
        st.subheader("Pass / Fail by Subject")
        breakdown = get_subject_pass_fail_breakdown()
        if breakdown:
            breakdown_df = pd.DataFrame(breakdown)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=breakdown_df["subject_code"], y=breakdown_df["pass_count"], name="Pass"))
            fig.add_trace(go.Bar(x=breakdown_df["subject_code"], y=breakdown_df["fail_count"], name="Fail"))
            fig.update_layout(barmode="stack", height=340, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No marks data yet.")

    st.divider()
    st.subheader("Average Score by Subject")
    subject_averages = get_subject_averages()
    if subject_averages:
        gauge_columns_per_row = 4
        for row_start in range(0, len(subject_averages), gauge_columns_per_row):
            row_entries = subject_averages[row_start:row_start + gauge_columns_per_row]
            gauge_cols = st.columns(gauge_columns_per_row)
            for col, entry in zip(gauge_cols, row_entries):
                with col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=entry["average_percentage"],
                        title={"text": entry["subject_code"]},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": _avatar_color(entry["subject_code"])},
                        },
                    ))
                    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No marks data yet.")
