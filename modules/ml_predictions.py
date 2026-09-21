"""
modules/ml_predictions.py
==========================
Loads the three models trained in ml/model_training.ipynb and exposes
prediction functions plus Streamlit pages for each: at-risk classification
(Task 1), final marks regression (Task 2), and student segmentation
(Task 3) -- plus a model comparison page.

THIS FILE NEVER TRAINS ANYTHING. It only ever calls joblib.load() on
files ml/model_training.ipynb already produced, and calls .predict() on
the result. If a model file is missing, that means the notebook has not
been run yet -- this file raises ModelNotFoundError (defined all the way
back in utils/exceptions.py in step 1) rather than trying to train a
replacement on the spot.

======================================================================
assignments_submitted: NOW COMPUTED FROM REAL RECORDS, NOT TYPED IN
======================================================================
All five of Task 1's features -- internal marks, attendance, previous
SGPA, backlog count, and assignments submitted -- are computed live from
this project's own database (students, subjects, marks, attendance,
assignments tables). This used to be the one manually-entered feature
(nothing tracked assignment submissions anywhere), which meant every
prediction depended on whoever was clicking "Predict" typing in an
honest number. modules/assignments.py closes that gap: Admin/Teacher
record real total-assigned/submitted counts per subject, and
_compute_assignment_engagement() below turns those into the same 0-10
"engagement score" scale ml/generate_data.py trained the models on (see
that function's docstring for exactly how the translation works, and
config.ASSIGNMENT_ENGAGEMENT_SCALE for why both files must agree on the
same scale).

======================================================================
WHY LIVE FEATURE COMPUTATION MUST MATCH THE TRAINING DATA EXACTLY
======================================================================
`_compute_live_features()` below rebuilds internal_pct, attendance_pct,
average_marks, consistency, and improvement_rate from real marks and
attendance rows, using the SAME formulas ml/generate_data.py used to
generate the synthetic training data (e.g. "consistency" is the standard
deviation ACROSS the three assessment-component percentages -- internal,
external, practical -- not across different subjects). If a live feature
were computed even slightly differently from its training-time
definition, every prediction made from it would be subtly wrong in a way
that is easy to miss -- this is a well-known category of real-world ML
bug called "training/serving skew". Matching the definitions exactly is
what makes these predictions trustworthy.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

import config
from modules import auth, students
from modules.marks import compute_sgpa_for_marks, list_marks_for_student
from modules.attendance import list_attendance_for_student
from modules.assignments import list_assignments_for_student
from utils.exceptions import ModelNotFoundError, ValidationError
from utils.logger import get_logger
from utils.table_view import render_data_table

logger = get_logger(__name__)

RESULTS_DIR = config.ML_DIR / "results"

PREDICTION_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)


# ---------------------------------------------------------------------------
# MODEL LOADING (cached -- joblib.load() from disk only happens once per
# Streamlit server process, not on every page interaction)
# ---------------------------------------------------------------------------

def _load_pickle(path):
    """Load a joblib-pickled object, or raise ModelNotFoundError with a
    clear, actionable message if the notebook has not produced it yet."""
    import joblib  # imported here, not at module top, so a missing model
                    # file fails with OUR message before joblib even runs

    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found: {path.name}. "
            "Run ml/model_training.ipynb first to train and save this model."
        )
    return joblib.load(path)


def _load_metadata(path) -> dict:
    """Load a model's metadata JSON, or raise ModelNotFoundError."""
    if not path.exists():
        raise ModelNotFoundError(
            f"Model metadata not found: {path.name}. Run ml/model_training.ipynb first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def _get_at_risk_model():
    model = _load_pickle(config.ML_MODELS_DIR / "at_risk_classifier.pkl")
    scaler = _load_pickle(config.ML_MODELS_DIR / "at_risk_scaler.pkl")
    metadata = _load_metadata(config.ML_MODELS_DIR / "at_risk_classifier_metadata.json")
    return model, scaler, metadata


@st.cache_resource
def _get_final_marks_model():
    model = _load_pickle(config.ML_MODELS_DIR / "final_marks_regressor.pkl")
    metadata = _load_metadata(config.ML_MODELS_DIR / "final_marks_regressor_metadata.json")
    return model, metadata


@st.cache_resource
def _get_segmentation_model():
    model = _load_pickle(config.ML_MODELS_DIR / "student_segmentation.pkl")
    scaler = _load_pickle(config.ML_MODELS_DIR / "segmentation_scaler.pkl")
    metadata = _load_metadata(config.ML_MODELS_DIR / "student_segmentation_metadata.json")
    return model, scaler, metadata


# ---------------------------------------------------------------------------
# LIVE FEATURE COMPUTATION (see module docstring: must match
# ml/generate_data.py's definitions exactly)
# ---------------------------------------------------------------------------

def _compute_sgpa_for_semester(roll_no: str, semester: int) -> float | None:
    """SGPA for one semester, computed live from that semester's marks.
    Thin wrapper around modules.marks.compute_sgpa_for_marks() -- see that
    function's docstring for why this calculation is centralised there
    rather than duplicated here."""
    semester_marks = list_marks_for_student(roll_no, semester=semester)
    return compute_sgpa_for_marks(semester_marks)


def _compute_backlog_count(roll_no: str) -> int:
    """Count subjects the student has NEVER passed, across every semester
    and every attempt on record -- a subject with at least one passing
    attempt (regular, backlog, or improvement) does not count, even if an
    earlier attempt failed."""
    all_marks = list_marks_for_student(roll_no)
    ever_passed = {}
    for row in all_marks:
        code = row["subject_code"]
        ever_passed[code] = ever_passed.get(code, False) or row["passed"]
    return sum(1 for passed in ever_passed.values() if not passed)


def _compute_assignment_engagement(roll_no: str, semester: int) -> int:
    """
    Compute the "assignments submitted" ML feature (config.
    ASSIGNMENT_ENGAGEMENT_SCALE, i.e. 0-10) from real assignment records
    -- see the module docstring for why this replaces a manually-typed
    number.

    HOW THE TRANSLATION WORKS: a real assignments row is per-subject (a
    subject might assign 5 things, another 8), but the trained model
    expects a single flat 0-10 number for the whole semester (matching
    ml/generate_data.py's synthetic "engagement score"). To bridge that,
    this function averages the SUBMISSION RATE (submitted / total_assigned)
    across every subject this student has an assignment record for this
    semester, then scales that average rate (0.0-1.0) onto the 0-10 range
    the model was trained on. E.g. an average 80% submission rate across
    all subjects becomes an engagement score of 8.

    This is an approximation of what the synthetic training data's
    "assignments_submitted" represented, not a re-derivation of the exact
    same quantity (the real world doesn't have a single "assignments
    submitted out of 10" number to look up) -- worth stating plainly
    rather than implying false precision.

    Returns:
        An integer 0-config.ASSIGNMENT_ENGAGEMENT_SCALE. Returns 0 if no
        assignment records exist yet for this semester (treated as "no
        engagement recorded yet", not an error -- a student legitimately
        might not have any assignment records early in a semester).
    """
    assignment_rows = list_assignments_for_student(roll_no, semester=semester)
    rates = [
        row["submitted"] / row["total_assigned"]
        for row in assignment_rows if row["total_assigned"] > 0
    ]
    if not rates:
        return 0

    average_rate = sum(rates) / len(rates)
    return round(average_rate * config.ASSIGNMENT_ENGAGEMENT_SCALE)


def _compute_live_features(roll_no: str, semester: int) -> dict | None:
    """
    Compute every ML feature for one student in one semester, live from
    the database.

    Returns:
        A dict with keys internal_pct, attendance_pct, practical_pct,
        average_marks, consistency, improvement_rate, previous_sgpa,
        backlog_count, assignments_submitted -- or None if the student has
        no marks recorded yet for this semester (nothing to compute a
        prediction from).
    """
    semester_marks = list_marks_for_student(roll_no, semester=semester)
    if not semester_marks:
        return None

    # Average percentage per ASSESSMENT COMPONENT (internal/external/
    # practical), across every subject this semester -- matching
    # ml/generate_data.py's aggregate internal_pct/external_pct/
    # practical_pct exactly. The maximums are fixed system-wide constants
    # (config.MAX_INTERNAL_MARKS etc.), never zero, so no zero-division
    # guard is needed here.
    internal_components = [row["internal"] / config.MAX_INTERNAL_MARKS * 100 for row in semester_marks]
    external_components = [row["external"] / config.MAX_EXTERNAL_MARKS * 100 for row in semester_marks]
    practical_components = [row["practical"] / config.MAX_PRACTICAL_MARKS * 100 for row in semester_marks]

    internal_pct = float(np.mean(internal_components)) if internal_components else 0.0
    external_pct = float(np.mean(external_components)) if external_components else 0.0
    practical_pct = float(np.mean(practical_components)) if practical_components else 0.0

    average_marks = float(np.mean([row["percentage"] for row in semester_marks]))
    # Same definition as ml/generate_data.py: variability ACROSS the three
    # component types, not across subjects.
    consistency = float(np.std([internal_pct, external_pct, practical_pct]))

    attendance_rows = list_attendance_for_student(roll_no, semester=semester)
    known_attendance = [
        row["attendance_percentage"] for row in attendance_rows
        if row["attendance_percentage"] is not None
    ]
    attendance_pct = float(np.mean(known_attendance)) if known_attendance else 0.0

    previous_semester = semester - 1
    previous_sgpa = None
    previous_average_marks = None
    if previous_semester >= config.MIN_SEMESTER:
        previous_sgpa = _compute_sgpa_for_semester(roll_no, previous_semester)
        previous_marks = list_marks_for_student(roll_no, semester=previous_semester)
        if previous_marks:
            previous_average_marks = float(np.mean([row["percentage"] for row in previous_marks]))

    improvement_rate = (
        average_marks - previous_average_marks if previous_average_marks is not None else 0.0
    )

    return {
        "internal_pct": round(internal_pct, config.ROUND_DECIMALS),
        "attendance_pct": round(attendance_pct, config.ROUND_DECIMALS),
        "practical_pct": round(practical_pct, config.ROUND_DECIMALS),
        "average_marks": round(average_marks, config.ROUND_DECIMALS),
        "consistency": round(consistency, config.ROUND_DECIMALS),
        "improvement_rate": round(improvement_rate, config.ROUND_DECIMALS),
        "previous_sgpa": round(previous_sgpa, config.ROUND_DECIMALS) if previous_sgpa is not None else 0.0,
        "backlog_count": _compute_backlog_count(roll_no),
        "assignments_submitted": _compute_assignment_engagement(roll_no, semester),
    }


# ---------------------------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------------------------

def predict_at_risk_for_student(roll_no: str, semester: int) -> dict:
    """
    Predict at-risk status for an existing student.

    Args:
        roll_no: The student.
        semester: Which semester's marks/attendance/assignments to compute
            features from.

    Returns:
        A dict: at_risk (bool), risk_probability (float, 0-1), and
        features_used (the exact feature values the model saw, for
        transparency).

    Raises:
        ModelNotFoundError: if ml/model_training.ipynb has not been run.
        ValidationError: if the student has no marks yet this semester.
    """
    live_features = _compute_live_features(roll_no, semester)
    if live_features is None:
        raise ValidationError(
            f"No marks recorded yet for {roll_no} in semester {semester} -- cannot predict."
        )

    model, scaler, metadata = _get_at_risk_model()

    feature_row = live_features
    X = pd.DataFrame([feature_row])[metadata["features"]]
    X_scaled = scaler.transform(X)

    prediction = bool(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])  # P(at_risk = 1)

    logger.info("At-risk prediction for %s (semester %s): at_risk=%s prob=%.4f", roll_no, semester, prediction, probability)

    return {
        "at_risk": prediction,
        "risk_probability": round(probability, 4),
        "features_used": feature_row,
    }


def generate_recommendations(features: dict, semester: int) -> list[str]:
    """
    Rule-based, explainable improvement recommendations, built directly
    from the SAME 9 features the at-risk model itself predicts from (see
    predict_at_risk_for_student()'s "features_used") -- not a separate,
    disconnected "advice" system layered on top.

    DELIBERATELY RULE-BASED, NOT A CALL TO AN EXTERNAL AI/LLM SERVICE: no
    new credentials, no per-call cost, no network dependency, and every
    recommendation is exactly reproducible from the input features -- a
    fixed set of feature values always produces the exact same
    recommendations, which matters for something a Teacher might refer
    back to later and expect to still make sense. Each threshold below
    reuses an EXISTING config constant wherever this exact real-world
    rule already exists elsewhere in the app (config.ATTENDANCE_
    SHORTAGE_THRESHOLD, config.PASS_PERCENTAGE) -- see config.py's own
    comments on the RECOMMENDATION_* constants for the few genuinely new
    judgment calls this function needed its own threshold for.

    Args:
        features: A features_used dict (see predict_at_risk_for_student()) --
            internal_pct, attendance_pct, practical_pct, average_marks,
            consistency, improvement_rate, previous_sgpa, backlog_count,
            assignments_submitted.
        semester: The semester these features were computed for -- used
            only to skip the previous_sgpa rule for semester 1, where
            "previous_sgpa" is 0.0 because there IS no previous semester
            (not because it was actually low; _compute_live_features()
            cannot tell those two cases apart from the number alone, so
            this function uses semester instead of guessing).

    Returns:
        A list of human-readable recommendation strings, most urgent
        first (attendance and backlogs before finer-grained component
        weaknesses). Empty if every feature is already within its
        reasonable range -- expected for most students, and always
        expected for anyone the model does not flag at_risk.
    """
    recommendations = []

    if features["attendance_pct"] < config.ATTENDANCE_SHORTAGE_THRESHOLD:
        recommendations.append(
            f"Attendance is {features['attendance_pct']}%, below the "
            f"{config.ATTENDANCE_SHORTAGE_THRESHOLD}% requirement. Prioritize attending "
            "classes regularly -- this is usually the single fastest lever to pull."
        )

    if features["backlog_count"] > 0:
        subject_word = "subject" if features["backlog_count"] == 1 else "subjects"
        recommendations.append(
            f"{features['backlog_count']} {subject_word} not yet passed in any attempt. "
            "Clearing backlogs should take priority alongside current coursework, before "
            "they compound further."
        )

    if features["average_marks"] < config.PASS_PERCENTAGE:
        recommendations.append(
            f"Overall average this semester is {features['average_marks']}%, below the "
            f"{config.PASS_PERCENTAGE}% pass mark. Needs focused revision across subjects "
            "broadly, not just one weak area."
        )

    if features["internal_pct"] < config.PASS_PERCENTAGE:
        recommendations.append(
            f"Internal assessment average is {features['internal_pct']}%. Internal marks "
            "(class tests, quizzes, participation) are usually the most directly "
            "controllable part of the final grade -- start there."
        )

    if features["practical_pct"] < config.PASS_PERCENTAGE:
        recommendations.append(
            f"Practical average is {features['practical_pct']}%. Spend more time on lab "
            "work and practical assessments specifically."
        )

    if features["assignments_submitted"] < config.RECOMMENDATION_LOW_ENGAGEMENT_THRESHOLD:
        recommendations.append(
            f"Assignment engagement score is {features['assignments_submitted']}/"
            f"{config.ASSIGNMENT_ENGAGEMENT_SCALE}. Submitting assignments consistently "
            "and on time directly raises this and reflects steady effort to instructors."
        )

    if features["improvement_rate"] < 0:
        recommendations.append(
            f"Performance has DECLINED by {abs(features['improvement_rate'])} percentage "
            "points compared to last semester -- this trend needs to be reversed, not "
            "just this semester's marks improved in isolation."
        )

    if semester > config.MIN_SEMESTER and features["previous_sgpa"] < config.RECOMMENDATION_LOW_SGPA_THRESHOLD:
        recommendations.append(
            f"Previous semester's SGPA was {features['previous_sgpa']}, below "
            f"{config.RECOMMENDATION_LOW_SGPA_THRESHOLD}. Consider seeking extra academic "
            "support (tutoring, office hours) rather than relying on self-study alone."
        )

    if features["consistency"] > config.RECOMMENDATION_HIGH_INCONSISTENCY_THRESHOLD:
        recommendations.append(
            f"Performance is uneven across internal/external/practical components "
            f"(spread of {features['consistency']} points). Identify the specific weak "
            "component and focus effort there, rather than spreading it evenly."
        )

    return recommendations


def predict_final_marks_for_student(roll_no: str, semester: int) -> dict:
    """
    Predict final percentage for an existing student.

    Args:
        roll_no: The student.
        semester: Which semester's marks/attendance/assignments to compute
            features from.

    Returns:
        A dict: predicted_final_percentage (float), features_used (dict).

    Raises:
        ModelNotFoundError: if ml/model_training.ipynb has not been run.
        ValidationError: if the student has no marks yet this semester.
    """
    live_features = _compute_live_features(roll_no, semester)
    if live_features is None:
        raise ValidationError(
            f"No marks recorded yet for {roll_no} in semester {semester} -- cannot predict."
        )

    model, metadata = _get_final_marks_model()

    feature_row = live_features
    X = pd.DataFrame([feature_row])[metadata["features"]]

    predicted = float(model.predict(X)[0])

    logger.info("Final marks prediction for %s (semester %s): %.2f", roll_no, semester, predicted)

    return {
        "predicted_final_percentage": round(predicted, config.ROUND_DECIMALS),
        "features_used": feature_row,
    }


@st.cache_data(ttl=300)
def get_at_risk_count() -> int | None:
    """
    How many ACTIVE students the deployed classifier currently predicts as
    at-risk, evaluated at each student's own current semester
    (students.semester) -- feeds the "At-Risk Count" KPI card on the
    Dashboard page (see modules/analytics.py's render_dashboard_page()).

    WHY THIS IS CACHED FOR 5 MINUTES, NOT 60 SECONDS LIKE MOST OF
    modules/analytics.py's DASHBOARD QUERIES: computing this means running
    a real model prediction PER STUDENT, which itself means several
    database round-trips per student inside _compute_live_features()
    (their marks, attendance, and assignment records, plus their previous
    semester's marks for improvement_rate). Against the Turso cloud
    backend, every one of those round-trips carries real network latency
    -- for a class of any size, doing this on every 60-second cache expiry
    (or worse, every page load) would make the Dashboard noticeably slow.
    A student's at-risk status also does not realistically change minute
    to minute, so a slightly longer TTL costs nothing in practical
    accuracy. Contrast this with get_dashboard_summary()'s pass_rate/
    avg_attendance, which are single cheap aggregate SQL queries and so
    can afford to refresh much more often.

    Returns:
        The number of active students currently predicted at-risk, or
        None if the at-risk model has not been trained yet (see
        ModelNotFoundError) -- displayed as "N/A" rather than a
        misleading 0, exactly like get_dashboard_summary()'s pass_rate/
        avg_attendance already do when there is no data to compute from.
    """
    try:
        _get_at_risk_model()
    except ModelNotFoundError:
        return None

    at_risk_count = 0
    for student in students.list_students():
        try:
            result = predict_at_risk_for_student(student["roll_no"], student["semester"])
        except ValidationError:
            # No marks recorded yet for this student's current semester --
            # nothing to predict from, so they are simply excluded from
            # this count rather than counted as either at-risk or safe.
            continue
        if result["at_risk"]:
            at_risk_count += 1

    return at_risk_count


def get_at_risk_report() -> list[dict]:
    """
    Per-student at-risk predictions for every ACTIVE student, evaluated
    at each student's own current semester -- the full, row-level detail
    behind get_at_risk_count()'s single number. Lets Admin/Teacher see
    WHO is flagged, not just how many, without clicking through the
    single-student picker on render_at_risk_page() one roll_no at a time.

    NOT CACHED, UNLIKE get_at_risk_count(): that function returns one
    number shown on every Dashboard visit, so caching it protects
    against running this same per-student loop constantly. This function
    is only ever called on demand (a button click on the At-Risk
    Prediction page -- see render_at_risk_page() below), so a fresh
    cache buys little while risking a report that's a few minutes stale
    right when someone specifically asked to see current results.

    Returns:
        A list of dicts: roll_no, student_name, semester, at_risk (bool,
        or None if unavailable), risk_probability (float 0-1, or None),
        status ("At Risk" / "On Track" / "No marks yet"). Sorted with
        At Risk students first (highest risk_probability first), then
        On Track students, then students with no marks yet to predict
        from.
    """
    results = []
    for student in students.list_students():
        try:
            prediction = predict_at_risk_for_student(student["roll_no"], student["semester"])
            results.append({
                "roll_no": student["roll_no"],
                "student_name": student["name"],
                "semester": student["semester"],
                "at_risk": prediction["at_risk"],
                "risk_probability": prediction["risk_probability"],
                "status": "At Risk" if prediction["at_risk"] else "On Track",
            })
        except ValidationError:
            # No marks recorded yet for this student's current semester
            # -- included in the report (so nobody is silently missing
            # from it), but clearly labelled rather than guessed at.
            results.append({
                "roll_no": student["roll_no"],
                "student_name": student["name"],
                "semester": student["semester"],
                "at_risk": None,
                "risk_probability": None,
                "status": "No marks yet",
            })

    def _sort_key(row: dict) -> tuple:
        priority = 0 if row["at_risk"] is True else 1 if row["at_risk"] is False else 2
        return (priority, -(row["risk_probability"] or 0))

    results.sort(key=_sort_key)
    return results


def predict_segment_for_student(roll_no: str, semester: int) -> dict:
    """
    Assign an existing student to one of the named performance segments.

    Args:
        roll_no: The student.
        semester: Which semester's marks/attendance to compute features from.

    Returns:
        A dict: cluster_id (int), cluster_name (str), features_used (dict).

    Raises:
        ModelNotFoundError: if ml/model_training.ipynb has not been run.
        ValidationError: if the student has no marks yet this semester.
    """
    live_features = _compute_live_features(roll_no, semester)
    if live_features is None:
        raise ValidationError(
            f"No marks recorded yet for {roll_no} in semester {semester} -- cannot predict."
        )

    model, scaler, metadata = _get_segmentation_model()

    feature_row = {k: live_features[k] for k in metadata["features"]}
    X = pd.DataFrame([feature_row])[metadata["features"]]
    X_scaled = scaler.transform(X)

    cluster_id = int(model.predict(X_scaled)[0])
    cluster_name = metadata["cluster_names"][str(cluster_id)]

    logger.info("Segment prediction for %s (semester %s): cluster=%s (%s)", roll_no, semester, cluster_id, cluster_name)

    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "features_used": feature_row,
    }


# ---------------------------------------------------------------------------
# STREAMLIT PAGES
# ---------------------------------------------------------------------------

def _select_student_and_semester(key_prefix: str):
    """Shared UI: a student picker + semester number input, used by all
    three prediction pages below. Returns (roll_no, semester) or
    (None, None) if there are no students yet."""
    student_list = students.list_students()
    if not student_list:
        st.info("No students found. Add students first.")
        return None, None

    labels = {f"{s['roll_no']} - {s['name']}": s for s in student_list}
    picked_label = st.selectbox("Student", options=list(labels.keys()), key=f"{key_prefix}_student")
    picked_student = labels[picked_label]

    semester = st.number_input(
        "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
        value=picked_student["semester"], step=1, key=f"{key_prefix}_semester",
    )
    return picked_student["roll_no"], int(semester)


def render_at_risk_page() -> None:
    """Streamlit page: Task 1 -- at-risk classification for one student."""
    auth.require_role(*PREDICTION_ROLES)
    st.title("At-Risk Student Prediction")

    try:
        _, _, metadata = _get_at_risk_model()
    except ModelNotFoundError as error:
        st.warning(str(error))
        return

    st.caption(f"Deployed model: {metadata['algorithm']} "
               f"(test recall = {metadata['metrics']['recall']}, "
               f"cross-validated recall = {metadata['metrics']['cv_recall_mean']} "
               f"+/- {metadata['metrics']['cv_recall_std']})")

    roll_no, semester = _select_student_and_semester("at_risk")
    if roll_no is None:
        return

    if st.button("Predict At-Risk Status"):
        try:
            result = predict_at_risk_for_student(roll_no, semester)
        except ValidationError as error:
            st.error(str(error))
            return

        if result["at_risk"]:
            st.error(f"At risk of failing -- predicted probability: {result['risk_probability']:.1%}")

            recommendations = generate_recommendations(result["features_used"], semester)
            if recommendations:
                st.subheader(":material/lightbulb: How to improve")
                st.caption(
                    "Rule-based, built from this student's own feature values above -- "
                    "not a generic list. See modules/ml_predictions.py's "
                    "generate_recommendations() for exactly which threshold triggered each one."
                )
                for tip in recommendations:
                    st.warning(tip, icon=":material/priority_high:")
        else:
            st.success(f"On track to pass -- predicted risk probability: {result['risk_probability']:.1%}")

        with st.expander("Feature values used for this prediction"):
            st.json(result["features_used"])

    st.divider()
    st.subheader("Class-wide At-Risk Report")
    st.caption(
        "Runs the same model above for every active student at once, instead of "
        "checking one roll number at a time."
    )
    if st.button("Generate Full Class Report"):
        report = get_at_risk_report()
        if not report:
            st.info("No students found.")
        else:
            at_risk_total = sum(1 for row in report if row["at_risk"] is True)
            st.caption(f"{at_risk_total} of {len(report)} student(s) currently flagged at-risk.")
            display_rows = [
                {
                    "Roll No": row["roll_no"],
                    "Student": row["student_name"],
                    "Semester": row["semester"],
                    "Status": row["status"],
                    "Risk Probability": f"{row['risk_probability']:.1%}" if row["risk_probability"] is not None else "N/A",
                }
                for row in report
            ]
            render_data_table(display_rows, key_prefix="at_risk_report", filename_prefix="at_risk_report")


def render_final_marks_page() -> None:
    """Streamlit page: Task 2 -- final marks regression for one student."""
    auth.require_role(*PREDICTION_ROLES)
    st.title("Final Marks Prediction")

    try:
        _, metadata = _get_final_marks_model()
    except ModelNotFoundError as error:
        st.warning(str(error))
        return

    st.caption(f"Deployed model: {metadata['algorithm']} "
               f"(R2 = {metadata['metrics']['r2']}, MAE = {metadata['metrics']['mae']} percentage points)")

    roll_no, semester = _select_student_and_semester("final_marks")
    if roll_no is None:
        return

    if st.button("Predict Final Percentage"):
        try:
            result = predict_final_marks_for_student(roll_no, semester)
        except ValidationError as error:
            st.error(str(error))
            return

        st.metric("Predicted final percentage", f"{result['predicted_final_percentage']}%")
        st.caption(
            f"Typical prediction error on unseen students is about "
            f"+/-{metadata['metrics']['mae']} percentage points (MAE) -- treat this as an "
            "estimate, not an exact result."
        )

        with st.expander("Feature values used for this prediction"):
            st.json(result["features_used"])


def render_segmentation_page() -> None:
    """Streamlit page: Task 3 -- student segment assignment for one student."""
    auth.require_role(*PREDICTION_ROLES)
    st.title("Student Segmentation")

    try:
        _, _, metadata = _get_segmentation_model()
    except ModelNotFoundError as error:
        st.warning(str(error))
        return

    st.caption(f"{metadata['k']} segments discovered via K-Means "
               f"(silhouette score = {metadata['k_selection']['chosen_k_silhouette']})")

    roll_no, semester = _select_student_and_semester("segment")
    if roll_no is None:
        return

    if st.button("Assign Segment"):
        try:
            result = predict_segment_for_student(roll_no, semester)
        except ValidationError as error:
            st.error(str(error))
            return

        st.info(f"Segment: **{result['cluster_name']}**")

        with st.expander("Feature values used for this prediction"):
            st.json(result["features_used"])


def render_model_comparison_page() -> None:
    """
    Streamlit page: model comparison tables and deployment justification
    for all three ML tasks, read from ml/results/*.csv and each model's
    metadata JSON -- generated from the SAME numbers the notebook itself
    produced, not separately maintained prose that could drift out of
    sync if the notebook is ever retrained.
    """
    auth.require_role(*PREDICTION_ROLES)
    st.title("Model Comparison")

    st.subheader("Task 1 -- At-Risk Classification")
    task1_csv = RESULTS_DIR / "task1_classification_results.csv"
    if task1_csv.exists():
        st.dataframe(pd.read_csv(task1_csv), use_container_width=True, hide_index=True)
        try:
            _, _, meta1 = _get_at_risk_model()
            st.caption(
                f"Deployed: **{meta1['algorithm']}** -- chosen for the highest recall "
                f"({meta1['metrics']['recall']}) among candidates with reasonable precision/F1, "
                "since missing an at-risk student is worse than a false alarm."
            )
        except ModelNotFoundError:
            pass
    else:
        st.info("Run ml/model_training.ipynb to generate this comparison.")

    st.subheader("Task 2 -- Final Marks Regression")
    task2_csv = RESULTS_DIR / "task2_regression_results.csv"
    if task2_csv.exists():
        st.dataframe(pd.read_csv(task2_csv), use_container_width=True, hide_index=True)
        try:
            _, meta2 = _get_final_marks_model()
            st.caption(
                f"Deployed: **{meta2['algorithm']}** -- R2 = {meta2['metrics']['r2']}, "
                f"MAE = {meta2['metrics']['mae']} percentage points, well ahead of the mean-only baseline."
            )
        except ModelNotFoundError:
            pass
    else:
        st.info("Run ml/model_training.ipynb to generate this comparison.")

    st.subheader("Task 3 -- Student Segmentation")
    task3_csv = RESULTS_DIR / "task3_clustering_k_selection.csv"
    if task3_csv.exists():
        st.dataframe(pd.read_csv(task3_csv), use_container_width=True, hide_index=True)
        try:
            _, _, meta3 = _get_segmentation_model()
            st.caption(
                f"Deployed: k={meta3['k']} (silhouette = {meta3['k_selection']['chosen_k_silhouette']}) -- "
                "chosen for practical segment usefulness over the silhouette-optimal k=2; "
                "see ml/model_training.ipynb for the full reasoning."
            )
        except ModelNotFoundError:
            pass
    else:
        st.info("Run ml/model_training.ipynb to generate this comparison.")
