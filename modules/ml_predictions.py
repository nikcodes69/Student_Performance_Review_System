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
A REAL GAP THIS FILE HAS TO WORK AROUND: assignments_submitted
======================================================================
Three of Task 1's five features -- internal marks, attendance, previous
SGPA, backlog count -- can be computed live from this project's own
database (students, subjects, marks, attendance tables). The fifth,
assignments_submitted, CANNOT: nothing in database/db_setup.py's schema
tracks assignment submissions anywhere (this was flagged already, back in
ml/generate_data.py's module docstring, as a feature invented for the ML
task that a real system might track in a separate LMS). Rather than
silently inventing a number, the prediction pages below ask the user to
type it in directly, with a visible caption explaining why -- an honest
gap is better than a fabricated one.

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
from utils.exceptions import ModelNotFoundError, ValidationError
from utils.logger import get_logger

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


def _compute_live_features(roll_no: str, semester: int) -> dict | None:
    """
    Compute every ML feature (except assignments_submitted -- see module
    docstring) for one student in one semester, live from the database.

    Returns:
        A dict with keys internal_pct, attendance_pct, practical_pct,
        average_marks, consistency, improvement_rate, previous_sgpa,
        backlog_count -- or None if the student has no marks recorded yet
        for this semester (nothing to compute a prediction from).
    """
    semester_marks = list_marks_for_student(roll_no, semester=semester)
    if not semester_marks:
        return None

    # Average percentage per ASSESSMENT COMPONENT (internal/external/
    # practical), across every subject this semester -- matching
    # ml/generate_data.py's aggregate internal_pct/external_pct/
    # practical_pct exactly. A subject with max_internal == 0 (no internal
    # component at all) is excluded from that component's average rather
    # than causing a division by zero.
    internal_components = [
        row["internal"] / row["max_internal"] * 100
        for row in semester_marks if row["max_internal"] > 0
    ]
    external_components = [
        row["external"] / row["max_external"] * 100
        for row in semester_marks if row["max_external"] > 0
    ]
    practical_components = [
        row["practical"] / row["max_practical"] * 100
        for row in semester_marks if row["max_practical"] > 0
    ]

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
    }


# ---------------------------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------------------------

def predict_at_risk_for_student(roll_no: str, semester: int, assignments_submitted: int) -> dict:
    """
    Predict at-risk status for an existing student.

    Args:
        roll_no: The student.
        semester: Which semester's marks/attendance to compute features from.
        assignments_submitted: Manually supplied (see module docstring) --
            count out of 10.

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

    feature_row = {**live_features, "assignments_submitted": assignments_submitted}
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


def predict_final_marks_for_student(roll_no: str, semester: int, assignments_submitted: int) -> dict:
    """
    Predict final percentage for an existing student.

    Args:
        roll_no: The student.
        semester: Which semester's marks/attendance to compute features from.
        assignments_submitted: Manually supplied -- count out of 10.

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

    feature_row = {**live_features, "assignments_submitted": assignments_submitted}
    X = pd.DataFrame([feature_row])[metadata["features"]]

    predicted = float(model.predict(X)[0])

    logger.info("Final marks prediction for %s (semester %s): %.2f", roll_no, semester, predicted)

    return {
        "predicted_final_percentage": round(predicted, config.ROUND_DECIMALS),
        "features_used": feature_row,
    }


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


def _assignments_input(key_prefix: str) -> int:
    """Shared UI: the one manually-entered feature. See module docstring
    for why this cannot be computed automatically."""
    st.caption(
        "This system does not track assignment submissions anywhere in its database "
        "(there is no assignments table) -- enter this student's own record directly."
    )
    return int(st.number_input(
        "Assignments submitted (out of 10)", min_value=0, max_value=10, value=5, step=1,
        key=f"{key_prefix}_assignments",
    ))


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

    assignments_submitted = _assignments_input("at_risk")

    if st.button("Predict At-Risk Status"):
        try:
            result = predict_at_risk_for_student(roll_no, semester, assignments_submitted)
        except ValidationError as error:
            st.error(str(error))
            return

        if result["at_risk"]:
            st.error(f"At risk of failing -- predicted probability: {result['risk_probability']:.1%}")
        else:
            st.success(f"On track to pass -- predicted risk probability: {result['risk_probability']:.1%}")

        with st.expander("Feature values used for this prediction"):
            st.json(result["features_used"])


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

    assignments_submitted = _assignments_input("final_marks")

    if st.button("Predict Final Percentage"):
        try:
            result = predict_final_marks_for_student(roll_no, semester, assignments_submitted)
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
