"""
modules/assignments.py
=======================
Assignment tracking: record and correct total-assigned/submitted counts
for a student in a subject, plus the Streamlit page for it.

WHY THIS FILE EXISTS: the ML models (Task 1 and Task 2 in
ml/model_training.ipynb) were trained on an "assignments submitted"
feature -- but earlier versions of this app had nowhere to actually
RECORD that, anywhere in the database. Predictions asked a Teacher to
type a number in by hand, every single time, which is exactly the kind
of "hardcoded" input the rest of this app was deliberately built to avoid
(every other ML feature is computed live from real records). This module
closes that gap: assignments are now tracked the same way marks and
attendance are, and modules/ml_predictions.py computes the ML feature
from these real rows instead of asking anyone to type a number in.

This file follows the exact same shape as modules/attendance.py (RBAC:
Admin or Teacher can write; no soft delete -- corrections go through an
audited UPDATE, never a DELETE; record_id is a natural composite key, not
the surrogate assignment_id, for the same reason explained in
modules/marks.py's _marks_record_id(); submitted <= total_assigned is a
rule about TWO FIELDS TOGETHER, validated as a combination on partial
updates, exactly like attendance's classes_attended <= classes_held). The
Streamlit page also uses the same live-capping technique as
modules/attendance.py's bulk entry form (plain widgets, not st.form, so
"Submitted"'s max_value can track "Total Assigned" as it's typed) --
see that file's render_attendance_page() for the full reasoning.
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, fetch_all, fetch_one
from modules import auth, students, subjects
from modules.audit import build_audit_entry
from modules.teacher_subjects import check_teacher_subject_access, list_subjects_for_marks_entry
from utils.bulk_import import render_bulk_import
from utils.exceptions import AuthorizationError, DuplicateRecordError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.table_view import render_data_table
from utils.validators import (
    validate_assignment_values,
    validate_roll_no,
    validate_semester,
    validate_subject_code,
)

logger = get_logger(__name__)

ASSIGNMENT_WRITE_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)

ASSIGNMENT_COLUMNS = (
    "assignment_id, roll_no, subject_code, total_assigned, submitted, "
    "semester, created_at, updated_at"
)


def _assignment_record_id(roll_no: str, subject_code: str, semester: int) -> str:
    """Build the audit_log record_id for one logical assignment entry --
    the natural composite key (roll_no, subject_code, semester), for the
    same reason explained in modules/marks.py's _marks_record_id()."""
    return f"{roll_no}:{subject_code}:{semester}"


def _compute_submission_rate(total_assigned: int, submitted: int) -> float | None:
    """
    Compute the submission rate as a percentage.

    Returns:
        The percentage, rounded to config.ROUND_DECIMALS places, or None
        if total_assigned is 0 -- 0/0 is mathematically undefined, and
        semantically means "no assignments have been given yet", which is
        different from "0% submitted".
    """
    if total_assigned == 0:
        return None
    return round((submitted / total_assigned) * 100, config.ROUND_DECIMALS)


def _with_rate(row: dict) -> dict:
    """Attach 'submission_rate' to an assignment row dict."""
    row["submission_rate"] = _compute_submission_rate(row["total_assigned"], row["submitted"])
    return row


# ---------------------------------------------------------------------------
# WRITE OPERATIONS (Admin or Teacher)
# ---------------------------------------------------------------------------

def record_assignment(
    roll_no: str,
    subject_code: str,
    total_assigned: int,
    submitted: int,
    semester: int,
    acting_user: dict,
) -> int:
    """
    Record a NEW assignment entry for a student/subject/semester.

    Args:
        roll_no: The student.
        subject_code: The subject.
        total_assigned: Total assignments given so far this semester.
        submitted: Assignments this student submitted.
        semester: The semester this record belongs to.
        acting_user: The logged-in user performing this action.

    Returns:
        The new assignment row's assignment_id.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor Teacher.
        ValidationError: if the values fail validation (negative, or
            submitted > total_assigned).
        RecordNotFoundError: if the student or subject does not exist (or
            is inactive).
        DuplicateRecordError: if an assignment entry already exists for
            this student/subject/semester -- use update_assignment() instead.
    """
    auth.check_permission(acting_user["role"], ASSIGNMENT_WRITE_ROLES)

    roll_no = validate_roll_no(roll_no)
    subject_code = validate_subject_code(subject_code)
    semester = validate_semester(semester)
    validate_assignment_values(total_assigned, submitted)

    check_teacher_subject_access(acting_user, subject_code)

    students.get_student(roll_no)
    subjects.get_subject(subject_code)

    if get_assignment_entry(roll_no, subject_code, semester) is not None:
        raise DuplicateRecordError(
            f"Assignment record already exists for {roll_no} in {subject_code} "
            f"(semester {semester}). Use update_assignment to correct it."
        )

    insert_statement = (
        "INSERT INTO assignments (roll_no, subject_code, total_assigned, submitted, semester) "
        "VALUES (?, ?, ?, ?, ?)",
        (roll_no, subject_code, total_assigned, submitted, semester),
    )
    new_value = {
        "roll_no": roll_no, "subject_code": subject_code,
        "total_assigned": total_assigned, "submitted": submitted, "semester": semester,
    }
    record_id = _assignment_record_id(roll_no, subject_code, semester)
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "assignments", record_id,
        old_value=None, new_value=new_value,
    )

    results = execute_transaction([insert_statement, audit_statement])
    assignment_id = results[0]
    logger.info(
        "Assignment record created for %s in %s (semester=%s) by user_id=%s.",
        roll_no, subject_code, semester, acting_user["user_id"],
    )
    return assignment_id


def update_assignment(
    assignment_id: int,
    acting_user: dict,
    total_assigned: int | None = None,
    submitted: int | None = None,
) -> None:
    """
    Correct an existing assignment entry. Only fields passed as something
    other than None are changed.

    WHY THE "EFFECTIVE" VALUES ARE VALIDATED TOGETHER: submitted <=
    total_assigned is a rule about the pair, not either field alone --
    exactly the same situation as modules/attendance.py's
    update_attendance(). If only submitted were being changed, we still
    must confirm the NEW submitted together with the EXISTING
    total_assigned forms a valid pair.

    Args:
        assignment_id: The assignment row to update.
        acting_user: The logged-in user performing this action.
        total_assigned, submitted: New values; leave as None to keep the
            existing value.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor Teacher.
        RecordNotFoundError: if assignment_id does not exist.
        ValidationError: if the resulting combination is invalid, or no
            fields at all were provided.
    """
    auth.check_permission(acting_user["role"], ASSIGNMENT_WRITE_ROLES)

    existing = get_assignment_by_id(assignment_id)
    check_teacher_subject_access(acting_user, existing["subject_code"])

    if total_assigned is None and submitted is None:
        raise ValidationError("No fields were provided to update.")

    effective_total = total_assigned if total_assigned is not None else existing["total_assigned"]
    effective_submitted = submitted if submitted is not None else existing["submitted"]
    validate_assignment_values(effective_total, effective_submitted)

    set_clauses = []
    params: list = []
    new_value = {}

    if total_assigned is not None:
        set_clauses.append("total_assigned = ?")
        params.append(total_assigned)
        new_value["total_assigned"] = total_assigned

    if submitted is not None:
        set_clauses.append("submitted = ?")
        params.append(submitted)
        new_value["submitted"] = submitted

    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    params.append(assignment_id)

    update_statement = (
        f"UPDATE assignments SET {', '.join(set_clauses)} WHERE assignment_id = ?",
        tuple(params),
    )
    old_value = {field: existing[field] for field in new_value}
    record_id = _assignment_record_id(existing["roll_no"], existing["subject_code"], existing["semester"])
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "assignments", record_id,
        old_value=old_value, new_value=new_value,
    )

    execute_transaction([update_statement, audit_statement])
    logger.info(
        "Assignment (assignment_id=%s) updated by user_id=%s. Fields changed: %s",
        assignment_id, acting_user["user_id"], list(new_value.keys()),
    )


# ---------------------------------------------------------------------------
# READ OPERATIONS (see modules/students.py's module docstring for why
# these are not individually role-gated)
# ---------------------------------------------------------------------------

def get_assignment_by_id(assignment_id: int) -> dict:
    """Fetch one assignment row by assignment_id. Raises RecordNotFoundError if missing."""
    row = fetch_one(
        f"SELECT {ASSIGNMENT_COLUMNS} FROM assignments WHERE assignment_id = ?", (assignment_id,)
    )
    if row is None:
        raise RecordNotFoundError(f"No assignment entry found with assignment_id {assignment_id}.")
    return dict(row)


def get_assignment_entry(roll_no: str, subject_code: str, semester: int) -> dict | None:
    """
    Look up an assignment row by its natural identity, used internally to
    decide whether record_assignment() or update_assignment() is
    appropriate.

    Returns:
        A dict if a matching row exists, otherwise None.
    """
    roll_no = validate_roll_no(roll_no)
    subject_code = validate_subject_code(subject_code)

    row = fetch_one(
        f"SELECT {ASSIGNMENT_COLUMNS} FROM assignments "
        "WHERE roll_no = ? AND subject_code = ? AND semester = ?",
        (roll_no, subject_code, semester),
    )
    return dict(row) if row is not None else None


def list_assignments_for_student(roll_no: str, semester: int | None = None) -> list[dict]:
    """
    Fetch every assignment row for one student, joined with each
    subject's name, with submission_rate computed live.

    Args:
        roll_no: The student to fetch assignment records for.
        semester: If given, only this semester.

    Returns:
        A list of dicts with the raw assignment columns plus subject_name
        and submission_rate.
    """
    roll_no = validate_roll_no(roll_no)

    query = (
        "SELECT a.assignment_id, a.roll_no, a.subject_code, s.name AS subject_name, "
        "a.total_assigned, a.submitted, a.semester, a.created_at, a.updated_at "
        "FROM assignments a JOIN subjects s ON a.subject_code = s.subject_code "
        "WHERE a.roll_no = ?"
    )
    params: list = [roll_no]

    if semester is not None:
        query += " AND a.semester = ?"
        params.append(semester)

    query += " ORDER BY a.semester, a.subject_code"

    return [_with_rate(dict(row)) for row in fetch_all(query, tuple(params))]


def list_assignments_for_subject(subject_code: str, semester: int | None = None) -> list[dict]:
    """
    Fetch every assignment row for one subject, joined with each
    student's name, with submission_rate computed live.

    Args:
        subject_code: The subject to fetch assignment records for.
        semester: If given, only this semester.

    Returns:
        A list of dicts with the raw assignment columns plus student_name
        and submission_rate.
    """
    subject_code = validate_subject_code(subject_code)

    query = (
        "SELECT a.assignment_id, a.roll_no, st.name AS student_name, a.subject_code, "
        "a.total_assigned, a.submitted, a.semester, a.created_at, a.updated_at "
        "FROM assignments a "
        "JOIN students st ON a.roll_no = st.roll_no "
        "WHERE a.subject_code = ?"
    )
    params: list = [subject_code]

    if semester is not None:
        query += " AND a.semester = ?"
        params.append(semester)

    query += " ORDER BY a.roll_no"

    return [_with_rate(dict(row)) for row in fetch_all(query, tuple(params))]


# ---------------------------------------------------------------------------
# BULK IMPORT (see utils/bulk_import.py's module docstring, and
# modules/marks.py's _validate_bulk_marks_row() for the identical
# reasoning applied there, including why teacher-subject access is
# checked in the validator itself)
# ---------------------------------------------------------------------------

def _validate_bulk_assignment_row(row: dict, acting_user: dict) -> None:
    """
    Validate one row of a bulk assignment import, run during the PREVIEW
    step. Reuses validate_assignment_values() -- the same "submitted
    cannot exceed total_assigned" rule record_assignment()/
    update_assignment() themselves enforce.

    Args:
        row: One row from the uploaded file.
        acting_user: The logged-in Admin or Teacher running the import.

    Raises:
        ValidationError: if any field is missing, malformed, or
            submitted exceeds total_assigned.
        RecordNotFoundError: if the student or subject does not exist
            (or is inactive).
        AuthorizationError: if acting_user is a Teacher not assigned to
            this subject.
    """
    roll_no = validate_roll_no(row.get("roll_no", ""))
    subject_code = validate_subject_code(row.get("subject_code", ""))

    try:
        semester = int(row.get("semester", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"Semester must be a whole number, got '{row.get('semester')}'.")
    validate_semester(semester)

    check_teacher_subject_access(acting_user, subject_code)

    students.get_student(roll_no)
    subjects.get_subject(subject_code)

    try:
        total_assigned = int(row.get("total_assigned", ""))
        submitted = int(row.get("submitted", ""))
    except (TypeError, ValueError):
        raise ValidationError("total_assigned/submitted must both be whole numbers.")

    validate_assignment_values(total_assigned, submitted)


def _commit_bulk_assignment_row(row: dict, acting_user: dict) -> None:
    """Insert or update one bulk-imported assignment row -- an UPSERT,
    the same "insert if new, else update" pattern
    modules/marks.py's _commit_bulk_marks_row() uses, for the same
    reason (a re-imported, corrected spreadsheet should fix an existing
    entry, not fail with DuplicateRecordError)."""
    roll_no = row["roll_no"]
    subject_code = row["subject_code"]
    semester = int(row["semester"])
    total_assigned = int(row["total_assigned"])
    submitted = int(row["submitted"])

    existing = get_assignment_entry(roll_no, subject_code, semester)
    if existing is None:
        record_assignment(roll_no, subject_code, total_assigned, submitted, semester, acting_user)
    else:
        update_assignment(
            existing["assignment_id"], acting_user,
            total_assigned=total_assigned, submitted=submitted,
        )


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_assignments_page() -> None:
    """
    Streamlit page: bulk assignment entry for one subject/semester at a
    time, showing every active student with editable total-assigned/
    submitted fields, followed by a live-computed submission-rate table.

    See modules/attendance.py's render_attendance_page() for why these
    widgets are deliberately NOT wrapped in st.form() -- the same live-
    capping technique is used here so "Submitted" can never be typed
    higher than "Total Assigned" in the first place.
    """
    user = auth.require_role(*ASSIGNMENT_WRITE_ROLES)

    st.title("Assignments")
    st.caption(
        "Tracking assignment submissions here is what lets the ML predictions "
        "compute a real engagement score automatically, instead of asking for "
        "a manually-typed number every time."
    )

    with st.expander("Bulk Import Assignments (CSV/Excel)"):
        render_bulk_import(
            key_prefix="assignments_import",
            required_columns=("roll_no", "subject_code", "semester", "total_assigned", "submitted"),
            key_columns=("roll_no", "subject_code", "semester"),
            validate_row=lambda row: _validate_bulk_assignment_row(row, user),
            commit_row=lambda row: _commit_bulk_assignment_row(row, user),
        )

    subject_list = list_subjects_for_marks_entry(user)
    if not subject_list:
        if user["role"] == config.ROLE_TEACHER:
            st.warning("You are not assigned to any subjects yet. Contact an administrator.")
        else:
            st.warning("No subjects have been configured yet. Add a subject first.")
        return

    subject_options = {f"{s['subject_code']} - {s['name']}": s for s in subject_list}
    subject_label = st.selectbox("Subject", options=list(subject_options.keys()))
    selected_subject = subject_options[subject_label]

    selected_semester = st.number_input(
        "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
        value=selected_subject["semester"], step=1,
    )
    semester_value = int(selected_semester)

    class_students = students.list_students(semester=semester_value)
    if not class_students:
        st.info(f"No active students found in semester {semester_value}.")
        return

    entered_values = {}
    for student in class_students:
        existing_entry = get_assignment_entry(
            student["roll_no"], selected_subject["subject_code"], semester_value
        )
        st.markdown(f"**{student['roll_no']} — {student['name']}**")
        total_col, submitted_col = st.columns(2)
        with total_col:
            total_input = st.number_input(
                "Total Assigned", min_value=0,
                value=existing_entry["total_assigned"] if existing_entry else 0,
                key=f"total_{student['roll_no']}",
            )
        with submitted_col:
            # max_value tracks the CURRENT total_input value from above,
            # live -- see the module docstring for why this makes an
            # invalid pair impossible to enter, not just rejected after
            # "Save All" is clicked.
            submitted_input = st.number_input(
                "Submitted", min_value=0, max_value=int(total_input),
                value=min(existing_entry["submitted"], int(total_input)) if existing_entry else 0,
                key=f"submitted_{student['roll_no']}",
            )
        entered_values[student["roll_no"]] = {
            "total_assigned": int(total_input),
            "submitted": int(submitted_input),
            "existing": existing_entry,
        }

    save_submitted = st.button("Save All Assignments", type="primary")

    if save_submitted:
        saved_count = 0
        skipped_count = 0
        errors = []

        for roll_no, data in entered_values.items():
            existing_entry = data["existing"]
            new_pair = (data["total_assigned"], data["submitted"])

            try:
                if existing_entry is None:
                    record_assignment(
                        roll_no, selected_subject["subject_code"],
                        data["total_assigned"], data["submitted"], semester_value, user,
                    )
                    saved_count += 1
                elif (existing_entry["total_assigned"], existing_entry["submitted"]) != new_pair:
                    update_assignment(
                        existing_entry["assignment_id"], user,
                        total_assigned=data["total_assigned"], submitted=data["submitted"],
                    )
                    saved_count += 1
                else:
                    skipped_count += 1
            except (ValidationError, DuplicateRecordError, RecordNotFoundError, AuthorizationError) as error:
                errors.append(f"{roll_no}: {error}")

        if saved_count:
            st.success(f"Saved assignment records for {saved_count} student(s).")
        if skipped_count:
            st.caption(f"{skipped_count} student(s) unchanged -- nothing written.")
        for error_message in errors:
            st.error(error_message)
        if saved_count:
            st.rerun()

    st.subheader("Current Assignments")
    current_assignments = list_assignments_for_subject(
        selected_subject["subject_code"], semester=semester_value
    )

    if not current_assignments:
        st.info("No assignment records yet for this selection.")
        return

    display_rows = [
        {
            "Roll No": entry["roll_no"],
            "Student": entry["student_name"],
            "Total Assigned": entry["total_assigned"],
            "Submitted": entry["submitted"],
            "Submission Rate": f"{entry['submission_rate']}%" if entry["submission_rate"] is not None else "N/A",
        }
        for entry in current_assignments
    ]
    render_data_table(display_rows, key_prefix="assignments_table", filename_prefix="assignments")
