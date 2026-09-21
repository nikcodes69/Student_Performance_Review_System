"""
modules/attendance.py
======================
Attendance tracking: record and correct classes-held/classes-attended
counts for a student in a subject, plus the Streamlit page for it.

This file follows the same shape as modules/marks.py (RBAC: Admin or
Teacher can write, gated additionally by modules/teacher_subjects.py's
check_teacher_subject_access() so a Teacher can only record attendance
for a subject they are assigned to; no soft delete -- corrections go
through an audited UPDATE, never a DELETE; record_id is a natural
composite key, not the surrogate att_id, for the same reason marks.py's
record_id is not mark_id -- att_id is only assigned after INSERT, too
late for build_audit_entry()). One thing is different from marks.py, and
mirrors modules/subjects.py instead: classes_attended <= classes_held is
a rule about TWO FIELDS TOGETHER (like subjects' max_internal/external/
practical sum), so update_attendance() validates the EFFECTIVE
combination after a partial update, not just whichever single field
changed -- see the comment inside update_attendance() for the same
reasoning applied there.

ATTENDANCE PERCENTAGE IS NEVER STORED, for the same 3NF reason marks
percentage is never stored (see modules/marks.py) -- it is entirely
derivable from classes_held and classes_attended, so it is computed fresh
by _compute_attendance_percentage() every time a row is read, never
written to a column.
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
    validate_attendance_values,
    validate_roll_no,
    validate_semester,
    validate_subject_code,
)

logger = get_logger(__name__)

ATTENDANCE_WRITE_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)

ATTENDANCE_COLUMNS = (
    "att_id, roll_no, subject_code, classes_held, classes_attended, "
    "semester, created_at, updated_at"
)


def _attendance_record_id(roll_no: str, subject_code: str, semester: int) -> str:
    """Build the audit_log record_id for one logical attendance entry --
    the natural composite key (roll_no, subject_code, semester), for the
    same reason explained in modules/marks.py's _marks_record_id()."""
    return f"{roll_no}:{subject_code}:{semester}"


def _compute_attendance_percentage(classes_held: int, classes_attended: int) -> float | None:
    """
    Compute attendance percentage.

    Returns:
        The percentage, rounded to config.ROUND_DECIMALS places, or None
        if classes_held is 0 -- 0/0 is mathematically undefined, and
        semantically means "no classes have been held yet to measure
        attendance against", which is different from "0% attendance".
    """
    if classes_held == 0:
        return None
    return round((classes_attended / classes_held) * 100, config.ROUND_DECIMALS)


def _with_percentage(row: dict) -> dict:
    """Attach 'attendance_percentage' and 'shortage' (True if below
    config.ATTENDANCE_SHORTAGE_THRESHOLD) to an attendance row dict."""
    percentage = _compute_attendance_percentage(row["classes_held"], row["classes_attended"])
    row["attendance_percentage"] = percentage
    row["shortage"] = percentage is not None and percentage < config.ATTENDANCE_SHORTAGE_THRESHOLD
    return row


# ---------------------------------------------------------------------------
# WRITE OPERATIONS (Admin or Teacher)
# ---------------------------------------------------------------------------

def record_attendance(
    roll_no: str,
    subject_code: str,
    classes_held: int,
    classes_attended: int,
    semester: int,
    acting_user: dict,
) -> int:
    """
    Record a NEW attendance entry for a student/subject/semester.

    Args:
        roll_no: The student.
        subject_code: The subject.
        classes_held: Total classes held so far this semester.
        classes_attended: Classes this student attended.
        semester: The semester this attendance record belongs to.
        acting_user: The logged-in user performing this action.

    Returns:
        The new attendance row's att_id.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor Teacher.
        ValidationError: if the values fail validation (negative, or
            attended > held).
        RecordNotFoundError: if the student or subject does not exist (or
            is inactive).
        DuplicateRecordError: if an attendance entry already exists for
            this student/subject/semester -- use update_attendance() instead.
    """
    auth.check_permission(acting_user["role"], ATTENDANCE_WRITE_ROLES)

    roll_no = validate_roll_no(roll_no)
    subject_code = validate_subject_code(subject_code)
    semester = validate_semester(semester)
    validate_attendance_values(classes_held, classes_attended)

    check_teacher_subject_access(acting_user, subject_code)

    students.get_student(roll_no)
    subjects.get_subject(subject_code)

    if get_attendance_entry(roll_no, subject_code, semester) is not None:
        raise DuplicateRecordError(
            f"Attendance already recorded for {roll_no} in {subject_code} "
            f"(semester {semester}). Use update_attendance to correct it."
        )

    insert_statement = (
        "INSERT INTO attendance (roll_no, subject_code, classes_held, classes_attended, semester) "
        "VALUES (?, ?, ?, ?, ?)",
        (roll_no, subject_code, classes_held, classes_attended, semester),
    )
    new_value = {
        "roll_no": roll_no, "subject_code": subject_code,
        "classes_held": classes_held, "classes_attended": classes_attended, "semester": semester,
    }
    record_id = _attendance_record_id(roll_no, subject_code, semester)
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "attendance", record_id,
        old_value=None, new_value=new_value,
    )

    results = execute_transaction([insert_statement, audit_statement])
    att_id = results[0]
    logger.info(
        "Attendance recorded for %s in %s (semester=%s) by user_id=%s.",
        roll_no, subject_code, semester, acting_user["user_id"],
    )
    return att_id


def update_attendance(
    att_id: int,
    acting_user: dict,
    classes_held: int | None = None,
    classes_attended: int | None = None,
) -> None:
    """
    Correct an existing attendance entry. Only fields passed as something
    other than None are changed.

    WHY THE "EFFECTIVE" VALUES ARE VALIDATED TOGETHER: classes_attended
    <= classes_held is a rule about the pair, not either field alone --
    exactly the same situation as modules/subjects.py's max_internal/
    max_external/max_practical needing to be validated as a combination
    during a partial update (see that file's update_subject() for the
    fuller explanation). If only classes_attended were being changed, we
    still must confirm the NEW classes_attended together with the
    EXISTING classes_held forms a valid pair -- not just that the new
    number on its own is non-negative.

    Args:
        att_id: The attendance row to update.
        acting_user: The logged-in user performing this action.
        classes_held, classes_attended: New values; leave as None to keep
            the existing value.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor Teacher.
        RecordNotFoundError: if att_id does not exist.
        ValidationError: if the resulting combination is invalid, or no
            fields at all were provided.
    """
    auth.check_permission(acting_user["role"], ATTENDANCE_WRITE_ROLES)

    existing = get_attendance_by_id(att_id)
    check_teacher_subject_access(acting_user, existing["subject_code"])

    if classes_held is None and classes_attended is None:
        raise ValidationError("No fields were provided to update.")

    effective_held = classes_held if classes_held is not None else existing["classes_held"]
    effective_attended = classes_attended if classes_attended is not None else existing["classes_attended"]
    validate_attendance_values(effective_held, effective_attended)

    set_clauses = []
    params: list = []
    new_value = {}

    if classes_held is not None:
        set_clauses.append("classes_held = ?")
        params.append(classes_held)
        new_value["classes_held"] = classes_held

    if classes_attended is not None:
        set_clauses.append("classes_attended = ?")
        params.append(classes_attended)
        new_value["classes_attended"] = classes_attended

    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    params.append(att_id)

    update_statement = (
        f"UPDATE attendance SET {', '.join(set_clauses)} WHERE att_id = ?",
        tuple(params),
    )
    old_value = {field: existing[field] for field in new_value}
    record_id = _attendance_record_id(existing["roll_no"], existing["subject_code"], existing["semester"])
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "attendance", record_id,
        old_value=old_value, new_value=new_value,
    )

    execute_transaction([update_statement, audit_statement])
    logger.info(
        "Attendance (att_id=%s) updated by user_id=%s. Fields changed: %s",
        att_id, acting_user["user_id"], list(new_value.keys()),
    )


# ---------------------------------------------------------------------------
# READ OPERATIONS (see modules/students.py's module docstring for why
# these are not individually role-gated)
# ---------------------------------------------------------------------------

def get_attendance_by_id(att_id: int) -> dict:
    """Fetch one attendance row by att_id. Raises RecordNotFoundError if missing."""
    row = fetch_one(f"SELECT {ATTENDANCE_COLUMNS} FROM attendance WHERE att_id = ?", (att_id,))
    if row is None:
        raise RecordNotFoundError(f"No attendance entry found with att_id {att_id}.")
    return dict(row)


def get_attendance_entry(roll_no: str, subject_code: str, semester: int) -> dict | None:
    """
    Look up an attendance row by its natural identity, used internally to
    decide whether record_attendance() or update_attendance() is
    appropriate.

    Returns:
        A dict if a matching row exists, otherwise None.
    """
    roll_no = validate_roll_no(roll_no)
    subject_code = validate_subject_code(subject_code)

    row = fetch_one(
        f"SELECT {ATTENDANCE_COLUMNS} FROM attendance "
        "WHERE roll_no = ? AND subject_code = ? AND semester = ?",
        (roll_no, subject_code, semester),
    )
    return dict(row) if row is not None else None


def list_attendance_for_student(roll_no: str, semester: int | None = None) -> list[dict]:
    """
    Fetch every attendance row for one student, joined with each
    subject's name, with attendance_percentage/shortage computed live.

    Args:
        roll_no: The student to fetch attendance for.
        semester: If given, only this semester.

    Returns:
        A list of dicts with the raw attendance columns plus subject_name,
        attendance_percentage, and shortage.
    """
    roll_no = validate_roll_no(roll_no)

    query = (
        "SELECT a.att_id, a.roll_no, a.subject_code, s.name AS subject_name, "
        "a.classes_held, a.classes_attended, a.semester, a.created_at, a.updated_at "
        "FROM attendance a JOIN subjects s ON a.subject_code = s.subject_code "
        "WHERE a.roll_no = ?"
    )
    params: list = [roll_no]

    if semester is not None:
        query += " AND a.semester = ?"
        params.append(semester)

    query += " ORDER BY a.semester, a.subject_code"

    return [_with_percentage(dict(row)) for row in fetch_all(query, tuple(params))]


def list_attendance_for_subject(subject_code: str, semester: int | None = None) -> list[dict]:
    """
    Fetch every attendance row for one subject, joined with each
    student's name, with attendance_percentage/shortage computed live.

    Args:
        subject_code: The subject to fetch attendance for.
        semester: If given, only this semester.

    Returns:
        A list of dicts with the raw attendance columns plus student_name,
        attendance_percentage, and shortage.
    """
    subject_code = validate_subject_code(subject_code)

    query = (
        "SELECT a.att_id, a.roll_no, st.name AS student_name, a.subject_code, "
        "a.classes_held, a.classes_attended, a.semester, a.created_at, a.updated_at "
        "FROM attendance a "
        "JOIN students st ON a.roll_no = st.roll_no "
        "WHERE a.subject_code = ?"
    )
    params: list = [subject_code]

    if semester is not None:
        query += " AND a.semester = ?"
        params.append(semester)

    query += " ORDER BY a.roll_no"

    return [_with_percentage(dict(row)) for row in fetch_all(query, tuple(params))]


# ---------------------------------------------------------------------------
# BULK IMPORT (see utils/bulk_import.py's module docstring, and
# modules/marks.py's _validate_bulk_marks_row() for the identical
# reasoning applied there, including why teacher-subject access is
# checked in the validator itself)
# ---------------------------------------------------------------------------

def _validate_bulk_attendance_row(row: dict, acting_user: dict) -> None:
    """
    Validate one row of a bulk attendance import, run during the PREVIEW
    step. Reuses validate_attendance_values() -- the same "classes
    attended cannot exceed classes held" rule record_attendance()/
    update_attendance() themselves enforce.

    Args:
        row: One row from the uploaded file.
        acting_user: The logged-in Admin or Teacher running the import.

    Raises:
        ValidationError: if any field is missing, malformed, or classes
            attended exceeds classes held.
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
        classes_held = int(row.get("classes_held", ""))
        classes_attended = int(row.get("classes_attended", ""))
    except (TypeError, ValueError):
        raise ValidationError("classes_held/classes_attended must both be whole numbers.")

    validate_attendance_values(classes_held, classes_attended)


def _commit_bulk_attendance_row(row: dict, acting_user: dict) -> None:
    """Insert or update one bulk-imported attendance row -- an UPSERT,
    the same "insert if new, else update" pattern
    modules/marks.py's _commit_bulk_marks_row() uses, for the same
    reason (a re-imported, corrected spreadsheet should fix an existing
    entry, not fail with DuplicateRecordError)."""
    roll_no = row["roll_no"]
    subject_code = row["subject_code"]
    semester = int(row["semester"])
    classes_held = int(row["classes_held"])
    classes_attended = int(row["classes_attended"])

    existing = get_attendance_entry(roll_no, subject_code, semester)
    if existing is None:
        record_attendance(roll_no, subject_code, classes_held, classes_attended, semester, acting_user)
    else:
        update_attendance(
            existing["att_id"], acting_user,
            classes_held=classes_held, classes_attended=classes_attended,
        )


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_attendance_page() -> None:
    """
    Streamlit page: bulk attendance entry for one subject/semester at a
    time, showing every active student with editable classes-held/
    classes-attended fields, followed by a live-computed attendance table.
    """
    user = auth.require_role(*ATTENDANCE_WRITE_ROLES)

    st.title("Attendance Tracking")

    with st.expander("Bulk Import Attendance (CSV/Excel)"):
        render_bulk_import(
            key_prefix="attendance_import",
            required_columns=("roll_no", "subject_code", "semester", "classes_held", "classes_attended"),
            key_columns=("roll_no", "subject_code", "semester"),
            validate_row=lambda row: _validate_bulk_attendance_row(row, user),
            commit_row=lambda row: _commit_bulk_attendance_row(row, user),
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

    # THESE ARE DELIBERATELY *NOT* WRAPPED IN st.form(). A form only
    # reruns the script when its submit button is clicked, not on every
    # keystroke -- which meant "Classes Attended" could not dynamically
    # cap itself to whatever "Classes Held" currently holds (this was a
    # known, explicitly documented limitation of the earlier version of
    # this page). Using plain widgets instead means every change reruns
    # the script immediately, so max_value=held_input below is always the
    # CURRENT value the user just typed for that same student -- making it
    # physically impossible for the "Classes Attended" widget to accept a
    # number greater than "Classes Held", rather than accepting it and
    # only rejecting it after "Save All" is clicked. The trade-off is more
    # reruns while filling the form in, which our caching (see
    # modules/students.py's list_students()) keeps cheap.
    entered_values = {}
    for student in class_students:
        existing_entry = get_attendance_entry(
            student["roll_no"], selected_subject["subject_code"], semester_value
        )
        st.markdown(f"**{student['roll_no']} — {student['name']}**")
        held_col, attended_col = st.columns(2)
        with held_col:
            held_input = st.number_input(
                "Classes Held", min_value=0,
                value=existing_entry["classes_held"] if existing_entry else 0,
                key=f"held_{student['roll_no']}",
            )
        with attended_col:
            # max_value is set to the CURRENT held_input value from above
            # -- this is the live constraint described in the comment
            # above. validate_attendance_values() in update_attendance()/
            # record_attendance() still enforces the same rule server-side
            # too (the UI constraint is a convenience, never the only gate
            # -- the same defense-in-depth principle used throughout this
            # project).
            attended_input = st.number_input(
                "Classes Attended", min_value=0, max_value=int(held_input),
                value=min(existing_entry["classes_attended"], int(held_input)) if existing_entry else 0,
                key=f"attended_{student['roll_no']}",
            )
        entered_values[student["roll_no"]] = {
            "classes_held": int(held_input),
            "classes_attended": int(attended_input),
            "existing": existing_entry,
        }

    save_submitted = st.button("Save All Attendance", type="primary")

    if save_submitted:
        saved_count = 0
        skipped_count = 0
        errors = []

        for roll_no, data in entered_values.items():
            existing_entry = data["existing"]
            new_pair = (data["classes_held"], data["classes_attended"])

            try:
                if existing_entry is None:
                    record_attendance(
                        roll_no, selected_subject["subject_code"],
                        data["classes_held"], data["classes_attended"], semester_value, user,
                    )
                    saved_count += 1
                elif (existing_entry["classes_held"], existing_entry["classes_attended"]) != new_pair:
                    update_attendance(
                        existing_entry["att_id"], user,
                        classes_held=data["classes_held"], classes_attended=data["classes_attended"],
                    )
                    saved_count += 1
                else:
                    skipped_count += 1
            except (ValidationError, DuplicateRecordError, RecordNotFoundError, AuthorizationError) as error:
                errors.append(f"{roll_no}: {error}")

        if saved_count:
            # st.toast(), not st.success() -- see app.py's
            # render_role_login_form() for why, wherever a message is
            # immediately followed by st.rerun().
            st.toast(f"Saved attendance for {saved_count} student(s).", icon=":material/check_circle:")
        if skipped_count:
            st.caption(f"{skipped_count} student(s) unchanged -- nothing written.")
        for error_message in errors:
            st.error(error_message)
        if saved_count:
            st.rerun()

    st.subheader("Current Attendance")
    current_attendance = list_attendance_for_subject(selected_subject["subject_code"], semester=semester_value)

    if not current_attendance:
        st.info("No attendance recorded yet for this selection.")
        return

    display_rows = [
        {
            "Roll No": entry["roll_no"],
            "Student": entry["student_name"],
            "Classes Held": entry["classes_held"],
            "Classes Attended": entry["classes_attended"],
            "Attendance %": entry["attendance_percentage"],
            "Shortage": "Yes" if entry["shortage"] else "No",
        }
        for entry in current_attendance
    ]
    render_data_table(display_rows, key_prefix="attendance_table", filename_prefix="attendance")
