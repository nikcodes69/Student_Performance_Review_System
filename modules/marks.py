"""
modules/marks.py
=================
Marks entry: record and correct internal/external/practical marks for a
student in a subject, plus the Streamlit page for it.

WHO CAN WRITE MARKS: unlike students.py/subjects.py (Admin only), marks
entry allows BOTH Admin and Teacher -- entering marks is a Teacher's
day-to-day job in real life, not an administrative action. A Teacher may
only enter marks for a subject they are assigned to (see
modules/teacher_subjects.py) -- enter_marks() and update_marks() both
call teacher_subjects.check_teacher_subject_access() right after their
own check_permission() call, and render_marks_page() below builds its
subject picker from teacher_subjects.list_subjects_for_marks_entry()
rather than subjects.list_subjects(), so a Teacher never even sees a
subject they are not assigned to. Admin is unaffected by this check --
see that module's docstring for the full two-layer reasoning.

NO SOFT DELETE HERE: unlike students/subjects, marks has no is_active
column and this file exposes no delete-style function at all -- see
database/db_setup.py's comment on the marks table. A wrong mark is fixed
with update_marks() (an audited UPDATE), never removed.

WHY record_id FOR MARKS IS NOT mark_id: audit_log.record_id normally
holds the row's own primary key (see modules/students.py using roll_no,
modules/subjects.py using subject_code). marks.mark_id is different: it
is an AUTOINCREMENT surrogate key that SQLite only assigns once the
INSERT actually runs -- so it does not exist yet at the moment
build_audit_entry() needs to be called (which happens BEFORE the
statements are sent to execute_transaction(), see modules/audit.py).
Instead, this file identifies a marks entry by the same four columns its
own UNIQUE constraint is built from: roll_no, subject_code, semester,
exam_type (see _marks_record_id() below). Every INSERT and every later
UPDATE for the same logical mark entry then shares that one record_id, so
modules/audit.py's audit log viewer can show its complete history
together -- and it reads far more meaningfully in that viewer than a bare
integer would ("BCA001:CACS201:3:regular" vs. "17").

PERCENTAGE/GRADE ARE NEVER STORED: list_marks_for_student() and
list_marks_for_subject() below compute percentage/grade/pass-fail on the
fly, for every row, by calling modules.grades.evaluate_subject_marks()
with that row's OWN subject's max_internal/external/practical. Storing a
"percentage" column on the marks table itself would make it a value that
depends entirely on other columns in two different tables (marks and
subjects) -- exactly the kind of redundant, derivable data 3NF asks us to
avoid (see database/db_setup.py's normalisation notes). Computing it fresh
every time also means it is IMPOSSIBLE for a stored percentage to go
stale after a mark correction.
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, fetch_all, fetch_one
from modules import auth, students, subjects
from modules.audit import build_audit_entry
from modules.grades import calculate_sgpa, evaluate_subject_marks
from modules.teacher_subjects import check_teacher_subject_access, list_subjects_for_marks_entry
from utils.exceptions import AuthorizationError, DuplicateRecordError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.table_view import render_data_table
from utils.validators import (
    validate_exam_type,
    validate_mark_value,
    validate_roll_no,
    validate_semester,
    validate_subject_code,
)

logger = get_logger(__name__)

MARKS_WRITE_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)

MARKS_COLUMNS = (
    "mark_id, roll_no, subject_code, internal, external, practical, "
    "semester, exam_type, created_at, updated_at"
)


def _marks_record_id(roll_no: str, subject_code: str, semester: int, exam_type: str) -> str:
    """Build the audit_log record_id for one logical marks entry. See the
    module docstring's "WHY record_id FOR MARKS IS NOT mark_id" section."""
    return f"{roll_no}:{subject_code}:{semester}:{exam_type}"


# ---------------------------------------------------------------------------
# WRITE OPERATIONS (Admin or Teacher)
# ---------------------------------------------------------------------------

def enter_marks(
    roll_no: str,
    subject_code: str,
    internal: int,
    external: int,
    practical: int,
    semester: int,
    exam_type: str,
    acting_user: dict,
) -> int:
    """
    Record a NEW marks entry (first time this student/subject/semester/
    exam_type combination is being graded).

    Args:
        roll_no: The student's roll number.
        subject_code: The subject.
        internal: Internal marks obtained.
        external: External marks obtained.
        practical: Practical marks obtained.
        semester: The semester this exam belongs to.
        exam_type: One of config.EXAM_TYPES.
        acting_user: The logged-in user performing this action.

    Returns:
        The new marks row's mark_id.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor Teacher.
        ValidationError: if any field fails validation.
        RecordNotFoundError: if the student or subject does not exist (or
            is inactive).
        DuplicateRecordError: if a marks entry already exists for this
            exact combination -- use update_marks() to correct it instead.
    """
    auth.check_permission(acting_user["role"], MARKS_WRITE_ROLES)

    roll_no = validate_roll_no(roll_no)
    subject_code = validate_subject_code(subject_code)
    semester = validate_semester(semester)
    exam_type = validate_exam_type(exam_type)

    check_teacher_subject_access(acting_user, subject_code)

    # Confirm the student and subject actually exist (and are active)
    # BEFORE validating the mark values against the subject's own maximums
    # -- we need subject's max_internal/external/practical to do that.
    students.get_student(roll_no)
    subject = subjects.get_subject(subject_code)

    internal = validate_mark_value(internal, subject["max_internal"], "internal")
    external = validate_mark_value(external, subject["max_external"], "external")
    practical = validate_mark_value(practical, subject["max_practical"], "practical")

    if get_marks_entry(roll_no, subject_code, semester, exam_type) is not None:
        raise DuplicateRecordError(
            f"Marks already exist for {roll_no} in {subject_code} "
            f"(semester {semester}, {exam_type}). Use update_marks to correct them."
        )

    insert_statement = (
        "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (roll_no, subject_code, internal, external, practical, semester, exam_type),
    )
    new_value = {
        "roll_no": roll_no, "subject_code": subject_code, "internal": internal,
        "external": external, "practical": practical, "semester": semester, "exam_type": exam_type,
    }
    record_id = _marks_record_id(roll_no, subject_code, semester, exam_type)
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "marks", record_id,
        old_value=None, new_value=new_value,
    )

    results = execute_transaction([insert_statement, audit_statement])
    mark_id = results[0]
    logger.info(
        "Marks entered for %s in %s (semester=%s, exam_type=%s) by user_id=%s.",
        roll_no, subject_code, semester, exam_type, acting_user["user_id"],
    )
    return mark_id


def update_marks(
    mark_id: int,
    acting_user: dict,
    internal: int | None = None,
    external: int | None = None,
    practical: int | None = None,
) -> None:
    """
    Correct one or more mark components of an EXISTING marks entry. Only
    components passed as something other than None are changed.

    Args:
        mark_id: The marks row to update.
        acting_user: The logged-in user performing this action.
        internal, external, practical: New values; leave as None to keep
            the existing value.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor Teacher.
        RecordNotFoundError: if mark_id does not exist.
        ValidationError: if a provided value fails validation, or no
            fields at all were provided.
    """
    auth.check_permission(acting_user["role"], MARKS_WRITE_ROLES)

    existing = get_marks_by_id(mark_id)
    check_teacher_subject_access(acting_user, existing["subject_code"])
    # include_inactive=True: we must still be able to correct an old mark
    # even if the subject has since been deactivated/retired.
    subject = subjects.get_subject(existing["subject_code"], include_inactive=True)

    set_clauses = []
    params: list = []
    new_value = {}

    if internal is not None:
        internal = validate_mark_value(internal, subject["max_internal"], "internal")
        set_clauses.append("internal = ?")
        params.append(internal)
        new_value["internal"] = internal

    if external is not None:
        external = validate_mark_value(external, subject["max_external"], "external")
        set_clauses.append("external = ?")
        params.append(external)
        new_value["external"] = external

    if practical is not None:
        practical = validate_mark_value(practical, subject["max_practical"], "practical")
        set_clauses.append("practical = ?")
        params.append(practical)
        new_value["practical"] = practical

    if not set_clauses:
        raise ValidationError("No fields were provided to update.")

    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    params.append(mark_id)

    update_statement = (
        f"UPDATE marks SET {', '.join(set_clauses)} WHERE mark_id = ?",
        tuple(params),
    )
    old_value = {field: existing[field] for field in new_value}
    record_id = _marks_record_id(
        existing["roll_no"], existing["subject_code"], existing["semester"], existing["exam_type"]
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "marks", record_id,
        old_value=old_value, new_value=new_value,
    )

    execute_transaction([update_statement, audit_statement])
    logger.info(
        "Marks (mark_id=%s) updated by user_id=%s. Fields changed: %s",
        mark_id, acting_user["user_id"], list(new_value.keys()),
    )


# ---------------------------------------------------------------------------
# READ OPERATIONS (see modules/students.py's module docstring for why
# these are not individually role-gated)
# ---------------------------------------------------------------------------

def get_marks_by_id(mark_id: int) -> dict:
    """
    Fetch one marks row by its mark_id.

    Raises:
        RecordNotFoundError: if mark_id does not exist.
    """
    row = fetch_one(f"SELECT {MARKS_COLUMNS} FROM marks WHERE mark_id = ?", (mark_id,))
    if row is None:
        raise RecordNotFoundError(f"No marks entry found with mark_id {mark_id}.")
    return dict(row)


def get_marks_entry(roll_no: str, subject_code: str, semester: int, exam_type: str) -> dict | None:
    """
    Look up a marks row by its natural identity (the same four columns
    the UNIQUE constraint is built from), used internally to decide
    whether enter_marks() or update_marks() is appropriate.

    Returns:
        A dict if a matching row exists, otherwise None (deliberately NOT
        an exception here -- "no marks yet" is an expected, normal outcome
        for this lookup, not an error).
    """
    roll_no = validate_roll_no(roll_no)
    subject_code = validate_subject_code(subject_code)

    row = fetch_one(
        f"SELECT {MARKS_COLUMNS} FROM marks "
        "WHERE roll_no = ? AND subject_code = ? AND semester = ? AND exam_type = ?",
        (roll_no, subject_code, semester, exam_type),
    )
    return dict(row) if row is not None else None


def list_marks_for_student(
    roll_no: str, semester: int | None = None, exam_type: str | None = None
) -> list[dict]:
    """
    Fetch every marks row for one student, joined with each subject's name
    and maximums, with percentage/grade/pass-fail computed live for each
    row via modules.grades.evaluate_subject_marks().

    Args:
        roll_no: The student to fetch marks for.
        semester: If given, only this semester.
        exam_type: If given, only this exam type.

    Returns:
        A list of dicts, each with the raw marks columns PLUS
        subject_name, max_internal/external/practical, and the computed
        "percentage", "grade_letter", "grade_point", "passed" keys.
    """
    roll_no = validate_roll_no(roll_no)

    query = (
        "SELECT m.mark_id, m.roll_no, m.subject_code, s.name AS subject_name, "
        "m.internal, m.external, m.practical, "
        "s.max_internal, s.max_external, s.max_practical, "
        "m.semester, m.exam_type, m.created_at, m.updated_at "
        "FROM marks m JOIN subjects s ON m.subject_code = s.subject_code "
        "WHERE m.roll_no = ?"
    )
    params: list = [roll_no]

    if semester is not None:
        query += " AND m.semester = ?"
        params.append(semester)
    if exam_type is not None:
        query += " AND m.exam_type = ?"
        params.append(exam_type)

    query += " ORDER BY m.semester, m.subject_code"

    return [_with_evaluation(dict(row)) for row in fetch_all(query, tuple(params))]


def list_marks_for_subject(
    subject_code: str, semester: int | None = None, exam_type: str | None = None
) -> list[dict]:
    """
    Fetch every marks row for one subject, joined with each student's
    name, with percentage/grade/pass-fail computed live for each row.

    Args:
        subject_code: The subject to fetch marks for.
        semester: If given, only this semester.
        exam_type: If given, only this exam type.

    Returns:
        A list of dicts, each with the raw marks columns PLUS
        student_name, max_internal/external/practical, and the computed
        "percentage", "grade_letter", "grade_point", "passed" keys.
    """
    subject_code = validate_subject_code(subject_code)

    query = (
        "SELECT m.mark_id, m.roll_no, st.name AS student_name, m.subject_code, "
        "m.internal, m.external, m.practical, "
        "s.max_internal, s.max_external, s.max_practical, "
        "m.semester, m.exam_type, m.created_at, m.updated_at "
        "FROM marks m "
        "JOIN students st ON m.roll_no = st.roll_no "
        "JOIN subjects s ON m.subject_code = s.subject_code "
        "WHERE m.subject_code = ?"
    )
    params: list = [subject_code]

    if semester is not None:
        query += " AND m.semester = ?"
        params.append(semester)
    if exam_type is not None:
        query += " AND m.exam_type = ?"
        params.append(exam_type)

    query += " ORDER BY m.roll_no"

    return [_with_evaluation(dict(row)) for row in fetch_all(query, tuple(params))]


def _with_evaluation(row: dict) -> dict:
    """Attach percentage/grade_letter/grade_point/passed to a marks row
    dict, computed via modules.grades.evaluate_subject_marks() using that
    row's own subject maximums. Shared by both list_* functions above."""
    evaluation = evaluate_subject_marks(
        row["internal"], row["external"], row["practical"],
        row["max_internal"], row["max_external"], row["max_practical"],
    )
    row.update(evaluation)
    return row


def compute_sgpa_for_marks(subject_marks: list[dict]) -> float | None:
    """
    Compute SGPA from a list of already-fetched marks rows (each must
    have "subject_code" and "grade_point" -- exactly what
    list_marks_for_student() returns), by looking up each subject's
    credits and calling modules.grades.calculate_sgpa() -- the same grade
    engine every other part of this app uses.

    Centralised here, rather than left as three near-identical private
    copies: modules/ml_predictions.py (computing a student's PREVIOUS
    semester SGPA as an ML feature), utils/pdf_generator.py (the report
    card's SGPA line), and modules/student_portal.py (a student's own
    SGPA) all need exactly this same "marks rows -> SGPA" computation.
    modules/marks.py is the natural shared home for it: it already
    imports both modules.subjects (for credits) and modules.grades (for
    the SGPA formula itself), which the other three files would otherwise
    each need to import solely for this one calculation.

    Args:
        subject_marks: Marks rows for ONE semester (mixing semesters
            would credit-weight subjects from different semesters
            together, which is not what SGPA means).

    Returns:
        The SGPA, or None if subject_marks is empty.
    """
    if not subject_marks:
        return None

    subject_results = [
        {
            "credits": subjects.get_subject(row["subject_code"], include_inactive=True)["credits"],
            "grade_point": row["grade_point"],
        }
        for row in subject_marks
    ]
    return calculate_sgpa(subject_results)


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_marks_page() -> None:
    """
    Streamlit page: bulk marks entry for one subject/semester/exam_type at
    a time, showing every active student in that semester with editable
    internal/external/practical fields, followed by a live-graded table of
    marks already entered.
    """
    user = auth.require_role(*MARKS_WRITE_ROLES)

    st.title("Marks Entry")

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

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_semester = st.number_input(
            "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
            value=selected_subject["semester"], step=1,
        )
    with filter_col2:
        selected_exam_type = st.selectbox("Exam Type", options=list(config.EXAM_TYPES))

    semester_value = int(selected_semester)
    class_students = students.list_students(semester=semester_value)

    if not class_students:
        st.info(f"No active students found in semester {semester_value}.")
        return

    st.caption(
        f"Maximum marks for {selected_subject['subject_code']} -- "
        f"Internal: {selected_subject['max_internal']}, "
        f"External: {selected_subject['max_external']}, "
        f"Practical: {selected_subject['max_practical']}"
    )

    with st.form("marks_entry_form"):
        entered_values = {}
        for student in class_students:
            existing_entry = get_marks_entry(
                student["roll_no"], selected_subject["subject_code"], semester_value, selected_exam_type
            )
            st.markdown(f"**{student['roll_no']} — {student['name']}**")
            mark_col1, mark_col2, mark_col3 = st.columns(3)
            with mark_col1:
                internal_input = st.number_input(
                    "Internal", min_value=0, max_value=selected_subject["max_internal"],
                    value=existing_entry["internal"] if existing_entry else 0,
                    key=f"internal_{student['roll_no']}",
                )
            with mark_col2:
                external_input = st.number_input(
                    "External", min_value=0, max_value=selected_subject["max_external"],
                    value=existing_entry["external"] if existing_entry else 0,
                    key=f"external_{student['roll_no']}",
                )
            with mark_col3:
                practical_input = st.number_input(
                    "Practical", min_value=0, max_value=selected_subject["max_practical"],
                    value=existing_entry["practical"] if existing_entry else 0,
                    key=f"practical_{student['roll_no']}",
                )
            entered_values[student["roll_no"]] = {
                "internal": int(internal_input),
                "external": int(external_input),
                "practical": int(practical_input),
                "existing": existing_entry,
            }

        save_submitted = st.form_submit_button("Save All Marks")

    if save_submitted:
        saved_count = 0
        skipped_count = 0
        errors = []

        for roll_no, data in entered_values.items():
            existing_entry = data["existing"]
            new_triplet = (data["internal"], data["external"], data["practical"])

            try:
                if existing_entry is None:
                    enter_marks(
                        roll_no, selected_subject["subject_code"],
                        data["internal"], data["external"], data["practical"],
                        semester_value, selected_exam_type, user,
                    )
                    saved_count += 1
                elif (existing_entry["internal"], existing_entry["external"], existing_entry["practical"]) != new_triplet:
                    # Only write (and audit) an UPDATE if something actually
                    # changed -- otherwise clicking "Save All" on an
                    # unmodified form would flood audit_log with no-op
                    # UPDATE entries for every student in the class.
                    update_marks(
                        existing_entry["mark_id"], user,
                        internal=data["internal"], external=data["external"], practical=data["practical"],
                    )
                    saved_count += 1
                else:
                    skipped_count += 1
            except (ValidationError, DuplicateRecordError, RecordNotFoundError, AuthorizationError) as error:
                errors.append(f"{roll_no}: {error}")

        if saved_count:
            st.success(f"Saved marks for {saved_count} student(s).")
        if skipped_count:
            st.caption(f"{skipped_count} student(s) unchanged -- nothing written.")
        for error_message in errors:
            st.error(error_message)
        if saved_count:
            st.rerun()

    st.subheader("Current Marks (with computed grade)")
    current_marks = list_marks_for_subject(
        selected_subject["subject_code"], semester=semester_value, exam_type=selected_exam_type
    )

    if not current_marks:
        st.info("No marks entered yet for this selection.")
        return

    display_rows = [
        {
            "Roll No": entry["roll_no"],
            "Student": entry["student_name"],
            "Internal": entry["internal"],
            "External": entry["external"],
            "Practical": entry["practical"],
            "Percentage": entry["percentage"],
            "Grade": entry["grade_letter"],
            "Passed": "Yes" if entry["passed"] else "No",
        }
        for entry in current_marks
    ]
    render_data_table(display_rows, key_prefix="marks_table", filename_prefix="marks")
