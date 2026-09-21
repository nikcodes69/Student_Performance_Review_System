"""
modules/grade_appeals.py
=========================
Grade appeals: a Student who believes a PUBLISHED mark is wrong can
submit an appeal (subject/semester/exam_type + a written reason); a
Teacher assigned to that subject, or an Admin, reviews it and responds
approved or rejected with a written response.

WHAT THIS DELIBERATELY DOES NOT DO: approving an appeal does not, by
itself, change any marks row. This module is a communication/tracking
layer, not an auto-correction mechanism -- a Teacher who approves an
appeal still goes and fixes the mark by hand via modules/marks.py's
update_marks(), exactly as they would for any other correction, which
keeps that one audited, validated write path as the ONLY way marks ever
change (see modules/marks.py's own module docstring). Coupling "approve
this appeal" to "silently rewrite these marks" would be a second,
parallel path to change a grade, with its own risk of drifting out of
sync with update_marks()'s own validation and audit trail.

WHY ONLY A PUBLISHED MARK CAN BE APPEALED: a Student cannot even SEE a
draft mark (see modules/marks.py's publish_marks()) -- there is nothing
to dispute yet. submit_appeal() confirms a published marks row exists
for the exact roll_no/subject_code/semester/exam_type combination before
accepting an appeal.

WHO SEES WHAT: a Student sees only their OWN appeals (roll_no = their own
username, the same convention modules/student_portal.py relies on for
"my own data only"). A Teacher sees pending appeals only for subjects
they are assigned to (modules/teacher_subjects.py). Admin sees
everything.
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, fetch_all, fetch_one
from modules import auth, marks
from modules.audit import build_audit_entry
from modules.teacher_subjects import check_teacher_subject_access, list_subjects_for_teacher
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.validators import (
    validate_appeal_reason,
    validate_appeal_response,
    validate_exam_type,
    validate_roll_no,
    validate_semester,
    validate_subject_code,
)

logger = get_logger(__name__)

APPEAL_COLUMNS = (
    "a.appeal_id, a.roll_no, a.subject_code, a.semester, a.exam_type, a.reason, "
    "a.status, a.response, a.reviewed_by, a.created_at, a.reviewed_at"
)


def submit_appeal(
    roll_no: str, subject_code: str, semester: int, exam_type: str, reason: str, acting_user: dict,
) -> int:
    """
    A Student submits an appeal against their own PUBLISHED mark.

    Args:
        roll_no: Must equal acting_user's own username -- a Student can
            only appeal their own result, never another student's.
        subject_code: The subject.
        semester: The semester.
        exam_type: One of config.EXAM_TYPES.
        reason: Why the Student believes this mark is wrong.
        acting_user: The logged-in user submitting this (must be Student).

    Returns:
        The new appeal_id.

    Raises:
        AuthorizationError: if acting_user's role is not Student, or
            roll_no is not their own.
        ValidationError: if any field fails validation, no PUBLISHED mark
            exists for this exact combination, or a pending appeal for
            the same combination already exists.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_STUDENT,))

    roll_no = validate_roll_no(roll_no)
    if roll_no != acting_user["username"]:
        raise AuthorizationError("You can only appeal your own marks.")

    subject_code = validate_subject_code(subject_code)
    semester = validate_semester(semester)
    exam_type = validate_exam_type(exam_type)
    reason = validate_appeal_reason(reason)

    published_mark = marks.get_marks_entry(roll_no, subject_code, semester, exam_type)
    if published_mark is None or not published_mark["is_published"]:
        raise ValidationError(
            "No published mark found for this subject/semester/exam type -- "
            "there is nothing to appeal yet."
        )

    existing_pending = fetch_one(
        "SELECT appeal_id FROM grade_appeals WHERE roll_no = ? AND subject_code = ? "
        "AND semester = ? AND exam_type = ? AND status = ?",
        (roll_no, subject_code, semester, exam_type, config.APPEAL_PENDING),
    )
    if existing_pending is not None:
        raise ValidationError(
            "You already have a pending appeal for this subject/semester/exam type."
        )

    insert_statement = (
        "INSERT INTO grade_appeals (roll_no, subject_code, semester, exam_type, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (roll_no, subject_code, semester, exam_type, reason),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "grade_appeals",
        f"{roll_no}:{subject_code}:{semester}:{exam_type}",
        new_value={"reason": reason},
    )
    results = execute_transaction([insert_statement, audit_statement])
    appeal_id = results[0]

    logger.info(
        "Grade appeal #%s submitted by %s for %s (semester=%s, exam_type=%s).",
        appeal_id, roll_no, subject_code, semester, exam_type,
    )
    return appeal_id


def list_appeals_for_student(roll_no: str) -> list[dict]:
    """Every appeal a Student has submitted, newest first -- their own
    history, whatever its current status."""
    roll_no = validate_roll_no(roll_no)
    return fetch_all(
        f"SELECT {APPEAL_COLUMNS} FROM grade_appeals a "
        "WHERE a.roll_no = ? ORDER BY a.created_at DESC, a.appeal_id DESC",
        (roll_no,),
    )


def list_appeals_for_reviewer(acting_user: dict, status: str | None = config.APPEAL_PENDING) -> list[dict]:
    """
    Appeals visible to a Teacher/Admin reviewer.

    Args:
        acting_user: The logged-in Teacher or Admin.
        status: If given, only appeals in this status (default: only
            "pending", the reviewer's actual queue). Pass None for every
            status, e.g. to show a resolved-appeals history too.

    Returns:
        A list of dicts (see APPEAL_COLUMNS), plus student_name and
        subject_name, newest first. For a Teacher, only appeals for
        subjects they are assigned to; for Admin, every appeal.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN, config.ROLE_TEACHER))

    query = (
        f"SELECT {APPEAL_COLUMNS}, st.name AS student_name, sub.name AS subject_name "
        "FROM grade_appeals a "
        "JOIN students st ON a.roll_no = st.roll_no "
        "JOIN subjects sub ON a.subject_code = sub.subject_code"
    )
    conditions = []
    params: list = []

    if acting_user["role"] == config.ROLE_TEACHER:
        assigned_codes = [s["subject_code"] for s in list_subjects_for_teacher(acting_user["user_id"])]
        if not assigned_codes:
            return []
        placeholders = ", ".join("?" for _ in assigned_codes)
        conditions.append(f"a.subject_code IN ({placeholders})")
        params.extend(assigned_codes)

    if status is not None:
        conditions.append("a.status = ?")
        params.append(status)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY a.created_at DESC, a.appeal_id DESC"

    return fetch_all(query, tuple(params))


def respond_to_appeal(appeal_id: int, approved: bool, response: str, acting_user: dict) -> None:
    """
    Resolve a pending appeal: mark it approved or rejected, with a
    written response. Does NOT change any marks row -- see this module's
    docstring for why that stays a deliberate, separate, manual step
    through modules/marks.py's update_marks().

    Args:
        appeal_id: The appeal to resolve.
        approved: True to approve, False to reject.
        response: Explanation shown back to the Student.
        acting_user: The logged-in user resolving this -- must be Admin,
            or the Teacher assigned to the appeal's subject.

    Raises:
        AuthorizationError: if acting_user's role is neither Admin nor
            Teacher, or (for a Teacher) they are not assigned to this subject.
        RecordNotFoundError: if appeal_id does not exist.
        ValidationError: if response fails validation, or this appeal is
            no longer pending (already resolved).
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN, config.ROLE_TEACHER))

    existing = fetch_one(
        f"SELECT {APPEAL_COLUMNS} FROM grade_appeals a WHERE a.appeal_id = ?", (appeal_id,),
    )
    if existing is None:
        raise RecordNotFoundError(f"Appeal #{appeal_id} not found.")

    check_teacher_subject_access(acting_user, existing["subject_code"])

    if existing["status"] != config.APPEAL_PENDING:
        raise ValidationError(f"Appeal #{appeal_id} has already been {existing['status']}.")

    response = validate_appeal_response(response)
    new_status = config.APPEAL_APPROVED if approved else config.APPEAL_REJECTED

    update_statement = (
        "UPDATE grade_appeals SET status = ?, response = ?, reviewed_by = ?, "
        "reviewed_at = CURRENT_TIMESTAMP WHERE appeal_id = ?",
        (new_status, response, acting_user["user_id"], appeal_id),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "grade_appeals",
        f"{existing['roll_no']}:{existing['subject_code']}:{existing['semester']}:{existing['exam_type']}",
        old_value={"status": config.APPEAL_PENDING},
        new_value={"status": new_status, "response": response},
    )
    execute_transaction([update_statement, audit_statement])

    logger.info(
        "Appeal #%s %s by user_id=%s.", appeal_id, new_status, acting_user["user_id"],
    )


def render_grade_appeals_page() -> None:
    """Streamlit page: a Student submits/tracks their own appeals;
    Admin/Teacher review and resolve pending ones."""
    user = auth.require_role(config.ROLE_ADMIN, config.ROLE_TEACHER, config.ROLE_STUDENT)

    st.title("Grade Appeals")

    if user["role"] == config.ROLE_STUDENT:
        _render_student_view(user)
    else:
        _render_reviewer_view(user)


def _render_student_view(user: dict) -> None:
    roll_no = user["username"]

    published_marks = marks.list_marks_for_student(roll_no, published_only=True)
    if published_marks:
        with st.expander(":material/gavel: Submit a new appeal"):
            options = {
                f"{row['subject_code']} - {row['subject_name']} "
                f"(semester {row['semester']}, {row['exam_type']})": row
                for row in published_marks
            }
            choice = st.selectbox("Which result are you appealing?", options=list(options.keys()))
            selected = options[choice]
            reason = st.text_area("Reason")

            if st.button("Submit appeal", type="primary"):
                try:
                    submit_appeal(
                        roll_no, selected["subject_code"], selected["semester"],
                        selected["exam_type"], reason, user,
                    )
                    st.success("Appeal submitted.")
                    st.rerun()
                except ValidationError as error:
                    st.error(str(error))
    else:
        st.info("You have no published results yet to appeal.")

    st.divider()
    st.subheader("Your appeals")
    appeals = list_appeals_for_student(roll_no)
    if not appeals:
        st.info("You haven't submitted any appeals.")
        return

    for entry in appeals:
        with st.container(border=True):
            st.markdown(f"**{entry['subject_code']}** -- semester {entry['semester']}, {entry['exam_type']}")
            st.caption(f"Status: {entry['status'].capitalize()} -- submitted {entry['created_at']}")
            st.write(f"Your reason: {entry['reason']}")
            if entry["response"]:
                st.write(f"Response: {entry['response']}")


def _render_reviewer_view(user: dict) -> None:
    pending = list_appeals_for_reviewer(user, status=config.APPEAL_PENDING)

    st.subheader(f"Pending appeals ({len(pending)})")
    if not pending:
        st.info("No pending appeals.")
    for entry in pending:
        with st.container(border=True):
            st.markdown(
                f"**{entry['student_name']} ({entry['roll_no']})** -- "
                f"{entry['subject_code']} - {entry['subject_name']}, "
                f"semester {entry['semester']}, {entry['exam_type']}"
            )
            st.caption(f"Submitted {entry['created_at']}")
            st.write(f"Reason: {entry['reason']}")

            response = st.text_area("Your response", key=f"appeal_response_{entry['appeal_id']}")
            approve_col, reject_col = st.columns(2)
            with approve_col:
                if st.button("Approve", key=f"approve_{entry['appeal_id']}", use_container_width=True):
                    try:
                        respond_to_appeal(entry["appeal_id"], True, response, user)
                        st.success("Appeal approved.")
                        st.rerun()
                    except ValidationError as error:
                        st.error(str(error))
            with reject_col:
                if st.button("Reject", key=f"reject_{entry['appeal_id']}", use_container_width=True):
                    try:
                        respond_to_appeal(entry["appeal_id"], False, response, user)
                        st.warning("Appeal rejected.")
                        st.rerun()
                    except ValidationError as error:
                        st.error(str(error))

    st.divider()
    with st.expander("Resolved appeals"):
        resolved = [
            entry for entry in list_appeals_for_reviewer(user, status=None)
            if entry["status"] != config.APPEAL_PENDING
        ]
        if not resolved:
            st.info("No resolved appeals yet.")
        for entry in resolved:
            st.markdown(
                f"**{entry['student_name']} ({entry['roll_no']})** -- "
                f"{entry['subject_code']}, semester {entry['semester']}, {entry['exam_type']} "
                f"-- {entry['status'].capitalize()}"
            )
            st.caption(f"Response: {entry['response']}")
