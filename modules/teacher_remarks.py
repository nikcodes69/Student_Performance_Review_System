"""
modules/teacher_remarks.py
============================
Teacher remarks: a short note a Teacher (or Admin) leaves on a student's
record -- e.g. "Great improvement this semester" or "Needs to submit
assignments on time" -- visible to that student on their own portal, the
same way a real report card carries a handwritten remark from a teacher.

SUBJECT SCOPING: subject_code is OPTIONAL. A remark about one specific
subject (e.g. tied to a particular exam) sets it, and then follows the
exact same write-access rule as modules/marks.py -- a Teacher must be
assigned to that subject (modules/teacher_subjects.py's
check_teacher_subject_access()), Admin is unrestricted. A GENERAL remark
about the student overall (not about any one subject) leaves subject_code
NULL -- there is no subject to scope it to, so any Teacher may leave one,
the same way any Teacher may post an Announcement to any audience
(modules/announcements.py) without being restricted to their own
subjects' students.

WHO CAN TAKE ONE DOWN: Admin can take down any remark. A Teacher can only
take down their OWN -- otherwise one Teacher could silence another's
note about a student, the same permission boundary
modules/announcements.py's deactivate_announcement() enforces, and for
the same reason (checked in code, not just hidden in the UI).
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, execute_write, fetch_all, fetch_one
from modules import auth, students
from modules.audit import build_audit_entry
from modules.teacher_subjects import check_teacher_subject_access, list_subjects_for_marks_entry
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.validators import validate_remark_text, validate_roll_no, validate_subject_code

logger = get_logger(__name__)

REMARK_WRITE_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)


def create_remark(roll_no: str, subject_code: str | None, remark: str, acting_user: dict) -> int:
    """
    Leave a new remark on a student's record.

    Args:
        roll_no: The student this remark is about.
        subject_code: A subject_code to tie this remark to a specific
            subject, or None for a general remark -- see this module's
            docstring for the access-control difference.
        remark: The remark text.
        acting_user: The logged-in user leaving this (must be Admin or Teacher).

    Returns:
        The new remark_id.

    Raises:
        AuthorizationError: if acting_user's role is not Admin or
            Teacher, or (when subject_code is given) a Teacher not
            assigned to that subject.
        RecordNotFoundError: if roll_no does not exist.
        ValidationError: if remark or subject_code fails validation.
    """
    auth.check_permission(acting_user["role"], REMARK_WRITE_ROLES)

    roll_no = validate_roll_no(roll_no)
    students.get_student(roll_no)

    if subject_code is not None:
        subject_code = validate_subject_code(subject_code)
        check_teacher_subject_access(acting_user, subject_code)

    remark = validate_remark_text(remark)

    insert_statement = (
        "INSERT INTO teacher_remarks (roll_no, subject_code, teacher_id, remark) VALUES (?, ?, ?, ?)",
        (roll_no, subject_code, acting_user["user_id"], remark),
    )
    remark_id = execute_write(*insert_statement)

    audit_statement = build_audit_entry(
        user_id=acting_user["user_id"],
        action=config.AUDIT_INSERT,
        table_name="teacher_remarks",
        record_id=str(remark_id),
        new_value={"roll_no": roll_no, "subject_code": subject_code, "remark": remark},
    )
    execute_write(*audit_statement)

    logger.info(
        "Remark #%s left on %s by user_id=%s (subject_code=%s).",
        remark_id, roll_no, acting_user["user_id"], subject_code,
    )
    return remark_id


def list_remarks_for_student(roll_no: str, include_inactive: bool = False) -> list[dict]:
    """
    Every remark left on one student's record, newest first.

    Args:
        roll_no: The student.
        include_inactive: If True, also include taken-down remarks (used
            by the Admin/Teacher management view so a poster can see
            what they've retracted, not just what's currently visible).

    Returns:
        A list of dicts: remark_id, roll_no, subject_code, subject_name
        (None for a general remark), teacher_id, teacher_username,
        remark, is_active, created_at.
    """
    roll_no = validate_roll_no(roll_no)

    query = (
        "SELECT r.remark_id, r.roll_no, r.subject_code, sub.name AS subject_name, "
        "r.teacher_id, u.username AS teacher_username, r.remark, r.is_active, r.created_at "
        "FROM teacher_remarks r "
        "JOIN users u ON r.teacher_id = u.user_id "
        "LEFT JOIN subjects sub ON r.subject_code = sub.subject_code "
        "WHERE r.roll_no = ?"
    )
    params: list = [roll_no]

    if not include_inactive:
        query += " AND r.is_active = 1"

    # remark_id DESC as a tiebreaker for the same reason
    # modules/announcements.py's list_announcements_for_role() needs one:
    # created_at has only second resolution.
    query += " ORDER BY r.created_at DESC, r.remark_id DESC"

    return fetch_all(query, tuple(params))


def deactivate_remark(remark_id: int, acting_user: dict) -> None:
    """
    Retract a remark: sets is_active = 0. Never a real SQL DELETE -- see
    this module's docstring for why.

    Args:
        remark_id: The remark to take down.
        acting_user: The logged-in user performing this action -- must be
            Admin, or the Teacher who originally left it.

    Raises:
        AuthorizationError: if acting_user is neither Admin nor the
            original author.
        RecordNotFoundError: if remark_id does not exist.
        ValidationError: if the remark is already inactive.
    """
    existing = fetch_one(
        "SELECT remark_id, teacher_id, is_active FROM teacher_remarks WHERE remark_id = ?",
        (remark_id,),
    )
    if existing is None:
        raise RecordNotFoundError(f"Remark #{remark_id} not found.")

    is_own_remark = acting_user["user_id"] == existing["teacher_id"]
    if acting_user["role"] != config.ROLE_ADMIN and not is_own_remark:
        raise AuthorizationError("You can only take down remarks you left yourself.")

    if not existing["is_active"]:
        raise ValidationError(f"Remark #{remark_id} is already inactive.")

    update_statement = ("UPDATE teacher_remarks SET is_active = 0 WHERE remark_id = ?", (remark_id,))
    audit_statement = build_audit_entry(
        user_id=acting_user["user_id"],
        action=config.AUDIT_SOFT_DELETE,
        table_name="teacher_remarks",
        record_id=str(remark_id),
        old_value={"is_active": True},
        new_value=None,
    )
    execute_transaction([update_statement, audit_statement])

    logger.info("Remark #%s taken down by user_id=%s.", remark_id, acting_user["user_id"])


def render_remarks_page() -> None:
    """Streamlit page (Admin/Teacher): pick a student, leave a remark,
    view and retract remarks already left on their record."""
    user = auth.require_role(*REMARK_WRITE_ROLES)
    st.title("Student Remarks")

    student_list = students.list_students()
    if not student_list:
        st.info("No students found.")
        return

    labels = {f"{s['roll_no']} - {s['name']}": s for s in student_list}
    picked_label = st.selectbox("Student", options=list(labels.keys()))
    picked_student = labels[picked_label]

    with st.expander(":material/rate_review: Leave a new remark"):
        subject_options = {"General (not tied to a subject)": None}
        subject_options |= {
            f"{s['subject_code']} - {s['name']}": s["subject_code"]
            for s in list_subjects_for_marks_entry(user)
        }
        subject_choice = st.selectbox("Subject", options=list(subject_options.keys()))
        remark_text = st.text_area("Remark")

        if st.button("Post remark", type="primary"):
            try:
                create_remark(
                    picked_student["roll_no"], subject_options[subject_choice], remark_text, user,
                )
                # st.toast(), not st.success() -- see app.py's
                # render_role_login_form() for why, wherever a message is
                # immediately followed by st.rerun().
                st.toast("Remark posted.", icon=":material/check_circle:")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))

    st.divider()
    st.subheader(f"Remarks on {picked_student['name']}'s record")

    remarks = list_remarks_for_student(picked_student["roll_no"])
    if not remarks:
        st.info("No remarks yet.")
        return

    for entry in remarks:
        subject_label = entry["subject_name"] or "General"
        with st.container(border=True):
            st.markdown(f"**{subject_label}**")
            st.caption(f"By {entry['teacher_username']} on {entry['created_at']}")
            st.write(entry["remark"])

            can_take_down = user["role"] == config.ROLE_ADMIN or entry["teacher_id"] == user["user_id"]
            if can_take_down:
                if st.button("Take down", key=f"deactivate_remark_{entry['remark_id']}"):
                    deactivate_remark(entry["remark_id"], user)
                    st.toast("Remark taken down.", icon=":material/check_circle:")
                    st.rerun()
