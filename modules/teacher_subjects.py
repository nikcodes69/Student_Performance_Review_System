"""
modules/teacher_subjects.py
============================
Teacher-to-subject assignment: which Teacher accounts are allowed to
enter marks/attendance/assignments for which subjects.

WHY THIS MODULE EXISTS: before it did, ANY Teacher account could enter
data for ANY subject in the system -- there was no mapping anywhere
saying "this teacher teaches that subject". That was fine for a small
demo with one teacher and one subject, but is a real access-control gap
for anything resembling a real deployment (a Physics teacher should not
be able to enter Chemistry marks just because both subjects exist in the
same database). See database/db_setup.py's teacher_subjects table for
the schema.

ENFORCEMENT IS TWO-LAYERED, THE SAME "DEFENSE IN DEPTH" PATTERN USED
EVERYWHERE ELSE IN THIS PROJECT:
  - UI LEVEL: modules/marks.py, attendance.py, and assignments.py's
    render_*_page() functions call list_subjects_for_teacher() (not
    subjects.list_subjects()) to build a Teacher's subject picker, so a
    Teacher never even SEES a subject they are not assigned to.
  - BUSINESS-LOGIC LEVEL: check_teacher_subject_access() below is called
    by record_marks()/update_marks(), record_attendance()/
    update_attendance(), and record_assignment()/update_assignment()
    immediately after their existing check_permission() call -- so even
    a request that somehow bypassed the UI (a bug, a future API) is
    still stopped by the business logic itself, not just kept out of
    the menu.

ADMIN IS NEVER RESTRICTED BY THIS TABLE: check_teacher_subject_access()
only checks anything when acting_user's role is Teacher. Admin can
always enter data for any subject, exactly as before this feature
existed -- this table narrows what a TEACHER can do, it does not add a
second gate in front of Admin.
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, fetch_all, fetch_one
from modules import auth
from modules.audit import build_audit_entry
from utils.exceptions import AuthorizationError, DuplicateRecordError, RecordNotFoundError
from utils.logger import get_logger
from utils.validators import validate_subject_code

logger = get_logger(__name__)


def _assignment_record_id(teacher_id: int, subject_code: str) -> str:
    """Composite natural key for audit_log's record_id -- this table has
    no surrogate primary key (see database/db_setup.py's teacher_subjects
    table comment), so its own (teacher_id, subject_code) pair is used
    directly, the same convention modules/marks.py and attendance.py use
    for their own composite-key tables."""
    return f"{teacher_id}:{subject_code}"


def assign_teacher_to_subject(teacher_id: int, subject_code: str, acting_user: dict) -> None:
    """
    Give a Teacher account access to enter data for one subject.

    Args:
        teacher_id: The user_id of the account being granted access. Must
            already exist as an active Teacher-role user.
        subject_code: The subject being granted.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if teacher_id does not belong to an active
            Teacher account, or subject_code does not exist.
        DuplicateRecordError: if this exact assignment already exists.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    subject_code = validate_subject_code(subject_code)

    teacher = fetch_one(
        "SELECT user_id, username FROM users WHERE user_id = ? AND role = ? AND is_active = 1",
        (teacher_id, config.ROLE_TEACHER),
    )
    if teacher is None:
        raise RecordNotFoundError(f"No active Teacher account found with user_id {teacher_id}.")

    subject = fetch_one("SELECT subject_code FROM subjects WHERE subject_code = ?", (subject_code,))
    if subject is None:
        raise RecordNotFoundError(f"No subject found with code '{subject_code}'.")

    existing = fetch_one(
        "SELECT teacher_id FROM teacher_subjects WHERE teacher_id = ? AND subject_code = ?",
        (teacher_id, subject_code),
    )
    if existing is not None:
        raise DuplicateRecordError(
            f"'{teacher['username']}' is already assigned to subject '{subject_code}'."
        )

    insert_statement = (
        "INSERT INTO teacher_subjects (teacher_id, subject_code) VALUES (?, ?)",
        (teacher_id, subject_code),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "teacher_subjects",
        _assignment_record_id(teacher_id, subject_code),
        old_value=None, new_value={"teacher_id": teacher_id, "subject_code": subject_code},
    )

    execute_transaction([insert_statement, audit_statement])
    list_subjects_for_teacher.clear()  # invalidate the cached lookup -- see that function's docstring
    logger.info(
        "Teacher '%s' (user_id=%s) assigned to subject '%s' by user_id=%s.",
        teacher["username"], teacher_id, subject_code, acting_user["user_id"],
    )


def unassign_teacher_from_subject(teacher_id: int, subject_code: str, acting_user: dict) -> None:
    """
    Revoke a Teacher's access to one subject.

    Args:
        teacher_id: The user_id of the account being revoked.
        subject_code: The subject being revoked.
        acting_user: The logged-in Admin performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if this assignment does not exist.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    subject_code = validate_subject_code(subject_code)

    existing = fetch_one(
        "SELECT teacher_id FROM teacher_subjects WHERE teacher_id = ? AND subject_code = ?",
        (teacher_id, subject_code),
    )
    if existing is None:
        raise RecordNotFoundError(
            f"No assignment found for user_id {teacher_id} and subject '{subject_code}'."
        )

    delete_statement = (
        "DELETE FROM teacher_subjects WHERE teacher_id = ? AND subject_code = ?",
        (teacher_id, subject_code),
    )
    audit_statement = build_audit_entry(
        # AUDIT_UPDATE with new_value=None, not a fourth AUDIT_DELETE
        # action -- see config.py's AUDIT_ACTIONS comment for why.
        acting_user["user_id"], config.AUDIT_UPDATE, "teacher_subjects",
        _assignment_record_id(teacher_id, subject_code),
        old_value={"teacher_id": teacher_id, "subject_code": subject_code}, new_value=None,
    )

    execute_transaction([delete_statement, audit_statement])
    list_subjects_for_teacher.clear()
    logger.info(
        "Teacher (user_id=%s) unassigned from subject '%s' by user_id=%s.",
        teacher_id, subject_code, acting_user["user_id"],
    )


# ---------------------------------------------------------------------------
# READ OPERATIONS (see modules/students.py's module docstring for why
# these are not individually role-gated -- render_*_page() functions
# elsewhere call these directly to build a Teacher's subject picker)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def list_subjects_for_teacher(teacher_id: int) -> list[dict]:
    """
    Every subject one Teacher account is assigned to.

    CACHED for 60 seconds, with explicit .clear() calls in
    assign_teacher_to_subject()/unassign_teacher_from_subject() above --
    the exact same caching pattern as modules/students.py's
    list_students() (see that function's docstring for the full
    reasoning): this is called on every visit to Marks Entry/Attendance/
    Assignments by a Teacher, so it needs to be cheap, but a freshly
    granted assignment should be usable immediately, not up to 60
    seconds later.

    Args:
        teacher_id: The Teacher account's user_id.

    Returns:
        A list of dicts (subject_code, name, semester, credits), active
        subjects only, ordered by subject_code.
    """
    return fetch_all(
        "SELECT s.subject_code, s.name, s.semester, s.credits "
        "FROM teacher_subjects ts JOIN subjects s ON ts.subject_code = s.subject_code "
        "WHERE ts.teacher_id = ? AND s.is_active = 1 ORDER BY s.subject_code",
        (teacher_id,),
    )


def list_teachers_for_subject(subject_code: str) -> list[dict]:
    """Every active Teacher account assigned to one subject.

    Args:
        subject_code: The subject to look up.

    Returns:
        A list of dicts (user_id, username), ordered by username.
    """
    subject_code = validate_subject_code(subject_code)
    return fetch_all(
        "SELECT u.user_id, u.username FROM teacher_subjects ts "
        "JOIN users u ON ts.teacher_id = u.user_id "
        "WHERE ts.subject_code = ? AND u.is_active = 1 ORDER BY u.username",
        (subject_code,),
    )


def check_teacher_subject_access(acting_user: dict, subject_code: str) -> None:
    """
    Confirm acting_user may enter data for subject_code -- the business-
    logic half of this module's two-layer enforcement (see module
    docstring). A no-op for any role other than Teacher: Admin is never
    restricted by this table.

    Args:
        acting_user: The logged-in user attempting a marks/attendance/
            assignment write.
        subject_code: The subject being written to.

    Raises:
        AuthorizationError: if acting_user is a Teacher not assigned to
            this subject.
    """
    if acting_user["role"] != config.ROLE_TEACHER:
        return

    assigned_codes = {row["subject_code"] for row in list_subjects_for_teacher(acting_user["user_id"])}
    if subject_code not in assigned_codes:
        raise AuthorizationError(
            f"You are not assigned to subject '{subject_code}'. Contact an administrator."
        )


def list_subjects_for_marks_entry(acting_user: dict) -> list[dict]:
    """
    The subject list a marks/attendance/assignments picker should offer:
    every subject for Admin, only assigned subjects for a Teacher. Used
    by modules/marks.py, attendance.py, and assignments.py so all three
    pages apply this rule identically instead of three separate copies
    of the same if/else.

    Args:
        acting_user: The logged-in user.

    Returns:
        A list of subject dicts (see modules/subjects.py's
        list_subjects()), active subjects only.
    """
    from modules.subjects import list_subjects  # lazy import: subjects.py does not import this module, so no cycle risk -- kept local anyway for the same "cross-module call, visible at its call site" reasoning used elsewhere in this project

    if acting_user["role"] == config.ROLE_ADMIN:
        return list_subjects()
    return list_subjects_for_teacher(acting_user["user_id"])


# ---------------------------------------------------------------------------
# STREAMLIT PAGE SECTION (embedded into modules/subjects.py's Subject
# Configuration page, Admin only -- not a standalone sidebar entry)
# ---------------------------------------------------------------------------

def render_teacher_assignment_section(subject_code: str, acting_user: dict) -> None:
    """
    Streamlit component: manage which Teachers are assigned to ONE
    subject. Called from modules/subjects.py's render_subjects_page(),
    inside the "Edit / Deactivate Subject" section, once a subject is
    selected -- assignment is naturally a per-subject action, so it lives
    next to the other per-subject Admin controls rather than as its own
    separate page.

    Args:
        subject_code: The currently selected subject.
        acting_user: The logged-in Admin.
    """
    st.caption("Only Teachers assigned to a subject can enter marks/attendance/assignments for it.")

    assigned_teachers = list_teachers_for_subject(subject_code)
    if assigned_teachers:
        st.write(", ".join(t["username"] for t in assigned_teachers))
    else:
        st.info("No teachers assigned to this subject yet.")

    all_teachers = fetch_all(
        "SELECT user_id, username FROM users WHERE role = ? AND is_active = 1 ORDER BY username",
        (config.ROLE_TEACHER,),
    )
    if not all_teachers:
        st.caption("No active Teacher accounts exist yet -- create one on the User Management page.")
        return

    assigned_ids = {t["user_id"] for t in assigned_teachers}
    unassigned_teachers = [t for t in all_teachers if t["user_id"] not in assigned_ids]

    assign_col, unassign_col = st.columns(2)
    with assign_col:
        if unassigned_teachers:
            teacher_labels = {t["username"]: t["user_id"] for t in unassigned_teachers}
            pick_label = st.selectbox(
                "Assign a teacher", options=list(teacher_labels.keys()), key=f"assign_pick_{subject_code}",
            )
            if st.button("Assign", key=f"assign_btn_{subject_code}"):
                try:
                    assign_teacher_to_subject(teacher_labels[pick_label], subject_code, acting_user)
                    st.success(f"'{pick_label}' assigned to '{subject_code}'.")
                    st.rerun()
                except (RecordNotFoundError, DuplicateRecordError) as error:
                    st.error(str(error))
        else:
            st.caption("Every active teacher is already assigned.")

    with unassign_col:
        if assigned_teachers:
            teacher_labels = {t["username"]: t["user_id"] for t in assigned_teachers}
            pick_label = st.selectbox(
                "Unassign a teacher", options=list(teacher_labels.keys()), key=f"unassign_pick_{subject_code}",
            )
            if st.button("Unassign", key=f"unassign_btn_{subject_code}"):
                try:
                    unassign_teacher_from_subject(teacher_labels[pick_label], subject_code, acting_user)
                    st.success(f"'{pick_label}' unassigned from '{subject_code}'.")
                    st.rerun()
                except RecordNotFoundError as error:
                    st.error(str(error))
