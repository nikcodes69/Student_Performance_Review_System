"""
modules/subjects.py
====================
Subject/curriculum configuration: create, read, update, and soft-delete
subjects, plus the Streamlit page for it.

This file mirrors modules/students.py's structure and reasoning closely
(same RBAC split between write and read functions, same
build_audit_entry() + execute_transaction() pattern for every write) --
see that file's module docstring for the full explanation of why reads
are not individually permission-checked.

NO PER-SUBJECT MARKS CONFIGURATION: every subject uses the exact same
fixed marks breakdown (config.MAX_INTERNAL_MARKS/MAX_EXTERNAL_MARKS/
MAX_PRACTICAL_MARKS -- 25/50/25, 100 total), so create_subject()/
update_subject() below no longer take max_internal/max_external/
max_practical parameters at all -- there is nothing left to configure
per subject. See database/db_setup.py's migrate_schema() for the
one-time migration that dropped those columns from a database created
before this change.
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, fetch_all, fetch_one
from modules import auth
from modules.audit import build_audit_entry
from modules.teacher_subjects import render_teacher_assignment_section
from utils.bulk_import import render_bulk_import
from utils.exceptions import DuplicateRecordError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.table_view import render_data_table
from utils.validators import (
    validate_credits,
    validate_semester,
    validate_subject_code,
    validate_subject_name,
)

logger = get_logger(__name__)

SUBJECT_COLUMNS = "subject_code, name, semester, credits, is_active, created_at, updated_at"


# ---------------------------------------------------------------------------
# WRITE OPERATIONS (Admin only)
# ---------------------------------------------------------------------------

def create_subject(
    subject_code: str,
    name: str,
    semester: int,
    credits: int,
    acting_user: dict,
) -> str:
    """
    Create a new subject.

    Args:
        subject_code: Unique subject code (becomes the primary key).
        name: Subject/course name.
        semester: Which semester this subject belongs to (1-8).
        credits: Credit value (config.MIN_CREDITS..MAX_CREDITS).
        acting_user: The logged-in user performing this action.

    Returns:
        The new subject's subject_code (normalised, e.g. uppercased).

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if any field fails validation.
        DuplicateRecordError: if subject_code is already in use.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    subject_code = validate_subject_code(subject_code)
    name = validate_subject_name(name)
    semester = validate_semester(semester)
    credits = validate_credits(credits)

    existing = fetch_one("SELECT subject_code FROM subjects WHERE subject_code = ?", (subject_code,))
    if existing is not None:
        raise DuplicateRecordError(f"A subject with code '{subject_code}' already exists.")

    insert_statement = (
        "INSERT INTO subjects (subject_code, name, semester, credits) VALUES (?, ?, ?, ?)",
        (subject_code, name, semester, credits),
    )
    new_value = {"subject_code": subject_code, "name": name, "semester": semester, "credits": credits}
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "subjects", subject_code,
        old_value=None, new_value=new_value,
    )

    execute_transaction([insert_statement, audit_statement])
    list_subjects.clear()  # invalidate the cached list -- see list_subjects()'s docstring
    logger.info("Subject '%s' created by user_id=%s.", subject_code, acting_user["user_id"])
    return subject_code


def update_subject(
    subject_code: str,
    acting_user: dict,
    name: str | None = None,
    semester: int | None = None,
    credits: int | None = None,
) -> None:
    """
    Update one or more fields of an existing subject. Only fields passed
    as something other than None are changed.

    Args:
        subject_code: The subject to update.
        acting_user: The logged-in user performing this action.
        name, semester, credits: New values; leave as None to keep the
            existing value.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if a provided field fails validation, or if no
            fields at all were provided.
        RecordNotFoundError: if subject_code does not exist.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    subject_code = validate_subject_code(subject_code)

    existing = get_subject(subject_code, include_inactive=True)

    set_clauses = []
    params: list = []
    new_value = {}

    if name is not None:
        name = validate_subject_name(name)
        set_clauses.append("name = ?")
        params.append(name)
        new_value["name"] = name

    if semester is not None:
        semester = validate_semester(semester)
        set_clauses.append("semester = ?")
        params.append(semester)
        new_value["semester"] = semester

    if credits is not None:
        credits = validate_credits(credits)
        set_clauses.append("credits = ?")
        params.append(credits)
        new_value["credits"] = credits

    if not set_clauses:
        raise ValidationError("No fields were provided to update.")

    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    params.append(subject_code)

    update_statement = (
        f"UPDATE subjects SET {', '.join(set_clauses)} WHERE subject_code = ?",
        tuple(params),
    )
    old_value = {field: existing[field] for field in new_value}
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "subjects", subject_code,
        old_value=old_value, new_value=new_value,
    )

    execute_transaction([update_statement, audit_statement])
    list_subjects.clear()  # invalidate the cached list -- see list_subjects()'s docstring
    logger.info(
        "Subject '%s' updated by user_id=%s. Fields changed: %s",
        subject_code, acting_user["user_id"], list(new_value.keys()),
    )


def deactivate_subject(subject_code: str, acting_user: dict) -> None:
    """
    Soft-delete a subject: sets is_active = 0. Never uses SQL DELETE.

    Args:
        subject_code: The subject to deactivate.
        acting_user: The logged-in user performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if subject_code does not exist.
        ValidationError: if the subject is already inactive.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    subject_code = validate_subject_code(subject_code)

    existing = get_subject(subject_code, include_inactive=True)
    if not existing["is_active"]:
        raise ValidationError(f"Subject '{subject_code}' is already inactive.")

    update_statement = (
        "UPDATE subjects SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE subject_code = ?",
        (subject_code,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_SOFT_DELETE, "subjects", subject_code,
        old_value={"is_active": 1}, new_value={"is_active": 0},
    )

    execute_transaction([update_statement, audit_statement])
    list_subjects.clear()  # invalidate the cached list -- see list_subjects()'s docstring
    logger.info("Subject '%s' deactivated by user_id=%s.", subject_code, acting_user["user_id"])


def reactivate_subject(subject_code: str, acting_user: dict) -> None:
    """
    Reverse a soft-delete: sets is_active back to 1.

    Args:
        subject_code: The subject to reactivate.
        acting_user: The logged-in user performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if subject_code does not exist.
        ValidationError: if the subject is already active.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    subject_code = validate_subject_code(subject_code)

    existing = get_subject(subject_code, include_inactive=True)
    if existing["is_active"]:
        raise ValidationError(f"Subject '{subject_code}' is already active.")

    update_statement = (
        "UPDATE subjects SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE subject_code = ?",
        (subject_code,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "subjects", subject_code,
        old_value={"is_active": 0}, new_value={"is_active": 1},
    )

    execute_transaction([update_statement, audit_statement])
    list_subjects.clear()  # invalidate the cached list -- see list_subjects()'s docstring
    logger.info("Subject '%s' reactivated by user_id=%s.", subject_code, acting_user["user_id"])


# ---------------------------------------------------------------------------
# READ OPERATIONS (see modules/students.py's module docstring for why
# these are not individually role-gated)
# ---------------------------------------------------------------------------

def get_subject(subject_code: str, include_inactive: bool = False) -> dict:
    """
    Fetch one subject by subject_code.

    Args:
        subject_code: The subject code to look up (case-insensitive).
        include_inactive: If False (default), a deactivated subject is
            treated as not found.

    Returns:
        A dict with all of SUBJECT_COLUMNS.

    Raises:
        RecordNotFoundError: if no matching (active, unless
            include_inactive) subject exists.
    """
    subject_code = validate_subject_code(subject_code)

    row = fetch_one(f"SELECT {SUBJECT_COLUMNS} FROM subjects WHERE subject_code = ?", (subject_code,))

    if row is None:
        raise RecordNotFoundError(f"No subject found with code '{subject_code}'.")

    if not include_inactive and not row["is_active"]:
        raise RecordNotFoundError(f"No active subject found with code '{subject_code}'.")

    return dict(row)


@st.cache_data(ttl=60)
def list_subjects(semester: int | None = None, include_inactive: bool = False) -> list[dict]:
    """
    Fetch multiple subjects, optionally filtered by semester.

    CACHED for 60 seconds, with every write function below calling
    list_subjects.clear() immediately on a successful change -- see
    modules/students.py's list_students() docstring for the full
    reasoning (this function is used across just as many pages: Subjects,
    Marks Entry, Attendance, ML predictions, Report Card).

    Args:
        semester: If given, only subjects in this semester.
        include_inactive: If False (default), deactivated subjects are
            excluded.

    Returns:
        A list of dicts (possibly empty), ordered by subject_code.
    """
    query = f"SELECT {SUBJECT_COLUMNS} FROM subjects"
    conditions = []
    params: list = []

    if not include_inactive:
        conditions.append("is_active = 1")
    if semester is not None:
        conditions.append("semester = ?")
        params.append(semester)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY subject_code"

    rows = fetch_all(query, tuple(params))
    return [dict(row) for row in rows]


def _validate_bulk_subject_row(row: dict) -> None:
    """
    Validate one row of a bulk subject import, run during the PREVIEW
    step -- see modules/students.py's _validate_bulk_student_row() for
    the identical reasoning (and utils/bulk_import.py's module docstring
    for why previewing before committing matters). Reuses the exact same
    validators AND the "does this subject_code already exist" check
    create_subject() itself performs.

    Args:
        row: One row from the uploaded file, as a dict of column name ->
            string cell value.

    Raises:
        ValidationError: if any field is missing, malformed, or not the
            expected type.
        DuplicateRecordError: if subject_code already belongs to an
            existing subject.
    """
    subject_code = validate_subject_code(row.get("subject_code", ""))
    validate_subject_name(row.get("name", ""))

    try:
        semester = int(row.get("semester", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"Semester must be a whole number, got '{row.get('semester')}'.")
    validate_semester(semester)

    try:
        credits = int(row.get("credits", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"Credits must be a whole number, got '{row.get('credits')}'.")
    validate_credits(credits)

    existing = fetch_one("SELECT subject_code FROM subjects WHERE subject_code = ?", (subject_code,))
    if existing is not None:
        raise DuplicateRecordError(f"A subject with code '{subject_code}' already exists.")


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_subjects_page() -> None:
    """
    Streamlit page: subject list for Admin and Teacher, with create/edit/
    deactivate/reactivate controls visible to Admin only.
    """
    user = auth.require_role(config.ROLE_ADMIN, config.ROLE_TEACHER)
    is_admin = user["role"] == config.ROLE_ADMIN

    st.title("Subject Configuration")
    st.caption(
        f"Every subject uses the same fixed marks breakdown: Internal (max "
        f"{config.MAX_INTERNAL_MARKS}) + External (max {config.MAX_EXTERNAL_MARKS}) + "
        f"Practical (max {config.MAX_PRACTICAL_MARKS}) = {config.MAX_TOTAL_MARKS} total."
    )

    if is_admin:
        with st.expander("Add New Subject"):
            with st.form("create_subject_form", clear_on_submit=True):
                new_code = st.text_input("Subject Code")
                new_name = st.text_input("Subject Name")
                new_semester = st.number_input(
                    "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER, step=1
                )
                new_credits = st.number_input(
                    "Credits", min_value=config.MIN_CREDITS, max_value=config.MAX_CREDITS, step=1
                )
                create_submitted = st.form_submit_button("Create Subject")

            if create_submitted:
                try:
                    created_code = create_subject(
                        new_code, new_name, int(new_semester), int(new_credits), user,
                    )
                    # st.toast(), not st.success() -- see app.py's
                    # render_role_login_form() for why, wherever a message
                    # is immediately followed by st.rerun().
                    st.toast(f"Subject '{created_code}' created.", icon=":material/check_circle:")
                    st.rerun()
                except (ValidationError, DuplicateRecordError) as error:
                    st.error(str(error))

        with st.expander("Bulk Import Subjects (CSV/Excel)"):
            render_bulk_import(
                key_prefix="subjects_import",
                required_columns=("subject_code", "name", "semester", "credits"),
                key_columns=("subject_code",),
                validate_row=_validate_bulk_subject_row,
                commit_row=lambda row: create_subject(
                    row["subject_code"], row["name"], int(row["semester"]), int(row["credits"]), user,
                ),
            )

    st.subheader("Subject List")
    show_inactive = st.checkbox("Show deactivated subjects", value=False) if is_admin else False
    subjects = list_subjects(include_inactive=show_inactive)

    if not subjects:
        st.info("No subjects found.")
        return

    render_data_table(subjects, key_prefix="subjects_table", filename_prefix="subjects")

    if not is_admin:
        return

    st.subheader("Edit / Deactivate Subject")
    code_choice = st.selectbox("Select a subject", options=[s["subject_code"] for s in subjects])
    subject = next(s for s in subjects if s["subject_code"] == code_choice)

    with st.form("edit_subject_form"):
        edit_name = st.text_input("Subject Name", value=subject["name"])
        edit_semester = st.number_input(
            "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
            value=subject["semester"], step=1,
        )
        edit_credits = st.number_input(
            "Credits", min_value=config.MIN_CREDITS, max_value=config.MAX_CREDITS,
            value=subject["credits"], step=1,
        )
        update_submitted = st.form_submit_button("Save Changes")

    if update_submitted:
        try:
            update_subject(
                code_choice, user, name=edit_name, semester=int(edit_semester),
                credits=int(edit_credits),
            )
            st.toast("Subject updated.", icon=":material/check_circle:")
            st.rerun()
        except (ValidationError, RecordNotFoundError) as error:
            st.error(str(error))

    deactivate_col, reactivate_col = st.columns(2)
    with deactivate_col:
        if subject["is_active"] and st.button("Deactivate this subject"):
            try:
                deactivate_subject(code_choice, user)
                st.toast(f"Subject '{code_choice}' deactivated.", icon=":material/check_circle:")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))
    with reactivate_col:
        if not subject["is_active"] and st.button("Reactivate this subject"):
            try:
                reactivate_subject(code_choice, user)
                st.toast(f"Subject '{code_choice}' reactivated.", icon=":material/check_circle:")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))

    st.subheader("Assigned Teachers")
    render_teacher_assignment_section(code_choice, user)
