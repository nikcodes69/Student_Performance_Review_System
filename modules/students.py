"""
modules/students.py
====================
Student management: create, read, update, and soft-delete (deactivate)
student records, plus the Streamlit page for it.

RBAC IN THIS FILE:
  - WRITE operations (create_student, update_student, deactivate_student,
    reactivate_student) call auth.check_permission() as their very first
    line, restricting them to Admin. This is enforced at the business-logic
    level, not just the page level -- see modules/auth.py's
    check_permission() docstring for why that matters.
  - READ operations (get_student, list_students) do NOT check permissions
    themselves. This is a deliberate choice: the assignment's own wording
    is "role-based access control enforced on every PAGE" -- so the real
    gate is render_students_page() calling auth.require_role() once, at
    the top. Reads are lower-risk than writes and are exactly the kind of
    thing OTHER pages legitimately need too -- and indeed do:
    modules/analytics.py, modules/ml_predictions.py, and
    utils/pdf_generator.py all call get_student()/list_students() directly,
    confirming this was the right call. Keeping these reads permission-
    agnostic meant those modules could call them directly instead of having
    to fake an "acting user" just to read data. If this application ever
    grew a public API with entry points
    OTHER than Streamlit pages, each read would need its own check too --
    but as long as Streamlit pages remain the only entry point, and every
    page that exposes student data gates itself with require_role() first,
    this is safe.

EVERY WRITE IS AUDITED: create_student/update_student/deactivate_student/
reactivate_student all build TWO statements -- the real data change, and
an audit_log entry describing it -- and run them together through
database.db_manager.execute_transaction(), so they can never happen apart
(see modules/audit.py's module docstring for the full explanation).
"""

from datetime import datetime

import streamlit as st

import config
from database.db_manager import execute_transaction, fetch_all, fetch_one
from modules import auth
from modules.audit import build_audit_entry
from utils.exceptions import DuplicateRecordError, RecordNotFoundError, ValidationError
from utils.bulk_import import render_bulk_import
from utils.logger import get_logger
from utils.table_view import render_data_table
from utils.validators import (
    validate_admission_year,
    validate_branch,
    validate_email,
    validate_name,
    validate_phone,
    validate_roll_no,
    validate_semester,
)

logger = get_logger(__name__)

STUDENT_COLUMNS = (
    "roll_no, name, semester, branch, email, phone, admission_year, "
    "is_active, created_at, updated_at"
)


# ---------------------------------------------------------------------------
# WRITE OPERATIONS (Admin only)
# ---------------------------------------------------------------------------

def create_student(
    roll_no: str,
    name: str,
    semester: int,
    branch: str,
    email: str,
    phone: str,
    admission_year: int,
    acting_user: dict,
) -> str:
    """
    Create a new student record.

    Args:
        roll_no: The student's roll number (becomes the primary key).
        name: Full name.
        semester: Current semester (1-8).
        branch: Program/branch, e.g. "BCA".
        email: Email address.
        phone: Phone number.
        admission_year: Year of admission.
        acting_user: The logged-in user performing this action (dict with
            "user_id" and "role"), used for the permission check and the
            audit log entry.

    Returns:
        The new student's roll_no (normalised, e.g. uppercased).

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if any field fails validation.
        DuplicateRecordError: if roll_no is already in use.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    roll_no = validate_roll_no(roll_no)
    name = validate_name(name)
    semester = validate_semester(semester)
    branch = validate_branch(branch)
    email = validate_email(email)
    phone = validate_phone(phone)
    admission_year = validate_admission_year(admission_year)

    existing = fetch_one("SELECT roll_no FROM students WHERE roll_no = ?", (roll_no,))
    if existing is not None:
        raise DuplicateRecordError(f"A student with roll number '{roll_no}' already exists.")

    insert_statement = (
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (roll_no, name, semester, branch, email, phone, admission_year),
    )
    new_value = {
        "roll_no": roll_no, "name": name, "semester": semester, "branch": branch,
        "email": email, "phone": phone, "admission_year": admission_year,
    }
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_INSERT, "students", roll_no,
        old_value=None, new_value=new_value,
    )

    execute_transaction([insert_statement, audit_statement])
    list_students.clear()  # invalidate the cached list -- see list_students()'s docstring
    logger.info("Student '%s' created by user_id=%s.", roll_no, acting_user["user_id"])
    return roll_no


def update_student(
    roll_no: str,
    acting_user: dict,
    name: str | None = None,
    semester: int | None = None,
    branch: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    admission_year: int | None = None,
) -> None:
    """
    Update one or more fields of an existing student record. Only fields
    passed as something other than None are changed -- e.g. calling
    update_student("BCA001", user, phone="9812345678") changes only the
    phone number, leaving every other field untouched.

    HOW THE UPDATE STATEMENT IS BUILT SAFELY: like modules/audit.py's
    get_audit_logs(), the list of "column = ?" fragments is assembled by
    Python control flow (which arguments were provided), never from the
    VALUES themselves -- every value still flows through the params tuple
    and a "?" placeholder.

    Args:
        roll_no: The student to update.
        acting_user: The logged-in user performing this action.
        name, semester, branch, email, phone, admission_year: New values;
            leave as None to keep the existing value.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if a provided field fails validation, or if no
            fields at all were provided.
        RecordNotFoundError: if roll_no does not exist.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    roll_no = validate_roll_no(roll_no)

    existing = get_student(roll_no, include_inactive=True)  # raises RecordNotFoundError if missing

    set_clauses = []
    params: list = []
    new_value = {}

    if name is not None:
        name = validate_name(name)
        set_clauses.append("name = ?")
        params.append(name)
        new_value["name"] = name

    if semester is not None:
        semester = validate_semester(semester)
        set_clauses.append("semester = ?")
        params.append(semester)
        new_value["semester"] = semester

    if branch is not None:
        branch = validate_branch(branch)
        set_clauses.append("branch = ?")
        params.append(branch)
        new_value["branch"] = branch

    if email is not None:
        email = validate_email(email)
        set_clauses.append("email = ?")
        params.append(email)
        new_value["email"] = email

    if phone is not None:
        phone = validate_phone(phone)
        set_clauses.append("phone = ?")
        params.append(phone)
        new_value["phone"] = phone

    if admission_year is not None:
        admission_year = validate_admission_year(admission_year)
        set_clauses.append("admission_year = ?")
        params.append(admission_year)
        new_value["admission_year"] = admission_year

    if not set_clauses:
        raise ValidationError("No fields were provided to update.")

    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
    params.append(roll_no)

    update_statement = (
        f"UPDATE students SET {', '.join(set_clauses)} WHERE roll_no = ?",
        tuple(params),
    )
    # old_value only records the fields that are actually changing, read
    # from the row we fetched BEFORE this update -- a focused "diff",
    # rather than dumping every unrelated column into the audit entry.
    old_value = {field: existing[field] for field in new_value}
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "students", roll_no,
        old_value=old_value, new_value=new_value,
    )

    execute_transaction([update_statement, audit_statement])
    list_students.clear()  # invalidate the cached list -- see list_students()'s docstring
    logger.info(
        "Student '%s' updated by user_id=%s. Fields changed: %s",
        roll_no, acting_user["user_id"], list(new_value.keys()),
    )


def deactivate_student(roll_no: str, acting_user: dict) -> None:
    """
    Soft-delete a student: sets is_active = 0. The row is never removed
    with SQL DELETE -- see database/db_setup.py's students table for why
    academic records are never hard-deleted in this system.

    Args:
        roll_no: The student to deactivate.
        acting_user: The logged-in user performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if roll_no does not exist.
        ValidationError: if the student is already inactive.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    roll_no = validate_roll_no(roll_no)

    existing = get_student(roll_no, include_inactive=True)
    if not existing["is_active"]:
        raise ValidationError(f"Student '{roll_no}' is already inactive.")

    update_statement = (
        "UPDATE students SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE roll_no = ?",
        (roll_no,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_SOFT_DELETE, "students", roll_no,
        old_value={"is_active": 1}, new_value={"is_active": 0},
    )

    execute_transaction([update_statement, audit_statement])
    list_students.clear()  # invalidate the cached list -- see list_students()'s docstring
    logger.info("Student '%s' deactivated by user_id=%s.", roll_no, acting_user["user_id"])


def reactivate_student(roll_no: str, acting_user: dict) -> None:
    """
    Reverse a soft-delete: sets is_active back to 1. Recorded as a normal
    AUDIT_UPDATE (not AUDIT_SOFT_DELETE, which specifically means
    deactivation) since this is undoing that state, not repeating it.

    Args:
        roll_no: The student to reactivate.
        acting_user: The logged-in user performing this action.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if roll_no does not exist.
        ValidationError: if the student is already active.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    roll_no = validate_roll_no(roll_no)

    existing = get_student(roll_no, include_inactive=True)
    if existing["is_active"]:
        raise ValidationError(f"Student '{roll_no}' is already active.")

    update_statement = (
        "UPDATE students SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE roll_no = ?",
        (roll_no,),
    )
    audit_statement = build_audit_entry(
        acting_user["user_id"], config.AUDIT_UPDATE, "students", roll_no,
        old_value={"is_active": 0}, new_value={"is_active": 1},
    )

    execute_transaction([update_statement, audit_statement])
    list_students.clear()  # invalidate the cached list -- see list_students()'s docstring
    logger.info("Student '%s' reactivated by user_id=%s.", roll_no, acting_user["user_id"])


def promote_students(from_semester: int, acting_user: dict) -> list[str]:
    """
    Advance every ACTIVE student currently in from_semester to
    from_semester + 1, all at once -- e.g. moving an entire class from
    semester 1 to semester 2 at the start of a new term, instead of
    editing each student record by hand.

    WHY THIS IS ONE ATOMIC TRANSACTION FOR THE WHOLE BATCH, UNLIKE
    utils/bulk_import.py'S PER-ROW COMMIT (WHERE ONE BAD ROW DOES NOT
    STOP THE OTHERS): bulk import rows are independently validated
    spreadsheet data, where a typo in one row genuinely has nothing to
    do with any other row succeeding or failing. Promotion has no such
    per-student validation risk -- every matching student is promoted by
    the exact same simple, uniform rule (their current semester plus
    one), driven entirely from data already known to be consistent (the
    SELECT below only ever returns active students actually in
    from_semester). A partial promotion here would be confusing, not
    useful -- "why did half the class move to semester 2 and the other
    half didn't" has no good answer -- so this either promotes everyone
    matching, or (if anything at all went wrong) promotes no one.

    WHY A STUDENT IN config.MAX_SEMESTER CANNOT BE PROMOTED: there is no
    semester beyond the maximum for them to move into. A student
    finishing the final semester is a "graduation" event, not a
    "promotion" -- deliberately a different, out-of-scope action from
    this function (nothing here deactivates or archives a graduating
    student).

    Args:
        from_semester: The semester every matching active student is
            currently in.
        acting_user: The logged-in Admin performing this action.

    Returns:
        The roll_no of every student that was promoted, in the same
        order list_students() returns them (possibly empty, if no active
        student is currently in from_semester).

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if from_semester is outside the valid range, or
            is already config.MAX_SEMESTER.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))
    from_semester = validate_semester(from_semester)

    if from_semester >= config.MAX_SEMESTER:
        raise ValidationError(
            f"Cannot promote students out of semester {config.MAX_SEMESTER} -- "
            "there is no semester beyond the maximum."
        )
    to_semester = from_semester + 1

    matching_students = list_students(semester=from_semester)
    if not matching_students:
        return []

    statements = []
    for student in matching_students:
        statements.append((
            "UPDATE students SET semester = ?, updated_at = CURRENT_TIMESTAMP WHERE roll_no = ?",
            (to_semester, student["roll_no"]),
        ))
        statements.append(build_audit_entry(
            acting_user["user_id"], config.AUDIT_UPDATE, "students", student["roll_no"],
            old_value={"semester": from_semester}, new_value={"semester": to_semester},
        ))

    execute_transaction(statements)
    list_students.clear()  # invalidate the cached list -- see list_students()'s docstring
    logger.info(
        "Promoted %s student(s) from semester %s to %s by user_id=%s.",
        len(matching_students), from_semester, to_semester, acting_user["user_id"],
    )
    return [student["roll_no"] for student in matching_students]


# ---------------------------------------------------------------------------
# READ OPERATIONS (see module docstring for why these are not role-gated)
# ---------------------------------------------------------------------------

def get_student(roll_no: str, include_inactive: bool = False) -> dict:
    """
    Fetch one student by roll number.

    Args:
        roll_no: The roll number to look up (case-insensitive -- it is
            normalised the same way validate_roll_no() normalises it at
            creation time, so lookups always match regardless of case).
        include_inactive: If False (default), a deactivated student is
            treated as not found. Pass True to fetch it anyway (e.g. to
            show it on an "inactive students" view, or to reactivate it).

    Returns:
        A dict with all of STUDENT_COLUMNS.

    Raises:
        RecordNotFoundError: if no matching (active, unless
            include_inactive) student exists.
    """
    roll_no = validate_roll_no(roll_no)

    row = fetch_one(f"SELECT {STUDENT_COLUMNS} FROM students WHERE roll_no = ?", (roll_no,))

    if row is None:
        raise RecordNotFoundError(f"No student found with roll number '{roll_no}'.")

    if not include_inactive and not row["is_active"]:
        raise RecordNotFoundError(f"No active student found with roll number '{roll_no}'.")

    return dict(row)


@st.cache_data(ttl=60)
def list_students(
    semester: int | None = None,
    branch: str | None = None,
    include_inactive: bool = False,
) -> list[dict]:
    """
    Fetch multiple students, optionally filtered.

    CACHED for 60 seconds: this function is called on nearly every page in
    the app (as the source of a student picker dropdown, a data table, or
    both), so without caching it would re-run this query on every single
    click anywhere in the app -- Streamlit reruns the whole script on every
    interaction. @st.cache_data keys its cache by this function's actual
    argument values, so list_students(semester=3) and list_students() are
    cached separately, exactly as they should be.

    STAYING FRESH AFTER A WRITE: the 60-second TTL alone would mean a
    newly-created student might not appear in a dropdown for up to a
    minute -- unacceptable, since a Teacher creating a student usually
    wants to immediately select them for marks entry. So every write
    function below (create_student, update_student, deactivate_student,
    reactivate_student) calls `list_students.clear()` immediately after a
    successful change, wiping the cache instantly rather than waiting for
    the TTL. The TTL is just a safety net in case some future write path
    ever forgets to call .clear().

    Args:
        semester: If given, only students in this semester.
        branch: If given, only students in this branch.
        include_inactive: If False (default), deactivated students are
            excluded.

    Returns:
        A list of dicts (possibly empty), ordered by roll_no.
    """
    query = f"SELECT {STUDENT_COLUMNS} FROM students"
    conditions = []
    params: list = []

    if not include_inactive:
        conditions.append("is_active = 1")
    if semester is not None:
        conditions.append("semester = ?")
        params.append(semester)
    if branch is not None:
        conditions.append("branch = ?")
        params.append(branch)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY roll_no"

    rows = fetch_all(query, tuple(params))
    return [dict(row) for row in rows]


def _validate_bulk_student_row(row: dict) -> None:
    """
    Validate one row of a bulk student import, run during the PREVIEW
    step (before anything is committed) -- see utils/bulk_import.py's
    module docstring for why previewing matters here. Reuses the exact
    same utils/validators.py functions AND the "does this roll_no
    already exist" check create_student() itself performs, so a row from
    a spreadsheet is held to identical rules as one typed into the "Add
    New Student" form by hand -- just checked here EARLY, so the preview
    can already show "already exists" instead of only discovering that
    at commit time.

    Args:
        row: One row from the uploaded file, as a dict of column name ->
            string cell value (every value arrives as text -- see
            utils.bulk_import.parse_uploaded_file()'s docstring for why).

    Raises:
        ValidationError: if any field is missing, malformed, or not the
            expected type (e.g. "semester" is not a whole number).
        DuplicateRecordError: if roll_no already belongs to an existing
            student record.
    """
    roll_no = validate_roll_no(row.get("roll_no", ""))
    validate_name(row.get("name", ""))

    try:
        semester = int(row.get("semester", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"Semester must be a whole number, got '{row.get('semester')}'.")
    validate_semester(semester)

    validate_branch(row.get("branch", ""))
    validate_email(row.get("email", ""))
    validate_phone(row.get("phone", ""))

    try:
        admission_year = int(row.get("admission_year", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"Admission year must be a whole number, got '{row.get('admission_year')}'.")
    validate_admission_year(admission_year)

    existing = fetch_one("SELECT roll_no FROM students WHERE roll_no = ?", (roll_no,))
    if existing is not None:
        raise DuplicateRecordError(f"A student with roll number '{roll_no}' already exists.")


def _render_phone_input(key: str, existing_full_number: str | None = None) -> str:
    """
    Render a phone number field as a disabled "+977" box next to an
    editable 10-digit local-number box -- see
    utils.validators.validate_phone()'s docstring for why the country
    code is fixed by the system rather than typed by the user. Shared
    between the Add and Edit forms below so this layout is defined in
    exactly one place, not copy-pasted twice.

    Args:
        key: Unique Streamlit widget key for this field (every widget in
            a form needs a distinct key, same as every other field here).
        existing_full_number: An already-stored "+977XXXXXXXXXX" value
            to pre-fill (Edit form only) -- the "+977" prefix is
            stripped off before showing it, since the editable box
            should only ever contain the part the user is allowed to
            change.

    Returns:
        Whatever the user typed into the local-number box, UNVALIDATED
        -- the caller still passes this through validate_phone() before
        it reaches the database, exactly like every other field here.
    """
    local_number = ""
    if existing_full_number:
        local_number = existing_full_number.removeprefix(config.PHONE_COUNTRY_CODE)

    prefix_col, number_col = st.columns([1, 3])
    with prefix_col:
        st.text_input(
            "Country Code", value=config.PHONE_COUNTRY_CODE, disabled=True, key=f"{key}_prefix",
        )
    with number_col:
        return st.text_input(
            "Phone Number", value=local_number, placeholder="98XXXXXXXX", key=key,
        )


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_students_page() -> None:
    """
    Streamlit page: student list for Admin and Teacher, with create/edit/
    deactivate/reactivate controls visible to Admin only.
    """
    user = auth.require_role(config.ROLE_ADMIN, config.ROLE_TEACHER)
    is_admin = user["role"] == config.ROLE_ADMIN

    st.title("Student Management")

    if is_admin:
        with st.expander("Add New Student"):
            with st.form("create_student_form", clear_on_submit=True):
                new_roll_no = st.text_input("Roll Number")
                new_name = st.text_input("Name")
                new_semester = st.number_input(
                    "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER, step=1
                )
                new_branch = st.text_input("Branch", value="BCA")
                new_email = st.text_input("Email")
                new_phone = _render_phone_input("new_phone")
                new_admission_year = st.number_input(
                    "Admission Year",
                    min_value=config.ADMISSION_YEAR_MIN,
                    max_value=datetime.now().year,
                    value=datetime.now().year,
                    step=1,
                )
                create_submitted = st.form_submit_button("Create Student")

            if create_submitted:
                try:
                    created_roll_no = create_student(
                        new_roll_no, new_name, int(new_semester), new_branch,
                        new_email, new_phone, int(new_admission_year), user,
                    )
                    # st.toast(), not st.success() -- see app.py's
                    # render_role_login_form() for why, wherever a message
                    # is immediately followed by st.rerun().
                    st.toast(f"Student '{created_roll_no}' created.", icon=":material/check_circle:")
                    st.rerun()
                except (ValidationError, DuplicateRecordError) as error:
                    st.error(str(error))

        with st.expander("Bulk Import Students (CSV/Excel)"):
            render_bulk_import(
                key_prefix="students_import",
                required_columns=("roll_no", "name", "semester", "branch", "email", "phone", "admission_year"),
                key_columns=("roll_no",),
                validate_row=_validate_bulk_student_row,
                commit_row=lambda row: create_student(
                    row["roll_no"], row["name"], int(row["semester"]), row["branch"],
                    row["email"], row["phone"], int(row["admission_year"]), user,
                ),
            )

    st.subheader("Student List")
    show_inactive = st.checkbox("Show deactivated students", value=False) if is_admin else False
    students = list_students(include_inactive=show_inactive)

    if not students:
        st.info("No students found.")
        return

    render_data_table(students, key_prefix="students_table", filename_prefix="students")

    if not is_admin:
        return  # Teachers can view the list above, but not edit/deactivate.

    st.subheader("Edit / Deactivate Student")
    roll_no_choice = st.selectbox("Select a student", options=[s["roll_no"] for s in students])
    student = next(s for s in students if s["roll_no"] == roll_no_choice)

    with st.form("edit_student_form"):
        edit_name = st.text_input("Name", value=student["name"])
        edit_semester = st.number_input(
            "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
            value=student["semester"], step=1,
        )
        edit_branch = st.text_input("Branch", value=student["branch"])
        edit_email = st.text_input("Email", value=student["email"])
        edit_phone = _render_phone_input("edit_phone", existing_full_number=student["phone"])
        edit_admission_year = st.number_input(
            "Admission Year", min_value=config.ADMISSION_YEAR_MIN,
            max_value=datetime.now().year, value=student["admission_year"], step=1,
        )
        update_submitted = st.form_submit_button("Save Changes")

    if update_submitted:
        try:
            update_student(
                roll_no_choice, user, name=edit_name, semester=int(edit_semester),
                branch=edit_branch, email=edit_email, phone=edit_phone,
                admission_year=int(edit_admission_year),
            )
            st.toast("Student updated.", icon=":material/check_circle:")
            st.rerun()
        except (ValidationError, RecordNotFoundError) as error:
            st.error(str(error))

    deactivate_col, reactivate_col = st.columns(2)
    with deactivate_col:
        if student["is_active"] and st.button("Deactivate this student"):
            try:
                deactivate_student(roll_no_choice, user)
                st.toast(f"Student '{roll_no_choice}' deactivated.", icon=":material/check_circle:")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))
    with reactivate_col:
        if not student["is_active"] and st.button("Reactivate this student"):
            try:
                reactivate_student(roll_no_choice, user)
                st.toast(f"Student '{roll_no_choice}' reactivated.", icon=":material/check_circle:")
                st.rerun()
            except ValidationError as error:
                st.error(str(error))

    st.divider()
    st.subheader("Semester Promotion")
    st.caption(
        "Advance every active student in one semester to the next -- e.g. at the "
        "start of a new term. This affects every matching student at once."
    )
    promote_from = st.selectbox(
        "Promote students from semester",
        options=range(config.MIN_SEMESTER, config.MAX_SEMESTER),  # MAX_SEMESTER itself has nowhere to promote TO
        key="promote_from_semester",
    )
    affected_students = list_students(semester=int(promote_from))
    if not affected_students:
        st.info(f"No active students currently in semester {promote_from}.")
    else:
        st.write(
            f"This will move **{len(affected_students)}** student(s) from semester "
            f"{promote_from} to semester {promote_from + 1}:"
        )
        st.dataframe(
            [{"Roll No": s["roll_no"], "Name": s["name"]} for s in affected_students],
            use_container_width=True, hide_index=True,
        )
        confirm_promotion = st.checkbox(
            f"I understand this will move all {len(affected_students)} student(s) above "
            f"to semester {promote_from + 1}.",
            key="confirm_promotion",
        )
        if st.button("Promote These Students", type="primary", disabled=not confirm_promotion):
            try:
                promoted = promote_students(int(promote_from), user)
                st.toast(
                    f"Promoted {len(promoted)} student(s) to semester {promote_from + 1}.",
                    icon=":material/check_circle:",
                )
                st.rerun()
            except ValidationError as error:
                st.error(str(error))
