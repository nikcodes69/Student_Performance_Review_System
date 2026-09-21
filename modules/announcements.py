"""
modules/announcements.py
=========================
In-app announcements: Admin or Teacher posts a short message, optionally
aimed at just one role (e.g. "Teacher-only: submit mid-term marks by
Friday") or left untargeted so every role sees it.

WHY target_role IS NULLABLE RATHER THAN "post once per role": an
announcement meant for everyone (e.g. "College closed Monday for a
public holiday") is ONE real-world event, not three independent ones --
storing it as one row with target_role=NULL keeps that 1:1 correspondence
between "one thing that happened" and "one row", instead of needing three
near-duplicate rows that could drift out of sync if one were edited
without the others (this table has no UPDATE path at all, only
create/deactivate, so drift specifically isn't possible here, but the
one-row-per-real-event principle is the same reasoning applied elsewhere
in this project, e.g. students/subjects).

WHO CAN DEACTIVATE WHAT: Admin can take down any announcement. A Teacher
can only take down their OWN -- otherwise one Teacher could silence
another's message, which is a real permission boundary, not just a UI
nicety (enforced in deactivate_announcement() itself, not only hidden in
the UI -- see modules/auth.py's check_permission() docstring for why that
matters).
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, execute_write, fetch_all, fetch_one
from modules import auth
from modules.audit import build_audit_entry
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.validators import (
    validate_announcement_message,
    validate_announcement_target_role,
    validate_announcement_title,
)

logger = get_logger(__name__)

ANNOUNCEMENT_POST_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)


def create_announcement(
    title: str, message: str, target_role: str | None, acting_user: dict
) -> int:
    """
    Post a new announcement.

    Args:
        title: Short headline.
        message: The announcement body.
        target_role: One of config.VALID_ROLES to aim this at just that
            role, or None for every role.
        acting_user: The logged-in user posting this (must be Admin or Teacher).

    Returns:
        The new announcement_id.

    Raises:
        AuthorizationError: if acting_user's role is not Admin or Teacher.
        ValidationError: if title, message, or target_role fails validation.
    """
    auth.check_permission(acting_user["role"], ANNOUNCEMENT_POST_ROLES)

    title = validate_announcement_title(title)
    message = validate_announcement_message(message)
    target_role = validate_announcement_target_role(target_role)

    insert_statement = (
        "INSERT INTO announcements (title, message, posted_by, target_role) VALUES (?, ?, ?, ?)",
        (title, message, acting_user["user_id"], target_role),
    )
    announcement_id = execute_write(*insert_statement)

    audit_statement = build_audit_entry(
        user_id=acting_user["user_id"],
        action=config.AUDIT_INSERT,
        table_name="announcements",
        record_id=str(announcement_id),
        new_value={"title": title, "target_role": target_role},
    )
    execute_write(*audit_statement)

    logger.info(
        "Announcement #%s posted by user_id=%s (target_role=%s).",
        announcement_id, acting_user["user_id"], target_role,
    )
    return announcement_id


def list_announcements_for_role(role: str, include_inactive: bool = False) -> list[dict]:
    """
    Every announcement visible to `role` -- either untargeted
    (target_role IS NULL) or aimed specifically at this role -- newest
    first.

    Args:
        role: The viewer's role.
        include_inactive: If True, also include deactivated announcements
            (used by the Admin/Teacher management view so a poster can
            see what they've taken down, not just what's currently live).

    Returns:
        A list of dicts: announcement_id, title, message, target_role,
        is_active, created_at, posted_by, posted_by_username.
    """
    query = (
        "SELECT a.announcement_id, a.title, a.message, a.target_role, a.is_active, "
        "a.created_at, a.posted_by, u.username AS posted_by_username "
        "FROM announcements a JOIN users u ON a.posted_by = u.user_id "
        "WHERE (a.target_role IS NULL OR a.target_role = ?)"
    )
    params: list = [role]

    if not include_inactive:
        query += " AND a.is_active = 1"

    # announcement_id DESC as a tiebreaker, not just created_at DESC:
    # created_at has only SECOND resolution, so two announcements posted
    # within the same second (e.g. two calls in a test, or an Admin
    # posting several in quick succession) would otherwise tie and sort
    # in an unspecified order -- the auto-increment id is monotonically
    # increasing and never ties, so it reliably breaks that tie newest-first.
    query += " ORDER BY a.created_at DESC, a.announcement_id DESC"

    return fetch_all(query, tuple(params))


def deactivate_announcement(announcement_id: int, acting_user: dict) -> None:
    """
    Soft-delete an announcement: sets is_active = 0. Never a real SQL
    DELETE -- see this module's docstring and database/db_setup.py's
    students table for why academic-adjacent records in this project are
    never hard-deleted.

    Args:
        announcement_id: The announcement to take down.
        acting_user: The logged-in user performing this action -- must be
            Admin, or the Teacher who originally posted it.

    Raises:
        AuthorizationError: if acting_user is neither Admin nor the
            original poster.
        RecordNotFoundError: if announcement_id does not exist.
        ValidationError: if the announcement is already inactive.
    """
    existing = fetch_one(
        "SELECT announcement_id, posted_by, is_active FROM announcements WHERE announcement_id = ?",
        (announcement_id,),
    )
    if existing is None:
        raise RecordNotFoundError(f"Announcement #{announcement_id} not found.")

    is_own_post = acting_user["user_id"] == existing["posted_by"]
    if acting_user["role"] != config.ROLE_ADMIN and not is_own_post:
        raise AuthorizationError("You can only take down announcements you posted yourself.")

    if not existing["is_active"]:
        raise ValidationError(f"Announcement #{announcement_id} is already inactive.")

    update_statement = (
        "UPDATE announcements SET is_active = 0 WHERE announcement_id = ?",
        (announcement_id,),
    )
    audit_statement = build_audit_entry(
        user_id=acting_user["user_id"],
        action=config.AUDIT_SOFT_DELETE,
        table_name="announcements",
        record_id=str(announcement_id),
        old_value={"is_active": True},
        new_value=None,
    )
    execute_transaction([update_statement, audit_statement])

    logger.info("Announcement #%s deactivated by user_id=%s.", announcement_id, acting_user["user_id"])


def render_announcements_page() -> None:
    """Streamlit page: view announcements (every role), post a new one
    or take one down (Admin/Teacher only)."""
    user = auth.require_role(*config.VALID_ROLES)
    can_post = user["role"] in ANNOUNCEMENT_POST_ROLES

    st.title("Announcements")

    if can_post:
        with st.expander(":material/campaign: Post a new announcement"):
            with st.form("new_announcement_form", clear_on_submit=True):
                title = st.text_input("Title")
                message = st.text_area("Message")
                audience_choice = st.selectbox(
                    "Audience",
                    options=["Everyone", "Admin only", "Teacher only", "Student only"],
                )
                submitted = st.form_submit_button("Post", type="primary")

            if submitted:
                target_role = {
                    "Everyone": None,
                    "Admin only": config.ROLE_ADMIN,
                    "Teacher only": config.ROLE_TEACHER,
                    "Student only": config.ROLE_STUDENT,
                }[audience_choice]
                try:
                    create_announcement(title, message, target_role, user)
                    st.success("Announcement posted.")
                    st.rerun()
                except ValidationError as error:
                    st.error(str(error))

    st.divider()

    announcements = list_announcements_for_role(user["role"])
    if not announcements:
        st.info("No announcements right now.")
        return

    for entry in announcements:
        audience_label = (
            "Everyone" if entry["target_role"] is None else f"{entry['target_role'].capitalize()} only"
        )
        with st.container(border=True):
            st.markdown(f"**{entry['title']}**")
            st.caption(
                f"Posted by {entry['posted_by_username']} on {entry['created_at']} -- {audience_label}"
            )
            st.write(entry["message"])

            can_take_down = (
                user["role"] == config.ROLE_ADMIN or entry["posted_by"] == user["user_id"]
            )
            if can_take_down:
                if st.button(
                    "Take down", key=f"deactivate_announcement_{entry['announcement_id']}",
                ):
                    deactivate_announcement(entry["announcement_id"], user)
                    st.rerun()
