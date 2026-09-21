"""
modules/academic_calendar.py
==============================
The academic calendar: semester start/end dates, exam windows, holidays,
and other scheduled events -- one row per event, visible to every role.

WHY THIS IS ADMIN-ONLY TO WRITE, UNLIKE ANNOUNCEMENTS (WHERE A TEACHER CAN
ALSO POST): an announcement is a day-to-day notice from whoever is
closest to the situation. A calendar entry is official institutional
scheduling -- "semester 3 ends on this date", "the mid-term exam window
is this week" -- the kind of fact that must have exactly ONE source of
truth across the whole institution, not one a given Teacher could set
independently for their own subjects. Keeping this Admin-only avoids two
Teachers (or a Teacher and the Admin) publishing conflicting dates for
the same real-world event.

DATES ARE ORDERED CHRONOLOGICALLY (start_date ASC), NOT NEWEST-FIRST:
unlike modules/announcements.py's feed (where "what was just posted"
matters most) or modules/teacher_remarks.py's history (where "the most
recent note" matters most), a calendar is naturally browsed by WHEN an
event happens, not when it was entered into the system -- so
list_calendar_events() sorts by start_date, not created_at.
"""

import streamlit as st

import config
from database.db_manager import execute_transaction, execute_write, fetch_all, fetch_one
from modules import auth
from modules.audit import build_audit_entry
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.validators import (
    validate_calendar_date,
    validate_calendar_description,
    validate_calendar_event_type,
    validate_calendar_title,
)

logger = get_logger(__name__)

CALENDAR_EVENT_TYPE_LABELS = {
    config.CALENDAR_EVENT_SEMESTER_START: "Semester Start",
    config.CALENDAR_EVENT_SEMESTER_END: "Semester End",
    config.CALENDAR_EVENT_EXAM: "Exam",
    config.CALENDAR_EVENT_HOLIDAY: "Holiday",
    config.CALENDAR_EVENT_OTHER: "Other",
}


def create_calendar_event(
    title: str, description: str | None, event_type: str,
    start_date: str, end_date: str | None, acting_user: dict,
) -> int:
    """
    Publish a new academic calendar event.

    Args:
        title: Short headline (e.g. "Mid-term Examinations").
        description: Optional extra detail, or None.
        event_type: One of config.CALENDAR_EVENT_TYPES.
        start_date: 'YYYY-MM-DD'. For a single-day event, this is the
            only date needed.
        end_date: 'YYYY-MM-DD', or None for a single-day event. Must not
            be earlier than start_date.
        acting_user: The logged-in user publishing this (must be Admin).

    Returns:
        The new event_id.

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        ValidationError: if any field fails validation, or end_date is
            earlier than start_date.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    title = validate_calendar_title(title)
    description = validate_calendar_description(description)
    event_type = validate_calendar_event_type(event_type)
    start_date = validate_calendar_date(start_date, "Start date")

    if end_date is not None and end_date != "":
        end_date = validate_calendar_date(end_date, "End date")
        if end_date < start_date:
            raise ValidationError("End date cannot be earlier than start date.")
    else:
        end_date = None

    insert_statement = (
        "INSERT INTO academic_calendar_events "
        "(title, description, event_type, start_date, end_date, created_by) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (title, description, event_type, start_date, end_date, acting_user["user_id"]),
    )
    event_id = execute_write(*insert_statement)

    audit_statement = build_audit_entry(
        user_id=acting_user["user_id"],
        action=config.AUDIT_INSERT,
        table_name="academic_calendar_events",
        record_id=str(event_id),
        new_value={"title": title, "event_type": event_type, "start_date": start_date, "end_date": end_date},
    )
    execute_write(*audit_statement)

    logger.info(
        "Calendar event #%s ('%s', %s) published by user_id=%s.",
        event_id, title, event_type, acting_user["user_id"],
    )
    return event_id


def list_calendar_events(include_inactive: bool = False) -> list[dict]:
    """
    Every calendar event, soonest-starting first -- see this module's
    docstring for why this sorts by start_date rather than newest-first.

    Args:
        include_inactive: If True, also include taken-down events (used
            by the Admin management view).

    Returns:
        A list of dicts: event_id, title, description, event_type,
        start_date, end_date, created_by, created_by_username,
        is_active, created_at.
    """
    query = (
        "SELECT c.event_id, c.title, c.description, c.event_type, c.start_date, c.end_date, "
        "c.created_by, u.username AS created_by_username, c.is_active, c.created_at "
        "FROM academic_calendar_events c JOIN users u ON c.created_by = u.user_id"
    )
    if not include_inactive:
        query += " WHERE c.is_active = 1"

    # event_id ASC as a tiebreaker: two events sharing a start_date should
    # still come back in a stable, predictable order rather than an
    # unspecified one.
    query += " ORDER BY c.start_date ASC, c.event_id ASC"

    return fetch_all(query)


def deactivate_calendar_event(event_id: int, acting_user: dict) -> None:
    """
    Take down a calendar event: sets is_active = 0. Never a real SQL
    DELETE -- see modules/announcements.py's deactivate_announcement()
    for the same soft-delete reasoning applied here.

    Args:
        event_id: The event to take down.
        acting_user: The logged-in user performing this action (must be Admin).

    Raises:
        AuthorizationError: if acting_user's role is not Admin.
        RecordNotFoundError: if event_id does not exist.
        ValidationError: if the event is already inactive.
    """
    auth.check_permission(acting_user["role"], (config.ROLE_ADMIN,))

    existing = fetch_one(
        "SELECT event_id, is_active FROM academic_calendar_events WHERE event_id = ?", (event_id,),
    )
    if existing is None:
        raise RecordNotFoundError(f"Calendar event #{event_id} not found.")

    if not existing["is_active"]:
        raise ValidationError(f"Calendar event #{event_id} is already inactive.")

    update_statement = (
        "UPDATE academic_calendar_events SET is_active = 0 WHERE event_id = ?", (event_id,),
    )
    audit_statement = build_audit_entry(
        user_id=acting_user["user_id"],
        action=config.AUDIT_SOFT_DELETE,
        table_name="academic_calendar_events",
        record_id=str(event_id),
        old_value={"is_active": True},
        new_value=None,
    )
    execute_transaction([update_statement, audit_statement])

    logger.info("Calendar event #%s taken down by user_id=%s.", event_id, acting_user["user_id"])


def render_calendar_page() -> None:
    """Streamlit page: every role views the calendar; Admin can also
    publish new events or take down existing ones."""
    user = auth.require_role(*config.VALID_ROLES)
    is_admin = user["role"] == config.ROLE_ADMIN

    st.title("Academic Calendar")

    if is_admin:
        with st.expander(":material/event: Publish a new event"):
            with st.form("new_calendar_event_form", clear_on_submit=True):
                title = st.text_input("Title")
                event_type_choice = st.selectbox(
                    "Type", options=list(CALENDAR_EVENT_TYPE_LABELS.keys()),
                    format_func=lambda key: CALENDAR_EVENT_TYPE_LABELS[key],
                )
                date_cols = st.columns(2)
                start_date = date_cols[0].date_input("Start date")
                has_end_date = date_cols[1].checkbox("Spans multiple days")
                end_date = date_cols[1].date_input("End date", disabled=not has_end_date) if has_end_date else None
                description = st.text_area("Description (optional)")
                submitted = st.form_submit_button("Publish", type="primary")

            if submitted:
                try:
                    create_calendar_event(
                        title, description or None, event_type_choice,
                        start_date.isoformat(), end_date.isoformat() if end_date else None, user,
                    )
                    # st.toast(), not st.success() -- see app.py's
                    # render_role_login_form() for why, wherever a message
                    # is immediately followed by st.rerun().
                    st.toast("Event published.", icon=":material/check_circle:")
                    st.rerun()
                except ValidationError as error:
                    st.error(str(error))

    st.divider()

    events = list_calendar_events()
    if not events:
        st.info("No events on the calendar right now.")
        return

    for entry in events:
        date_label = entry["start_date"] if not entry["end_date"] else f"{entry['start_date']} to {entry['end_date']}"
        with st.container(border=True):
            st.markdown(f"**{entry['title']}** -- {CALENDAR_EVENT_TYPE_LABELS[entry['event_type']]}")
            st.caption(f"{date_label} -- published by {entry['created_by_username']}")
            if entry["description"]:
                st.write(entry["description"])

            if is_admin:
                if st.button("Take down", key=f"deactivate_calendar_event_{entry['event_id']}"):
                    deactivate_calendar_event(entry["event_id"], user)
                    st.toast("Event taken down.", icon=":material/check_circle:")
                    st.rerun()
