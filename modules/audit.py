"""
modules/audit.py
=================
Records and displays the audit trail: a permanent history of who changed
what, and when, across the tables listed in config.AUDITED_TABLES --
including "auth_events" (login/logout/failed-login attempts, written by
modules/auth.py's _record_auth_audit_event(), not by this file).

HOW FUTURE MODULES WILL USE THIS (starting with modules/students.py,
marks.py, etc. in step 8): whenever one of those modules inserts, updates,
or soft-deletes a row, it will build TWO statements -- the actual data
change, and an audit entry describing it (via build_audit_entry() below)
-- and pass BOTH to database.db_manager.execute_transaction() together.
That guarantees the data change and its audit record either both happen,
or neither does; see the docstring on execute_transaction() in
database/db_manager.py for why that matters. Example (illustrative -- the
real version of this appears once modules/students.py exists):

    from database.db_manager import execute_transaction
    from modules.audit import build_audit_entry

    update_statement = (
        "UPDATE students SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE roll_no = ?",
        (new_name, roll_no),
    )
    audit_statement = build_audit_entry(
        user_id=current_user["user_id"],
        action=config.AUDIT_UPDATE,
        table_name="students",
        record_id=roll_no,
        old_value={"name": old_name},
        new_value={"name": new_name},
    )
    execute_transaction([update_statement, audit_statement])

CONTRACT FOR old_value/new_value: always a plain Python dict (or None),
never a sqlite3.Row -- convert with dict(row) first if needed. This module
only stores and retrieves them (as JSON text); it does not know, and does
not need to know, what columns any particular table has.
"""

import json

import streamlit as st

import config
from database.db_manager import execute_write, fetch_all
from modules import auth
from utils.logger import get_logger
from utils.table_view import render_data_table
from utils.validators import validate_audit_action, validate_table_name

logger = get_logger(__name__)


def build_audit_entry(
    user_id: int,
    action: str,
    table_name: str,
    record_id: str,
    old_value: dict | None = None,
    new_value: dict | None = None,
) -> tuple[str, tuple]:
    """
    Build the (query, params) tuple for ONE audit_log INSERT, without
    executing it. Meant to be combined with the actual data-change
    statement inside database.db_manager.execute_transaction() -- see the
    module docstring above for the full pattern.

    Args:
        user_id: The user_id of whoever made the change.
        action: One of config.AUDIT_ACTIONS.
        table_name: One of config.AUDITED_TABLES.
        record_id: The primary key of the affected row (e.g. a roll_no,
            or str(mark_id)) -- always converted to str, since audit_log
            stores record_id as TEXT to handle both kinds of primary key
            (see database/db_setup.py's comment on the audit_log table).
        old_value: A plain dict snapshot of the row before the change, or
            None for an INSERT (there is no "before").
        new_value: A plain dict snapshot of the row after the change.

    Returns:
        A (query, params) tuple ready to pass to execute_write() or
        execute_transaction().

    Raises:
        ValidationError: if action or table_name is not recognised.
    """
    action = validate_audit_action(action)
    table_name = validate_table_name(table_name)

    query = (
        "INSERT INTO audit_log (user_id, action, table_name, record_id, old_value, new_value) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    params = (
        user_id,
        action,
        table_name,
        str(record_id),
        json.dumps(old_value) if old_value is not None else None,
        json.dumps(new_value) if new_value is not None else None,
    )
    return query, params


def record_change(
    user_id: int,
    action: str,
    table_name: str,
    record_id: str,
    old_value: dict | None = None,
    new_value: dict | None = None,
) -> int:
    """
    Convenience function: build AND immediately execute one audit_log
    entry, on its own (not combined with another statement).

    Use this only for a change that has no separate "data write" of its
    own to pair it with. Every module built from step 8 onward that
    changes an academic record should prefer build_audit_entry() +
    execute_transaction() instead, so the data change and its audit entry
    are atomic together (see the module docstring above).

    Args:
        Same as build_audit_entry().

    Returns:
        The new audit_log row's log_id.
    """
    query, params = build_audit_entry(user_id, action, table_name, record_id, old_value, new_value)
    log_id = execute_write(query, params)
    logger.info(
        "Audit entry recorded: user_id=%s action=%s table=%s record_id=%s",
        user_id, action, table_name, record_id,
    )
    return log_id


def get_audit_logs(
    table_name: str | None = None,
    record_id: str | None = None,
    user_id: int | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Query audit_log, most recent first, with optional filters.

    HOW THIS STAYS SAFE FROM SQL INJECTION EVEN THOUGH THE QUERY STRING IS
    BUILT UP PIECE BY PIECE: the pieces being assembled below are only
    ever fixed, hard-coded clause fragments like "table_name = ?" -- which
    fragments get INCLUDED is decided by Python control flow (whether a
    filter argument is None), never by the CONTENTS of user-supplied data.
    Every actual VALUE (table_name, record_id, user_id) still flows
    through the params list and a "?" placeholder, exactly like every
    other query in this project. This is different from, and safe in a
    way that, e.g., `f"WHERE table_name = '{table_name}'"` would not be.

    Args:
        table_name: If given, only entries for this table.
        record_id: If given, only entries for this record.
        user_id: If given, only entries made by this user.
        limit: Maximum number of entries to return.

    Returns:
        A list of dicts, most recent first, with old_value/new_value
        parsed back from JSON text into plain dicts (or None).
    """
    query = (
        "SELECT log_id, user_id, action, table_name, record_id, old_value, new_value, timestamp "
        "FROM audit_log"
    )
    conditions = []
    params: list = []

    if table_name is not None:
        conditions.append("table_name = ?")
        params.append(table_name)
    if record_id is not None:
        conditions.append("record_id = ?")
        params.append(str(record_id))
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # log_id DESC as a tiebreaker, not just timestamp DESC: timestamp has
    # only SECOND resolution, so two entries written within the same
    # second (e.g. an INSERT immediately followed by its own correction,
    # or several bulk-import rows committed in a tight loop) would
    # otherwise tie and sort in an unspecified order -- the auto-increment
    # log_id is monotonically increasing and never ties, so it reliably
    # breaks that tie newest-first (same fix already applied to
    # modules/announcements.py's list_announcements_for_role()).
    query += " ORDER BY timestamp DESC, log_id DESC LIMIT ?"
    params.append(limit)

    rows = fetch_all(query, tuple(params))

    return [
        {
            "log_id": row["log_id"],
            "user_id": row["user_id"],
            "action": row["action"],
            "table_name": row["table_name"],
            "record_id": row["record_id"],
            "old_value": json.loads(row["old_value"]) if row["old_value"] else None,
            "new_value": json.loads(row["new_value"]) if row["new_value"] else None,
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def render_audit_log_page() -> None:
    """
    Streamlit page: the audit log viewer, restricted to Admin.

    Called from app.py's sidebar navigation, the same way every other
    module's own render_*_page() function will be from step 8 onward.
    """
    auth.require_role(config.ROLE_ADMIN)

    st.title("Audit Log")
    st.write(
        "A permanent record of every change made to academic records in this system, "
        "plus every login, logout, and failed login attempt (filter by table "
        "\"auth_events\" for just those)."
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        table_choice = st.selectbox("Filter by table", options=["(all)"] + list(config.AUDITED_TABLES))
    with filter_col2:
        record_filter = st.text_input("Filter by record ID (optional)")

    table_name = None if table_choice == "(all)" else table_choice
    record_id = record_filter.strip() or None

    logs = get_audit_logs(table_name=table_name, record_id=record_id)

    if not logs:
        st.info("No audit log entries match these filters.")
        return

    # old_value/new_value are Python dicts at this point (parsed back from
    # JSON by get_audit_logs) -- re-serialised to a compact JSON string
    # here purely for a readable table cell; the underlying stored data
    # stays exactly as it was written.
    display_rows = [
        {
            "Timestamp": entry["timestamp"],
            "User ID": entry["user_id"],
            "Action": entry["action"],
            "Table": entry["table_name"],
            "Record ID": entry["record_id"],
            "Old Value": json.dumps(entry["old_value"]) if entry["old_value"] else "",
            "New Value": json.dumps(entry["new_value"]) if entry["new_value"] else "",
        }
        for entry in logs
    ]

    render_data_table(display_rows, key_prefix="audit_table", filename_prefix="audit_log")
