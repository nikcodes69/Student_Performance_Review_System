"""
modules/messaging.py
======================
Direct 1:1 messages, exclusively between a Teacher and a Student --
distinct from modules/announcements.py's Announcements (a one-to-many
broadcast: one Admin/Teacher post, visible to a whole role or everyone)
and from modules/teacher_remarks.py's Remarks (a one-way note a Teacher
leaves ON a student's record). A message is a genuine back-and-forth
conversation between exactly two people.

WHO CAN MESSAGE WHOM: only a Teacher<->Student pair -- never Teacher<-
>Teacher, Student<->Student, or anything involving Admin at all.
send_message() checks the SENDER's own role via check_permission(), then
independently confirms the RECIPIENT's role is the other one of the pair
(a Teacher can only message a Student, and vice versa) -- two separate
checks, not one, the same "both ends of the pair matter" reasoning
modules/grade_appeals.py's check_teacher_subject_access() applies to a
different pair (Teacher/subject).

NO ROSTER RESTRICTION: unlike a subject-scoped feature (marks,
attendance), this project has no per-class roster to restrict a
Teacher/Student pair to "people who actually share a class" -- the same
gap modules/teacher_remarks.py's general (non-subject-tied) remarks
already accepted as a deliberate, documented limitation. Restricting
messaging to only students in a Teacher's assigned subjects would need
that same missing roster concept invented from scratch; instead, any
Teacher may message any Student and vice versa, exactly like a general
remark.

NO ADMIN VISIBILITY, UNLIKE EVERY OTHER COMMUNICATION FEATURE IN THIS
PROJECT: Announcements and Remarks are Admin-postable/Admin-manageable
because they are institutional records. A direct message between one
Teacher and one Student is closer to a private conversation -- Admin has
no read access to message content at all here, a deliberate privacy
boundary this module keeps even though Admin is unrestricted almost
everywhere else in this codebase.

NO SOFT DELETE: unlike an announcement or a remark, a sent message is
never taken down -- there is no deactivate_message() at all. is_read is
the only field this module ever updates after a message is sent.
"""

import streamlit as st

import config
from database.db_manager import execute_write, fetch_all, fetch_one
from modules import auth
from modules.audit import build_audit_entry
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError
from utils.logger import get_logger
from utils.validators import validate_message_body

logger = get_logger(__name__)

MESSAGING_ROLES = (config.ROLE_TEACHER, config.ROLE_STUDENT)


def _get_user_row(user_id: int) -> dict | None:
    """Minimal user lookup (id, username, role, is_active) -- this module's
    own small query rather than a new function on modules/auth.py, the
    same "self-contained need, own query" pattern modules/grade_appeals.py
    and modules/teacher_remarks.py already use."""
    return fetch_one("SELECT user_id, username, role, is_active FROM users WHERE user_id = ?", (user_id,))


def send_message(sender: dict, recipient_user_id: int, body: str) -> int:
    """
    Send a direct message.

    Args:
        sender: The logged-in user sending this (must be Teacher or Student).
        recipient_user_id: The user_id of the other party -- must be a
            Teacher if sender is a Student, or a Student if sender is a
            Teacher.
        body: The message text.

    Returns:
        The new message_id.

    Raises:
        AuthorizationError: if sender's role is not Teacher/Student, or
            recipient_user_id does not belong to the OTHER of that pair.
        RecordNotFoundError: if recipient_user_id does not exist.
        ValidationError: if body fails validation, or recipient_user_id
            is the sender themselves.
    """
    auth.check_permission(sender["role"], MESSAGING_ROLES)

    if recipient_user_id == sender["user_id"]:
        raise ValidationError("You cannot message yourself.")

    recipient = _get_user_row(recipient_user_id)
    if recipient is None:
        raise RecordNotFoundError(f"No user found with id {recipient_user_id}.")

    if {sender["role"], recipient["role"]} != set(MESSAGING_ROLES):
        raise AuthorizationError("Messages can only be exchanged between a Teacher and a Student.")

    body = validate_message_body(body)

    insert_statement = (
        "INSERT INTO messages (sender_id, recipient_id, body) VALUES (?, ?, ?)",
        (sender["user_id"], recipient_user_id, body),
    )
    message_id = execute_write(*insert_statement)

    audit_statement = build_audit_entry(
        user_id=sender["user_id"],
        action=config.AUDIT_INSERT,
        table_name="messages",
        record_id=str(message_id),
        new_value={"recipient_id": recipient_user_id},
    )
    execute_write(*audit_statement)

    logger.info(
        "Message #%s sent from user_id=%s to user_id=%s.", message_id, sender["user_id"], recipient_user_id,
    )
    return message_id


def list_conversation(user_id: int, other_user_id: int) -> list[dict]:
    """
    Every message exchanged between exactly these two users, oldest
    first -- a chat thread, not a feed.

    Args:
        user_id: One party.
        other_user_id: The other party.

    Returns:
        A list of dicts: message_id, sender_id, recipient_id, body,
        is_read, created_at, sender_username.
    """
    return fetch_all(
        "SELECT m.message_id, m.sender_id, m.recipient_id, m.body, m.is_read, m.created_at, "
        "u.username AS sender_username "
        "FROM messages m JOIN users u ON m.sender_id = u.user_id "
        "WHERE (m.sender_id = ? AND m.recipient_id = ?) OR (m.sender_id = ? AND m.recipient_id = ?) "
        "ORDER BY m.created_at ASC, m.message_id ASC",
        (user_id, other_user_id, other_user_id, user_id),
    )


def list_conversations_for_user(user_id: int) -> list[dict]:
    """
    One row per distinct person user_id has exchanged messages with,
    newest conversation first, each with the last message and an unread
    count.

    IMPLEMENTED IN PYTHON, NOT ONE "GROUP BY OTHER PARTY" SQL QUERY: SQL
    has no simple, backend-portable way to fetch "the single latest row
    per group" without window functions, and this project's two backends
    (plain SQLite locally, libsql/Turso in production -- see
    database/db_setup.py's get_connection()) are not both guaranteed to
    support the same window-function surface. A student project's message
    volume per user is small, so fetching every message this user is
    party to ONCE and grouping it in Python is simple, obviously correct,
    and fast enough -- the same "reuse + Python-side grouping over a
    clever query" choice modules/academic_calendar.py and others in this
    project already make when a query would otherwise get complicated for
    little benefit.

    Args:
        user_id: The logged-in user.

    Returns:
        A list of dicts: other_user_id, other_username, other_role,
        last_message_body, last_message_at, last_message_id, unread_count.
        Newest conversation first (last_message_id as a tiebreaker for
        two conversations whose latest message landed in the same second).
    """
    rows = fetch_all(
        "SELECT m.message_id, m.sender_id, m.recipient_id, m.body, m.is_read, m.created_at, "
        "su.username AS sender_username, ru.username AS recipient_username, "
        "su.role AS sender_role, ru.role AS recipient_role "
        "FROM messages m "
        "JOIN users su ON m.sender_id = su.user_id "
        "JOIN users ru ON m.recipient_id = ru.user_id "
        "WHERE m.sender_id = ? OR m.recipient_id = ? "
        "ORDER BY m.created_at ASC, m.message_id ASC",
        (user_id, user_id),
    )

    conversations: dict[int, dict] = {}
    for row in rows:
        is_outgoing = row["sender_id"] == user_id
        other_id = row["recipient_id"] if is_outgoing else row["sender_id"]
        other_username = row["recipient_username"] if is_outgoing else row["sender_username"]
        other_role = row["recipient_role"] if is_outgoing else row["sender_role"]

        entry = conversations.setdefault(other_id, {
            "other_user_id": other_id, "other_username": other_username, "other_role": other_role,
            "last_message_body": None, "last_message_at": None, "last_message_id": None, "unread_count": 0,
        })
        # Rows arrive oldest-first (message_id ascending), so the LAST one
        # processed for this partner is always the most recent -- simply
        # overwriting on every row naturally leaves the latest one in
        # place. last_message_id is kept alongside last_message_at purely
        # as a sort tiebreaker below -- created_at has only SECOND
        # resolution (the same issue modules/announcements.py's
        # list_announcements_for_role() already documents), so two
        # conversations whose latest message landed in the same second
        # would otherwise sort in an unspecified order.
        entry["last_message_body"] = row["body"]
        entry["last_message_at"] = row["created_at"]
        entry["last_message_id"] = row["message_id"]
        if not is_outgoing and not row["is_read"]:
            entry["unread_count"] += 1

    return sorted(
        conversations.values(),
        key=lambda entry: (entry["last_message_at"], entry["last_message_id"]),
        reverse=True,
    )


def mark_conversation_read(acting_user: dict, other_user_id: int) -> None:
    """
    Mark every unread message FROM other_user_id TO acting_user as read.
    Not audited -- see this module's docstring and
    modules/announcements.py's mark_announcement_read() for the same
    "read tracking is not a change to a real record" reasoning.

    Args:
        acting_user: The logged-in user viewing the conversation.
        other_user_id: The conversation partner whose messages are being
            marked read.
    """
    execute_write(
        "UPDATE messages SET is_read = 1 WHERE recipient_id = ? AND sender_id = ? AND is_read = 0",
        (acting_user["user_id"], other_user_id),
    )


def get_unread_message_count(acting_user: dict) -> int:
    """
    How many messages addressed to acting_user, across every
    conversation, have not yet been marked read -- the number a sidebar
    badge shows.
    """
    row = fetch_one(
        "SELECT COUNT(*) AS unread_count FROM messages WHERE recipient_id = ? AND is_read = 0",
        (acting_user["user_id"],),
    )
    return row["unread_count"]


def _list_possible_recipients(acting_user: dict) -> list[dict]:
    """Every account acting_user is allowed to message: every Student for
    a Teacher, every Teacher for a Student -- see this module's docstring
    for why this has no further roster restriction."""
    other_role = config.ROLE_STUDENT if acting_user["role"] == config.ROLE_TEACHER else config.ROLE_TEACHER
    return fetch_all(
        "SELECT user_id, username FROM users WHERE role = ? AND is_active = 1 ORDER BY username",
        (other_role,),
    )


def render_messages_page() -> None:
    """Streamlit page (Teacher/Student only): pick a conversation partner
    (or an existing conversation), view the thread, and reply."""
    user = auth.require_role(*MESSAGING_ROLES)
    st.title("Messages")

    conversations = list_conversations_for_user(user["user_id"])
    possible_recipients = _list_possible_recipients(user)

    if not possible_recipients:
        st.info("No one to message yet.")
        return

    existing_labels = {
        f"{c['other_username']}" + (f" ({c['unread_count']} unread)" if c["unread_count"] else ""): c["other_user_id"]
        for c in conversations
    }
    new_labels = {
        f"{r['username']} (new conversation)": r["user_id"]
        for r in possible_recipients if r["user_id"] not in {c["other_user_id"] for c in conversations}
    }
    all_labels = existing_labels | new_labels

    picked_label = st.selectbox("Conversation", options=list(all_labels.keys()))
    other_user_id = all_labels[picked_label]

    mark_conversation_read(user, other_user_id)

    thread = list_conversation(user["user_id"], other_user_id)
    with st.container(border=True, height=400):
        if not thread:
            st.info("No messages yet -- say hello.")
        for entry in thread:
            is_mine = entry["sender_id"] == user["user_id"]
            speaker = "You" if is_mine else entry["sender_username"]
            st.markdown(f"**{speaker}** -- {entry['created_at']}")
            st.write(entry["body"])

    with st.form(f"reply_form_{other_user_id}", clear_on_submit=True):
        reply_body = st.text_area("Message")
        submitted = st.form_submit_button("Send", type="primary")

    if submitted:
        try:
            send_message(user, other_user_id, reply_body)
            # st.toast(), not st.success() -- see app.py's
            # render_role_login_form() for why, wherever a message is
            # immediately followed by st.rerun().
            st.toast("Message sent.", icon=":material/check_circle:")
            st.rerun()
        except ValidationError as error:
            st.error(str(error))
