"""
tests/test_messaging.py
=========================
pytest tests for modules/messaging.py: sending, listing conversations
and threads, and read tracking for direct Teacher<->Student messages.

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_messaging.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.auth as auth
import modules.messaging as messaging
from database.db_setup import create_indexes, create_tables, get_connection
from utils.exceptions import AuthorizationError, RecordNotFoundError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_messaging.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _admin_user() -> dict:
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    return {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}


def _teacher_user(username: str = "teach1") -> dict:
    teacher_id = auth.create_user(username, "TeachPass1!", config.ROLE_TEACHER)
    return {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": username}


def _student_user(username: str = "student1") -> dict:
    student_id = auth.create_user(username, "StudentPass1!", config.ROLE_STUDENT)
    return {"user_id": student_id, "role": config.ROLE_STUDENT, "username": username}


# ---------------------------------------------------------------------------
# send_message()
# ---------------------------------------------------------------------------

def test_teacher_can_message_student(test_db):
    teacher = _teacher_user()
    student = _student_user()

    message_id = messaging.send_message(teacher, student["user_id"], "Hello!")

    thread = messaging.list_conversation(teacher["user_id"], student["user_id"])
    assert len(thread) == 1
    assert thread[0]["message_id"] == message_id
    assert thread[0]["body"] == "Hello!"


def test_student_can_message_teacher(test_db):
    teacher = _teacher_user()
    student = _student_user()

    messaging.send_message(student, teacher["user_id"], "Question about the exam.")

    thread = messaging.list_conversation(student["user_id"], teacher["user_id"])
    assert len(thread) == 1
    assert thread[0]["sender_id"] == student["user_id"]


def test_admin_cannot_send_messages(test_db):
    admin = _admin_user()
    student = _student_user()

    with pytest.raises(AuthorizationError):
        messaging.send_message(admin, student["user_id"], "Hi.")


def test_teacher_cannot_message_another_teacher(test_db):
    teacher1 = _teacher_user("teach1")
    teacher2 = _teacher_user("teach2")

    with pytest.raises(AuthorizationError):
        messaging.send_message(teacher1, teacher2["user_id"], "Hi.")


def test_student_cannot_message_another_student(test_db):
    student1 = _student_user("student1")
    student2 = _student_user("student2")

    with pytest.raises(AuthorizationError):
        messaging.send_message(student1, student2["user_id"], "Hi.")


def test_cannot_message_admin(test_db):
    teacher = _teacher_user()
    admin = _admin_user()

    with pytest.raises(AuthorizationError):
        messaging.send_message(teacher, admin["user_id"], "Hi.")


def test_cannot_message_self(test_db):
    teacher = _teacher_user()

    with pytest.raises(ValidationError):
        messaging.send_message(teacher, teacher["user_id"], "Hi.")


def test_send_message_rejects_unknown_recipient(test_db):
    teacher = _teacher_user()

    with pytest.raises(RecordNotFoundError):
        messaging.send_message(teacher, 9999, "Hi.")


def test_send_message_rejects_empty_body(test_db):
    teacher = _teacher_user()
    student = _student_user()

    with pytest.raises(ValidationError):
        messaging.send_message(teacher, student["user_id"], "")


def test_send_message_rejects_oversized_body(test_db):
    teacher = _teacher_user()
    student = _student_user()
    too_long = "x" * (config.MESSAGE_BODY_MAX_LENGTH + 1)

    with pytest.raises(ValidationError):
        messaging.send_message(teacher, student["user_id"], too_long)


# ---------------------------------------------------------------------------
# list_conversation() -- chronological thread
# ---------------------------------------------------------------------------

def test_list_conversation_orders_oldest_first(test_db):
    teacher = _teacher_user()
    student = _student_user()

    messaging.send_message(teacher, student["user_id"], "First.")
    messaging.send_message(student, teacher["user_id"], "Second.")
    messaging.send_message(teacher, student["user_id"], "Third.")

    thread = messaging.list_conversation(teacher["user_id"], student["user_id"])
    assert [m["body"] for m in thread] == ["First.", "Second.", "Third."]


def test_list_conversation_excludes_unrelated_messages(test_db):
    teacher = _teacher_user()
    student1 = _student_user("student1")
    student2 = _student_user("student2")

    messaging.send_message(teacher, student1["user_id"], "To student1.")
    messaging.send_message(teacher, student2["user_id"], "To student2.")

    thread = messaging.list_conversation(teacher["user_id"], student1["user_id"])
    assert len(thread) == 1
    assert thread[0]["body"] == "To student1."


# ---------------------------------------------------------------------------
# list_conversations_for_user()
# ---------------------------------------------------------------------------

def test_list_conversations_shows_last_message_and_unread_count(test_db):
    teacher = _teacher_user()
    student = _student_user()

    messaging.send_message(student, teacher["user_id"], "Hi teacher.")
    messaging.send_message(teacher, student["user_id"], "Hi back.")

    conversations = messaging.list_conversations_for_user(student["user_id"])
    assert len(conversations) == 1
    assert conversations[0]["other_user_id"] == teacher["user_id"]
    assert conversations[0]["last_message_body"] == "Hi back."
    assert conversations[0]["unread_count"] == 1


def test_list_conversations_one_row_per_partner(test_db):
    teacher = _teacher_user()
    student1 = _student_user("student1")
    student2 = _student_user("student2")

    messaging.send_message(teacher, student1["user_id"], "To student1.")
    messaging.send_message(teacher, student2["user_id"], "To student2.")

    conversations = messaging.list_conversations_for_user(teacher["user_id"])
    assert len(conversations) == 2
    assert {c["other_user_id"] for c in conversations} == {student1["user_id"], student2["user_id"]}


def test_list_conversations_newest_first(test_db):
    teacher = _teacher_user()
    student1 = _student_user("student1")
    student2 = _student_user("student2")

    messaging.send_message(teacher, student1["user_id"], "First conversation.")
    messaging.send_message(teacher, student2["user_id"], "Second conversation.")

    conversations = messaging.list_conversations_for_user(teacher["user_id"])
    assert conversations[0]["other_user_id"] == student2["user_id"]
    assert conversations[1]["other_user_id"] == student1["user_id"]


# ---------------------------------------------------------------------------
# mark_conversation_read() / get_unread_message_count()
# ---------------------------------------------------------------------------

def test_new_message_is_unread_for_recipient(test_db):
    teacher = _teacher_user()
    student = _student_user()

    messaging.send_message(teacher, student["user_id"], "Hello.")

    assert messaging.get_unread_message_count(student) == 1
    assert messaging.get_unread_message_count(teacher) == 0


def test_mark_conversation_read_clears_unread_count(test_db):
    teacher = _teacher_user()
    student = _student_user()
    messaging.send_message(teacher, student["user_id"], "Hello.")

    messaging.mark_conversation_read(student, teacher["user_id"])

    assert messaging.get_unread_message_count(student) == 0


def test_mark_conversation_read_does_not_affect_other_conversations(test_db):
    teacher1 = _teacher_user("teach1")
    teacher2 = _teacher_user("teach2")
    student = _student_user()
    messaging.send_message(teacher1, student["user_id"], "From teacher1.")
    messaging.send_message(teacher2, student["user_id"], "From teacher2.")

    messaging.mark_conversation_read(student, teacher1["user_id"])

    assert messaging.get_unread_message_count(student) == 1


def test_own_sent_messages_never_count_as_unread(test_db):
    teacher = _teacher_user()
    student = _student_user()

    messaging.send_message(teacher, student["user_id"], "Hello.")

    assert messaging.get_unread_message_count(teacher) == 0
