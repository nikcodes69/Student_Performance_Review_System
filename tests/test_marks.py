"""
tests/test_marks.py
====================
pytest tests for modules/marks.py's result publishing workflow
(publish_marks()/unpublish_marks(), and list_marks_for_student()'s
published_only parameter) -- see that module's docstrings for the full
design: marks start as drafts, invisible to the Student they belong to
until explicitly published as a whole subject/semester/exam_type batch.

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_marks.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.marks as marks
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, subjects, teacher_subjects
from utils.exceptions import AuthorizationError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test, with
    teacher_subjects' cached lookup cleared first -- see
    tests/test_teacher_subjects.py's module docstring for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_marks.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    teacher_subjects.list_subjects_for_teacher.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _seed_admin_teacher_subject_and_students(student_count: int = 2):
    """Returns (admin_user, teacher_user), with one subject ('SUB1', the
    teacher already assigned to it) and `student_count` students
    ('S1', 'S2', ...) already created."""
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    admin_user = {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}
    teacher_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}

    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, admin_user)
    teacher_subjects.assign_teacher_to_subject(teacher_id, "SUB1", admin_user)

    for i in range(1, student_count + 1):
        students.create_student(
            f"S{i}", f"Student {chr(64 + i)}", 1, "BCA", f"student{i}@example.com", "9812345678", 2024, admin_user,
        )

    return admin_user, teacher_user


# ---------------------------------------------------------------------------
# enter_marks() -- new entries start as drafts
# ---------------------------------------------------------------------------

def test_enter_marks_defaults_to_draft(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)

    entry = marks.get_marks_entry("S1", "SUB1", 1, "regular")
    assert entry["is_published"] == 0


# ---------------------------------------------------------------------------
# publish_marks()
# ---------------------------------------------------------------------------

def test_publish_marks_publishes_every_draft_in_the_batch(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(2)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.enter_marks("S2", "SUB1", 18, 35, 12, 1, "regular", admin_user)

    published_count = marks.publish_marks("SUB1", 1, "regular", admin_user)

    assert published_count == 2
    assert marks.get_marks_entry("S1", "SUB1", 1, "regular")["is_published"] == 1
    assert marks.get_marks_entry("S2", "SUB1", 1, "regular")["is_published"] == 1


def test_publish_marks_does_not_touch_a_different_exam_type(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.enter_marks("S1", "SUB1", 20, 40, 15, 1, "backlog", admin_user)

    marks.publish_marks("SUB1", 1, "regular", admin_user)

    assert marks.get_marks_entry("S1", "SUB1", 1, "regular")["is_published"] == 1
    assert marks.get_marks_entry("S1", "SUB1", 1, "backlog")["is_published"] == 0


def test_publish_marks_rejects_when_nothing_to_publish(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    with pytest.raises(ValidationError):
        marks.publish_marks("SUB1", 1, "regular", admin_user)


def test_publish_marks_is_idempotent_on_already_published_rows(test_db):
    # Publishing again after everything is already published must report
    # "nothing to publish", not silently re-publish (which would also
    # write a spurious extra audit_log entry every time).
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.publish_marks("SUB1", 1, "regular", admin_user)

    with pytest.raises(ValidationError):
        marks.publish_marks("SUB1", 1, "regular", admin_user)


def test_publish_marks_requires_admin_or_teacher(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    student_id = auth.create_user("student1", "StudentPass1!", config.ROLE_STUDENT)
    student_user = {"user_id": student_id, "role": config.ROLE_STUDENT, "username": "student1"}

    with pytest.raises(AuthorizationError):
        marks.publish_marks("SUB1", 1, "regular", student_user)


def test_publish_marks_requires_teacher_subject_assignment(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)

    unassigned_teacher_id = auth.create_user("teach2", "TeachPass1!", config.ROLE_TEACHER)
    unassigned_teacher = {"user_id": unassigned_teacher_id, "role": config.ROLE_TEACHER, "username": "teach2"}

    with pytest.raises(AuthorizationError):
        marks.publish_marks("SUB1", 1, "regular", unassigned_teacher)


# ---------------------------------------------------------------------------
# unpublish_marks()
# ---------------------------------------------------------------------------

def test_unpublish_marks_reverts_to_draft(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.publish_marks("SUB1", 1, "regular", admin_user)

    unpublished_count = marks.unpublish_marks("SUB1", 1, "regular", admin_user)

    assert unpublished_count == 1
    assert marks.get_marks_entry("S1", "SUB1", 1, "regular")["is_published"] == 0


def test_unpublish_marks_rejects_when_nothing_published(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)

    with pytest.raises(ValidationError):
        marks.unpublish_marks("SUB1", 1, "regular", admin_user)


# ---------------------------------------------------------------------------
# list_marks_for_student(published_only=True) -- what modules/student_portal.py uses
# ---------------------------------------------------------------------------

def test_published_only_hides_draft_marks(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)

    assert marks.list_marks_for_student("S1", published_only=True) == []
    assert len(marks.list_marks_for_student("S1", published_only=False)) == 1


def test_published_only_shows_published_marks(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.publish_marks("SUB1", 1, "regular", admin_user)

    visible = marks.list_marks_for_student("S1", published_only=True)
    assert len(visible) == 1
    assert visible[0]["roll_no"] == "S1"


def test_list_marks_for_student_defaults_to_showing_everything(test_db):
    # Admin/Teacher's own views (Marks Entry) call this WITHOUT
    # published_only -- they must see drafts too, to review before publishing.
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(1)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)

    assert len(marks.list_marks_for_student("S1")) == 1


def test_list_marks_for_subject_includes_is_published_for_every_row(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_subject_and_students(2)
    marks.enter_marks("S1", "SUB1", 15, 30, 10, 1, "regular", admin_user)
    marks.enter_marks("S2", "SUB1", 18, 35, 12, 1, "regular", admin_user)
    marks.publish_marks("SUB1", 1, "regular", admin_user)
    marks.enter_marks("S1", "SUB1", 20, 40, 15, 1, "backlog", admin_user)  # stays draft

    rows = marks.list_marks_for_subject("SUB1", semester=1)
    by_exam_type = {(row["roll_no"], row["exam_type"]): row["is_published"] for row in rows}

    assert by_exam_type[("S1", "regular")] == 1
    assert by_exam_type[("S2", "regular")] == 1
    assert by_exam_type[("S1", "backlog")] == 0
