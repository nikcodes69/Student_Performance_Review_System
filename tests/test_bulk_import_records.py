"""
tests/test_bulk_import_records.py
==================================
pytest tests for bulk CSV/Excel import of Marks, Attendance, and
Assignments -- the three record types with a COMPOSITE natural key
(roll_no + subject_code + semester[+ exam_type]), unlike Students/
Subjects (tested in tests/test_bulk_import.py), which are single-key.
Covers:
  - modules/marks.py's _validate_bulk_marks_row()/_commit_bulk_marks_row()
  - modules/attendance.py's _validate_bulk_attendance_row()/_commit_bulk_attendance_row()
  - modules/assignments.py's _validate_bulk_assignment_row()/_commit_bulk_assignment_row()

WHY THESE THREE SHARE ONE TEST FILE: all three follow the exact same
shape (validate a row's fields + the subject's own numeric rules, check
teacher-subject access, then commit as an UPSERT -- insert if new, else
update) -- see modules/marks.py's _validate_bulk_marks_row() docstring
for the fullest explanation, which the attendance/assignments versions
point back to rather than repeating.

WHY TEACHER-SUBJECT ACCESS IS TESTED HERE, UNLIKE
tests/test_bulk_import.py's STUDENT/SUBJECT VALIDATORS: only these three
record types are gated by modules/teacher_subjects.py (see that module's
docstring) -- Students/Subjects bulk import is Admin-only already, so
there is no separate "Teacher not assigned" case to cover there.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_bulk_import_records.py -v
"""

import pytest

import config
import database.db_setup as db_setup
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, subjects, teacher_subjects
from modules.assignments import (
    _commit_bulk_assignment_row,
    _validate_bulk_assignment_row,
    get_assignment_entry,
    list_assignments_for_student,
)
from modules.attendance import (
    _commit_bulk_attendance_row,
    _validate_bulk_attendance_row,
    get_attendance_entry,
    list_attendance_for_student,
)
from modules.marks import _commit_bulk_marks_row, _validate_bulk_marks_row, get_marks_entry, list_marks_for_student
from utils.exceptions import AuthorizationError, ValidationError


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database, with
    teacher_subjects' cached lookup cleared first -- see
    tests/test_teacher_subjects.py's module docstring for why."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_bulk_import_records.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    teacher_subjects.list_subjects_for_teacher.clear()

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _seed_admin_teacher_student_subject(assign: bool = True):
    """Admin, Teacher (assigned to SUB1 unless assign=False), one
    student (S1), one subject (SUB1). Returns (admin_user, teacher_user)."""
    admin_id = auth.create_user("admin", "AdminPass1!", config.ROLE_ADMIN)
    teacher_id = auth.create_user("teach1", "TeachPass1!", config.ROLE_TEACHER)
    admin_user = {"user_id": admin_id, "role": config.ROLE_ADMIN, "username": "admin"}
    teacher_user = {"user_id": teacher_id, "role": config.ROLE_TEACHER, "username": "teach1"}

    students.create_student("S1", "Alice", 1, "BCA", "alice@example.com", "9812345678", 2024, admin_user)
    subjects.create_subject("SUB1", "Fixture Subject", 1, 3, 20, 80, 0, admin_user)

    if assign:
        teacher_subjects.assign_teacher_to_subject(teacher_id, "SUB1", admin_user)

    return admin_user, teacher_user


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

_MARKS_ROW = {
    "roll_no": "S1", "subject_code": "SUB1", "semester": "1", "exam_type": "regular",
    "internal": "15", "external": "60", "practical": "0",
}


def test_validate_bulk_marks_row_blocks_unassigned_teacher(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject(assign=False)
    with pytest.raises(AuthorizationError):
        _validate_bulk_marks_row(_MARKS_ROW, teacher_user)


def test_validate_bulk_marks_row_rejects_marks_over_subject_ceiling(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject()
    row = {**_MARKS_ROW, "internal": "999"}  # SUB1's max_internal is 20
    with pytest.raises(ValidationError):
        _validate_bulk_marks_row(row, teacher_user)


def test_commit_bulk_marks_row_is_an_upsert(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject()

    _commit_bulk_marks_row(_MARKS_ROW, teacher_user)
    entry = get_marks_entry("S1", "SUB1", 1, "regular")
    assert entry["internal"] == 15

    _commit_bulk_marks_row({**_MARKS_ROW, "internal": "18"}, teacher_user)
    entry = get_marks_entry("S1", "SUB1", 1, "regular")
    assert entry["internal"] == 18
    assert len(list_marks_for_student("S1")) == 1  # updated in place, not duplicated


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------

_ATTENDANCE_ROW = {"roll_no": "S1", "subject_code": "SUB1", "semester": "1", "classes_held": "20", "classes_attended": "18"}


def test_validate_bulk_attendance_row_blocks_unassigned_teacher(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject(assign=False)
    with pytest.raises(AuthorizationError):
        _validate_bulk_attendance_row(_ATTENDANCE_ROW, teacher_user)


def test_validate_bulk_attendance_row_rejects_attended_over_held(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject()
    row = {**_ATTENDANCE_ROW, "classes_attended": "25"}  # > classes_held (20)
    with pytest.raises(ValidationError):
        _validate_bulk_attendance_row(row, teacher_user)


def test_commit_bulk_attendance_row_is_an_upsert(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject()

    _commit_bulk_attendance_row(_ATTENDANCE_ROW, teacher_user)
    entry = get_attendance_entry("S1", "SUB1", 1)
    assert entry["classes_attended"] == 18

    _commit_bulk_attendance_row({**_ATTENDANCE_ROW, "classes_attended": "19"}, teacher_user)
    entry = get_attendance_entry("S1", "SUB1", 1)
    assert entry["classes_attended"] == 19
    assert len(list_attendance_for_student("S1")) == 1


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------

_ASSIGNMENT_ROW = {"roll_no": "S1", "subject_code": "SUB1", "semester": "1", "total_assigned": "5", "submitted": "4"}


def test_validate_bulk_assignment_row_blocks_unassigned_teacher(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject(assign=False)
    with pytest.raises(AuthorizationError):
        _validate_bulk_assignment_row(_ASSIGNMENT_ROW, teacher_user)


def test_validate_bulk_assignment_row_rejects_submitted_over_total(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject()
    row = {**_ASSIGNMENT_ROW, "submitted": "10"}  # > total_assigned (5)
    with pytest.raises(ValidationError):
        _validate_bulk_assignment_row(row, teacher_user)


def test_commit_bulk_assignment_row_is_an_upsert(test_db):
    _admin_user, teacher_user = _seed_admin_teacher_student_subject()

    _commit_bulk_assignment_row(_ASSIGNMENT_ROW, teacher_user)
    entry = get_assignment_entry("S1", "SUB1", 1)
    assert entry["submitted"] == 4

    _commit_bulk_assignment_row({**_ASSIGNMENT_ROW, "submitted": "5"}, teacher_user)
    entry = get_assignment_entry("S1", "SUB1", 1)
    assert entry["submitted"] == 5
    assert len(list_assignments_for_student("S1")) == 1


def test_admin_never_restricted_by_teacher_subject_access(test_db):
    admin_user, _teacher_user = _seed_admin_teacher_student_subject(assign=False)
    _validate_bulk_marks_row(_MARKS_ROW, admin_user)  # must not raise, even though no teacher is assigned
