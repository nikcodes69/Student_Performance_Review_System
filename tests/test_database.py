"""
tests/test_database.py
=======================
pytest tests for database/db_setup.py's schema: CHECK constraints,
UNIQUE constraints, FOREIGN KEY enforcement, and the PRAGMA foreign_keys
setting itself.

These constraints were originally verified by hand with one-off scripts
while building database/db_setup.py (deliberately attempting invalid
inserts and confirming SQLite rejected them). This file turns those same
checks into a permanent, re-runnable pytest suite instead of a record
that only existed in a terminal scrollback.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below
creates a brand-new, empty SQLite database in a pytest-managed temporary
directory for every single test function (a fresh one each time, never
shared between tests), and monkeypatches config.DB_PATH so
database.db_setup.get_connection() opens THAT file instead of the real
application database (student_data.db) -- which this file never reads or
writes. pytest's monkeypatch fixture automatically undoes the patch after
each test, so there is no risk of one test's changes leaking into another,
or into the real app.

THIS ALSO FORCIBLY DISABLES THE TURSO CLOUD BACKEND FOR EVERY TEST: since
database/db_setup.py added Turso (cloud) support, get_connection() checks
Streamlit secrets FIRST and uses the cloud database whenever credentials
are configured there -- which they now are, in .streamlit/secrets.toml,
for the deployed app. Without an extra safeguard, simply having that file
present on a developer's machine would make this ENTIRE test suite quietly
start running against the real, shared, production Turso database instead
of a disposable local file -- inserting deliberately-invalid rows into it
on every test run. The fixture below monkeypatches
database.db_setup._get_turso_credentials() to always return (None, None),
which forces get_connection() down the local-SQLite path no matter what
secrets exist on the machine running the tests.

HOW TO RUN (from the project root):
    python -m pytest tests/test_database.py -v
"""

import sqlite3

import pytest

import config
import database.db_setup as db_setup
from database.db_setup import create_indexes, create_tables, get_connection


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_student_data.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()

    yield conn

    conn.close()


def _insert_student(conn: sqlite3.Connection, roll_no: str) -> None:
    conn.execute(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (roll_no, "Fixture Student", 1, "BCA", f"{roll_no.lower()}@example.com", "9812345678", 2024),
    )
    conn.commit()


def _insert_subject(conn: sqlite3.Connection, subject_code: str) -> None:
    conn.execute(
        "INSERT INTO subjects (subject_code, name, semester, credits, max_internal, max_external, max_practical) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (subject_code, "Fixture Subject", 1, 3, 20, 80, 0),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# PRAGMA foreign_keys
# ---------------------------------------------------------------------------

def test_foreign_keys_pragma_is_enabled(test_db):
    # SQLite has foreign key enforcement OFF by default per connection --
    # get_connection() must turn it on itself (see database/db_setup.py).
    cursor = test_db.execute("PRAGMA foreign_keys")
    assert cursor.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# UNIQUE / PRIMARY KEY constraints
# ---------------------------------------------------------------------------

def test_students_roll_no_is_unique(test_db):
    _insert_student(test_db, "BCA001")
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BCA001", "Duplicate Roll Number", 2, "BCA", "dup@example.com", "9812345679", 2024),
        )


def test_marks_unique_constraint_on_natural_key(test_db):
    _insert_student(test_db, "BCA002")
    _insert_subject(test_db, "SUB001")
    test_db.execute(
        "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BCA002", "SUB001", 10, 50, 0, 1, "regular"),
    )
    test_db.commit()
    # Same roll_no + subject_code + semester + exam_type combination again
    # -- must be rejected, even with different mark values.
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BCA002", "SUB001", 15, 60, 0, 1, "regular"),
        )


def test_marks_allows_different_exam_type_for_same_subject(test_db):
    # A 'backlog' attempt for the same roll_no/subject/semester is a
    # DIFFERENT natural key (exam_type differs), so it must be allowed --
    # this is exactly what lets a student retake a failed subject.
    _insert_student(test_db, "BCA003")
    _insert_subject(test_db, "SUB002")
    test_db.execute(
        "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BCA003", "SUB002", 5, 10, 0, 1, "regular"),
    )
    test_db.execute(
        "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BCA003", "SUB002", 15, 60, 0, 1, "backlog"),
    )
    test_db.commit()
    count = test_db.execute(
        "SELECT COUNT(*) FROM marks WHERE roll_no = ? AND subject_code = ?", ("BCA003", "SUB002")
    ).fetchone()[0]
    assert count == 2


# ---------------------------------------------------------------------------
# CHECK constraints
# ---------------------------------------------------------------------------

def test_students_semester_check_constraint(test_db):
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BCA004", "Bad Semester", 9, "BCA", "bad@example.com", "9812345670", 2024),
        )


def test_users_must_change_password_check_constraint(test_db):
    # See modules/auth.py's change-password feature: must_change_password
    # is a 0/1 flag, same CHECK pattern as is_active on this same table.
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO users (username, password_hash, role, must_change_password) VALUES (?, ?, ?, ?)",
            ("baduser2", "somehash", "student", 2),
        )


def test_users_role_check_constraint(test_db):
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("baduser", "somehash", "superadmin"),
        )


def test_marks_internal_sanity_ceiling_check_constraint(test_db):
    # This is the STATIC, generic 0-100 ceiling the database itself can
    # enforce (config.MAX_MARK_CEILING) -- the precise, per-subject
    # ceiling is utils/validators.py's job, not the database's; see
    # database/db_setup.py's comment on the marks table.
    _insert_student(test_db, "BCA005")
    _insert_subject(test_db, "SUB003")
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BCA005", "SUB003", 150, 50, 0, 1, "regular"),
        )


def test_attendance_attended_cannot_exceed_held(test_db):
    _insert_student(test_db, "BCA006")
    _insert_subject(test_db, "SUB004")
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO attendance (roll_no, subject_code, classes_held, classes_attended, semester) "
            "VALUES (?, ?, ?, ?, ?)",
            ("BCA006", "SUB004", 10, 15, 1),
        )


def test_assignments_submitted_cannot_exceed_total_assigned(test_db):
    _insert_student(test_db, "BCA008")
    _insert_subject(test_db, "SUB007")
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO assignments (roll_no, subject_code, total_assigned, submitted, semester) "
            "VALUES (?, ?, ?, ?, ?)",
            ("BCA008", "SUB007", 5, 8, 1),
        )


def test_subjects_all_zero_max_marks_rejected(test_db):
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO subjects (subject_code, name, semester, credits, max_internal, max_external, max_practical) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("SUB005", "No Gradable Component", 1, 3, 0, 0, 0),
        )


# ---------------------------------------------------------------------------
# FOREIGN KEY constraints
# ---------------------------------------------------------------------------

def test_marks_foreign_key_rejects_unknown_student(test_db):
    _insert_subject(test_db, "SUB006")
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("NOSUCHROLL", "SUB006", 10, 50, 0, 1, "regular"),
        )


def test_marks_foreign_key_rejects_unknown_subject(test_db):
    _insert_student(test_db, "BCA007")
    with pytest.raises(sqlite3.IntegrityError):
        test_db.execute(
            "INSERT INTO marks (roll_no, subject_code, internal, external, practical, semester, exam_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BCA007", "NOSUCHSUBJECT", 10, 50, 0, 1, "regular"),
        )
