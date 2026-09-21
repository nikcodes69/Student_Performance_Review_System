"""
tests/test_search.py
=====================
pytest tests for modules/search.py: substring matching across students
(roll_no/name) and subjects (subject_code/name).

Uses the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_search.py -v
"""

import pytest

import config
import database.db_setup as db_setup
import modules.search as search
from database.db_setup import create_indexes, create_tables, get_connection
from modules import auth, students, subjects


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_search.db")
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


def _seed_students_and_subjects(admin_user: dict) -> None:
    students.create_student("BCA045", "Sita Sharma", 1, "BCA", "sita@example.com", "9812345678", 2024, admin_user)
    students.create_student("BCA046", "Ram Thapa", 1, "BCA", "ram@example.com", "9812345679", 2024, admin_user)
    subjects.create_subject("CACS301", "Data Structures", 3, 3, admin_user)
    subjects.create_subject("CACS302", "Operating Systems", 3, 3, admin_user)


# ---------------------------------------------------------------------------
# search_students()
# ---------------------------------------------------------------------------

def test_search_students_matches_roll_no_substring(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    results = search.search_students("bca045")  # lowercase, case-insensitive
    assert [row["roll_no"] for row in results] == ["BCA045"]


def test_search_students_matches_name_substring(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    results = search.search_students("sharma")
    assert [row["roll_no"] for row in results] == ["BCA045"]


def test_search_students_empty_query_matches_nothing(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    assert search.search_students("") == []
    assert search.search_students("   ") == []


def test_search_students_excludes_inactive_by_default(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)
    students.deactivate_student("BCA045", admin_user)

    assert search.search_students("BCA045") == []
    assert len(search.search_students("BCA045", include_inactive=True)) == 1


def test_search_students_no_match_returns_empty_list(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    assert search.search_students("nonexistent") == []


# ---------------------------------------------------------------------------
# search_subjects()
# ---------------------------------------------------------------------------

def test_search_subjects_matches_subject_code_substring(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    results = search.search_subjects("cacs301")
    assert [row["subject_code"] for row in results] == ["CACS301"]


def test_search_subjects_matches_name_substring(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    results = search.search_subjects("operating")
    assert [row["subject_code"] for row in results] == ["CACS302"]


def test_search_subjects_partial_code_matches_multiple(test_db):
    admin_user = _admin_user()
    _seed_students_and_subjects(admin_user)

    results = search.search_subjects("cacs30")
    assert {row["subject_code"] for row in results} == {"CACS301", "CACS302"}
