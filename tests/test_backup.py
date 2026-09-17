"""
tests/test_backup.py
=====================
pytest tests for utils/backup.py's generate_full_backup() -- the "Full
Database Backup" button on the User Management page (Admin only).

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see
that file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_backup.py -v
"""

import zipfile
from io import BytesIO

import pytest

import config
import database.db_setup as db_setup
from database.db_manager import execute_write
from database.db_setup import create_indexes, create_tables, get_connection
from utils.backup import backup_filename, generate_full_backup


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database -- see
    tests/test_database.py's test_db fixture for the full rationale."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_backup.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def test_backup_contains_one_csv_per_table_even_when_empty(test_db):
    backup_bytes = generate_full_backup()
    archive = zipfile.ZipFile(BytesIO(backup_bytes))

    expected_files = {
        "students.csv", "subjects.csv", "marks.csv", "attendance.csv",
        "assignments.csv", "semesters.csv", "teacher_subjects.csv",
        "users.csv", "audit_log.csv",
    }
    assert set(archive.namelist()) == expected_files

    # An empty table must still produce a valid, header-only CSV, not a
    # missing or malformed file.
    students_csv = archive.read("students.csv").decode("utf-8")
    assert students_csv.strip() == "roll_no,name,semester,branch,email,phone,admission_year,is_active,created_at,updated_at"


def test_backup_includes_real_data(test_db):
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S1", "Alice", 1, "BCA", "alice@example.com", "9812345678", 2024),
    )

    backup_bytes = generate_full_backup()
    archive = zipfile.ZipFile(BytesIO(backup_bytes))
    students_csv = archive.read("students.csv").decode("utf-8")

    assert "S1" in students_csv
    assert "Alice" in students_csv


def test_backup_excludes_password_hash(test_db):
    execute_write(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        ("admin", "$2b$12$thisisaverysecretbcryptdigest", config.ROLE_ADMIN),
    )

    backup_bytes = generate_full_backup()
    archive = zipfile.ZipFile(BytesIO(backup_bytes))
    users_csv = archive.read("users.csv").decode("utf-8")

    assert "admin" in users_csv  # the account itself is exported...
    assert "thisisaverysecretbcryptdigest" not in users_csv  # ...but never its hash
    assert "password_hash" not in users_csv.splitlines()[0]  # not even as a header


def test_backup_filename_ends_with_zip_extension():
    assert backup_filename().endswith(".zip")
