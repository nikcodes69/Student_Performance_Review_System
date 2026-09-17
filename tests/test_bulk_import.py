"""
tests/test_bulk_import.py
==========================
pytest tests for the bulk CSV/Excel import feature:
utils/bulk_import.py's parse_uploaded_file() (pure -- no Streamlit
runtime needed), and modules/students.py's/modules/subjects.py's
_validate_bulk_student_row()/_validate_bulk_subject_row() -- the
row-level validation logic that runs during the "preview" step, before
anything is committed (see utils/bulk_import.py's module docstring for
why that preview step exists).

WHY render_bulk_import() ITSELF IS NOT UNIT-TESTED HERE: it drives real
Streamlit widgets (st.file_uploader, st.button) the same way
render_data_table() does -- see tests/test_table_view.py's module
docstring for why that class of function is left to manual verification
against the real app instead of a heavier st.testing.v1.AppTest harness.
This file focuses on the two PURE, reusable pieces that do not depend on
any Streamlit runtime: parsing an uploaded file, and validating one row.

HOW A FAKE UPLOADED FILE IS BUILT: st.file_uploader() normally returns a
Streamlit UploadedFile, but parse_uploaded_file() only ever calls
pandas.read_csv()/read_excel() on it and reads its .name attribute --
both of which work identically on a plain io.BytesIO with a .name
attribute bolted on, so no real Streamlit upload is needed to test it.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py and
tests/test_auth.py -- see those files' module docstrings for the full
rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_bulk_import.py -v
"""

from io import BytesIO

import pytest

import config
import database.db_setup as db_setup
from database.db_manager import execute_write
from database.db_setup import create_indexes, create_tables, get_connection
from modules.students import _validate_bulk_student_row
from modules.subjects import _validate_bulk_subject_row
from utils.bulk_import import parse_uploaded_file
from utils.exceptions import DuplicateRecordError, ValidationError


class _FakeUploadedFile(BytesIO):
    """A BytesIO with a .name attribute -- everything
    parse_uploaded_file() actually needs from a real Streamlit
    UploadedFile (see this file's module docstring)."""

    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    see tests/test_database.py's test_db fixture for the full rationale."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_bulk_import.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


# ---------------------------------------------------------------------------
# parse_uploaded_file()
# ---------------------------------------------------------------------------

def test_parse_uploaded_file_reads_csv_as_strings():
    content = b"roll_no,semester\nS1,1\nS2,2\n"
    df = parse_uploaded_file(_FakeUploadedFile(content, "students.csv"))
    assert list(df.columns) == ["roll_no", "semester"]
    assert df["semester"].tolist() == ["1", "2"]  # NOT int64 -- see docstring on why


def test_parse_uploaded_file_strips_header_whitespace():
    content = b" roll_no , name \nS1,Alice\n"
    df = parse_uploaded_file(_FakeUploadedFile(content, "students.csv"))
    assert list(df.columns) == ["roll_no", "name"]


def test_parse_uploaded_file_blank_cells_become_empty_string_not_nan():
    content = b"roll_no,phone\nS1,\n"
    df = parse_uploaded_file(_FakeUploadedFile(content, "students.csv"))
    assert df.iloc[0]["phone"] == ""


def test_parse_uploaded_file_rejects_unsupported_extension():
    with pytest.raises(ValueError):
        parse_uploaded_file(_FakeUploadedFile(b"whatever", "students.txt"))


def test_parse_uploaded_file_reads_xlsx():
    import pandas as pd
    buffer = BytesIO()
    pd.DataFrame([{"roll_no": "S1", "name": "Alice"}]).to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    fake_file = _FakeUploadedFile(buffer.read(), "students.xlsx")
    df = parse_uploaded_file(fake_file)
    assert df.iloc[0]["roll_no"] == "S1"
    assert df.iloc[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# _validate_bulk_student_row()
# ---------------------------------------------------------------------------

_VALID_STUDENT_ROW = {
    "roll_no": "S100", "name": "Alice", "semester": "1", "branch": "BCA",
    "email": "alice@example.com", "phone": "9812345678", "admission_year": "2024",
}


def test_validate_bulk_student_row_accepts_a_valid_row(test_db):
    _validate_bulk_student_row(_VALID_STUDENT_ROW)  # must not raise


def test_validate_bulk_student_row_rejects_bad_email(test_db):
    row = {**_VALID_STUDENT_ROW, "email": "not-an-email"}
    with pytest.raises(ValidationError):
        _validate_bulk_student_row(row)


def test_validate_bulk_student_row_rejects_non_numeric_semester(test_db):
    row = {**_VALID_STUDENT_ROW, "semester": "first"}
    with pytest.raises(ValidationError):
        _validate_bulk_student_row(row)


def test_validate_bulk_student_row_rejects_existing_roll_no(test_db):
    execute_write(
        "INSERT INTO students (roll_no, name, semester, branch, email, phone, admission_year) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("S100", "Existing Student", 1, "BCA", "existing@example.com", "9800000000", 2024),
    )
    with pytest.raises(DuplicateRecordError):
        _validate_bulk_student_row(_VALID_STUDENT_ROW)


# ---------------------------------------------------------------------------
# _validate_bulk_subject_row()
# ---------------------------------------------------------------------------

_VALID_SUBJECT_ROW = {
    "subject_code": "SUB100", "name": "Fixture Subject", "semester": "1",
    "credits": "3", "max_internal": "20", "max_external": "80", "max_practical": "0",
}


def test_validate_bulk_subject_row_accepts_a_valid_row(test_db):
    _validate_bulk_subject_row(_VALID_SUBJECT_ROW)  # must not raise


def test_validate_bulk_subject_row_rejects_all_zero_max_marks(test_db):
    row = {**_VALID_SUBJECT_ROW, "max_internal": "0", "max_external": "0", "max_practical": "0"}
    with pytest.raises(ValidationError):
        _validate_bulk_subject_row(row)


def test_validate_bulk_subject_row_rejects_existing_subject_code(test_db):
    execute_write(
        "INSERT INTO subjects (subject_code, name, semester, credits, max_internal, max_external, max_practical) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("SUB100", "Existing Subject", 1, 3, 20, 80, 0),
    )
    with pytest.raises(DuplicateRecordError):
        _validate_bulk_subject_row(_VALID_SUBJECT_ROW)
