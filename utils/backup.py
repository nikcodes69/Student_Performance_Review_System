"""
utils/backup.py
================
Full-database backup: every table in the application, exported together
as one downloadable .zip of CSV files (one file per table). Admin-only
-- see modules/auth.py's render_user_management_page(), which is where
this is surfaced (the closest thing this app has to an "admin settings"
area).

WHY A ZIP OF CSVs, NOT A COPY OF THE .db FILE ITSELF: the deployed app
runs against Turso (a REMOTE cloud database over the libsql protocol),
not a local SQLite file that could simply be copied -- see
database/db_setup.py's get_connection() for the dual-backend design, and
its module docstring for why local SQLite and Turso must both keep
working identically. A zip of CSVs works the same way regardless of
which backend is active, is human-readable and importable into any
spreadsheet tool without needing SQLite-specific software, and reuses
the SAME CSV-encoding helper (utils/table_view.py's to_csv_bytes())
already used for every other export button in this app -- one
export format, not two.

WHY users.password_hash IS DELIBERATELY EXCLUDED: a backup file that
included bcrypt hashes would be a genuinely sensitive artifact to have
sitting in someone's Downloads folder, even though a bcrypt hash is not
the plaintext password itself -- nothing about restoring or inspecting a
backup needs it, so it is simply never selected out of the database in
the first place (not filtered out afterwards, which would risk a future
change accidentally re-including it).

THIS IS AN EXPORT, NOT A RESTORE TOOL: there is deliberately no "upload
a backup to restore it" counterpart. Restoring a full database backup is
a fundamentally different, much higher-risk operation (foreign key
ordering, conflicting primary keys, partial failures partway through)
than this project's existing bulk import (which only ever adds NEW,
individually-validated rows -- see utils/bulk_import.py) -- building a
safe restore path is a deliberately separate, out-of-scope piece of work
from "let an Admin download a copy of everything".
"""

import zipfile
from datetime import datetime
from io import BytesIO

from database.db_manager import fetch_all
from utils.table_view import to_csv_bytes

# Every table this backup exports, and the exact columns pulled from
# each -- explicit column lists (not "SELECT *") for two reasons: it
# is what lets users.password_hash be excluded (see module docstring),
# and it means a future new column on some table does not silently
# change what a backup contains without a deliberate decision to
# include it here.
_BACKUP_TABLES: dict[str, str] = {
    "students": (
        "SELECT roll_no, name, semester, branch, email, phone, admission_year, "
        "is_active, created_at, updated_at FROM students ORDER BY roll_no"
    ),
    "subjects": (
        "SELECT subject_code, name, semester, credits, "
        "is_active, created_at, updated_at FROM subjects ORDER BY subject_code"
    ),
    "marks": (
        "SELECT mark_id, roll_no, subject_code, internal, external, practical, "
        "semester, exam_type, created_at, updated_at FROM marks ORDER BY mark_id"
    ),
    "attendance": (
        "SELECT att_id, roll_no, subject_code, classes_held, classes_attended, "
        "semester, created_at, updated_at FROM attendance ORDER BY att_id"
    ),
    "assignments": (
        "SELECT assignment_id, roll_no, subject_code, total_assigned, submitted, "
        "semester, created_at, updated_at FROM assignments ORDER BY assignment_id"
    ),
    "semesters": (
        "SELECT roll_no, semester, sgpa, cgpa, result_status, created_at, updated_at "
        "FROM semesters ORDER BY roll_no, semester"
    ),
    "teacher_subjects": (
        "SELECT teacher_id, subject_code, created_at FROM teacher_subjects "
        "ORDER BY teacher_id, subject_code"
    ),
    "users": (
        # password_hash deliberately excluded -- see module docstring.
        "SELECT user_id, username, role, is_active, must_change_password, "
        "created_at, updated_at, last_login FROM users ORDER BY user_id"
    ),
    "audit_log": (
        "SELECT log_id, user_id, action, table_name, record_id, old_value, "
        "new_value, timestamp FROM audit_log ORDER BY log_id"
    ),
}


def generate_full_backup() -> bytes:
    """
    Export every table in _BACKUP_TABLES as one CSV file each, packaged
    into a single in-memory .zip archive.

    Returns:
        The zip archive's raw bytes, ready for st.download_button. An
        empty table still gets a CSV file in the archive (just a header
        row with no data rows), so the archive's table list is always
        complete and predictable regardless of what has data yet.
    """
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for table_name, query in _BACKUP_TABLES.items():
            rows = fetch_all(query)
            if rows:
                csv_bytes = to_csv_bytes(rows)
            else:
                # to_csv_bytes() needs at least one row to infer column
                # headers from (pandas has nothing to build a DataFrame's
                # columns out of otherwise) -- an empty table still gets
                # a valid, header-only CSV by reusing the same SELECT's
                # column names directly instead.
                columns = _select_columns(query)
                csv_bytes = (",".join(columns) + "\n").encode("utf-8")
            archive.writestr(f"{table_name}.csv", csv_bytes)

    return buffer.getvalue()


def _select_columns(query: str) -> list[str]:
    """Extract the column names from one of _BACKUP_TABLES' own
    hand-written "SELECT col1, col2, ... FROM ..." strings -- safe here
    ONLY because every query above is a fixed, developer-written
    constant (never built from user input), the same trusted-constant
    reasoning database/db_setup.py's module docstring explains for its
    own f-string-built DDL."""
    select_clause = query[len("SELECT "):query.index(" FROM")]
    return [column.strip() for column in select_clause.split(",")]


def backup_filename() -> str:
    """A timestamped filename for the backup download, e.g.
    'backup_20260917_153000.zip' -- the timestamp is what lets an Admin
    tell two downloaded backups apart later."""
    return f"backup_{datetime.now():%Y%m%d_%H%M%S}.zip"
