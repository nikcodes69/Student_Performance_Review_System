"""
database/db_setup.py
=====================
Creates the SQLite database file and every table, constraint, and index
the application needs. Run this file once, from the PROJECT ROOT folder
(student_performance_system/), using Python's "-m" module flag:

    python -m database.db_setup

WHY "-m database.db_setup" AND NOT "python database/db_setup.py":
This file does `import config`, and config.py lives one folder up, at the
project root. Running a file directly (python database/db_setup.py) only
adds THAT FILE'S OWN FOLDER (database/) to Python's import search path, so
`import config` fails with ModuleNotFoundError. Running it with -m instead
adds the CURRENT WORKING DIRECTORY (the project root, if that's where you
are standing) to the search path, so both database/db_setup.py and
config.py are visible. This is a project-wide convention -- every runnable
file in this project (later: modules/auth.py, etc.) is meant to be run the
same way, with -m, from the project root.

Every CREATE TABLE statement uses "IF NOT EXISTS", so running this script
again later is harmless -- it will not wipe existing data, it just confirms
the schema is already there.

IMPORTANT NOTE ON f-STRINGS IN THIS FILE:
Elsewhere in this project, the rule is "parameterized queries only, never
f-strings in SQL" -- that rule exists to stop SQL INJECTION, where
untrusted USER INPUT (e.g. something typed into a login box) gets pasted
directly into a query and changes its meaning. This file is different: it
only interpolates constants from config.py (numbers and words that WE, the
developers, wrote -- never anything a user typed) into one-time schema
definitions (DDL: CREATE TABLE/INDEX). There is no user input anywhere
near this file, so there is no injection risk. Every file that handles
real user input (modules/auth.py, modules/marks.py, etc.) will use
parameterized "?" placeholders exclusively -- that rule starts from
step 3 onward, where we start writing queries that touch user-supplied
values.

NORMALISATION SUMMARY (see chat explanation for full detail per table):
Every table below is in Third Normal Form (3NF): each table has a primary
key, every non-key column depends on the WHOLE primary key (2NF), and no
non-key column depends on another non-key column instead of the key (3NF).
"""

import sqlite3

import config
from utils.exceptions import DatabaseError
from utils.logger import get_logger

logger = get_logger(__name__)


def _quoted_list(values: tuple[str, ...]) -> str:
    """
    Turn a tuple of trusted, developer-defined strings into a SQL literal
    list for use inside a CHECK (... IN (...)) clause.

    Example: ("admin", "teacher", "student") -> "'admin', 'teacher', 'student'"

    Safe to use ONLY with constants from config.py -- never with data that
    came from a user, a form, or a file, because that would reopen the SQL
    injection risk explained at the top of this file.
    """
    return ", ".join(f"'{value}'" for value in values)


def get_connection() -> sqlite3.Connection:
    """
    Open a connection to the project's SQLite database file.

    SQLite has foreign-key enforcement turned OFF by default, and this is
    a setting stored on the CONNECTION, not saved inside the database file
    itself -- so every single connection we open, anywhere in this project,
    must run "PRAGMA foreign_keys = ON" or constraints like
    "FOREIGN KEY (roll_no) REFERENCES students(roll_no)" will be silently
    ignored. This function is the one place that does it, and later
    modules (starting with db_manager.py) will reuse this exact function
    instead of opening connections by hand.

    Returns:
        An open sqlite3.Connection with foreign key enforcement enabled.
    """
    # Make sure the database/ folder exists before sqlite3 tries to create
    # the .db file inside it.
    config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create every application table if it does not already exist.

    Tables are created in an order where "parent" tables (users, students,
    subjects) come before "child" tables that reference them via FOREIGN
    KEY (marks, attendance, semesters, audit_log). This is good practice
    for readability, though SQLite itself does not require parent tables
    to exist before a child table is CREATEd -- foreign keys are only
    checked when data is inserted/updated, not when the table is defined.

    Raises:
        DatabaseError: if any CREATE TABLE statement fails.
    """
    cursor = conn.cursor()

    try:
        # ---------------------------------------------------------------
        # users
        # ---------------------------------------------------------------
        # user_id is a SURROGATE key (a made-up auto-incrementing number)
        # rather than using username as the primary key. Reason: a username
        # is something a person might reasonably want to change later, but
        # a primary key should never change once other rows may reference
        # it. Using a surrogate key means "rename a username" is a simple
        # UPDATE, not a cascade of foreign key updates everywhere.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL
                                  CHECK (role IN ({_quoted_list(config.VALID_ROLES)})),
                is_active     INTEGER NOT NULL DEFAULT 1
                                  CHECK (is_active IN (0, 1)),
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login    TEXT
            )
        """)

        # ---------------------------------------------------------------
        # students
        # ---------------------------------------------------------------
        # roll_no is used directly as the PRIMARY KEY (a "natural key"),
        # unlike users.user_id. Reason: a roll number is assigned once by
        # the institution and never changes, and it is already the
        # human-facing identifier used on mark sheets, attendance
        # registers, everywhere -- adding a separate surrogate student_id
        # would just add an unnecessary extra column and extra joins with
        # no benefit. PRIMARY KEY already enforces uniqueness, which is
        # what satisfies the "UNIQUE constraint on roll_no" requirement.
        #
        # Convention used later in modules/auth.py: for a user whose role
        # is 'student', users.username IS their roll_no. That is how a
        # logged-in student account maps to a row in this table, without
        # needing an extra foreign key column on users (which the given
        # table spec does not include).
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS students (
                roll_no         TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                semester        INTEGER NOT NULL
                                    CHECK (semester BETWEEN {config.MIN_SEMESTER}
                                                     AND {config.MAX_SEMESTER}),
                branch          TEXT NOT NULL,
                email           TEXT NOT NULL,
                phone           TEXT NOT NULL,
                admission_year  INTEGER NOT NULL
                                    CHECK (admission_year >= {config.ADMISSION_YEAR_MIN}),
                is_active       INTEGER NOT NULL DEFAULT 1
                                    CHECK (is_active IN (0, 1)),
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ---------------------------------------------------------------
        # subjects
        # ---------------------------------------------------------------
        # subject_code (e.g. "CACS201") is a natural key for the same
        # reason as roll_no: it is institutionally assigned, stable, and
        # already the human-facing identifier.
        #
        # max_internal / max_external / max_practical are per-subject
        # ceilings (a lab-heavy subject might have max_practical=50 and
        # max_external=0, for instance). The final CHECK below simply
        # guards against a subject where ALL THREE are zero, which would
        # mean the subject has no gradable component at all -- almost
        # certainly a data-entry mistake.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS subjects (
                subject_code  TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                semester      INTEGER NOT NULL
                                  CHECK (semester BETWEEN {config.MIN_SEMESTER}
                                                   AND {config.MAX_SEMESTER}),
                credits       INTEGER NOT NULL
                                  CHECK (credits BETWEEN {config.MIN_CREDITS}
                                                   AND {config.MAX_CREDITS}),
                max_internal  INTEGER NOT NULL DEFAULT 0 CHECK (max_internal >= 0),
                max_external  INTEGER NOT NULL DEFAULT 0 CHECK (max_external >= 0),
                max_practical INTEGER NOT NULL DEFAULT 0 CHECK (max_practical >= 0),
                is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (max_internal + max_external + max_practical > 0)
            )
        """)

        # ---------------------------------------------------------------
        # marks
        # ---------------------------------------------------------------
        # The upper bound used here (MAX_MARK_CEILING = 100) is a generous,
        # STATIC sanity check -- it just stops absurd values like 9999 at
        # the database level. The REAL per-subject ceiling (internal must
        # not exceed THIS subject's max_internal) cannot be written as a
        # simple CHECK constraint, because SQLite's CHECK constraints
        # cannot reliably look up a value from a different table. That
        # exact, dynamic rule is enforced instead by utils/validators.py
        # BEFORE any INSERT/UPDATE reaches this table. This is a
        # deliberate two-layer defence: validators.py enforces precise,
        # per-subject business rules; the database CHECK constraints
        # enforce broad, absolute sanity limits as a last line of defence
        # even if a bug elsewhere skipped validation.
        #
        # UNIQUE(roll_no, subject_code, semester, exam_type) stops the same
        # exam attempt from being entered twice for the same student and
        # subject -- without it, a double-click on "Save" in the marks
        # entry form could silently create two conflicting rows and corrupt
        # the SGPA calculation.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS marks (
                mark_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no      TEXT NOT NULL,
                subject_code TEXT NOT NULL,
                internal     INTEGER NOT NULL
                                 CHECK (internal BETWEEN 0 AND {config.MAX_MARK_CEILING}),
                external     INTEGER NOT NULL
                                 CHECK (external BETWEEN 0 AND {config.MAX_MARK_CEILING}),
                practical    INTEGER NOT NULL
                                 CHECK (practical BETWEEN 0 AND {config.MAX_MARK_CEILING}),
                semester     INTEGER NOT NULL
                                 CHECK (semester BETWEEN {config.MIN_SEMESTER}
                                                  AND {config.MAX_SEMESTER}),
                exam_type    TEXT NOT NULL
                                 CHECK (exam_type IN ({_quoted_list(config.EXAM_TYPES)})),
                created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (roll_no) REFERENCES students (roll_no),
                FOREIGN KEY (subject_code) REFERENCES subjects (subject_code),
                UNIQUE (roll_no, subject_code, semester, exam_type)
            )
        """)

        # ---------------------------------------------------------------
        # attendance
        # ---------------------------------------------------------------
        # classes_attended <= classes_held is a CHECK that compares two
        # columns IN THE SAME ROW, which SQLite fully supports (unlike the
        # cross-TABLE comparison marks would need for its max_internal
        # rule above). A student physically cannot attend more classes
        # than were held.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                att_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no           TEXT NOT NULL,
                subject_code      TEXT NOT NULL,
                classes_held      INTEGER NOT NULL CHECK (classes_held >= 0),
                classes_attended  INTEGER NOT NULL
                                      CHECK (classes_attended >= 0
                                             AND classes_attended <= classes_held),
                semester          INTEGER NOT NULL,
                created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (roll_no) REFERENCES students (roll_no),
                FOREIGN KEY (subject_code) REFERENCES subjects (subject_code),
                UNIQUE (roll_no, subject_code, semester)
            )
        """)

        # ---------------------------------------------------------------
        # semesters
        # ---------------------------------------------------------------
        # This table has no single-column primary key -- (roll_no,
        # semester) together form a COMPOSITE PRIMARY KEY, because the
        # natural way to uniquely identify "one student's result for one
        # semester" is exactly that pair. sgpa/cgpa/result_status are
        # nullable (no NOT NULL) because a row can exist the moment a
        # student is registered for a semester, before marks.py and
        # grades.py have calculated a result yet -- result_status starts
        # as 'pending' in that case.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS semesters (
                roll_no       TEXT NOT NULL,
                semester      INTEGER NOT NULL
                                  CHECK (semester BETWEEN {config.MIN_SEMESTER}
                                                   AND {config.MAX_SEMESTER}),
                sgpa          REAL CHECK (sgpa BETWEEN 0 AND {config.MAX_GRADE_POINT}),
                cgpa          REAL CHECK (cgpa BETWEEN 0 AND {config.MAX_GRADE_POINT}),
                result_status TEXT
                                  CHECK (result_status IN ({_quoted_list(config.RESULT_STATUSES)})),
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (roll_no, semester),
                FOREIGN KEY (roll_no) REFERENCES students (roll_no)
            )
        """)

        # ---------------------------------------------------------------
        # audit_log
        # ---------------------------------------------------------------
        # record_id is stored as TEXT even though some tables it points to
        # (marks, users) use an INTEGER primary key. This is deliberate:
        # audit_log needs to record changes to ANY table, whose primary
        # keys are a mix of INTEGER (mark_id, user_id) and TEXT (roll_no,
        # subject_code). Storing record_id as TEXT lets one column handle
        # both -- SQLite will happily store "45" as text, and we convert
        # back to int only where needed when displaying it.
        #
        # old_value/new_value store a JSON-encoded snapshot of the row
        # before/after the change (built by modules/audit.py, not this
        # file). old_value is NULL for an INSERT (there was no "before").
        #
        # No updated_at column here, unlike every other table. That is
        # intentional, not an oversight: an audit log entry is written
        # once and never modified afterwards -- if it could be updated, it
        # would no longer be a trustworthy record of what happened. Its
        # single "timestamp" column already answers "when was this
        # written", which is all an immutable row needs.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                action     TEXT NOT NULL
                               CHECK (action IN ({_quoted_list(config.AUDIT_ACTIONS)})),
                table_name TEXT NOT NULL,
                record_id  TEXT NOT NULL,
                old_value  TEXT,
                new_value  TEXT,
                timestamp  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)

        conn.commit()
        logger.info("All tables created successfully (or already existed).")

    except sqlite3.Error as error:
        conn.rollback()
        logger.error("Failed to create tables: %s", error)
        raise DatabaseError(f"Could not create database tables: {error}") from error


def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Create indexes on columns that will be searched or joined on often.

    WHY THESE COLUMNS: an index speeds up WHERE/JOIN lookups on a column,
    at the small cost of extra disk space and slightly slower writes. We
    only index columns that will be queried frequently and are NOT already
    covered by an existing UNIQUE or PRIMARY KEY index (SQLite creates
    those automatically, so indexing them again would be redundant):
      - marks/attendance will constantly be filtered "WHERE roll_no = ?"
        (a student's own record) or "WHERE subject_code = ?" (a subject's
        class-wide marks).
      - students.semester and students.branch are used to filter student
        lists on the Admin/Teacher dashboards.
      - audit_log is searched by table_name + record_id ("show me the
        history of this one record") and by user_id ("show me everything
        this teacher changed").

    Raises:
        DatabaseError: if any CREATE INDEX statement fails.
    """
    cursor = conn.cursor()

    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_students_semester ON students (semester)",
        "CREATE INDEX IF NOT EXISTS idx_students_branch ON students (branch)",
        "CREATE INDEX IF NOT EXISTS idx_marks_roll_no ON marks (roll_no)",
        "CREATE INDEX IF NOT EXISTS idx_marks_subject_code ON marks (subject_code)",
        "CREATE INDEX IF NOT EXISTS idx_attendance_roll_no ON attendance (roll_no)",
        "CREATE INDEX IF NOT EXISTS idx_attendance_subject_code ON attendance (subject_code)",
        "CREATE INDEX IF NOT EXISTS idx_semesters_roll_no ON semesters (roll_no)",
        "CREATE INDEX IF NOT EXISTS idx_audit_log_table_record "
        "ON audit_log (table_name, record_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log (user_id)",
    ]

    try:
        for statement in index_statements:
            cursor.execute(statement)
        conn.commit()
        logger.info("All indexes created successfully (or already existed).")

    except sqlite3.Error as error:
        conn.rollback()
        logger.error("Failed to create indexes: %s", error)
        raise DatabaseError(f"Could not create database indexes: {error}") from error


def initialize_database() -> None:
    """
    Full setup entry point: open a connection, create all tables, create
    all indexes, then close the connection. This is the single function
    other code should call to make sure the database is ready to use.
    """
    conn = get_connection()
    try:
        create_tables(conn)
        create_indexes(conn)
        logger.info("Database initialised at %s", config.DB_PATH)
    finally:
        # finally guarantees the connection is closed even if an error
        # was raised above -- otherwise the .db file could be left locked.
        conn.close()


if __name__ == "__main__":
    # This block only runs when the file is executed directly
    # (python database/db_setup.py), not when it is imported by another
    # module -- that is what "if __name__ == '__main__'" means in Python.
    initialize_database()
