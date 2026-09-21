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


def _get_turso_credentials() -> tuple[str | None, str | None]:
    """
    Read Turso (cloud database) credentials from Streamlit secrets, if
    configured.

    Deliberately a SEPARATE function from get_connection(), rather than
    inlined, for one specific reason: tests/test_database.py monkeypatches
    THIS function directly to force (None, None), guaranteeing the test
    suite always uses a local, throwaway SQLite file and never touches a
    real Turso database -- even on a machine that has real Turso
    credentials configured in .streamlit/secrets.toml for the deployed
    app. Splitting this out is what makes that guarantee possible.

    Returns:
        (database_url, auth_token) if both are configured in
        st.secrets, otherwise (None, None) -- which happens on any local
        development machine or CI run without a .streamlit/secrets.toml,
        and is not treated as an error, just "cloud mode isn't configured
        here".
    """
    try:
        import streamlit as st
        return st.secrets.get("TURSO_DATABASE_URL"), st.secrets.get("TURSO_AUTH_TOKEN")
    except Exception:
        # Covers every way this can fail to matter here: no secrets.toml
        # file at all, a secrets.toml missing these specific keys, or this
        # code running completely outside any Streamlit context (plain
        # scripts, pytest) where st.secrets may not even be usable.
        return None, None


def get_connection():
    """
    Open a connection to the project's database -- Turso (a SQLite-
    compatible cloud database, via the `libsql` package) if credentials
    are configured in Streamlit secrets, otherwise a local SQLite file
    (config.DB_PATH) exactly as this function worked before Turso support
    was added.

    WHY TWO BACKENDS INSTEAD OF JUST SWITCHING TO TURSO EVERYWHERE: local
    development and the 85-test pytest suite should never depend on
    network access or risk writing throwaway/deliberately-invalid test
    data into a real, possibly-shared cloud database. Falling back to
    local SQLite whenever Turso credentials aren't configured (which is
    the normal case for `pytest`, and for `python -m database.db_setup`
    run by hand) keeps every test and every earlier verification script
    in this project working completely unchanged.

    SQLite (and libSQL, which is SQLite-compatible) has foreign-key
    enforcement turned OFF by default, and this is a setting stored on the
    CONNECTION, not saved inside the database itself -- so every
    connection this function returns has "PRAGMA foreign_keys = ON" run on
    it explicitly, regardless of which backend it is.

    Returns:
        An open connection (sqlite3.Connection, or libsql's Connection
        when using Turso) with foreign key enforcement enabled. Both
        expose the same query interface (.execute(), .commit(),
        .rollback(), "?" placeholders) that the rest of this project uses.
    """
    turso_url, turso_token = _get_turso_credentials()

    if turso_url and turso_token:
        import libsql
        conn = libsql.connect(database=turso_url, auth_token=turso_token)
    else:
        # Make sure the database/ folder exists before sqlite3 tries to
        # create the .db file inside it.
        config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DB_PATH)

    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn) -> None:
    """
    Create every application table if it does not already exist.

    Tables are created in an order where "parent" tables (users, students,
    subjects) come before "child" tables that reference them via FOREIGN
    KEY (marks, attendance, semesters, audit_log). This is good practice
    for readability, though SQLite itself does not require parent tables
    to exist before a child table is CREATEd -- foreign keys are only
    checked when data is inserted/updated, not when the table is defined.

    Args:
        conn: Whatever get_connection() returned -- a sqlite3.Connection
            (local) or a libsql Connection (Turso). Not type-hinted to one
            specific class since it accepts either; both expose the same
            .cursor()/.execute()/.commit() interface this function uses.

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
                must_change_password INTEGER NOT NULL DEFAULT 0
                                  CHECK (must_change_password IN (0, 1)),
                google_email  TEXT,
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login    TEXT,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until  TEXT
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
        # NO max_internal/max_external/max_practical COLUMNS HERE (unlike
        # an earlier version of this schema): every subject now uses the
        # EXACT SAME marks breakdown (config.MAX_INTERNAL_MARKS/
        # MAX_EXTERNAL_MARKS/MAX_PRACTICAL_MARKS -- 25/50/25, 100 total),
        # so a per-subject column would just repeat the identical value on
        # every single row -- storing data that never actually varies is
        # exactly what 3NF's "every non-key column must depend on the
        # key" principle asks us to avoid (these three values do not
        # depend on WHICH subject the row describes; they are the same
        # for all of them). See migrate_schema() below for the one-time
        # table-rebuild that removed these columns from a database
        # created before this change.
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
                is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ---------------------------------------------------------------
        # marks
        # ---------------------------------------------------------------
        # Each component's own EXACT ceiling is now a fixed, global
        # constant (config.MAX_INTERNAL_MARKS/MAX_EXTERNAL_MARKS/
        # MAX_PRACTICAL_MARKS), not a per-subject lookup -- so, unlike an
        # earlier version of this schema, the database CHECK constraint
        # below can express the REAL rule directly (internal <= 25,
        # external <= 50, practical <= 25), not just a generic 0-100
        # sanity ceiling. utils/validators.py's validate_mark_value()
        # still enforces the identical rule BEFORE an INSERT/UPDATE is
        # even attempted, for a friendlier error message -- this CHECK
        # constraint is the second, independent layer of the same
        # two-layer defence used throughout this project.
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
                                 CHECK (internal BETWEEN 0 AND {config.MAX_INTERNAL_MARKS}),
                external     INTEGER NOT NULL
                                 CHECK (external BETWEEN 0 AND {config.MAX_EXTERNAL_MARKS}),
                practical    INTEGER NOT NULL
                                 CHECK (practical BETWEEN 0 AND {config.MAX_PRACTICAL_MARKS}),
                semester     INTEGER NOT NULL
                                 CHECK (semester BETWEEN {config.MIN_SEMESTER}
                                                  AND {config.MAX_SEMESTER}),
                exam_type    TEXT NOT NULL
                                 CHECK (exam_type IN ({_quoted_list(config.EXAM_TYPES)})),
                is_published INTEGER NOT NULL DEFAULT 0
                                 CHECK (is_published IN (0, 1)),
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
        # assignments
        # ---------------------------------------------------------------
        # Same shape as attendance immediately above, for the same reason:
        # "submitted cannot exceed total_assigned" is a same-row CHECK,
        # exactly like "classes_attended <= classes_held". This table
        # exists so the "assignments submitted" feature the ML models use
        # (see ml/generate_data.py) can be computed from real records
        # instead of asking a Teacher to type a number in by hand every
        # time a prediction is requested -- see
        # modules/ml_predictions.py's _compute_assignment_engagement().
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assignments (
                assignment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no        TEXT NOT NULL,
                subject_code   TEXT NOT NULL,
                total_assigned INTEGER NOT NULL CHECK (total_assigned >= 0),
                submitted      INTEGER NOT NULL
                                   CHECK (submitted >= 0 AND submitted <= total_assigned),
                semester       INTEGER NOT NULL,
                created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (roll_no) REFERENCES students (roll_no),
                FOREIGN KEY (subject_code) REFERENCES subjects (subject_code),
                UNIQUE (roll_no, subject_code, semester)
            )
        """)

        # ---------------------------------------------------------------
        # teacher_subjects
        # ---------------------------------------------------------------
        # A many-to-many mapping: which Teacher accounts are allowed to
        # enter marks/attendance/assignments for which subjects. Closes a
        # real RBAC gap this project shipped with -- before this table
        # existed, ANY Teacher account could enter data for ANY subject
        # (documented as a known limitation in the README). teacher_id
        # references users.user_id rather than a separate "teachers"
        # table, since a Teacher IS a row in users (role='teacher') --
        # there is no separate teacher-profile table anywhere in this
        # schema, so there is nothing else to reference.
        #
        # No is_active flag, unlike students/subjects: an assignment
        # either exists or it doesn't -- "temporarily disabled" has no
        # real-world meaning here that a straight DELETE (removing the
        # row) doesn't already capture equally well, and the audit_log
        # entry written alongside every INSERT/DELETE on this table (see
        # modules/teacher_subjects.py) is what preserves the history of
        # who was assigned when, not a soft-delete flag on the row itself.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teacher_subjects (
                teacher_id    INTEGER NOT NULL,
                subject_code  TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (teacher_id, subject_code),
                FOREIGN KEY (teacher_id) REFERENCES users (user_id),
                FOREIGN KEY (subject_code) REFERENCES subjects (subject_code)
            )
        """)

        # ---------------------------------------------------------------
        # pending_accounts
        # ---------------------------------------------------------------
        # An Admin-maintained allowlist: "this email is pre-approved to
        # self-register as a Teacher or Admin" (see modules/auth.py's
        # invite_account()). This is what makes Teacher/Admin signup safe
        # to offer as a public form at all -- without it, ANY visitor
        # could grant themselves Teacher/Admin access, exactly the
        # security hole this project's signup design has deliberately
        # avoided since it was first built (see modules/auth.py's module
        # docstring). email is the PRIMARY KEY (not a surrogate id) since
        # "is this email currently invited" is the only lookup this table
        # ever needs -- the same reasoning as students.roll_no/
        # subjects.subject_code being natural keys (see those tables'
        # comments).
        #
        # A row here is consumed exactly once: the moment someone signs
        # up (by password, via modules/auth.py's signup_invited_account(),
        # or automatically via Google Sign-In, via
        # authenticate_with_google()) with a matching email, the account
        # is created AND this row is deleted in the same breath -- there
        # is no "used" flag, because a consumed invite has nothing left
        # to represent; the resulting user account IS the record of it.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS pending_accounts (
                email        TEXT PRIMARY KEY,
                role         TEXT NOT NULL
                                 CHECK (role IN ({_quoted_list(config.INVITABLE_ROLES)})),
                invited_by   INTEGER NOT NULL,
                created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (invited_by) REFERENCES users (user_id)
            )
        """)

        # ---------------------------------------------------------------
        # announcements
        # ---------------------------------------------------------------
        # target_role NULL means "visible to every role" -- a targeted
        # announcement (e.g. Teacher-only) sets it to one of
        # config.VALID_ROLES instead. is_active is the same soft-delete
        # pattern used by students/subjects (see those tables): taking an
        # announcement down keeps its row (and audit trail) rather than
        # erasing history with a real DELETE.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS announcements (
                announcement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT NOT NULL,
                message         TEXT NOT NULL,
                posted_by       INTEGER NOT NULL,
                target_role     TEXT
                                    CHECK (target_role IS NULL
                                           OR target_role IN ({_quoted_list(config.VALID_ROLES)})),
                is_active       INTEGER NOT NULL DEFAULT 1
                                    CHECK (is_active IN (0, 1)),
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (posted_by) REFERENCES users (user_id)
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

    except (sqlite3.Error, ValueError) as error:
        # ValueError, alongside sqlite3's own exception hierarchy, because
        # the `libsql` package (used for the Turso cloud backend -- see
        # get_connection() above) raises a plain ValueError for every kind
        # of database error, rather than sqlite3.IntegrityError/
        # OperationalError/etc. -- confirmed directly against a real Turso
        # database, not assumed. Scoped narrowly to this one try block
        # (which only ever runs CREATE TABLE statements), so this does not
        # risk masking an unrelated ValueError from somewhere else.
        conn.rollback()
        logger.error("Failed to create tables: %s", error)
        raise DatabaseError(f"Could not create database tables: {error}") from error


def create_indexes(conn) -> None:
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
        "CREATE INDEX IF NOT EXISTS idx_assignments_roll_no ON assignments (roll_no)",
        "CREATE INDEX IF NOT EXISTS idx_assignments_subject_code ON assignments (subject_code)",
        "CREATE INDEX IF NOT EXISTS idx_teacher_subjects_subject_code ON teacher_subjects (subject_code)",
        "CREATE INDEX IF NOT EXISTS idx_semesters_roll_no ON semesters (roll_no)",
        "CREATE INDEX IF NOT EXISTS idx_audit_log_table_record "
        "ON audit_log (table_name, record_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log (user_id)",
        "CREATE INDEX IF NOT EXISTS idx_announcements_target_role ON announcements (target_role)",
    ]

    try:
        for statement in index_statements:
            cursor.execute(statement)
        conn.commit()
        logger.info("All indexes created successfully (or already existed).")

    except (sqlite3.Error, ValueError) as error:
        # See create_tables()'s comment above for why ValueError is caught
        # here too.
        conn.rollback()
        logger.error("Failed to create indexes: %s", error)
        raise DatabaseError(f"Could not create database indexes: {error}") from error


def migrate_schema(conn) -> None:
    """
    Apply schema changes needed on a database that was created BEFORE a
    given column existed -- currently users.must_change_password (see
    modules/auth.py's forced-password-change feature), users.google_email
    (see modules/auth.py's Google Sign-In support),
    users.failed_login_attempts/locked_until (see modules/auth.py's
    failed-login lockout), and marks.is_published (see
    modules/marks.py's publish_marks()).

    WHY THIS FUNCTION EXISTS, SEPARATE FROM create_tables(): every
    CREATE TABLE statement above uses "IF NOT EXISTS", which is a no-op
    the moment the table already exists -- it does NOT retroactively add
    a new column to a table that is already there. That is fine for a
    brand-new install (the CREATE TABLE statement already includes both
    columns), but this project's real, deployed Turso database already
    has a "users" table from before either column was added, and it
    holds real accounts (including the live admin account) that must not
    be touched or lost. This function's job is exactly that retrofit:
    add whichever column a users table still predates.

    SAFE TO RUN EVERY TIME THE APP STARTS, on both a brand-new database
    (where users already has both columns, because create_tables() just
    created it that way) and an old one (where one or both are missing):
    PRAGMA table_info() is used to check whether each column is already
    there before trying to add it, so this never runs the same ALTER
    TABLE twice.

    Args:
        conn: Whatever get_connection() returned.

    Raises:
        DatabaseError: if the migration check or ALTER TABLE fails.
    """
    cursor = conn.cursor()
    try:
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(users)").fetchall()}
        if "must_change_password" not in existing_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN must_change_password "
                "INTEGER NOT NULL DEFAULT 0 CHECK (must_change_password IN (0, 1))"
            )
            conn.commit()
            logger.info("Migrated users table: added must_change_password column.")

        if "google_email" not in existing_columns:
            # No UNIQUE here -- confirmed directly against SQLite that
            # ALTER TABLE ADD COLUMN cannot carry a UNIQUE constraint at
            # all ("Cannot add a UNIQUE column"), on either backend. The
            # CREATE UNIQUE INDEX statement right below enforces the
            # exact same guarantee instead -- the standard way to
            # retrofit uniqueness onto an existing table.
            cursor.execute("ALTER TABLE users ADD COLUMN google_email TEXT")
            conn.commit()
            logger.info("Migrated users table: added google_email column.")

        # Idempotent (IF NOT EXISTS) and run unconditionally, not only
        # inside the "column was just added" branch above -- a fresh
        # install's users table already has google_email (see
        # create_tables()), but never gets this unique index unless it is
        # created here too. NULL is treated as distinct from every other
        # NULL for UNIQUE indexing purposes (confirmed directly against
        # both backends), which is exactly what is needed: most accounts
        # will never link a Google account at all, and none of those
        # NULLs should collide with each other.
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_email ON users (google_email)"
        )
        conn.commit()

        if "failed_login_attempts" not in existing_columns:
            cursor.execute(
                "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
            logger.info("Migrated users table: added failed_login_attempts column.")

        if "locked_until" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")
            conn.commit()
            logger.info("Migrated users table: added locked_until column.")

        # subjects/marks predating the fixed 25/50/25 marks breakdown --
        # see _rebuild_subjects_and_marks_tables() for why this needs a
        # full table rebuild rather than a simple ALTER TABLE (dropping a
        # column and tightening a CHECK constraint are both things
        # neither SQLite nor libsql can do in place).
        subjects_columns = {row[1] for row in cursor.execute("PRAGMA table_info(subjects)").fetchall()}
        if "max_internal" in subjects_columns:
            _rebuild_subjects_and_marks_tables(conn)
            create_indexes(conn)  # marks' indexes were dropped along with the old table -- recreate them
            logger.info("Migrated subjects/marks tables to the fixed 25/50/25 marks breakdown.")

        # is_published -- see modules/marks.py's publish_marks()/
        # unpublish_marks(). Unlike max_internal/etc. above, this is a
        # plain additive column with a CHECK that only references its own
        # value (not other columns) -- confirmed directly against both
        # backends that ALTER TABLE ADD COLUMN can carry that kind of
        # CHECK, unlike a UNIQUE constraint (see google_email above) -- so
        # a simple ALTER TABLE is enough here, no table rebuild needed.
        marks_columns = {row[1] for row in cursor.execute("PRAGMA table_info(marks)").fetchall()}
        if "is_published" not in marks_columns:
            cursor.execute(
                "ALTER TABLE marks ADD COLUMN is_published INTEGER NOT NULL DEFAULT 0 "
                "CHECK (is_published IN (0, 1))"
            )
            # Every row that already existed at migration time predates the
            # publish/draft distinction entirely -- it was already visible
            # to its student before this feature existed. Backfilling
            # is_published=1 for exactly those rows (not the DEFAULT 0
            # every NEW row gets from here on) preserves that visibility
            # instead of silently hiding real marks a student could
            # already see. This is a one-time UPDATE, not part of the
            # column's own DEFAULT, specifically so it applies ONLY to
            # rows that existed before this ALTER TABLE ran.
            cursor.execute("UPDATE marks SET is_published = 1")
            conn.commit()
            logger.info(
                "Migrated marks table: added is_published column, "
                "backfilled existing rows as published."
            )

    except (sqlite3.Error, ValueError) as error:
        # See create_tables()'s comment above for why ValueError is caught
        # here too -- this function touches the same two backends.
        conn.rollback()
        logger.error("Failed to migrate schema: %s", error)
        raise DatabaseError(f"Could not migrate database schema: {error}") from error


def _rebuild_subjects_and_marks_tables(conn) -> None:
    """
    Rebuild subjects (dropping max_internal/max_external/max_practical)
    and marks (tightening its CHECK constraints to the fixed per-
    component ceilings) -- called by migrate_schema() above, only for a
    database created before config.MAX_INTERNAL_MARKS/MAX_EXTERNAL_MARKS/
    MAX_PRACTICAL_MARKS existed.

    WHY A FULL REBUILD, NOT AN ALTER TABLE: SQLite (and libsql) cannot
    drop a column that participates in a CHECK constraint, and cannot
    modify an existing CHECK constraint's bounds at all -- both are
    permanently baked into a table's original CREATE TABLE statement.
    The only way to change either is SQLite's own documented procedure:
    create a new table with the desired final shape, copy every row
    across, drop the old table, then rename the new one into its place.
    This is the same pattern already used for the "12-step ALTER TABLE"
    style changes documented elsewhere in this project, applied here to
    TWO tables together (subjects and marks) because marks.subject_code
    foreign-keys to subjects.subject_code, and both changes shipped in
    the same feature.

    WHY FOREIGN KEYS ARE TURNED OFF FOR THE DURATION: with enforcement
    on, dropping "subjects" out from under "marks" (even for the split
    second before "subjects_new" is renamed into place) would be
    rejected. Turned back on immediately afterward, followed by
    `PRAGMA foreign_key_check` to positively confirm no orphaned or
    mismatched row was introduced by the rebuild -- not just assuming it
    worked because no exception was raised.

    WHY marks' INSERT WOULD FAIL LOUDLY, NOT SILENTLY CLAMP, IF ANY
    EXISTING ROW VIOLATED THE NEW TIGHTER RANGES: marks_new's CHECK
    constraints are the real, final ones (internal<=25, external<=50,
    practical<=25) -- copying a row that used to be legal under the old,
    more permissive ceiling (up to 100 on each component) but violates
    the new one raises a CHECK constraint error immediately, aborting
    the whole migration rather than truncating or silently altering
    anyone's real marks data. (Verified directly against this project's
    real production data before ever running this migration for real --
    every existing mark obtained was already comfortably within the new
    ranges.)

    Args:
        conn: Whatever get_connection() returned.
    """
    cursor = conn.cursor()

    # Must happen BEFORE the transaction below -- SQLite (and libsql)
    # ignore an attempt to change this PRAGMA while a transaction is
    # already open.
    cursor.execute("PRAGMA foreign_keys = OFF")

    # --- subjects: drop max_internal/max_external/max_practical ---
    cursor.execute(f"""
        CREATE TABLE subjects_new (
            subject_code  TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            semester      INTEGER NOT NULL
                              CHECK (semester BETWEEN {config.MIN_SEMESTER}
                                               AND {config.MAX_SEMESTER}),
            credits       INTEGER NOT NULL
                              CHECK (credits BETWEEN {config.MIN_CREDITS}
                                               AND {config.MAX_CREDITS}),
            is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
            created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        INSERT INTO subjects_new
            (subject_code, name, semester, credits, is_active, created_at, updated_at)
        SELECT subject_code, name, semester, credits, is_active, created_at, updated_at
        FROM subjects
    """)
    cursor.execute("DROP TABLE subjects")
    cursor.execute("ALTER TABLE subjects_new RENAME TO subjects")

    # --- marks: tighten CHECK constraints to the fixed per-component ceilings ---
    cursor.execute(f"""
        CREATE TABLE marks_new (
            mark_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no      TEXT NOT NULL,
            subject_code TEXT NOT NULL,
            internal     INTEGER NOT NULL
                             CHECK (internal BETWEEN 0 AND {config.MAX_INTERNAL_MARKS}),
            external     INTEGER NOT NULL
                             CHECK (external BETWEEN 0 AND {config.MAX_EXTERNAL_MARKS}),
            practical    INTEGER NOT NULL
                             CHECK (practical BETWEEN 0 AND {config.MAX_PRACTICAL_MARKS}),
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
    cursor.execute("""
        INSERT INTO marks_new
            (mark_id, roll_no, subject_code, internal, external, practical,
             semester, exam_type, created_at, updated_at)
        SELECT mark_id, roll_no, subject_code, internal, external, practical,
               semester, exam_type, created_at, updated_at
        FROM marks
    """)
    cursor.execute("DROP TABLE marks")
    cursor.execute("ALTER TABLE marks_new RENAME TO marks")

    conn.commit()
    cursor.execute("PRAGMA foreign_keys = ON")

    fk_violations = cursor.execute("PRAGMA foreign_key_check").fetchall()
    if fk_violations:
        raise DatabaseError(
            f"Foreign key violations detected after rebuilding subjects/marks: {fk_violations}"
        )


def initialize_database() -> None:
    """
    Full setup entry point: open a connection, create all tables, create
    all indexes, apply any pending schema migrations, then close the
    connection. This is the single function other code should call to
    make sure the database is ready to use.
    """
    conn = get_connection()
    try:
        create_tables(conn)
        create_indexes(conn)
        migrate_schema(conn)
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
