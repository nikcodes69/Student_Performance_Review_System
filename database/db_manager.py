"""
database/db_manager.py
=======================
The runtime data-access layer: small, reusable helper functions that run
SQL queries safely. Every module that needs to read or write the database
from here on (auth.py, and from step 8 onward: students.py, subjects.py,
marks.py, attendance.py, audit.py) calls THESE functions rather than
opening its own database connection or writing its own cursor.execute()
calls.

LAYERING IN THIS PROJECT:
    database/db_setup.py   -> defines the SCHEMA (tables, constraints),
                               and get_connection(), which opens either a
                               local SQLite file or a Turso cloud database
                               depending on whether Streamlit secrets are
                               configured (see that file for the full
                               explanation).
    database/db_manager.py -> (this file) runs QUERIES at runtime, using
                               get_connection(). Imported by business-logic
                               modules, which never touch a database
                               connection directly.
    modules/*.py            -> business logic (e.g. "log a user in", "add
                               a student") -- calls db_manager only.

PARAMETERIZED QUERIES -- ENFORCED BY THIS FILE'S OWN DESIGN:
Every function below takes the SQL text and its values as TWO SEPARATE
arguments (query, params). This is not just a style choice: it is what
makes parameterized queries possible in the first place. The database
driver replaces each "?" in the query string with the matching value from
params, treating that value purely as DATA -- it can never be interpreted
as part of the SQL command, no matter what characters it contains. This is
what stops SQL injection: an attacker typing `' OR '1'='1` into a login
box just becomes a literal (and useless) username to search for, not a
change to the query's logic. Contrast this with database/db_setup.py,
which builds its CREATE TABLE strings with f-strings -- that file only
ever interpolates trusted constants from config.py, never a value that
came from a user, which is why it is safe there but would NOT be safe here.

WHY RESULTS COME BACK AS PLAIN DICTS, NOT sqlite3.Row: the local SQLite
backend supports `conn.row_factory = sqlite3.Row`, which lets calling code
read a result by column name (row["username"]). The Turso cloud backend
(the `libsql` package) does NOT support row_factory at all -- confirmed
directly against a real Turso database, not assumed -- it only ever
returns plain tuples. Rather than have calling code behave differently
depending on which backend is active, fetch_one()/fetch_all() below
convert every row into a plain dict themselves, using cursor.description
(a standard part of the Python DB-API that both backends provide) to get
each column's name. Every module in this project already either does
dict(row) or reads row["column"] -- both work identically well on an
already-plain dict, so this change required editing NO calling code
anywhere else in the project.
"""

import sqlite3

from database.db_setup import get_connection
from utils.exceptions import DatabaseError
from utils.logger import get_logger

logger = get_logger(__name__)

# Both sqlite3 and libsql raise errors that should be treated as database
# errors here -- sqlite3 raises its own well-known exception hierarchy
# (sqlite3.Error and subclasses like IntegrityError), while libsql raises
# a plain ValueError for every failure (constraint violations, syntax
# errors, and everything else) -- confirmed empirically against a real
# Turso database. Catching both, in exactly these four functions (which
# only ever wrap actual database calls), lets this file behave the same
# way regardless of which backend get_connection() decided to use.
DATABASE_ERRORS = (sqlite3.Error, ValueError)


def _row_to_dict(cursor, row) -> dict:
    """Convert one fetched row (a tuple, on both backends -- see module
    docstring) into a plain dict, using cursor.description for column
    names."""
    columns = [column[0] for column in cursor.description]
    return dict(zip(columns, row))


def fetch_one(query: str, params: tuple = ()) -> dict | None:
    """
    Run a SELECT expected to match at most one row, and return it.

    Args:
        query: SQL text containing "?" placeholders for any values.
        params: The values to substitute for each "?", in order.

    Returns:
        A plain dict (accessible like row["username"]) if a matching row
        was found, otherwise None.

    Raises:
        DatabaseError: if the query fails to execute.
    """
    conn = get_connection()

    try:
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return _row_to_dict(cursor, row) if row is not None else None
    except DATABASE_ERRORS as error:
        logger.error("fetch_one failed for query %r: %s", query, error)
        raise DatabaseError(f"Database read failed: {error}") from error
    finally:
        # The connection is always closed, whether the query succeeded or
        # raised -- otherwise a failed query could leave the database
        # connection open for the rest of the program's run.
        conn.close()


def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    """
    Run a SELECT and return every matching row.

    Args:
        query: SQL text containing "?" placeholders for any values.
        params: The values to substitute for each "?", in order.

    Returns:
        A list of plain dicts (possibly empty, never None).

    Raises:
        DatabaseError: if the query fails to execute.
    """
    conn = get_connection()

    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [_row_to_dict(cursor, row) for row in rows]
    except DATABASE_ERRORS as error:
        logger.error("fetch_all failed for query %r: %s", query, error)
        raise DatabaseError(f"Database read failed: {error}") from error
    finally:
        conn.close()


def execute_write(query: str, params: tuple = ()) -> int:
    """
    Run an INSERT, UPDATE, or (soft-)delete-style UPDATE, and commit it.

    Args:
        query: SQL text containing "?" placeholders for any values.
        params: The values to substitute for each "?", in order.

    Returns:
        For an INSERT, the new row's auto-generated primary key
        (cursor.lastrowid). For an UPDATE, cursor.rowcount -- NOTE: on the
        Turso backend this number has been observed to not always match
        the exact row count sqlite3 would report (a `libsql` package
        quirk, confirmed empirically). No code in this project currently
        depends on the exact numeric value of an UPDATE's rowcount for any
        decision, only on lastrowid for INSERTs, which is correct on both
        backends -- but this is worth knowing if a future feature ever
        wants to check "did my update actually change a row?" precisely.

    Raises:
        DatabaseError: if the write fails. The transaction is rolled back
            first, so a failed write never leaves the database half-changed.

    NOTE ON UNIQUE/CHECK CONSTRAINT VIOLATIONS: this function does NOT try
    to detect "you inserted a duplicate username" and translate it into a
    friendlier error. That is deliberately handled by the CALLER instead
    (e.g. modules/auth.py's create_user() checks "does this username
    already exist?" with a fetch_one() BEFORE calling execute_write()).
    That check-first approach is easier to read and to unit-test than
    parsing the database driver's error text after the fact. The
    database's own UNIQUE constraint still fires underneath as a second,
    independent safety net (see database/db_setup.py) in the rare case of
    two requests racing each other -- this function will still raise
    DatabaseError if THAT happens, it just isn't the primary way we expect
    duplicates to be caught.
    """
    conn = get_connection()

    try:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
    except DATABASE_ERRORS as error:
        conn.rollback()
        logger.error("execute_write failed for query %r: %s", query, error)
        raise DatabaseError(f"Database write failed: {error}") from error
    finally:
        conn.close()


def execute_transaction(statements: list[tuple[str, tuple]]) -> list[int]:
    """
    Run multiple write statements as ONE atomic transaction: either every
    statement succeeds and all are committed together, or (if any single
    one fails) every statement in the list is rolled back, as if none of
    them had ever run.

    WHY THIS FUNCTION EXISTS: from modules/audit.py onward, most data
    changes need TWO writes to happen together -- e.g. "update this mark"
    AND "record an audit_log entry describing that update". If those were
    two separate execute_write() calls, it would be possible for the mark
    update to succeed but the audit entry to fail (or vice versa), leaving
    either an unaudited change or a "ghost" audit entry for a change that
    never actually happened. Passing both statements to THIS function
    instead guarantees they rise or fall together, because both run
    inside one connection and one transaction.

    Args:
        statements: A list of (query, params) tuples, executed in order,
            all within a single transaction.

    Returns:
        A list of results, one per statement, in the same order: for an
        INSERT, that statement's new lastrowid; for an UPDATE, its
        rowcount (see execute_write()'s docstring for a note on Turso's
        rowcount behaviour).

    Raises:
        DatabaseError: if any statement fails. Every statement already
            executed earlier in this same call is rolled back before the
            error is raised -- none of them take effect.
    """
    conn = get_connection()

    try:
        results = []
        for query, params in statements:
            cursor = conn.execute(query, params)
            results.append(cursor.lastrowid if cursor.lastrowid else cursor.rowcount)
        conn.commit()
        return results
    except DATABASE_ERRORS as error:
        conn.rollback()
        logger.error("execute_transaction failed: %s", error)
        raise DatabaseError(f"Database transaction failed: {error}") from error
    finally:
        conn.close()
