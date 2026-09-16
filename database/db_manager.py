"""
database/db_manager.py
=======================
The runtime data-access layer: small, reusable helper functions that run
SQL queries safely. Every module that needs to read or write the database
from here on (auth.py, and from step 8 onward: students.py, subjects.py,
marks.py, attendance.py, audit.py) calls THESE functions rather than
opening its own sqlite3 connection or writing its own cursor.execute()
calls.

LAYERING IN THIS PROJECT:
    database/db_setup.py   -> defines the SCHEMA (tables, constraints).
                               Run once, at setup time.
    database/db_manager.py -> (this file) runs QUERIES at runtime, using
                               the connection function db_setup.py already
                               defined. Imported by business-logic modules.
    modules/*.py            -> business logic (e.g. "log a user in", "add
                               a student") -- calls db_manager, never
                               touches sqlite3 directly.

PARAMETERIZED QUERIES -- ENFORCED BY THIS FILE'S OWN DESIGN:
Every function below takes the SQL text and its values as TWO SEPARATE
arguments (query, params). This is not just a style choice: it is what
makes parameterized queries possible in the first place. SQLite (via
Python's sqlite3 module) replaces each "?" in the query string with the
matching value from params, treating that value purely as DATA -- it can
never be interpreted as part of the SQL command, no matter what characters
it contains. This is what stops SQL injection: an attacker typing
`' OR '1'='1` into a login box just becomes a literal (and useless)
username to search for, not a change to the query's logic. Contrast this
with database/db_setup.py, which builds its CREATE TABLE strings with
f-strings -- that file only ever interpolates trusted constants from
config.py, never a value that came from a user, which is why it is safe
there but would NOT be safe here.
"""

import sqlite3

from database.db_setup import get_connection
from utils.exceptions import DatabaseError
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_one(query: str, params: tuple = ()) -> sqlite3.Row | None:
    """
    Run a SELECT expected to match at most one row, and return it.

    Args:
        query: SQL text containing "?" placeholders for any values.
        params: The values to substitute for each "?", in order.

    Returns:
        A sqlite3.Row (accessible like a dict, e.g. row["username"]) if a
        matching row was found, otherwise None.

    Raises:
        DatabaseError: if the query fails to execute.
    """
    conn = get_connection()
    # row_factory = sqlite3.Row lets calling code read columns by NAME
    # (row["username"]) instead of by fragile numeric position (row[1]),
    # which stays correct even if a column is added to the table later.
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(query, params)
        return cursor.fetchone()
    except sqlite3.Error as error:
        logger.error("fetch_one failed for query %r: %s", query, error)
        raise DatabaseError(f"Database read failed: {error}") from error
    finally:
        # The connection is always closed, whether the query succeeded or
        # raised -- otherwise a failed query could leave the .db file
        # locked for the rest of the program's run.
        conn.close()


def fetch_all(query: str, params: tuple = ()) -> list[sqlite3.Row]:
    """
    Run a SELECT and return every matching row.

    Args:
        query: SQL text containing "?" placeholders for any values.
        params: The values to substitute for each "?", in order.

    Returns:
        A list of sqlite3.Row objects (possibly empty, never None).

    Raises:
        DatabaseError: if the query fails to execute.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(query, params)
        return cursor.fetchall()
    except sqlite3.Error as error:
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
        (cursor.lastrowid). For an UPDATE, the number of rows that were
        changed (cursor.rowcount) -- callers can use this to detect "I
        tried to update a record that doesn't exist" (rowcount == 0).

    Raises:
        DatabaseError: if the write fails. The transaction is rolled back
            first, so a failed write never leaves the database half-changed.

    NOTE ON UNIQUE/CHECK CONSTRAINT VIOLATIONS: this function does NOT try
    to detect "you inserted a duplicate username" and translate it into a
    friendlier error. That is deliberately handled by the CALLER instead
    (e.g. modules/auth.py's create_user() checks "does this username
    already exist?" with a fetch_one() BEFORE calling execute_write()).
    That check-first approach is easier to read and to unit-test than
    parsing SQLite's error text after the fact. The database's own UNIQUE
    constraint still fires underneath as a second, independent safety net
    (see database/db_setup.py) in the rare case of two requests racing
    each other -- this function will still raise DatabaseError if THAT
    happens, it just isn't the primary way we expect duplicates to be caught.
    """
    conn = get_connection()

    try:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
    except sqlite3.Error as error:
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
        INSERT, that statement's new lastrowid; for an UPDATE, its rowcount.

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
    except sqlite3.Error as error:
        conn.rollback()
        logger.error("execute_transaction failed: %s", error)
        raise DatabaseError(f"Database transaction failed: {error}") from error
    finally:
        conn.close()
