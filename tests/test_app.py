"""
tests/test_app.py
==================
pytest tests for app.py's role-based page visibility: confirms a Teacher
and a Student each only ever see the sidebar pages their role is allowed
to see (app.py's PAGES dict), never the ones reserved for another role.

WHY THIS USES streamlit.testing.v1.AppTest, UNLIKE EVERY OTHER TEST FILE
IN THIS PROJECT: every other test file calls a modules/*.py function
directly (e.g. auth.login()), which needs no real Streamlit script-run
context. But the thing being verified here -- "which options does
st.sidebar.radio() actually offer this user" -- only exists once app.py's
whole script has actually run inside a real (simulated) browser session,
sidebar included. AppTest is Streamlit's own official testing framework
for exactly this: it executes app.py the same way `streamlit run app.py`
would, in an isolated ScriptRunContext, and lets a test inspect the
resulting widgets afterward -- see
https://docs.streamlit.io/develop/api-reference/app-testing.

WHY THIS IS A REAL SECURITY CHECK, NOT JUST A UI NICETY: app.py's PAGES
dict comment explains the defense-in-depth design -- the sidebar filter
controls what appears in the MENU, but every render_*_page() function
ALSO calls auth.require_role() itself, independently, so a page can never
actually be reached even if it appeared in the menu by mistake. This test
covers the menu-filter half; the require_role() calls themselves were
verified by reading every modules/*.py render_*_page() function's source
(each one's allowed roles match its PAGES entry exactly).

HOW A LOGGED-IN SESSION IS SIMULATED WITHOUT A REAL LOGIN: auth.py's
require_login()/get_current_user() only ever look at
st.session_state["auth_user"] and st.session_state["auth_last_activity"]
(see modules/auth.py's SESSION_KEY_USER/SESSION_KEY_LAST_ACTIVITY). This
test sets those two keys directly, on the AppTest instance's own
session_state, BEFORE calling at.run() -- the same effect as a real
login(), without needing a real username/password round-trip.

HOW THIS AVOIDS TOUCHING THE REAL DATABASE: the test_db fixture below is
the same throwaway-database pattern as tests/test_database.py -- see that
file's module docstring for the full rationale.

HOW TO RUN (from the project root):
    python -m pytest tests/test_app.py -v
"""

from datetime import datetime
from pathlib import Path

import pytest

import config
import database.db_setup as db_setup
from database.db_setup import create_indexes, create_tables, get_connection
from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves a relative path against the file that CALLS
# it (this file, tests/test_app.py) -- not the project root -- so app.py's
# path is built explicitly here rather than passed as a bare "app.py".
_APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """A fresh, fully-constrained, empty test database for one test --
    never the real local database, and never the real Turso database."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test_app.db")
    monkeypatch.setattr(db_setup, "_get_turso_credentials", lambda: (None, None))

    conn = get_connection()
    create_tables(conn)
    create_indexes(conn)
    conn.commit()
    conn.close()

    yield


def _run_as(role: str, username: str) -> AppTest:
    """Simulate a logged-in session for `role` and run app.py, the same
    way a real browser hitting a freshly-logged-in page would."""
    # default_timeout=3 (AppTest's own default) was intermittently too
    # tight for this specific page -- it renders block_password_clipboard()'s
    # JS component on every run, and the Home page's dashboard summary
    # query, which together occasionally push a cold run past a few
    # seconds when the whole test suite is running (CPU/IO contention
    # from every other test running around the same time), even though
    # this page always runs in a couple of seconds in isolation. 30s
    # gives real headroom under full-suite load without letting a
    # genuinely hung script run for a long time undetected.
    at = AppTest.from_file(str(_APP_PATH), default_timeout=30)
    at.session_state["auth_user"] = {
        "user_id": 1, "username": username, "role": role, "must_change_password": False,
    }
    at.session_state["auth_last_activity"] = datetime.now()
    at.run()
    assert not at.exception, f"Unexpected exception rendering as {role}: {at.exception}"
    return at


def _sidebar_page_labels(at: AppTest) -> list[str]:
    """The plain page labels (icon prefix stripped) currently offered by
    the sidebar's page-navigation radio -- i.e. exactly what this user
    could click through to."""
    [radio] = at.sidebar.radio
    return [option.split(": ", 1)[1] if ": " in option else option for option in radio.options]


# ---------------------------------------------------------------------------
# Expected page sets per role -- kept in sync with app.py's PAGES dict by
# hand; a mismatch here means either this test or app.py's PAGES dict has
# drifted and needs reconciling, not that this test should be "fixed" to
# match whatever app.py currently does.
# ---------------------------------------------------------------------------

_ADMIN_ONLY_PAGES = {"Audit Log", "User Management"}
_STUDENT_ONLY_PAGES = {"My Performance"}
_STAFF_PAGES = {  # Admin and Teacher both
    "Dashboard", "Search", "Students", "Subjects", "Marks Entry", "Attendance", "Assignments",
    "Analytics", "At-Risk Prediction", "Final Marks Prediction", "Student Segmentation",
    "Model Comparison", "Report Card", "Class Report",
}
_SHARED_PAGES = {"Home", "Announcements", "Grade Appeals", "Change Password"}  # every role


def test_admin_sees_every_page(test_db):
    labels = set(_sidebar_page_labels(_run_as(config.ROLE_ADMIN, "admin1")))
    assert labels == _SHARED_PAGES | _STAFF_PAGES | _ADMIN_ONLY_PAGES


def test_teacher_sees_only_staff_pages(test_db):
    labels = set(_sidebar_page_labels(_run_as(config.ROLE_TEACHER, "teach1")))
    assert labels == _SHARED_PAGES | _STAFF_PAGES
    # Explicit negative checks -- these are the ones a Teacher must never
    # see, spelled out so a future accidental PAGES change fails loudly
    # and specifically, not just as "the set doesn't match".
    assert "Audit Log" not in labels
    assert "User Management" not in labels
    assert "My Performance" not in labels


def test_student_sees_only_student_pages(test_db):
    labels = set(_sidebar_page_labels(_run_as(config.ROLE_STUDENT, "S1")))
    assert labels == _SHARED_PAGES | _STUDENT_ONLY_PAGES
    # A Student must never see any staff or admin page, academic records
    # of other students being exactly what this boundary protects.
    assert labels.isdisjoint(_STAFF_PAGES)
    assert labels.isdisjoint(_ADMIN_ONLY_PAGES)


# ---------------------------------------------------------------------------
# Server-side route protection, independent of the sidebar menu -- proves
# auth.require_role() itself refuses a Teacher, not just that the Admin-only
# page never appears as a clickable option (test_teacher_sees_only_staff_pages
# above already proves that half). Calls the page function directly, bypassing
# app.py's PAGES/sidebar routing entirely, via AppTest.from_string() so
# require_role()'s st.stop() runs inside a real ScriptRunContext (outside one,
# st.stop() is a silent no-op -- see utils/ui_security.py's module docstring
# for the same class of AppTest-vs-bare-mode distinction).
# ---------------------------------------------------------------------------

_DIRECT_ADMIN_PAGE_SCRIPT = """
import streamlit as st
from modules import audit

st.session_state["auth_user"] = {
    "user_id": 5, "username": "teach1", "role": "teacher", "must_change_password": False,
}
from datetime import datetime
st.session_state["auth_last_activity"] = datetime.now()

audit.render_audit_log_page()
st.write("REACHED_AUDIT_LOG_CONTENT")
"""


def test_admin_only_page_refuses_a_teacher_even_when_called_directly(test_db):
    at = AppTest.from_string(_DIRECT_ADMIN_PAGE_SCRIPT)
    at.run()

    assert not at.exception
    # require_role()'s own permission-denied message shown...
    assert any("permission" in e.value.lower() for e in at.error)
    # ...and the page's real content never rendered -- st.stop() inside
    # require_role() actually halted the script, not merely logged a warning.
    assert not any("REACHED_AUDIT_LOG_CONTENT" in m.value for m in at.markdown)
