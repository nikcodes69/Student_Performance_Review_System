"""
app.py
======
The Streamlit entry point. Run with:

    streamlit run app.py

(Streamlit is the one exception to the "run everything with -m" rule
explained in database/db_setup.py -- the `streamlit run` command sets up
Python's import path itself before running the script, so app.py can
`import config` and `from modules import auth` directly, because app.py
lives at the project root next to config.py.)

WHAT THIS FILE DOES: shows a login form, and once logged in, a sidebar
menu of every page the current user's role is allowed to see (the PAGES
dict below). Each entry maps a menu label to (allowed roles, the render
function that draws that page) -- adding a new feature module to this
app means adding one line here, nothing else.

THE TOP-LEVEL SAFETY NET (see main() below): every individual page
already catches the specific errors it expects (a bad form value, a
missing record, and so on -- see each modules/*.py file). But a page
could still raise something nobody anticipated -- a bug, a database
hiccup, a model file deleted mid-session. Without a final catch-all,
Streamlit's own default behaviour is to show the user a raw Python
traceback, which is both unprofessional and a real information leak (a
traceback can reveal file paths, internal logic, sometimes fragments of
data). main() wraps the whole page-rendering call in one last try/except:
domain errors we recognise still get their specific class name logged
and a readable message shown; anything else gets logged in full via
logger.exception() (which records the complete traceback to
logs/app.log for debugging) while the user only ever sees a generic,
safe message.
"""

import streamlit as st

import config
from modules import analytics, attendance, audit, auth, ml_predictions, marks, students, subjects
from utils.pdf_generator import render_report_card_page
from utils.exceptions import AppError, AuthenticationError, ValidationError
from utils.logger import get_logger

logger = get_logger(__name__)


def render_home_page() -> None:
    """Landing page, visible to every logged-in role."""
    st.title("Student Performance Review & Prediction System")
    st.write("Use the sidebar to navigate. Available pages depend on your role:")
    st.markdown(
        "- **Students / Subjects / Marks Entry / Attendance** -- day-to-day "
        "record keeping (Admin, Teacher)\n"
        "- **Analytics** -- averages, trends, grade distribution, "
        "attendance-vs-marks correlation, subject difficulty (Admin, Teacher)\n"
        "- **At-Risk Prediction / Final Marks Prediction / Student Segmentation** "
        "-- machine learning predictions for an individual student (Admin, Teacher)\n"
        "- **Model Comparison** -- metrics and deployment justification for every "
        "trained model (Admin, Teacher)\n"
        "- **Report Card** -- downloadable PDF report card, optionally including "
        "ML predictions (Admin, Teacher)\n"
        "- **Audit Log** -- full history of every change made to academic records "
        "(Admin only)"
    )


# Each entry maps a sidebar menu label to (roles allowed to see it, the
# function that renders that page). A page's render function ALSO enforces
# its own role check internally (e.g. audit.render_audit_log_page() calls
# auth.require_role(config.ROLE_ADMIN) itself) -- this dict only controls
# what appears in the menu. That is deliberate, not redundant: the menu
# filter is for a clean UI, but the real access control is the check
# inside each function, the same defense-in-depth principle used
# throughout this project (see modules/auth.py's check_permission()).
PAGES = {
    "Home": (config.VALID_ROLES, render_home_page),
    "Students": ((config.ROLE_ADMIN, config.ROLE_TEACHER), students.render_students_page),
    "Subjects": ((config.ROLE_ADMIN, config.ROLE_TEACHER), subjects.render_subjects_page),
    "Marks Entry": ((config.ROLE_ADMIN, config.ROLE_TEACHER), marks.render_marks_page),
    "Attendance": ((config.ROLE_ADMIN, config.ROLE_TEACHER), attendance.render_attendance_page),
    "Analytics": ((config.ROLE_ADMIN, config.ROLE_TEACHER), analytics.render_analytics_page),
    "At-Risk Prediction": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_at_risk_page),
    "Final Marks Prediction": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_final_marks_page),
    "Student Segmentation": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_segmentation_page),
    "Model Comparison": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_model_comparison_page),
    "Report Card": ((config.ROLE_ADMIN, config.ROLE_TEACHER), render_report_card_page),
    "Audit Log": ((config.ROLE_ADMIN,), audit.render_audit_log_page),
}


def render_login_form() -> None:
    """Show the login form and handle a submitted login attempt."""
    st.title("Student Performance Review & Prediction System")
    st.subheader("Log In")

    # st.form groups the two inputs and the button together so the page
    # only reruns (and only tries to log in) once, when "Log In" is
    # clicked -- not on every single keystroke in the username/password
    # boxes, which is what would happen without a form.
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        try:
            user = auth.login(username, password)
            st.success(f"Welcome, {user['username']} ({user['role']}).")
            # st.rerun() immediately restarts the script from the top.
            # This matters because is_session_valid() (checked in main(),
            # below) needs to run again to notice the session we JUST
            # created above -- without this, the user would still see the
            # login form for one extra click.
            st.rerun()
        except (ValidationError, AuthenticationError) as error:
            # Both exception types produce a message written specifically
            # to be shown to a human (see utils/validators.py and
            # modules/auth.py) -- so str(error) is safe and appropriate
            # to display directly here.
            st.error(str(error))


def render_authenticated_view(user: dict) -> None:
    """Show the sidebar menu and whichever page the user picked."""
    st.sidebar.write(f"Logged in as **{user['username']}**")
    st.sidebar.write(f"Role: **{user['role']}**")

    if st.sidebar.button("Log Out"):
        auth.logout()
        st.rerun()

    st.sidebar.divider()

    # Only offer menu entries this user's role is allowed to see.
    available_pages = [
        label for label, (roles, _render_fn) in PAGES.items() if user["role"] in roles
    ]
    choice = st.sidebar.radio("Navigate", available_pages)

    _roles, render_fn = PAGES[choice]

    # This is the top-level safety net described in the module docstring.
    # Note what this does NOT catch: auth.require_role()/require_login()
    # halt a page via Streamlit's own st.stop() mechanism, which works
    # through Streamlit's internal script-runner control flow rather than
    # a normal Python exception in this Streamlit version -- so an access-
    # control stop passes through this try/except completely undisturbed,
    # and a denied page still shows require_role()'s own message, not a
    # generic one from here.
    try:
        render_fn()
    except AppError as error:
        logger.exception("%s on page '%s'.", type(error).__name__, choice)
        st.error(f"Something went wrong: {error}")
    except Exception:
        logger.exception("Unexpected error on page '%s'.", choice)
        st.error("An unexpected error occurred. Please try again or contact an administrator.")


def main() -> None:
    st.set_page_config(page_title="Student Performance System", layout="wide")

    if auth.is_session_valid():
        render_authenticated_view(auth.get_current_user())
    else:
        render_login_form()


if __name__ == "__main__":
    main()
