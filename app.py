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
from modules import analytics, assignments, attendance, audit, auth, ml_predictions, marks, student_portal, students, subjects
from utils.pdf_generator import render_class_report_page, render_report_card_page
from utils.exceptions import (
    AppError,
    AuthenticationError,
    DuplicateRecordError,
    RecordNotFoundError,
    ValidationError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Page sections shown on the home dashboard below, grouped the same way
# they're grouped in the sidebar -- (label, icon, roles allowed, one-line
# description). Icons use Streamlit's built-in Material icon syntax
# (":material/name:"), not emoji -- a real, consistent icon set rather
# than a decorative flourish.
HOME_SECTIONS = [
    ("Dashboard", ":material/dashboard:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Spotlight cards, charts, and gauges at a glance"),
    ("Students", ":material/group:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Manage student records"),
    ("Subjects", ":material/menu_book:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Configure the curriculum"),
    ("Marks Entry", ":material/edit_note:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Record internal/external/practical marks"),
    ("Attendance", ":material/event_available:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Track classes held/attended"),
    ("Assignments", ":material/assignment_turned_in:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Track assignment submissions"),
    ("Analytics", ":material/monitoring:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Averages, trends, correlations"),
    ("At-Risk Prediction", ":material/warning:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Who needs early support"),
    ("Final Marks Prediction", ":material/query_stats:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Estimate a student's final percentage"),
    ("Student Segmentation", ":material/scatter_plot:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Behavioural performance groups"),
    ("Model Comparison", ":material/model_training:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Metrics behind every deployed model"),
    ("Report Card", ":material/picture_as_pdf:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Downloadable PDF report card"),
    ("Class Report", ":material/summarize:", (config.ROLE_ADMIN, config.ROLE_TEACHER), "Institution-wide PDF summary"),
    ("Audit Log", ":material/history:", (config.ROLE_ADMIN,), "Full history of every change"),
    ("User Management", ":material/manage_accounts:", (config.ROLE_ADMIN,), "Create Teacher/Admin accounts"),
    ("My Performance", ":material/person:", (config.ROLE_STUDENT,), "Your own marks, attendance, and predictions"),
    ("Change Password", ":material/password:", config.VALID_ROLES, "Update your own login password"),
]


def render_home_page() -> None:
    """Landing dashboard, visible to every logged-in role. The
    institution-wide metrics row is Admin/Teacher only -- a Student's own
    numbers belong on their dedicated "My Performance" page (see
    modules/student_portal.py), not mixed in with aggregate stats about
    every OTHER student too."""
    user = auth.get_current_user()
    st.title("Student Performance Review & Prediction System")

    if user["role"] in (config.ROLE_ADMIN, config.ROLE_TEACHER):
        summary = analytics.get_dashboard_summary()

        metric_cols = st.columns(5)
        metric_cols[0].metric(":material/group: Active Students", summary["student_count"])
        metric_cols[1].metric(":material/menu_book: Active Subjects", summary["subject_count"])
        metric_cols[2].metric(":material/edit_note: Marks Recorded", summary["marks_count"])
        metric_cols[3].metric(
            ":material/check_circle: Pass Rate",
            f"{summary['pass_rate']}%" if summary["pass_rate"] is not None else "N/A",
        )
        metric_cols[4].metric(
            ":material/event_available: Avg. Attendance",
            f"{summary['avg_attendance']}%" if summary["avg_attendance"] is not None else "N/A",
        )
        st.divider()

    st.subheader("Available pages")

    visible_sections = [s for s in HOME_SECTIONS if user["role"] in s[2]]
    section_cols = st.columns(3)
    for index, (label, icon, _roles, description) in enumerate(visible_sections):
        with section_cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{icon} {label}**")
                st.caption(description)


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
    "Dashboard": ((config.ROLE_ADMIN, config.ROLE_TEACHER), analytics.render_dashboard_page),
    "Students": ((config.ROLE_ADMIN, config.ROLE_TEACHER), students.render_students_page),
    "Subjects": ((config.ROLE_ADMIN, config.ROLE_TEACHER), subjects.render_subjects_page),
    "Marks Entry": ((config.ROLE_ADMIN, config.ROLE_TEACHER), marks.render_marks_page),
    "Attendance": ((config.ROLE_ADMIN, config.ROLE_TEACHER), attendance.render_attendance_page),
    "Assignments": ((config.ROLE_ADMIN, config.ROLE_TEACHER), assignments.render_assignments_page),
    "Analytics": ((config.ROLE_ADMIN, config.ROLE_TEACHER), analytics.render_analytics_page),
    "At-Risk Prediction": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_at_risk_page),
    "Final Marks Prediction": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_final_marks_page),
    "Student Segmentation": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_segmentation_page),
    "Model Comparison": ((config.ROLE_ADMIN, config.ROLE_TEACHER), ml_predictions.render_model_comparison_page),
    "Report Card": ((config.ROLE_ADMIN, config.ROLE_TEACHER), render_report_card_page),
    "Class Report": ((config.ROLE_ADMIN, config.ROLE_TEACHER), render_class_report_page),
    "Audit Log": ((config.ROLE_ADMIN,), audit.render_audit_log_page),
    "User Management": ((config.ROLE_ADMIN,), auth.render_user_management_page),
    "My Performance": ((config.ROLE_STUDENT,), student_portal.render_student_portal_page),
    "Change Password": (config.VALID_ROLES, auth.render_change_password_page),
}

# Icon per sidebar entry, reusing HOME_SECTIONS' icon choices (plus Home's
# own) so the same page is represented by the same icon everywhere in the
# app -- looked up by st.sidebar.radio's format_func below, which only
# changes how each option is DISPLAYED, not the underlying value used to
# look up PAGES[choice].
PAGE_ICONS = {"Home": ":material/home:"} | {label: icon for label, icon, _roles, _desc in HOME_SECTIONS}


def render_login_form() -> None:
    """
    Show the Log In / Sign Up screen and handle either submission.

    Sign Up here means STUDENT self-registration only -- see
    modules/auth.py's module docstring ("WHO CAN CREATE AN ACCOUNT") for
    why Teacher and Admin accounts are deliberately never created through
    a public-facing form like this one; those are created by an existing
    Admin, on the User Management page, once logged in.
    """
    # A centered, fixed-width column instead of a full-page-wide form --
    # purely a layout choice (st.columns with unused side columns to
    # center the middle one), no new widget behaviour.
    _left, center, _right = st.columns([1, 1.2, 1])
    with center:
        st.title(":material/school: Student Performance System")

        login_tab, signup_tab = st.tabs(["Log In", "Sign Up (Students)"])

        with login_tab:
            st.caption("Sign in to continue")

            # st.form groups the two inputs and the button together so the
            # page only reruns (and only tries to log in) once, when "Log
            # In" is clicked -- not on every single keystroke in the
            # username/password boxes, which is what would happen without
            # a form.
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")

            if submitted:
                try:
                    user = auth.login(username, password)
                    st.success(f"Welcome, {user['username']} ({user['role']}).")
                    # st.rerun() immediately restarts the script from the
                    # top. This matters because is_session_valid() (checked
                    # in main(), below) needs to run again to notice the
                    # session we JUST created above -- without this, the
                    # user would still see the login form for one extra click.
                    st.rerun()
                except (ValidationError, AuthenticationError) as error:
                    # Both exception types produce a message written
                    # specifically to be shown to a human (see
                    # utils/validators.py and modules/auth.py) -- so
                    # str(error) is safe and appropriate to display directly.
                    st.error(str(error))

        with signup_tab:
            st.caption(
                "For students only. Your roll number must already exist in the "
                "system (an Admin or Teacher enters it when you're enrolled) -- "
                "this just creates YOUR login for it."
            )
            with st.form("signup_form", clear_on_submit=True):
                signup_roll_no = st.text_input("Roll Number", placeholder="e.g. BCA078123")
                signup_email = st.text_input(
                    "Email", placeholder="the email on file for your roll number",
                )
                signup_password = st.text_input(
                    "Choose a Password", type="password", placeholder="At least 8 characters",
                )
                signup_confirm = st.text_input("Confirm Password", type="password")
                signup_submitted = st.form_submit_button(
                    "Create My Account", use_container_width=True,
                )

            if signup_submitted:
                if signup_password != signup_confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        auth.self_register_student(signup_roll_no, signup_email, signup_password)
                        st.success("Account created. Switch to the Log In tab to sign in.")
                    except (ValidationError, RecordNotFoundError, DuplicateRecordError) as error:
                        # RecordNotFoundError: no such roll_no on file yet
                        #   (an Admin/Teacher needs to create the student
                        #   record first -- see the caption above).
                        # ValidationError: bad email/password, or the email
                        #   didn't match what's on file for that roll_no.
                        # DuplicateRecordError: a login already exists for
                        #   this roll_no.
                        st.error(str(error))


def render_notifications_bell(user: dict) -> None:
    """
    A small, always-visible sidebar alert summary for Admin/Teacher:
    "N student(s) at-risk, M below attendance threshold" -- a lightweight
    nudge toward numbers the Dashboard page already computes
    (analytics.get_attendance_shortage_list(), ml_predictions.
    get_at_risk_count()), so an Admin/Teacher sees them on every page,
    not only when they happen to open the Dashboard.

    DELIBERATELY NOT A NEW ALERTING CHANNEL: this reuses the SAME
    already-cached functions the Dashboard's KPI row and shortage panel
    already call (see modules/analytics.py's render_dashboard_page()) --
    no new computation, no email/push notifications (out of scope for
    this project, see README's "Beyond the original spec" section).
    Hidden entirely when there is nothing to flag, rather than always
    showing "0 alerts" -- a permanent empty banner would just be visual
    noise on every single page. Also a no-op for a Student -- these
    numbers are about the whole class, not a fit for a Student-scoped
    view (see modules/student_portal.py's module docstring on "my own
    data only").
    """
    if user["role"] not in (config.ROLE_ADMIN, config.ROLE_TEACHER):
        return

    at_risk_count = ml_predictions.get_at_risk_count()
    shortage_count = len(analytics.get_attendance_shortage_list())
    total_alerts = (at_risk_count or 0) + shortage_count

    if total_alerts == 0:
        return

    with st.sidebar.expander(f":material/notifications: Alerts ({total_alerts})"):
        if at_risk_count:
            st.warning(f"{at_risk_count} student(s) flagged at-risk. See At-Risk Prediction.")
        if shortage_count:
            st.warning(f"{shortage_count} student(s) below the attendance threshold. See Dashboard.")


def render_authenticated_view(user: dict) -> None:
    """Show the sidebar menu and whichever page the user picked.

    FORCED PASSWORD CHANGE GATE: checked first, before the sidebar
    navigation is even built. If this account was created with
    must_change_password=1 (a freshly Admin-created account, the
    bootstrap admin, or one an Admin just reset -- see
    modules/auth.py's create_user()/admin_reset_password()), the ONLY
    thing shown is the change-password form; render_fn() for whatever
    page is nominally selected never runs. This is what makes it a real
    gate and not just a suggestion -- there is no menu item to click
    around it."""
    if user.get("must_change_password"):
        auth.render_change_password_page(forced=True)
        return

    st.sidebar.markdown(f"### :material/account_circle: {user['username']}")
    st.sidebar.caption(f"Role: {user['role'].capitalize()}")

    if st.sidebar.button("Log Out", icon=":material/logout:", use_container_width=True):
        auth.logout()
        st.rerun()

    render_notifications_bell(user)

    st.sidebar.divider()

    # Only offer menu entries this user's role is allowed to see.
    available_pages = [
        label for label, (roles, _render_fn) in PAGES.items() if user["role"] in roles
    ]
    # format_func only changes how each option is DISPLAYED (prefixing its
    # icon) -- the value st.sidebar.radio actually returns, and that
    # `choice` gets used to look up PAGES[choice] below, is still the
    # plain label string.
    choice = st.sidebar.radio(
        "Navigate", available_pages,
        format_func=lambda label: f"{PAGE_ICONS.get(label, '')} {label}".strip(),
    )

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
