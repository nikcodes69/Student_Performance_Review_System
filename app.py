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

WHAT THIS FILE DOES: before login, shows a landing page (project intro,
features, an "About" section, and three role-entry cards) rather than a
bare login form -- see render_landing_page() and the PRE-LOGIN
NAVIGATION section below for the full session-state-driven flow between
the landing page, each role's own login view, and the two signup views.
Once logged in, it shows a sidebar menu of every page the current user's
role is allowed to see (the PAGES dict below). Each PAGES entry maps a
menu label to (allowed roles, the render function that draws that page)
-- adding a new feature module to this app means adding one line here,
nothing else.

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
from streamlit.errors import StreamlitAuthError

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
from utils.ui_security import block_password_clipboard

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


# ---------------------------------------------------------------------------
# PRE-LOGIN NAVIGATION -- landing page, then role-specific login/signup
# views, all navigated via st.session_state rather than Streamlit's own
# page-routing (this app already has its own PAGES-dict routing for the
# AUTHENTICATED side -- see render_authenticated_view() below -- so the
# same session_state-driven pattern is reused here for consistency,
# rather than introducing a second, different navigation mechanism just
# for the handful of screens a visitor sees before logging in).
# ---------------------------------------------------------------------------

ENTRY_VIEW_KEY = "entry_view"
VIEW_LANDING = "landing"
VIEW_SIGNUP_STUDENT = "signup_student"
VIEW_SIGNUP_INVITED = "signup_invited"
# Login views are NOT given their own separate VIEW_* constants -- the
# role string itself (config.ROLE_ADMIN/ROLE_TEACHER/ROLE_STUDENT, e.g.
# "admin") doubles as its own view value, since it is already guaranteed
# distinct from "landing"/"signup_student"/"signup_invited" above. This
# is what lets render_pre_login_view() below check `view in
# _ROLE_LOGIN_META` directly, with no separate mapping between "which
# view string" and "which role" to keep in sync.

# Display metadata for each role's login view -- title and icon, keyed by
# the SAME config.ROLE_* constant auth.login()'s expected_role parameter
# takes, so there is only one place that maps "which portal" to "which
# role", not two definitions that could drift apart.
_ROLE_LOGIN_META = {
    config.ROLE_ADMIN: ("Admin Login", ":material/shield_person:"),
    config.ROLE_TEACHER: ("Teacher Login", ":material/school:"),
    config.ROLE_STUDENT: ("Student Login", ":material/person:"),
}


def _go_to(view: str) -> None:
    """Switch the pre-login view and immediately rerun, so the new view
    paints on this same click rather than one click later."""
    st.session_state[ENTRY_VIEW_KEY] = view
    st.rerun()


def render_landing_page() -> None:
    """
    The very first thing a first-time visitor sees -- NOT a bare login
    form (see this file's module docstring's "WHAT THIS FILE DOES" for
    why that matters: this is what an examiner, or a real institution's
    first-time user, sees before anything else). Three role cards route
    to a role-SPECIFIC login view (render_role_login_form()) rather than
    one generic form asking "which role are you" -- see
    modules/auth.py's authenticate() docstring for why that separation is
    a real security property, not just a visual one. A separate "Request
    an account" link goes straight to Student self-registration --
    signup is never reachable by picking a role card labelled "Login".
    """
    st.title(":material/school: Student Performance Review & Prediction System")
    st.caption(
        "A role-based platform for managing student records, tracking attendance, "
        "and predicting academic outcomes with machine learning."
    )
    st.write(
        "This system brings together everything a college's academic office, faculty, "
        "and students need in one place: marks and attendance records, a real-time "
        "analytics dashboard, machine-learning-based at-risk predictions, and "
        "downloadable report cards -- all protected by role-based access, so everyone "
        "sees exactly what they're meant to and nothing more."
    )

    st.divider()
    st.subheader("What this system does")
    features = [
        (":material/edit_note:", "Marks & Attendance", "Record and track every student's marks and attendance, semester by semester."),
        (":material/monitoring:", "Analytics Dashboard", "Class averages, rankings, attendance trends, and pass/fail breakdowns at a glance."),
        (":material/warning:", "At-Risk Prediction", "A trained machine learning model flags students who may need early academic support."),
        (":material/picture_as_pdf:", "Report Cards", "Downloadable PDF report cards, including grades, SGPA, and ML predictions."),
        (":material/group:", "Role-Based Access", "Admins, Teachers, and Students each see only the data relevant to their role."),
        (":material/history:", "Full Audit Trail", "Every change to a student record is logged -- who changed it, and when."),
    ]
    feature_cols = st.columns(3)
    for index, (icon, feature_title, description) in enumerate(features):
        with feature_cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{icon} {feature_title}**")
                st.caption(description)

    st.divider()
    st.subheader("About")
    st.write(
        "Built as an 8th-semester BCA final year project, this system was designed "
        "around real institutional workflows: strict role-based data access, an "
        "auditable history of every change made to a student record, and machine "
        "learning models trained specifically for early academic intervention -- "
        "not just a digital grade book."
    )

    st.divider()
    st.subheader("Sign in to continue")
    role_cols = st.columns(3)
    for col, role in zip(role_cols, (config.ROLE_ADMIN, config.ROLE_TEACHER, config.ROLE_STUDENT)):
        title, icon = _ROLE_LOGIN_META[role]
        with col:
            with st.container(border=True):
                st.markdown(f"### {icon} {role.capitalize()}")
                if role == config.ROLE_ADMIN:
                    st.caption("Full system access, user management, and the audit log.")
                elif role == config.ROLE_TEACHER:
                    st.caption("Enter marks and attendance for your assigned subjects.")
                else:
                    st.caption("View your own marks, attendance, and predictions.")
                if st.button(title, use_container_width=True, key=f"landing_{role}_login", type="primary"):
                    _go_to(role)

    st.divider()
    _left, center, _right = st.columns([1, 2, 1])
    with center:
        st.caption("New student? Your roll number must already be on file before you can sign up.")
        if st.button(
            ":material/person_add: New student? Request an account",
            use_container_width=True, key="landing_student_signup",
        ):
            _go_to(VIEW_SIGNUP_STUDENT)


def render_role_login_form(role: str) -> None:
    """
    A login view scoped to exactly ONE role. The role name is always
    shown clearly (the page title itself), so there is no ambiguity
    about which portal a visitor is in.

    THE SECURITY THAT MATTERS HERE ISN'T THE LABEL, IT'S expected_role:
    passing expected_role=role into auth.login() (see modules/auth.py's
    authenticate() docstring for the full reasoning) is what makes a
    valid Admin username/password combination actually FAIL when
    submitted here on the Teacher or Student view -- the database query
    itself only matches a row with both that username AND this role, so
    there is no separate "is this the right portal" check for an
    attacker to find a gap in.

    THE "SIGN IN WITH GOOGLE" BUTTON FOLLOWS THE SAME RULE, VIA A
    DIFFERENT MECHANISM: st.login(role) below calls Streamlit's Google
    OAuth flow through a NAMED PROVIDER matching this role ([auth.admin]/
    [auth.teacher]/[auth.student] in secrets.toml -- same underlying
    Google Client ID/Secret repeated three times under three names, not
    three real separate Google apps). Which named provider was used
    survives the redirect to Google and back inside Streamlit's OWN
    signed identity cookie (st.user.provider) -- st.session_state does
    NOT reliably survive that redirect (confirmed empirically; see
    modules/auth.py's try_google_login() docstring for the full story and
    why this replaced an earlier session_state-based attempt).

    Args:
        role: One of config.ROLE_ADMIN/ROLE_TEACHER/ROLE_STUDENT.
    """
    title, icon = _ROLE_LOGIN_META[role]

    _left, center, _right = st.columns([1, 1.2, 1])
    with center:
        if st.button(":material/arrow_back: Back to Home", key=f"{role}_back"):
            _go_to(VIEW_LANDING)

        st.title(f"{icon} {title}")
        st.caption(f"Sign in to the {role.capitalize()} portal.")

        # st.form groups the two inputs and the button together so the
        # page only reruns (and only tries to log in) once, when "Log
        # In" is clicked -- not on every single keystroke in the
        # username/password boxes, which is what would happen without
        # a form.
        with st.form(f"{role}_login_form"):
            username = st.text_input("Username", placeholder="e.g. admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")

        if submitted:
            try:
                user = auth.login(username, password, expected_role=role)
                st.success(f"Welcome, {user['username']}.")
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

        st.divider()
        if st.button(
            ":material/login: Sign in with Google", use_container_width=True, key=f"{role}_google_login",
        ):
            try:
                # role itself IS the provider name here -- see this
                # function's docstring and modules/auth.py's
                # try_google_login() for why (st.user.provider, not
                # session_state, is what carries this across the redirect).
                st.login(role)
            except StreamlitAuthError:
                # Only reachable if [auth.<role>] isn't configured in
                # secrets.toml (a misconfigured/undeployed setup) --
                # st.login() validates credentials itself before
                # redirecting anywhere (see modules/auth.py's
                # try_google_login() docstring for the same check
                # applied on the READ side).
                st.error("Google Sign-In is not configured on this deployment yet.")

        st.divider()
        if role == config.ROLE_STUDENT:
            st.caption("New student?")
            if st.button("Request an account", use_container_width=True, key=f"{role}_signup_link"):
                _go_to(VIEW_SIGNUP_STUDENT)
        else:
            st.caption("Have an invite from an Admin?")
            if st.button("Sign up with your invite", use_container_width=True, key=f"{role}_signup_link"):
                _go_to(VIEW_SIGNUP_INVITED)


def render_student_signup_view() -> None:
    """
    Student self-registration -- see modules/auth.py's module docstring
    ("WHO CAN CREATE AN ACCOUNT") for the full reasoning: this only
    CLAIMS a login for a roll_no an Admin/Teacher already entered into
    the students table, matched against the email already on file for
    it, and is the sole self-service signup path for students.
    """
    _left, center, _right = st.columns([1, 1.2, 1])
    with center:
        if st.button(":material/arrow_back: Back to Home", key="student_signup_back"):
            _go_to(VIEW_LANDING)

        st.title(":material/person_add: Student Sign Up")
        st.caption(
            "Your roll number must already exist in the system (an Admin or "
            "Teacher enters it when you're enrolled) -- this just creates YOUR "
            "login for it."
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
                    st.success("Account created! Go back and log in as a Student.")
                except (ValidationError, RecordNotFoundError, DuplicateRecordError) as error:
                    # RecordNotFoundError: no such roll_no on file yet
                    #   (an Admin/Teacher needs to create the student
                    #   record first -- see the caption above).
                    # ValidationError: bad email/password, or the email
                    #   didn't match what's on file for that roll_no.
                    # DuplicateRecordError: a login already exists for
                    #   this roll_no.
                    st.error(str(error))


def render_invited_signup_view() -> None:
    """
    Teacher/Admin self-registration -- see modules/auth.py's module
    docstring for why this only works for an email an existing Admin has
    already pre-approved via invite_account().
    """
    _left, center, _right = st.columns([1, 1.2, 1])
    with center:
        if st.button(":material/arrow_back: Back to Home", key="invited_signup_back"):
            _go_to(VIEW_LANDING)

        st.title(":material/person_add: Sign Up (Invited)")
        st.caption(
            "For Teachers and Admins an existing Admin has already invited. "
            "Ask an administrator to invite your email first if you don't have "
            "an account yet -- see User Management."
        )
        with st.form("invited_signup_form", clear_on_submit=True):
            invited_email = st.text_input("Your invited email")
            invited_username = st.text_input("Choose a Username")
            invited_password = st.text_input(
                "Choose a Password", type="password", placeholder="At least 8 characters",
            )
            invited_confirm = st.text_input("Confirm Password", type="password")
            invited_submitted = st.form_submit_button(
                "Create My Account", use_container_width=True,
            )

        if invited_submitted:
            if invited_password != invited_confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    auth.signup_invited_account(invited_email, invited_username, invited_password)
                    st.success("Account created! Go back and log in.")
                except (ValidationError, RecordNotFoundError, DuplicateRecordError) as error:
                    # RecordNotFoundError: no pending invite for this
                    #   email (an Admin needs to invite it first).
                    # ValidationError: bad username/password.
                    # DuplicateRecordError: that username is already taken.
                    st.error(str(error))


def render_pre_login_view() -> None:
    """
    Dispatch to whichever pre-login screen st.session_state[ENTRY_VIEW_KEY]
    currently points at, defaulting to the landing page. This is the
    ONLY place that reads ENTRY_VIEW_KEY -- every other function above
    only ever WRITES to it, via _go_to(), then reruns immediately, so
    this dispatch always sees the latest choice on the very next paint.
    """
    view = st.session_state.get(ENTRY_VIEW_KEY, VIEW_LANDING)

    if view in _ROLE_LOGIN_META:
        render_role_login_form(view)
    elif view == VIEW_SIGNUP_STUDENT:
        render_student_signup_view()
    elif view == VIEW_SIGNUP_INVITED:
        render_invited_signup_view()
    else:
        render_landing_page()


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
        # Land back on the landing page, not whichever role-login view
        # this same browser session last visited before logging in --
        # logging out should feel like a fresh start, not resume mid-flow.
        st.session_state[ENTRY_VIEW_KEY] = VIEW_LANDING
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

    # One call here covers every password field on every page (login,
    # signup, change password, admin create/reset) -- see
    # utils/ui_security.py's module docstring for why a single call at
    # the top of main() is enough, rather than one per page.
    block_password_clipboard()

    # Checked on every single rerun, before anything else: if this
    # browser already has a verified Google identity (from a completed
    # "Sign in with Google" round-trip) but our own session doesn't know
    # about it yet, map it to an app account and start a session for it.
    # A no-op, cheaply, whenever there is no Google identity to check --
    # see modules/auth.py's try_google_login() for the full reasoning,
    # including why this is safe even when Google Sign-In isn't
    # configured at all.
    if auth.try_google_login():
        st.rerun()

    if auth.is_session_valid():
        render_authenticated_view(auth.get_current_user())
    else:
        render_pre_login_view()


if __name__ == "__main__":
    main()
