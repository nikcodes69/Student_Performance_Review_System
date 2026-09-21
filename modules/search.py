"""
modules/search.py
==================
Global search: one search box, matched against both students (roll_no or
name) and subjects (subject_code or name) at once -- for an Admin/Teacher
who knows a name or code but not which page it lives on.

WHY THIS FILTERS IN PYTHON RATHER THAN A SQL LIKE QUERY: modules/students.py's
list_students() and modules/subjects.py's list_subjects() are already
@st.cache_data-cached, 60-second-TTL, cleared immediately on every write
(see those functions' own docstrings) -- reusing them here means this
page's search is instant (reading from cache, not hitting the database on
every keystroke) and can never drift out of sync with what those pages
show, since it is the literal same data. Writing a separate SQL LIKE
query would duplicate that caching/invalidation logic for a second time,
for a dataset small enough (this project's realistic scale: one BCA
program, at most a few hundred students) that a Python substring filter
over an already-cached list costs nothing noticeable.

WHY ADMIN/TEACHER ONLY: this searches across EVERY student and subject,
not just "your own" anything -- the same visibility Students/Subjects
pages already have. A Student already has their own scoped view
(modules/student_portal.py); a general cross-student lookup is exactly
the kind of access that page's docstring explains a Student must not have.
"""

import streamlit as st

import config
from modules import auth, students, subjects


def search_students(query: str, include_inactive: bool = False) -> list[dict]:
    """
    Every student whose roll_no or name contains `query` (case-insensitive).

    Args:
        query: The search text. An empty/whitespace-only query matches nothing.
        include_inactive: If True, also search deactivated students.

    Returns:
        A list of dicts (see modules/students.py's STUDENT_COLUMNS),
        ordered however list_students() itself orders them.
    """
    needle = query.strip().lower()
    if not needle:
        return []

    return [
        row for row in students.list_students(include_inactive=include_inactive)
        if needle in row["roll_no"].lower() or needle in row["name"].lower()
    ]


def search_subjects(query: str, include_inactive: bool = False) -> list[dict]:
    """
    Every subject whose subject_code or name contains `query`
    (case-insensitive).

    Args:
        query: The search text. An empty/whitespace-only query matches nothing.
        include_inactive: If True, also search deactivated subjects.

    Returns:
        A list of dicts (see modules/subjects.py's SUBJECT_COLUMNS),
        ordered however list_subjects() itself orders them.
    """
    needle = query.strip().lower()
    if not needle:
        return []

    return [
        row for row in subjects.list_subjects(include_inactive=include_inactive)
        if needle in row["subject_code"].lower() or needle in row["name"].lower()
    ]


def render_search_page() -> None:
    """Streamlit page: one search box, results split into Students and
    Subjects sections."""
    auth.require_role(config.ROLE_ADMIN, config.ROLE_TEACHER)

    st.title("Search")
    query = st.text_input(
        "Search by roll number, student name, subject code, or subject name",
        placeholder="e.g. BCA045 or Sharma or CACS301",
    )

    if not query.strip():
        st.info("Type something above to search.")
        return

    student_results = search_students(query)
    subject_results = search_subjects(query)

    st.subheader(f":material/group: Students ({len(student_results)})")
    if student_results:
        display_rows = [
            {
                "Roll No": row["roll_no"],
                "Name": row["name"],
                "Semester": row["semester"],
                "Branch": row["branch"],
                "Email": row["email"],
            }
            for row in student_results
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No matching students.")

    st.subheader(f":material/menu_book: Subjects ({len(subject_results)})")
    if subject_results:
        display_rows = [
            {
                "Subject Code": row["subject_code"],
                "Name": row["name"],
                "Semester": row["semester"],
                "Credits": row["credits"],
            }
            for row in subject_results
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No matching subjects.")
