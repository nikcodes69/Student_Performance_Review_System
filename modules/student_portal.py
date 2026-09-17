"""
modules/student_portal.py
==========================
The Student-facing view: a logged-in Student sees only THEIR OWN marks,
attendance, SGPA, ML predictions, and report card -- never another
student's data. This is the one page in the whole app scoped to
config.ROLE_STUDENT alone (every other page is Admin/Teacher, or
Admin-only).

HOW "MY OWN DATA ONLY" IS ENFORCED: recall the convention established
back in database/db_setup.py's students table comment, and used again in
modules/auth.py's self_register_student() -- a Student-role user's
username IS their roll_no. This page reads auth.get_current_user()
["username"] and uses it directly as the roll_no to query, rather than
offering any dropdown of "which student to view". There is no selectbox
of students anywhere on this page -- that absence is the actual access
control, not a decorative choice. A Student cannot type or click their
way into seeing a classmate's marks, because the roll_no this page uses
is never taken from user input at all.

THIS PAGE REUSES EXISTING FUNCTIONS ENTIRELY -- no new business logic:
modules.marks.list_marks_for_student(), modules.attendance.
list_attendance_for_student(), modules.marks.compute_sgpa_for_marks(),
modules.ml_predictions's three predict_*_for_student() functions, and
utils.pdf_generator.generate_report_card() were all already built for
the Admin/Teacher-facing pages. This file is only a new, narrower way of
calling them -- pinned to one roll_no instead of letting the caller pick.
"""

import streamlit as st

import config
from modules import auth
from modules.attendance import list_attendance_for_student
from modules.marks import compute_sgpa_for_marks, list_marks_for_student
from modules.ml_predictions import (
    predict_at_risk_for_student,
    predict_final_marks_for_student,
    predict_segment_for_student,
)
from utils.exceptions import ModelNotFoundError, ValidationError
from utils.logger import get_logger
from utils.pdf_generator import generate_report_card

logger = get_logger(__name__)


def render_student_portal_page() -> None:
    """Streamlit page: a Student's own marks, attendance, SGPA, ML
    predictions, and report card download -- see module docstring for
    how this stays scoped to only the logged-in student's own data."""
    user = auth.require_role(config.ROLE_STUDENT)
    roll_no = user["username"]  # see module docstring: this convention is what makes "my own data" safe

    st.title("My Performance")

    semester = st.number_input(
        "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
        value=config.MAX_SEMESTER,
    )
    semester_value = int(semester)

    marks_rows = list_marks_for_student(roll_no, semester=semester_value)
    attendance_rows = list_attendance_for_student(roll_no, semester=semester_value)

    st.subheader("Marks")
    if not marks_rows:
        st.info("No marks recorded yet for this semester.")
    else:
        display_rows = [
            {
                "Subject": f"{row['subject_code']} - {row['subject_name']}",
                "Internal": row["internal"],
                "External": row["external"],
                "Practical": row["practical"],
                "Percentage": f"{row['percentage']}%",
                "Grade": row["grade_letter"],
                "Result": "Pass" if row["passed"] else "Fail",
            }
            for row in marks_rows
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        sgpa = compute_sgpa_for_marks(marks_rows)
        st.metric("SGPA", sgpa if sgpa is not None else "N/A")

    st.subheader("Attendance")
    if not attendance_rows:
        st.info("No attendance recorded yet for this semester.")
    else:
        display_rows = [
            {
                "Subject": f"{row['subject_code']} - {row['subject_name']}",
                "Classes Held": row["classes_held"],
                "Classes Attended": row["classes_attended"],
                "Attendance %": f"{row['attendance_percentage']}%" if row["attendance_percentage"] is not None else "N/A",
                "Shortage": "Yes" if row["shortage"] else "No",
            }
            for row in attendance_rows
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Predictions")
    st.caption(
        "Based on your marks, attendance, and assignment records on file -- "
        "computed automatically, nothing to fill in here."
    )

    if st.button("Show My Predictions"):
        try:
            risk = predict_at_risk_for_student(roll_no, semester_value)
            if risk["at_risk"]:
                st.error(f"At risk of failing -- predicted probability: {risk['risk_probability']:.1%}")
            else:
                st.success(f"On track to pass -- predicted risk probability: {risk['risk_probability']:.1%}")
        except (ModelNotFoundError, ValidationError) as error:
            st.warning(f"At-risk prediction unavailable: {error}")

        try:
            final_marks = predict_final_marks_for_student(roll_no, semester_value)
            st.metric("Predicted final percentage", f"{final_marks['predicted_final_percentage']}%")
        except (ModelNotFoundError, ValidationError) as error:
            st.warning(f"Final marks prediction unavailable: {error}")

        try:
            segment = predict_segment_for_student(roll_no, semester_value)
            st.info(f"Performance segment: **{segment['cluster_name']}**")
        except (ModelNotFoundError, ValidationError) as error:
            st.warning(f"Segmentation unavailable: {error}")

    st.divider()
    st.subheader("Report Card")
    if st.button("Generate My Report Card"):
        pdf_bytes = generate_report_card(roll_no, semester_value)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{roll_no}_semester{semester_value}_report_card.pdf",
            mime="application/pdf",
        )
