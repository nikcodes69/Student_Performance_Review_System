"""
utils/pdf_generator.py
=======================
Builds a downloadable PDF report card for one student's semester --
personal details, marks per subject with computed grades, SGPA,
attendance per subject, and (optionally) ML predictions -- using
ReportLab's "platypus" flowables API.

WHY THIS FILE HAS A STREAMLIT PAGE, UNLIKE utils/validators.py,
utils/exceptions.py, AND utils/logger.py: those three are pure logic with
zero UI, used by many other files. This file is different -- its entire
purpose is to produce one downloadable artefact for a human to click a
button and receive, so its generation logic and its trigger page are
tightly coupled and belong together. This mirrors a precedent already set
by modules/auth.py, which is likewise split into a pure-logic half and a
Streamlit-session half in one file -- the same reasoning applies here.

HOW THE PDF IS BUILT: a list of "flowables" (Paragraph, Table, Spacer --
ReportLab's term for a piece of content that can flow across page breaks)
is assembled in `generate_report_card()`, then handed to a
SimpleDocTemplate that lays them out and renders the final PDF. The
finished PDF is returned as raw bytes, written to an in-memory BytesIO
buffer -- never to a temporary file on disk -- so it can be handed
directly to Streamlit's st.download_button() and cleaned up automatically
when the buffer goes out of scope.
"""

from datetime import datetime
from io import BytesIO

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config
from modules import auth, students
from modules.attendance import list_attendance_for_student
from modules.marks import compute_sgpa_for_marks, list_marks_for_student
from utils.exceptions import ModelNotFoundError, ValidationError
from utils.logger import get_logger

logger = get_logger(__name__)

REPORT_CARD_ROLES = (config.ROLE_ADMIN, config.ROLE_TEACHER)

TABLE_HEADER_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
]


# ---------------------------------------------------------------------------
# PURE PDF-BUILDING LOGIC (no Streamlit here, down to generate_report_card)
# ---------------------------------------------------------------------------
# SGPA is computed via modules.marks.compute_sgpa_for_marks() -- see that
# function's docstring for why this calculation is centralised in one
# shared place (modules/ml_predictions.py and modules/student_portal.py
# both need the exact same "marks rows -> SGPA" computation) rather than
# a private copy living in this file.


def _build_student_info_section(student: dict, semester: int, styles) -> list:
    """A small key-value table of the student's own details."""
    data = [
        ["Roll No.", student["roll_no"], "Name", student["name"]],
        ["Branch", student["branch"], "Semester", str(semester)],
        ["Email", student["email"], "Phone", student["phone"]],
    ]
    table = Table(data, colWidths=[2.5 * cm, 5 * cm, 2.5 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [Paragraph("Student Information", styles["Heading3"]), table]


def _build_marks_section(subject_marks: list[dict], sgpa: float | None, styles) -> list:
    """The marks table, with each row's grade/pass-fail already computed
    by modules.marks.list_marks_for_student() via modules/grades.py --
    this file never recomputes a grade, only displays one."""
    elements = [Paragraph("Marks", styles["Heading3"])]

    if not subject_marks:
        elements.append(Paragraph("No marks recorded for this semester.", styles["Normal"]))
        return elements

    header = ["Subject", "Internal", "External", "Practical", "Percentage", "Grade", "Result"]
    rows = [header]
    for row in subject_marks:
        rows.append([
            f"{row['subject_code']} - {row['subject_name']}",
            str(row["internal"]),
            str(row["external"]),
            str(row["practical"]),
            f"{row['percentage']}%",
            row["grade_letter"],
            "Pass" if row["passed"] else "Fail",
        ])

    table = Table(
        rows,
        colWidths=[5.5 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.2 * cm, 1.5 * cm, 1.5 * cm],
        repeatRows=1,
    )
    table.setStyle(TableStyle(TABLE_HEADER_STYLE))

    # A second setStyle() call ADDS to the table's style rather than
    # replacing it, so this highlights failing rows in red on top of the
    # header formatting already applied above.
    for row_index, row in enumerate(subject_marks, start=1):
        if not row["passed"]:
            table.setStyle(TableStyle([("TEXTCOLOR", (0, row_index), (-1, row_index), colors.red)]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * cm))

    sgpa_text = f"SGPA: {sgpa}" if sgpa is not None else "SGPA: N/A (no marks recorded)"
    elements.append(Paragraph(f"<b>{sgpa_text}</b>", styles["Normal"]))
    return elements


def _build_attendance_section(subject_attendance: list[dict], styles) -> list:
    """The attendance table -- attendance_percentage/shortage are already
    computed by modules.attendance.list_attendance_for_student()."""
    elements = [Paragraph("Attendance", styles["Heading3"])]

    if not subject_attendance:
        elements.append(Paragraph("No attendance recorded for this semester.", styles["Normal"]))
        return elements

    header = ["Subject", "Classes Held", "Classes Attended", "Attendance %", "Shortage"]
    rows = [header]
    for row in subject_attendance:
        percentage = row["attendance_percentage"]
        rows.append([
            f"{row['subject_code']} - {row['subject_name']}",
            str(row["classes_held"]),
            str(row["classes_attended"]),
            f"{percentage}%" if percentage is not None else "N/A",
            "Yes" if row["shortage"] else "No",
        ])

    table = Table(rows, colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm, 2 * cm], repeatRows=1)
    table.setStyle(TableStyle(TABLE_HEADER_STYLE))
    elements.append(table)
    return elements


def _build_ml_predictions_section(roll_no: str, semester: int, styles) -> list:
    """
    The ML predictions section. Always included -- even if every
    individual prediction turns out to be unavailable, the section still
    appears explaining why, rather than silently vanishing, which could
    look like the predictions were simply forgotten rather than genuinely
    unavailable (e.g. ml/model_training.ipynb has not been run yet, or
    this is the student's very first semester so there is no "previous"
    data to compute some features from). All three predictions, including
    the assignment-engagement feature, are computed entirely from real
    records -- see modules/ml_predictions.py's module docstring.
    """
    from modules.ml_predictions import (
        predict_at_risk_for_student,
        predict_final_marks_for_student,
        predict_segment_for_student,
    )

    elements = [Paragraph("Machine Learning Predictions", styles["Heading3"])]

    try:
        risk = predict_at_risk_for_student(roll_no, semester)
        status = "AT RISK" if risk["at_risk"] else "On track"
        elements.append(Paragraph(
            f"At-risk status: <b>{status}</b> (predicted probability: {risk['risk_probability']:.0%})",
            styles["Normal"],
        ))
    except (ModelNotFoundError, ValidationError) as error:
        elements.append(Paragraph(f"At-risk prediction unavailable: {error}", styles["Normal"]))

    try:
        final_marks = predict_final_marks_for_student(roll_no, semester)
        elements.append(Paragraph(
            f"Predicted final percentage: <b>{final_marks['predicted_final_percentage']}%</b>",
            styles["Normal"],
        ))
    except (ModelNotFoundError, ValidationError) as error:
        elements.append(Paragraph(f"Final marks prediction unavailable: {error}", styles["Normal"]))

    try:
        segment = predict_segment_for_student(roll_no, semester)
        elements.append(Paragraph(f"Performance segment: <b>{segment['cluster_name']}</b>", styles["Normal"]))
    except (ModelNotFoundError, ValidationError) as error:
        elements.append(Paragraph(f"Segmentation unavailable: {error}", styles["Normal"]))

    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph(
        "<i>These are statistical estimates from a trained model, not guarantees. "
        "See the Model Comparison page for accuracy figures.</i>",
        styles["Normal"],
    ))
    return elements


def generate_report_card(roll_no: str, semester: int) -> bytes:
    """
    Build a complete PDF report card for one student's semester, always
    including the ML predictions section (every feature it needs,
    including assignment engagement, is computed from real records --
    see modules/ml_predictions.py's module docstring).

    Args:
        roll_no: The student.
        semester: Which semester to report on.

    Returns:
        The finished PDF as raw bytes, ready for st.download_button() --
        never written to a temporary file on disk.

    Raises:
        RecordNotFoundError: if roll_no does not exist.
    """
    student = students.get_student(roll_no)
    subject_marks = list_marks_for_student(roll_no, semester=semester)
    subject_attendance = list_attendance_for_student(roll_no, semester=semester)
    sgpa = compute_sgpa_for_marks(subject_marks)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Student Performance Review &amp; Prediction System", styles["Title"]),
        Paragraph("Semester Report Card", styles["Heading2"]),
        Spacer(1, 0.5 * cm),
    ]
    story.extend(_build_student_info_section(student, semester, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_marks_section(subject_marks, sgpa, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_attendance_section(subject_attendance, styles))

    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_ml_predictions_section(roll_no, semester, styles))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"],
    ))

    doc.build(story)
    logger.info("Report card generated for %s (semester %s).", roll_no, semester)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------------------------

def render_report_card_page() -> None:
    """Streamlit page: pick a student and semester, then generate and
    download the PDF (always includes the ML predictions section -- every
    feature it needs, including assignment engagement, is computed from
    real records now, not typed in by hand)."""
    auth.require_role(*REPORT_CARD_ROLES)
    st.title("Report Card Export")

    student_list = students.list_students()
    if not student_list:
        st.info("No students found.")
        return

    labels = {f"{s['roll_no']} - {s['name']}": s for s in student_list}
    picked_label = st.selectbox("Student", options=list(labels.keys()))
    picked_student = labels[picked_label]

    semester = st.number_input(
        "Semester", min_value=config.MIN_SEMESTER, max_value=config.MAX_SEMESTER,
        value=picked_student["semester"], step=1,
    )

    if st.button("Generate Report Card"):
        pdf_bytes = generate_report_card(picked_student["roll_no"], int(semester))
        st.success("Report card generated.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{picked_student['roll_no']}_semester{int(semester)}_report_card.pdf",
            mime="application/pdf",
        )
