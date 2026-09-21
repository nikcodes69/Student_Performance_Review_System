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
from modules import auth, students, subjects
from modules.attendance import list_attendance_for_student
from modules.grades import calculate_cgpa
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


def generate_report_card(roll_no: str, semester: int, published_only: bool = False) -> bytes:
    """
    Build a complete PDF report card for one student's semester, always
    including the ML predictions section (every feature it needs,
    including assignment engagement, is computed from real records --
    see modules/ml_predictions.py's module docstring).

    Args:
        roll_no: The student.
        semester: Which semester to report on.
        published_only: If True, only PUBLISHED marks are included (see
            modules/marks.py's publish_marks()) -- modules/student_portal.py
            passes True for a Student downloading their OWN report card,
            so a draft mark they can't even see on-screen doesn't leak
            into a PDF they can download instead. Staff's own
            render_report_card_page() leaves this False -- an Admin/
            Teacher reviewing a report card needs to see drafts too.

    Returns:
        The finished PDF as raw bytes, ready for st.download_button() --
        never written to a temporary file on disk.

    Raises:
        RecordNotFoundError: if roll_no does not exist.
    """
    student = students.get_student(roll_no)
    subject_marks = list_marks_for_student(roll_no, semester=semester, published_only=published_only)
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


def _build_transcript_header_section(student: dict, styles) -> list:
    """Student details for a transcript -- admission year instead of a
    single semester, since a transcript (unlike a per-semester report
    card) spans every semester on record at once."""
    data = [
        ["Roll No.", student["roll_no"], "Name", student["name"]],
        ["Branch", student["branch"], "Admission Year", str(student["admission_year"])],
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


def generate_transcript(roll_no: str, published_only: bool = False) -> bytes:
    """
    Build a complete PDF transcript for one student -- EVERY semester
    they have marks recorded for, one after another, ending with the
    cumulative CGPA -- as opposed to generate_report_card()'s single
    semester.

    HOW CGPA IS COMPUTED: modules.grades.calculate_cgpa() needs a
    {"sgpa": ..., "credits": ...} pair per completed semester, where
    "credits" is the total credits of the subjects actually taken that
    semester (not every subject that exists) -- built here by summing
    subjects.credits for exactly the subject_codes appearing in that
    semester's marks, using ONE subjects.list_subjects() call up front
    (include_inactive=True, so a transcript still shows correctly for a
    subject that has since been retired) rather than a database query
    per semester.

    Args:
        roll_no: The student.
        published_only: If True, only PUBLISHED marks count towards each
            semester's entries and CGPA (see modules/marks.py's
            publish_marks()) -- modules/student_portal.py passes True for
            a Student downloading their OWN transcript, same reasoning as
            generate_report_card()'s published_only. Staff's own
            render_report_card_page() leaves this False.

    Returns:
        The finished PDF as raw bytes, ready for st.download_button().

    Raises:
        RecordNotFoundError: if roll_no does not exist.
    """
    student = students.get_student(roll_no)
    credits_by_code = {s["subject_code"]: s["credits"] for s in subjects.list_subjects(include_inactive=True)}

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Student Performance Review &amp; Prediction System", styles["Title"]),
        Paragraph("Official Academic Transcript", styles["Heading2"]),
        Spacer(1, 0.5 * cm),
    ]
    story.extend(_build_transcript_header_section(student, styles))
    story.append(Spacer(1, 0.5 * cm))

    semester_results = []
    any_marks_at_all = False
    for semester in range(config.MIN_SEMESTER, config.MAX_SEMESTER + 1):
        semester_marks = list_marks_for_student(roll_no, semester=semester, published_only=published_only)
        if not semester_marks:
            continue
        any_marks_at_all = True

        sgpa = compute_sgpa_for_marks(semester_marks)
        if sgpa is not None:
            semester_credits = sum(credits_by_code.get(row["subject_code"], 0) for row in semester_marks)
            semester_results.append({"sgpa": sgpa, "credits": semester_credits})

        story.append(Paragraph(f"Semester {semester}", styles["Heading3"]))
        story.extend(_build_marks_section(semester_marks, sgpa, styles))
        story.append(Spacer(1, 0.4 * cm))

    if not any_marks_at_all:
        story.append(Paragraph("No marks recorded for this student yet.", styles["Normal"]))
    elif semester_results:
        cgpa = calculate_cgpa(semester_results)
        story.append(Paragraph(f"<b>Cumulative CGPA: {cgpa}</b>", styles["Heading3"]))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"],
    ))

    doc.build(story)
    logger.info("Transcript generated for %s.", roll_no)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# CLASS-WIDE SUMMARY REPORT (an institution/department-level PDF, as
# opposed to generate_report_card()'s single-student report above --
# built for handing to a faculty meeting or department review)
# ---------------------------------------------------------------------------

CLASS_REPORT_RANKINGS_LIMIT = 30


def _build_class_kpi_section(summary: dict, at_risk_count: int | None, styles) -> list:
    """The same headline numbers shown on the Dashboard page's KPI row
    (see modules/analytics.py's render_dashboard_page()), as a small
    table instead of Streamlit metric cards."""
    data = [
        ["Active Students", str(summary["student_count"])],
        ["Active Subjects", str(summary["subject_count"])],
        ["Class Average", f"{summary['class_average']}%" if summary["class_average"] is not None else "N/A"],
        ["Pass Rate", f"{summary['pass_rate']}%" if summary["pass_rate"] is not None else "N/A"],
        ["Avg. Attendance", f"{summary['avg_attendance']}%" if summary["avg_attendance"] is not None else "N/A"],
        ["At-Risk Count", str(at_risk_count) if at_risk_count is not None else "N/A"],
    ]
    table = Table(data, colWidths=[5 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [Paragraph("Summary", styles["Heading3"]), table]


def _build_subject_averages_section(subject_averages: list[dict], styles) -> list:
    elements = [Paragraph("Subject Averages", styles["Heading3"])]
    if not subject_averages:
        elements.append(Paragraph("No marks data available.", styles["Normal"]))
        return elements

    rows = [["Subject", "Average %", "Entries"]]
    for row in subject_averages:
        rows.append([f"{row['subject_code']} - {row['subject_name']}", f"{row['average_percentage']}%", str(row["entry_count"])])

    table = Table(rows, colWidths=[8 * cm, 3 * cm, 3 * cm], repeatRows=1)
    table.setStyle(TableStyle(TABLE_HEADER_STYLE))
    elements.append(table)
    return elements


def _build_grade_distribution_section(grade_distribution: dict, styles) -> list:
    elements = [Paragraph("Grade Distribution", styles["Heading3"])]
    if sum(grade_distribution.values()) == 0:
        elements.append(Paragraph("No marks data available.", styles["Normal"]))
        return elements

    rows = [["Grade", "Count"]] + [[letter, str(count)] for letter, count in grade_distribution.items()]
    table = Table(rows, colWidths=[5 * cm, 5 * cm], repeatRows=1)
    table.setStyle(TableStyle(TABLE_HEADER_STYLE))
    elements.append(table)
    return elements


def _build_top_bottom_section(top_performers: list[dict], bottom_performers: list[dict], styles) -> list:
    elements = [Paragraph("Top and Bottom Performers", styles["Heading3"])]
    if not top_performers and not bottom_performers:
        elements.append(Paragraph("No marks data available.", styles["Normal"]))
        return elements

    def _performers_table(performers: list[dict]) -> Table:
        rows = [["Roll No", "Student", "Average %"]]
        for row in performers:
            rows.append([row["roll_no"], row["student_name"], f"{row['average_percentage']}%"])
        table = Table(rows, colWidths=[3 * cm, 6 * cm, 3 * cm], repeatRows=1)
        table.setStyle(TableStyle(TABLE_HEADER_STYLE))
        return table

    if top_performers:
        elements.append(Paragraph("Top Performers", styles["Normal"]))
        elements.append(_performers_table(top_performers))
        elements.append(Spacer(1, 0.3 * cm))
    if bottom_performers:
        elements.append(Paragraph("Bottom Performers", styles["Normal"]))
        elements.append(_performers_table(bottom_performers))
    return elements


def _build_shortage_section(shortage_list: list[dict], styles) -> list:
    elements = [Paragraph("Attendance Shortage Alerts", styles["Heading3"])]
    elements.append(Paragraph(
        f"Students below {config.ATTENDANCE_SHORTAGE_THRESHOLD}% average attendance.", styles["Normal"],
    ))
    if not shortage_list:
        elements.append(Paragraph("No students currently below the threshold.", styles["Normal"]))
        return elements

    rows = [["Roll No", "Student", "Average Attendance %"]]
    for row in shortage_list:
        rows.append([row["roll_no"], row["student_name"], f"{row['average_attendance']}%"])
    table = Table(rows, colWidths=[3 * cm, 6 * cm, 4 * cm], repeatRows=1)
    table.setStyle(TableStyle(TABLE_HEADER_STYLE))
    elements.append(table)
    return elements


def _build_rankings_section(rankings: list[dict], styles) -> list:
    """
    Class rankings, capped at CLASS_REPORT_RANKINGS_LIMIT rows -- a real
    department could have far more students than comfortably fits a PDF
    handed out at a meeting. The FULL, unlimited list is already
    available elsewhere in the app (the Analytics page's Class Rankings
    table, which has its own CSV/Excel export via utils/table_view.py) --
    this section is a print-friendly summary, not the only place to get
    the complete data.
    """
    elements = [Paragraph("Class Rankings", styles["Heading3"])]
    if not rankings:
        elements.append(Paragraph("No marks data available.", styles["Normal"]))
        return elements

    shown = rankings[:CLASS_REPORT_RANKINGS_LIMIT]
    if len(rankings) > CLASS_REPORT_RANKINGS_LIMIT:
        elements.append(Paragraph(
            f"Showing top {CLASS_REPORT_RANKINGS_LIMIT} of {len(rankings)} students -- "
            "see the Analytics page for the complete, exportable list.",
            styles["Normal"],
        ))

    rows = [["Rank", "Roll No", "Student", "Average %", "Percentile"]]
    for row in shown:
        rows.append([str(row["rank"]), row["roll_no"], row["student_name"], f"{row['average_percentage']}%", f"{row['percentile']}"])
    table = Table(rows, colWidths=[1.5 * cm, 2.5 * cm, 6 * cm, 3 * cm, 3 * cm], repeatRows=1)
    table.setStyle(TableStyle(TABLE_HEADER_STYLE))
    elements.append(table)
    return elements


def generate_class_report(semester: int | None = None) -> bytes:
    """
    Build an institution/class-wide PDF summary report -- KPIs, subject
    averages, grade distribution, top/bottom performers, attendance
    shortage alerts, and class rankings -- for handing to a faculty
    meeting or department review. This is the class-wide counterpart to
    generate_report_card()'s single-student report above; every section
    reuses modules/analytics.py's existing query functions directly
    rather than recomputing any of this data a second way.

    Args:
        semester: If given, restricts every section to that semester;
            otherwise covers every semester on record. The Summary KPI
            section (student/subject counts, at-risk count) is always
            institution-wide regardless of this filter, matching what
            the Dashboard page itself shows -- there is no meaningful
            "per-semester active student count" distinct from the
            semester filter already narrowing every OTHER section.

    Returns:
        The finished PDF as raw bytes, ready for st.download_button().
    """
    from modules import analytics
    from modules.ml_predictions import get_at_risk_count

    summary = analytics.get_dashboard_summary()
    at_risk_count = get_at_risk_count()
    subject_averages = analytics.get_subject_averages(semester=semester)
    grade_distribution = analytics.get_grade_distribution(semester=semester)
    top_performers = analytics.get_top_performers(n=10, semester=semester)
    bottom_performers = analytics.get_bottom_performers(n=10, semester=semester)
    shortage_list = analytics.get_attendance_shortage_list(semester=semester)
    rankings = analytics.get_class_rankings(semester=semester)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()

    scope_text = f"Semester {semester}" if semester is not None else "All Semesters"
    story = [
        Paragraph("Student Performance Review &amp; Prediction System", styles["Title"]),
        Paragraph(f"Class Summary Report -- {scope_text}", styles["Heading2"]),
        Spacer(1, 0.5 * cm),
    ]
    story.extend(_build_class_kpi_section(summary, at_risk_count, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_subject_averages_section(subject_averages, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_grade_distribution_section(grade_distribution, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_top_bottom_section(top_performers, bottom_performers, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_shortage_section(shortage_list, styles))
    story.append(Spacer(1, 0.5 * cm))
    story.extend(_build_rankings_section(rankings, styles))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"],
    ))

    doc.build(story)
    logger.info("Class report generated (semester=%s).", semester)
    return buffer.getvalue()


def render_class_report_page() -> None:
    """Streamlit page: pick an optional semester filter, then generate
    and download the class-wide summary PDF."""
    auth.require_role(*REPORT_CARD_ROLES)
    st.title("Class Report Export")
    st.caption(
        "A single PDF summarising the whole class/institution -- KPIs, subject averages, "
        "grade distribution, top/bottom performers, attendance alerts, and class rankings. "
        "For one student's own report, use Report Card Export instead."
    )

    semester_choice = st.selectbox(
        "Filter by semester", options=["(all)"] + list(range(config.MIN_SEMESTER, config.MAX_SEMESTER + 1)),
    )
    semester = None if semester_choice == "(all)" else int(semester_choice)

    if st.button("Generate Class Report"):
        pdf_bytes = generate_class_report(semester=semester)
        st.success("Class report generated.")
        scope_label = f"semester{semester}" if semester is not None else "all_semesters"
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"class_report_{scope_label}.pdf",
            mime="application/pdf",
        )


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
        # published_only=False: staff reviewing a report card needs to see
        # drafts too, unlike a student downloading their own (see
        # generate_report_card()'s docstring).
        pdf_bytes = generate_report_card(picked_student["roll_no"], int(semester), published_only=False)
        st.success("Report card generated.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{picked_student['roll_no']}_semester{int(semester)}_report_card.pdf",
            mime="application/pdf",
        )

    st.divider()
    st.subheader("Full Transcript")
    st.caption(
        f"Every semester {picked_student['roll_no']} has marks recorded for, one after "
        "another, ending with the cumulative CGPA -- as opposed to the single-semester "
        "report card above."
    )
    if st.button("Generate Transcript"):
        transcript_bytes = generate_transcript(picked_student["roll_no"], published_only=False)
        st.success("Transcript generated.")
        st.download_button(
            "Download Transcript PDF",
            data=transcript_bytes,
            file_name=f"{picked_student['roll_no']}_transcript.pdf",
            mime="application/pdf",
        )
