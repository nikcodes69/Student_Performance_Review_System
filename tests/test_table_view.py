"""
tests/test_table_view.py
=========================
pytest tests for utils/table_view.py's export helpers (_to_csv_bytes,
_to_excel_bytes) -- the pure, reusable logic behind every "Export CSV" /
"Export Excel" button in this app (see modules/students.py, subjects.py,
marks.py, attendance.py, assignments.py, audit.py, auth.py, and
modules/analytics.py's class rankings table, all of which call
render_data_table()/render_export_buttons() from this module).

WHY THE SEARCH/PAGINATION LOGIC ITSELF IS NOT UNIT-TESTED HERE:
render_data_table() drives real Streamlit widgets (st.text_input,
st.number_input) and reads/writes st.session_state -- exercising that
properly needs Streamlit's own st.testing.v1.AppTest harness (a full
simulated script run), which is significantly heavier to set up than the
value justifies for this project. The search/pagination arithmetic
itself (substring filtering, page-count math, slicing) was verified by
hand against real data while building the feature -- see the chat
history around this feature's implementation. This file focuses on the
two PURE functions that do not depend on any Streamlit runtime at all,
and so are cleanly testable exactly like utils/validators.py and
modules/grades.py's tests already are.

HOW TO RUN (from the project root):
    python -m pytest tests/test_table_view.py -v
"""

from io import BytesIO

import pandas as pd

from utils.table_view import _to_csv_bytes, _to_excel_bytes


def test_to_csv_bytes_round_trips_exactly():
    rows = [{"roll_no": "S1", "name": "Alice"}, {"roll_no": "S2", "name": "Bob"}]
    csv_bytes = _to_csv_bytes(rows)
    roundtrip = pd.read_csv(BytesIO(csv_bytes)).to_dict("records")
    assert roundtrip == rows


def test_to_csv_bytes_handles_unicode_and_commas():
    # A name containing a comma is exactly the case CSV's own quoting
    # rules exist for -- if this were built by hand with string
    # concatenation instead of pandas' csv writer, a comma in the data
    # would corrupt the column structure. Also covers a non-ASCII name,
    # since roll_no/name are free-text fields real students can enter.
    rows = [{"name": "Thapa, Nikhil"}, {"name": "Nguyễn Văn A"}]
    csv_bytes = _to_csv_bytes(rows)
    roundtrip = pd.read_csv(BytesIO(csv_bytes)).to_dict("records")
    assert roundtrip == rows


def test_to_excel_bytes_round_trips_exactly():
    rows = [{"roll_no": "S1", "average_percentage": 90.5}, {"roll_no": "S2", "average_percentage": 15.73}]
    excel_bytes = _to_excel_bytes(rows, sheet_name="students")
    roundtrip = pd.read_excel(BytesIO(excel_bytes)).to_dict("records")
    assert roundtrip == rows


def test_to_excel_bytes_truncates_long_sheet_names():
    # Excel's own file format caps sheet names at 31 characters -- a
    # longer name would raise from openpyxl if not truncated first (see
    # _to_excel_bytes()'s docstring). filename_prefix values used
    # elsewhere in this project (e.g. "class_rankings") are already
    # under that limit, but this guards the general case regardless.
    rows = [{"a": 1}]
    long_name = "x" * 50
    excel_bytes = _to_excel_bytes(rows, sheet_name=long_name)  # must not raise
    roundtrip = pd.read_excel(BytesIO(excel_bytes)).to_dict("records")
    assert roundtrip == rows
