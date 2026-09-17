"""
utils/table_view.py
====================
A single, reusable helper for every "list of records" table in this
app: search (client-side substring filter), pagination, and CSV/Excel
export -- used by modules/students.py, subjects.py, marks.py,
attendance.py, assignments.py, audit.py, and analytics.py, instead of
half a dozen near-identical copies of the same three controls.

SORTING IS DELIBERATELY NOT REIMPLEMENTED HERE: st.dataframe (built on
the glide-data-grid widget) already lets a user click any column header
to sort that column, ascending or descending, entirely client-side, with
no code needed on our end -- adding a second, custom sort control next
to a table that can already sort itself would just be a confusing,
redundant control. This file's job is the things st.dataframe cannot
already do by itself: text search, pagination (a table with hundreds of
rows renders slowly without cutting it into pages), and exporting to a
file.

WHY SEARCH IS PLAIN PYTHON, NOT SQL: every list_*() function this is
used with has already run its query and returned plain dicts by the time
it reaches here -- searching in Python over an already-fetched result
list is simpler than threading a search term back down into a dozen
different SQL queries, each with different columns, and the row counts
in this project are small enough (a class or department, not millions of
rows) that this is never a real performance concern.

HOW THE PAGE-NUMBER WIDGET AVOIDS A KNOWN STREAMLIT FOOTGUN: once a
widget has been created with a given `key` in an earlier script run,
Streamlit uses whatever is in st.session_state[key] for that widget on
every later rerun -- a `value=` argument passed alongside `key=` is only
ever used the VERY FIRST time that key appears, not as a way to force-
update it afterwards. That matters here because search narrowing the
result set can make a previously valid page number (e.g. page 3) larger
than the new total_pages (now only 1 page) -- passing value=1 in that
situation would silently be ignored, and the widget would keep showing
"3" while this code sliced an empty page. The fix used below is the
Streamlit-documented one: write the CLAMPED number directly into
st.session_state[key] *before* the widget with that same key is
instantiated in this run, and never pass `value=` to that widget at all.
"""

from io import BytesIO

import pandas as pd
import streamlit as st


def _to_csv_bytes(rows: list[dict]) -> bytes:
    """Encode a list of dicts as CSV bytes, ready for st.download_button."""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _to_excel_bytes(rows: list[dict], sheet_name: str = "Sheet1") -> bytes:
    """
    Encode a list of dicts as .xlsx bytes.

    Written to an in-memory BytesIO buffer rather than a temporary file
    on disk -- there is nothing here that needs to survive past this one
    function call, and a deployed Streamlit Cloud instance's filesystem
    is not guaranteed to be writable/persistent anyway.

    Excel sheet names have a hard 31-character limit (an Excel file
    format rule, not a pandas/openpyxl one) -- sheet_name is truncated to
    fit, since a name any longer would raise an error from openpyxl.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
    return buffer.getvalue()


def render_export_buttons(rows: list[dict], filename_prefix: str, key_prefix: str) -> None:
    """
    Two download buttons -- CSV and Excel -- for any list of plain dicts.

    Args:
        rows: The exact rows to export. When called from
            render_data_table() below, this is the SEARCH-FILTERED set,
            not necessarily every row that exists -- exporting "what I'm
            currently looking at" is the more useful default once a
            search has narrowed things down.
        filename_prefix: Used to build both file names, e.g. "students"
            -> "students.csv" / "students.xlsx".
        key_prefix: Unique per call site, so two export button pairs
            elsewhere on the same page never collide (Streamlit requires
            every widget's key to be unique across the whole app).
    """
    if not rows:
        return

    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "Export CSV", data=_to_csv_bytes(rows),
            file_name=f"{filename_prefix}.csv", mime="text/csv",
            key=f"{key_prefix}_export_csv", use_container_width=True,
            icon=":material/download:",
        )
    with export_cols[1]:
        st.download_button(
            "Export Excel", data=_to_excel_bytes(rows, sheet_name=filename_prefix),
            file_name=f"{filename_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_export_excel", use_container_width=True,
            icon=":material/download:",
        )


def render_data_table(
    rows: list[dict],
    key_prefix: str,
    filename_prefix: str = "export",
    page_size: int = 20,
    search_placeholder: str = "Search this table...",
) -> None:
    """
    Render one list of dicts as a searchable, paginated, exportable
    table -- the standard way every "list of records" page in this
    project should show its data from here on.

    Args:
        rows: The full, unfiltered list of dicts to show. Column names
            are whatever keys the dicts already have -- pass an already-
            relabelled "display_rows" list, the same way every
            render_*_page() in this project already builds one before
            calling st.dataframe() directly.
        key_prefix: Unique per call site (see render_export_buttons()).
        filename_prefix: Used for the exported file's name and the
            Excel sheet name.
        page_size: Rows shown per page.
        search_placeholder: Placeholder text for the search box.

    SORTING: click any column header in the rendered table -- see the
    module docstring for why that is st.dataframe's own built-in
    behaviour rather than something this function implements.
    """
    if not rows:
        st.info("No data to display.")
        return

    search_term = st.text_input(
        "Search", key=f"{key_prefix}_search", placeholder=search_placeholder,
        icon=":material/search:", label_visibility="collapsed",
    ).strip().lower()

    if search_term:
        filtered_rows = [
            row for row in rows
            if any(search_term in str(value).lower() for value in row.values())
        ]
    else:
        filtered_rows = rows

    total_rows = len(filtered_rows)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # See module docstring for why the clamp happens by writing directly
    # into session_state BEFORE the widget below is created, rather than
    # via a `value=` argument.
    page_input_key = f"{key_prefix}_page_input"
    if page_input_key not in st.session_state:
        st.session_state[page_input_key] = 1
    elif st.session_state[page_input_key] > total_pages:
        st.session_state[page_input_key] = total_pages

    caption_col, page_col = st.columns([3, 1])
    with caption_col:
        found_caption = f"{total_rows} record(s) found"
        if search_term:
            found_caption += f" (filtered from {len(rows)})"
        st.caption(found_caption)
    with page_col:
        st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1,
            key=page_input_key, label_visibility="collapsed",
        )

    current_page = st.session_state[page_input_key]
    st.caption(f"Page {current_page} of {total_pages}")

    start = (current_page - 1) * page_size
    page_rows = filtered_rows[start:start + page_size]

    st.dataframe(pd.DataFrame(page_rows), use_container_width=True, hide_index=True)

    render_export_buttons(filtered_rows, filename_prefix, key_prefix)
