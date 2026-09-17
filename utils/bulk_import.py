"""
utils/bulk_import.py
=====================
Bulk CSV/Excel import: upload a spreadsheet of records, see a per-row
VALIDATION PREVIEW (which rows pass, which don't and exactly why) BEFORE
anything is written to the database, then commit only the valid rows.

WHY VALIDATE-THEN-PREVIEW-THEN-COMMIT, NOT "VALIDATE AND INSERT
IMMEDIATELY": a spreadsheet handed over by an admin office is exactly
the kind of input most likely to have a few bad rows mixed in with many
good ones (a typo'd email, a semester typed as "1st" instead of 1, a
roll number that's already in the system). Showing every row's status
FIRST, with the precise reason for each failure, lets the person
importing catch and fix a spreadsheet mistake before anything is
written -- rather than discovering row 47 of 60 failed only after 46
rows were already committed, leaving a half-imported, inconsistent
batch. This mirrors the "preview before you commit" pattern real bulk-
import tools (a payroll system, a CRM's contact importer) use, for
exactly this reason.

THIS FILE IS GENERIC, NOT TIED TO ANY ONE RECORD TYPE:
render_bulk_import() takes the required columns, a per-row validate
function, and a per-row commit function as arguments -- see
modules/students.py's render_students_page() and modules/subjects.py's
render_subjects_page() for the two places that call it, each supplying
their OWN validation/commit logic (which itself reuses the exact same
utils/validators.py functions and create_*()/DuplicateRecordError checks
the manual "Add" form on each page already uses -- a row from a
spreadsheet is held to identical rules as one typed in by hand). Adding
bulk import for a further record type means writing that type's own
small validate/commit functions, not copying this file.
"""

from typing import Callable

import pandas as pd
import streamlit as st


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a DataFrame of STRING values.

    WHY EVERYTHING IS READ AS TEXT, NOT LEFT AS WHATEVER PANDAS WOULD
    OTHERWISE INFER: pandas' automatic type inference is a liability
    here, not a convenience -- a "semester" column containing only whole
    numbers would normally be inferred as int64, but a single blank cell
    anywhere in that column flips the WHOLE column to float64 (a missing
    value forces it), silently turning "1" into "1.0" for every row, not
    just the blank one. Reading everything as text and letting this
    project's OWN utils/validators.py functions do the real parsing (the
    exact same functions every manual form already goes through, via
    each caller's validate_row function) avoids a second, parallel set
    of import-only parsing rules that could quietly drift out of sync
    with the manual-entry rules.

    Args:
        uploaded_file: A Streamlit UploadedFile, from st.file_uploader.

    Returns:
        A DataFrame with every cell as a string (empty string for a
        blank cell, never NaN), and column headers stripped of
        surrounding whitespace (spreadsheets are frequently pasted with
        stray leading/trailing spaces in header cells).

    Raises:
        ValueError: if the file extension is not .csv, .xlsx, or .xls.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, dtype=str)
        df = df.fillna("")  # .xlsx blanks come back as NaN, not "" like keep_default_na=False gives CSV
    else:
        raise ValueError("Only .csv, .xlsx, and .xls files are supported.")

    df.columns = [str(column).strip() for column in df.columns]
    return df


def _row_key(row: dict, key_columns: tuple[str, ...]) -> str:
    """Build one row's composite key string, e.g. ("roll_no",
    "subject_code", "semester") -> "S1:SUB1:1" -- used both to detect
    in-file duplicates and to label a row in error messages. A tuple of
    one column (most callers) still works fine -- it just produces a
    plain, unjoined value."""
    return ":".join(str(row.get(column, "")) for column in key_columns)


def render_bulk_import(
    key_prefix: str,
    required_columns: tuple[str, ...],
    key_columns: tuple[str, ...],
    validate_row: Callable[[dict], None],
    commit_row: Callable[[dict], None],
) -> None:
    """
    Streamlit component: upload a file, preview per-row validation
    results, then commit only the rows that passed.

    Args:
        key_prefix: Unique per call site (widget keys).
        required_columns: Column names the uploaded file MUST contain
            (checked once, up front, against the file's own header row).
        key_columns: The column(s) that together must be unique WITHIN
            the uploaded file itself -- e.g. ("roll_no",) for a students
            import, or ("roll_no", "subject_code", "semester") for an
            attendance import, where no single column alone identifies
            one logical record. Checked here, generically, before
            validate_row runs on each row, since a single row in
            isolation cannot tell that another row later in the SAME file
            repeats its own key. validate_row is still responsible for
            checking uniqueness against the EXISTING database (a row can
            be unique within the file but still collide with a record
            already on file).
        validate_row: Called once per row (a dict of column name ->
            string cell value). Should raise ValidationError,
            DuplicateRecordError, or RecordNotFoundError if the row is
            invalid -- the exact same exceptions this project's normal
            single-record forms already raise, reusing their messages
            verbatim in the preview. Returning normally means the row is
            valid.
        commit_row: Called once per VALID row, only after the user
            clicks "Commit". Whatever this raises (e.g. a race condition
            between preview and commit -- someone else just took that
            roll number) is caught and reported per-row, the same way
            modules/attendance.py's bulk entry form already handles
            partial failures within one batch.
    """
    from utils.exceptions import AppError  # lazy import: avoids this generic utility depending on app-specific exceptions at module load time

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key=f"{key_prefix}_uploader",
    )
    if uploaded_file is None:
        st.caption(f"Required columns: {', '.join(required_columns)}")
        return

    try:
        df = parse_uploaded_file(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The uploaded file is missing required column(s): {', '.join(missing_columns)}")
        return

    rows = df.to_dict("records")
    if not rows:
        st.warning("The uploaded file has no data rows.")
        return

    # In-file duplicate check on key_columns -- see this function's
    # docstring for why this is handled here, generically, rather than
    # inside each caller's own validate_row.
    key_counts: dict[str, int] = {}
    for row in rows:
        key = _row_key(row, key_columns)
        key_counts[key] = key_counts.get(key, 0) + 1

    preview_rows = []
    valid_rows = []
    for row in rows:
        key = _row_key(row, key_columns)
        if key_counts.get(key, 0) > 1:
            status = f"Invalid: duplicate ({', '.join(key_columns)}) = ({key}) within this file"
        else:
            try:
                validate_row(row)
                status = "Valid"
            except AppError as error:
                status = f"Invalid: {error}"

        preview_rows.append({**row, "Import Status": status})
        if status == "Valid":
            valid_rows.append(row)

    valid_count = len(valid_rows)
    invalid_count = len(rows) - valid_count

    st.subheader("Preview")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Total Rows", len(rows))
    summary_cols[1].metric("Valid", valid_count)
    summary_cols[2].metric("Invalid", invalid_count)

    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    if valid_count == 0:
        st.warning("No valid rows to import -- fix the issues above and re-upload.")
        return

    if st.button(f"Commit {valid_count} Valid Row(s)", type="primary", key=f"{key_prefix}_commit"):
        committed_count = 0
        commit_errors = []
        for row in valid_rows:
            try:
                commit_row(row)
                committed_count += 1
            except AppError as error:
                # A row that passed the PREVIEW check can still fail at
                # commit time (e.g. another user just created the same
                # roll_no in the seconds since the preview ran) -- caught
                # per-row so one such collision does not abort the rest
                # of an otherwise-good batch.
                commit_errors.append(f"{_row_key(row, key_columns)}: {error}")

        if committed_count:
            st.success(f"Imported {committed_count} row(s) successfully.")
        for error_message in commit_errors:
            st.error(error_message)
        if committed_count:
            st.rerun()
