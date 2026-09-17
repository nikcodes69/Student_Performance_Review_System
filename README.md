# Student Performance Review & Prediction System

An 8th-semester BCA major project: a role-based Streamlit application for
managing students, subjects, marks, and attendance, with an analytics
dashboard and three machine-learning-driven prediction features (at-risk
classification, final-marks regression, and student segmentation).

## Tech stack

Python 3.13 | Streamlit | SQLite (local) / Turso-libSQL (cloud, optional) |
Pandas | NumPy | Plotly | scikit-learn | joblib | bcrypt | ReportLab | pytest

## Features

| # | Feature | Where |
|---|---|---|
| 1 | Authentication (Admin/Teacher/Student roles, bcrypt, session timeout, dark/light theme) | `modules/auth.py` |
| 2 | Student self-registration + Admin-managed User Management | `modules/auth.py` |
| 3 | Student management (CRUD, validation, soft delete) | `modules/students.py` |
| 4 | Subject & curriculum configuration | `modules/subjects.py` |
| 5 | Marks entry with audit logging | `modules/marks.py` |
| 6 | Attendance tracking (submitted-cannot-exceed-held enforced live in the UI) | `modules/attendance.py` |
| 7 | Assignment tracking (real records, not a manually-typed ML input) | `modules/assignments.py` |
| 8 | Grade engine (percentage, grade, SGPA, CGPA, pass/fail) | `modules/grades.py` |
| 9 | Analytics dashboard (averages, trends, distribution, correlation, difficulty index) | `modules/analytics.py` |
| 10 | ML: at-risk classification | `modules/ml_predictions.py` |
| 11 | ML: final marks regression | `modules/ml_predictions.py` |
| 12 | ML: student segmentation | `modules/ml_predictions.py` |
| 13 | Model comparison page | `modules/ml_predictions.py` |
| 14 | PDF report card export (incl. ML predictions) | `utils/pdf_generator.py` |
| 15 | Audit log viewer (Admin only) | `modules/audit.py` |
| 16 | Student-facing "My Performance" portal | `modules/student_portal.py` |
| 17 | Bulk CSV/Excel import with a validation preview before committing | `utils/bulk_import.py` |
| 18 | CSV/Excel export, search, and pagination on every data table | `utils/table_view.py` |
| 19 | Visual "Dashboard" page: KPI cards, at-risk count, attendance shortage alerts, spotlight cards, charts | `modules/analytics.py` |
| 20 | Class rank and percentile per student | `modules/analytics.py` |
| 21 | Student-vs-class comparative analytics | `modules/analytics.py` |
| 22 | Change password, with a forced change on a freshly created or admin-reset account | `modules/auth.py` |

### Beyond the original 13-module spec

This system grew past its original scope during development, in response
to real usage:
- **Cloud database (Turso/libSQL)**: `database/db_setup.py`'s
  `get_connection()` uses Turso when `TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN`
  are configured in Streamlit secrets, falling back to local SQLite
  otherwise -- local development and all 90 tests stay fully offline.
- **Dark mode**: `.streamlit/config.toml` defines both `[theme.light]`
  and `[theme.dark]` -- Streamlit's own built-in Settings menu lets each
  user switch, no custom code involved.
- **Assignments are a real, tracked feature**, not a number typed in by
  hand every time a prediction is requested (see `modules/assignments.py`
  and `modules/ml_predictions.py`'s `_compute_assignment_engagement()`).
- **Bulk import, export, search/pagination, rank/percentile, attendance
  shortage alerts, forced password change, and comparative analytics**:
  see `utils/bulk_import.py`, `utils/table_view.py`, and the rank/
  shortage/comparison functions in `modules/analytics.py`. The at-risk
  KPI on the Dashboard runs the real trained classifier for every active
  student (not a cheaper proxy), cached 5 minutes given the added
  Turso round-trips -- see `ml_predictions.get_at_risk_count()`.
- **Google Sign-In**: prepared but not yet wired up -- requires OAuth
  credentials from Google Cloud Console, which only the project owner can
  create.

## Project structure

```
student_performance_system/
├── app.py                     # Streamlit entry point / page router
├── config.py                  # every constant used across the project
├── requirements.txt
├── database/
│   ├── db_setup.py            # schema: tables, constraints, indexes (run once)
│   ├── db_manager.py          # runtime query layer (fetch_one/fetch_all/execute_write/execute_transaction)
│   └── student_data.db        # created by db_setup.py (gitignored)
├── modules/
│   ├── auth.py                # authentication, RBAC, self-registration, User Management, Streamlit session
│   ├── students.py            # student CRUD
│   ├── subjects.py            # subject/curriculum CRUD
│   ├── marks.py                # marks entry, correction, shared SGPA helper
│   ├── attendance.py          # attendance entry, correction (live UI-enforced held >= attended)
│   ├── assignments.py         # assignment tracking (feeds the ML engagement feature)
│   ├── grades.py              # grade engine (pure functions, no DB/UI)
│   ├── analytics.py           # dashboard queries + Plotly charts
│   ├── ml_predictions.py      # loads trained models, predicts, ML pages
│   ├── student_portal.py      # Student-only "My Performance" page
│   └── audit.py               # audit trail read/write + viewer page
├── ml/
│   ├── generate_data.py       # synthetic training data generator
│   ├── model_training.ipynb   # the only place models are trained
│   ├── data/                  # generated training CSV (tracked in git -- deterministic, small)
│   ├── models/                # trained .pkl + metadata JSON (tracked in git -- small, deterministic, needed for the deployed app to serve predictions)
│   └── results/               # model comparison CSVs (tracked -- used in the project report)
├── utils/
│   ├── validators.py          # validation layer, checked before every DB write
│   ├── exceptions.py          # domain-specific exception classes
│   ├── logger.py              # logging configuration (writes to logs/app.log)
│   ├── pdf_generator.py       # report card PDF builder + trigger page
│   ├── table_view.py          # search/paginate/export -- shared by every data table
│   └── bulk_import.py         # CSV/Excel bulk import with a validation preview
├── .streamlit/
│   ├── config.toml            # light/dark theme (tracked)
│   └── secrets.toml           # Turso credentials (gitignored -- never commit this)
├── tests/
│   ├── test_validators.py
│   ├── test_grades.py
│   ├── test_database.py
│   ├── test_auth.py
│   ├── test_analytics.py
│   ├── test_table_view.py
│   └── test_bulk_import.py
└── logs/
    └── app.log                # created at runtime (gitignored)
```

## Setup

Requires Python 3.11+ (developed and tested on 3.13).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create the database (tables, constraints, indexes)
python -m database.db_setup

# 4. Create the bootstrap admin account
python -m modules.auth

# 5. Generate synthetic ML training data (450 rows)
python -m ml.generate_data

# 6. Train the ML models (opens Jupyter -- run every cell top to bottom)
jupyter notebook ml/model_training.ipynb

# 7. Run the app
streamlit run app.py
```

**Why `python -m package.module` instead of `python path/to/file.py`**:
several files (`database/db_setup.py`, `modules/auth.py`, `ml/generate_data.py`)
`import config`, which lives at the project root. Running a file directly
only adds *that file's own folder* to Python's import search path; `-m`
adds the *current working directory* instead, so `config.py` is visible.
Run every command above from the project root.

**Windows-specific note**: if `streamlit run app.py` fails with
`"An Application Control policy has blocked this file"`, use
`python -m streamlit run app.py` instead — this runs Streamlit through the
already-trusted `python.exe` rather than the separate, unsigned
`streamlit.exe` launcher stub that some locked-down machines block.

### Default admin login

- **Username**: `admin`
- **Password**: `Admin@12345`

This account is created automatically by step 4 above, only if no admin
account exists yet (see `config.DEFAULT_ADMIN_USERNAME`/`DEFAULT_ADMIN_PASSWORD`).
**This password is visible in the source code** — it exists purely to
provide one working login on a fresh install. A real deployment would
need a forced password-change-on-first-login flow, which is out of scope
here; change this password immediately after first login.

## Running tests

```bash
python -m pytest -v
```

128 tests across seven files:
- `tests/test_validators.py` (58 tests) — every validation rule, including boundary values.
- `tests/test_grades.py` (20 tests) — percentage/grade/SGPA/CGPA calculations, including deliberately adversarial edge cases like division-by-zero guards and off-by-one grade boundaries.
- `tests/test_database.py` (13 tests) — CHECK/UNIQUE/FOREIGN KEY constraints and the `PRAGMA foreign_keys` setting itself, run against a throwaway temporary database (never the real `student_data.db`, and never the real Turso database either) created fresh for every test.
- `tests/test_auth.py` (14 tests) — account creation defaults, self-registration, login, and the change-password/admin-reset-password flow.
- `tests/test_analytics.py` (7 tests) — class rank/percentile (including tie handling), attendance shortage detection, and the student-vs-class comparison, each verified against hand-calculated expected values.
- `tests/test_table_view.py` (4 tests) — CSV/Excel export round-trips, including unicode and comma-containing values.
- `tests/test_bulk_import.py` (12 tests) — file parsing and per-row import validation, including duplicate detection both within a file and against existing records.

## Database design

7 tables (`users`, `students`, `subjects`, `marks`, `attendance`,
`semesters`, `audit_log`), all normalised to **3NF**: atomic columns
(1NF), every column depends on the whole primary key including composite
keys like `semesters(roll_no, semester)` (2NF), and no column depends on
another non-key column instead of the key (3NF) — see the extensive
comments in `database/db_setup.py` for table-by-table reasoning,
including why some standard-looking normalisations were deliberately
NOT done (e.g. a mark's percentage is never stored, only computed, to
avoid a value that's redundantly derivable from two other tables).

Every write to `students`, `subjects`, `marks`, or `attendance` is
recorded in `audit_log` (old value, new value, who, when), atomically
with the change itself via `database.db_manager.execute_transaction()` —
the data change and its audit entry can never happen apart. Academic
records are never hard-deleted; `students`/`subjects` use a soft-delete
`is_active` flag, while `marks`/`attendance` corrections go through an
audited `UPDATE` instead (see `database/db_setup.py` and
`modules/audit.py` for the full reasoning).

## Machine learning

Three tasks, five algorithms, all trained exclusively in
`ml/model_training.ipynb` and saved via `joblib` + a metadata JSON
(version, training date, features, hyperparameters, metrics) to
`ml/models/`. The Streamlit app (`modules/ml_predictions.py`) only ever
loads these files and calls `.predict()` — it never trains at runtime.

**Task 1 — At-risk classification** (Logistic Regression, Decision Tree,
Random Forest, plus a `DummyClassifier` baseline). Stratified 80/20
split, 5-fold cross-validation, `GridSearchCV` on the tree-based models,
`class_weight="balanced"` for the ~79/21 class imbalance, recall
prioritised as the primary metric (a missed at-risk student is worse than
a false alarm). **Deployed: Logistic Regression** (recall 0.737,
cross-validated 0.751 ± 0.112) — not the Random Forest the brief
anticipated; the real results said otherwise, reported honestly rather
than forced (see the notebook's "Deployment justification" cell for the
full, data-driven reasoning).

**Task 2 — Final marks regression** (Linear Regression, unscaled features
for coefficient interpretability, quartile-binned stratified split since
the target is continuous). R² = 0.61, MAE = 7.1 percentage points — capped
well below 1.0 by design, since 60% of the true target is driven by a
hidden exam signal deliberately excluded from every feature to avoid
target leakage (see `ml/generate_data.py`).

**Task 3 — Student segmentation** (K-Means, k=4, chosen via the elbow
method and silhouette score together with practical segment usefulness —
silhouette technically peaks at k=2, but that's too coarse to be useful;
see the notebook for the full reasoning). PCA to 2D for visualisation
(captures 67% of the original 4D variance). Clusters named programmatically
by inspecting real centroids: **Consistent High Performers**, **Needs
Intervention**, **Improving Students**, **Inconsistent Performers**.

Full metrics, comparison tables, and plots are in the notebook itself and
in `ml/results/*.csv`.

## Known limitations (documented deliberately, not discovered by accident)

- **No teacher-to-subject assignment table**: any Teacher account can
  currently enter marks/attendance for any subject.
- **The default admin password is hardcoded and visible in source** (see
  "Default admin login" above). This is now mitigated going FORWARD --
  `modules/auth.py`'s `create_user()` defaults every Admin-created (and
  the bootstrap) account to `must_change_password=1`, which forces a
  password change on first login before anything else is reachable (see
  `app.py`'s `render_authenticated_view()`). It does **not** retroactively
  force a change on an admin account that already existed before this
  feature was added and had never changed its password -- a schema
  migration can add the flag, but it cannot know whether a given
  already-existing account's password was ever actually changed.
  Anyone still using the original default password should use the new
  "Change Password" page immediately.
- **RandomForestClassifier and Windows Smart App Control**: on the
  development machine, a Windows security policy blocked one unrelated,
  unused scikit-learn submodule (`HistGradientBoosting*`) in a way that
  also broke importing `RandomForestClassifier` from the same package.
  Worked around with a harmless, fully-reversible import stub — see the
  "environment note" markdown cell near the top of
  `ml/model_training.ipynb` for the full explanation. This does not
  affect the deployed models (none of the three currently use Random
  Forest) but is documented in case a future retrain does.
