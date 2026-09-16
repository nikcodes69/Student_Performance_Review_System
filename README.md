# Student Performance Review & Prediction System

An 8th-semester BCA major project: a role-based Streamlit application for
managing students, subjects, marks, and attendance, with an analytics
dashboard and three machine-learning-driven prediction features (at-risk
classification, final-marks regression, and student segmentation).

## Tech stack

Python 3.13 | Streamlit | SQLite | Pandas | NumPy | Plotly | scikit-learn |
joblib | bcrypt | ReportLab | pytest

## Features

| # | Feature | Where |
|---|---|---|
| 1 | Authentication (Admin/Teacher/Student roles, bcrypt, session timeout) | `modules/auth.py` |
| 2 | Student management (CRUD, validation, soft delete) | `modules/students.py` |
| 3 | Subject & curriculum configuration | `modules/subjects.py` |
| 4 | Marks entry with audit logging | `modules/marks.py` |
| 5 | Attendance tracking with percentage calculation | `modules/attendance.py` |
| 6 | Grade engine (percentage, grade, SGPA, CGPA, pass/fail) | `modules/grades.py` |
| 7 | Analytics dashboard (averages, trends, distribution, correlation, difficulty index) | `modules/analytics.py` |
| 8 | ML: at-risk classification | `modules/ml_predictions.py` |
| 9 | ML: final marks regression | `modules/ml_predictions.py` |
| 10 | ML: student segmentation | `modules/ml_predictions.py` |
| 11 | Model comparison page | `modules/ml_predictions.py` |
| 12 | PDF report card export (incl. ML predictions) | `utils/pdf_generator.py` |
| 13 | Audit log viewer (Admin only) | `modules/audit.py` |

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
│   ├── auth.py                # authentication, RBAC, Streamlit session
│   ├── students.py            # student CRUD
│   ├── subjects.py            # subject/curriculum CRUD
│   ├── marks.py                # marks entry, correction
│   ├── attendance.py          # attendance entry, correction
│   ├── grades.py              # grade engine (pure functions, no DB/UI)
│   ├── analytics.py           # dashboard queries + Plotly charts
│   ├── ml_predictions.py      # loads trained models, predicts, ML pages
│   └── audit.py               # audit trail read/write + viewer page
├── ml/
│   ├── generate_data.py       # synthetic training data generator
│   ├── model_training.ipynb   # the only place models are trained
│   ├── data/                  # generated training CSV (tracked in git -- deterministic, small)
│   ├── models/                # trained .pkl + metadata JSON (gitignored -- regenerate from the notebook)
│   └── results/               # model comparison CSVs (tracked -- used in the project report)
├── utils/
│   ├── validators.py          # validation layer, checked before every DB write
│   ├── exceptions.py          # domain-specific exception classes
│   ├── logger.py              # logging configuration (writes to logs/app.log)
│   └── pdf_generator.py       # report card PDF builder + trigger page
├── tests/
│   ├── test_validators.py
│   ├── test_grades.py
│   └── test_database.py
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

85 tests across three files:
- `tests/test_validators.py` (54 tests) — every validation rule, including boundary values.
- `tests/test_grades.py` (20 tests) — percentage/grade/SGPA/CGPA calculations, including deliberately adversarial edge cases like division-by-zero guards and off-by-one grade boundaries.
- `tests/test_database.py` (11 tests) — CHECK/UNIQUE/FOREIGN KEY constraints and the `PRAGMA foreign_keys` setting itself, run against a throwaway temporary database (never the real `student_data.db`) created fresh for every test.

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

- **`assignments_submitted`** is a feature the ML models need but the
  live database schema doesn't track anywhere (no assignments table) —
  the prediction pages ask the user to enter it manually, with an
  on-screen explanation of why.
- **No teacher-to-subject assignment table**: any Teacher account can
  currently enter marks/attendance for any subject.
- **The default admin password is hardcoded and visible in source** (see
  "Default admin login" above) — acceptable for a bootstrap-only account
  on a fresh install, not for a production deployment.
- **RandomForestClassifier and Windows Smart App Control**: on the
  development machine, a Windows security policy blocked one unrelated,
  unused scikit-learn submodule (`HistGradientBoosting*`) in a way that
  also broke importing `RandomForestClassifier` from the same package.
  Worked around with a harmless, fully-reversible import stub — see the
  "environment note" markdown cell near the top of
  `ml/model_training.ipynb` for the full explanation. This does not
  affect the deployed models (none of the three currently use Random
  Forest) but is documented in case a future retrain does.
