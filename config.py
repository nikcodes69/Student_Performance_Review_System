"""
config.py
=========
Single source of truth for every constant used across this project.

WHY THIS FILE EXISTS:
Hard-coding values like "0.75" or "8" directly inside business logic
(this is called a "magic number") makes code hard to read and hard to
change -- six months later nobody remembers what 0.75 meant. By keeping
every tunable value here, with a name and a comment, any other file can
write `config.PASS_PERCENTAGE` instead of the bare number 40, and we only
ever have ONE place to update if a rule changes (e.g. the college revises
its pass percentage).

This file contains ONLY constants (values that do not change while the
program runs). It contains no functions and no logic, so it is safe to
import from anywhere (database code, UI code, ML code) without causing
circular imports.
"""

from pathlib import Path
import logging

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------------
# Path(__file__) is the path to THIS file (config.py). .resolve() turns it
# into an absolute path (e.g. E:\8thSem\student_performance_system\config.py)
# so it works no matter which folder we run the app from. .parent then gives
# us the project's root folder. Every other path below is built relative to
# this, so the project is portable -- it does not depend on being run from
# a specific drive or folder.
BASE_DIR = Path(__file__).resolve().parent

DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "student_data.db"

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

ML_DIR = BASE_DIR / "ml"
ML_MODELS_DIR = ML_DIR / "models"

# ---------------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# ---------------------------------------------------------------------------
# Used by utils/logger.py to configure the built-in `logging` module instead
# of using print(). Logging is better than print() because it can be turned
# on/off, tagged with severity levels (INFO, WARNING, ERROR), timestamped,
# and written to a file automatically -- print() output is lost the moment
# the terminal closes.
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# RotatingFileHandler settings: once app.log reaches LOG_MAX_BYTES, it is
# renamed to app.log.1 and a fresh app.log is started. LOG_BACKUP_COUNT
# controls how many old log files are kept before the oldest is deleted.
# This stops the log file from growing forever during a long viva demo.
LOG_MAX_BYTES = 1_000_000  # 1 MB
LOG_BACKUP_COUNT = 3

# ---------------------------------------------------------------------------
# 3. SECURITY / AUTHENTICATION
# ---------------------------------------------------------------------------
# bcrypt "rounds" (also called cost factor) controls how many times the
# hashing algorithm repeats internally. Higher = slower to compute = harder
# for an attacker to brute-force, but also slower for real logins. 12 is
# bcrypt's widely-recommended default balance between security and speed.
BCRYPT_ROUNDS = 12

# If a logged-in user is inactive for this many minutes, the session should
# be treated as expired and they must log in again. Protects an unattended
# lab PC from being used by someone else under a logged-in session.
SESSION_TIMEOUT_MINUTES = 30

# The three roles this system supports. Defined as constants (not raw
# strings scattered through the code) so a typo like "admn" becomes an
# obvious NameError instead of a silent bug.
ROLE_ADMIN = "admin"
ROLE_TEACHER = "teacher"
ROLE_STUDENT = "student"
VALID_ROLES = (ROLE_ADMIN, ROLE_TEACHER, ROLE_STUDENT)

# On a brand-new database there are no users yet, which is a chicken-and-
# egg problem: an Admin is needed to create every other account, but
# nobody can log in to create that first Admin. modules/auth.py solves
# this by creating exactly one bootstrap account, with these fixed
# credentials, ONLY if no admin account exists yet.
#
# SECURITY NOTE (also flagged again where it's used): this password is
# intentionally simple and is, by definition, published in this source
# file -- it exists purely to get one working login on a fresh install.
# A real deployment would need a "force password change on first login"
# flow, which is out of scope here; this is a known, deliberate
# limitation worth stating plainly in your report rather than glossing
# over.
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "Admin@12345"

# ---------------------------------------------------------------------------
# 4. FIELD LENGTH / FORMAT RULES (used by utils/validators.py)
# ---------------------------------------------------------------------------
USERNAME_MIN_LENGTH = 4
USERNAME_MAX_LENGTH = 30
PASSWORD_MIN_LENGTH = 8  # enforced at signup/creation time, before hashing

NAME_MAX_LENGTH = 100
ROLL_NO_MAX_LENGTH = 20

# A 10-digit local mobile number. Kept as a length constant rather than
# hard-coded "10" inside validators.py.
PHONE_LENGTH = 10

# Regex patterns are also "constants" -- they do not change at runtime --
# so they live here rather than being retyped inside validators.py.
EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
PHONE_REGEX = r"^[9][6-8]\d{8}$"  # Nepali mobile numbers start with 96-98

# ---------------------------------------------------------------------------
# 5. ACADEMIC STRUCTURE RULES
# ---------------------------------------------------------------------------
MIN_SEMESTER = 1
MAX_SEMESTER = 8  # BCA is an 8-semester program

MIN_CREDITS = 1
MAX_CREDITS = 6

# Earliest admission year we consider valid. There is no fixed MAX here --
# the maximum valid admission year is "the current year", which we compute
# dynamically with datetime.now().year inside validators.py rather than
# hard-coding a year that would go stale (e.g. writing 2026 here would be
# wrong the moment 2027 arrives).
ADMISSION_YEAR_MIN = 2000

# ---------------------------------------------------------------------------
# 6. ATTENDANCE RULES
# ---------------------------------------------------------------------------
# Below this attendance percentage, a student is flagged as having a
# "shortage" -- a common academic rule (many universities use 75%).
ATTENDANCE_SHORTAGE_THRESHOLD = 75.0

# ---------------------------------------------------------------------------
# 7. GRADING SCALE (used by modules/grades.py)
# ---------------------------------------------------------------------------
# Each tuple is (minimum_percentage, grade_letter, grade_point).
# MUST stay sorted from highest threshold to lowest -- grades.py will walk
# down this list and return the first grade whose threshold the student's
# percentage meets or exceeds.
#
# This is a standard 10-point absolute grading scale (a common pattern used
# by many autonomous colleges/universities). It is intentionally kept as
# plain data here so it can be swapped for your institution's exact
# official grading table by editing ONLY this list -- no other file needs
# to change.
GRADE_SCALE = [
    (90, "O", 10.0),   # Outstanding
    (80, "A+", 9.0),
    (70, "A", 8.0),
    (60, "B+", 7.0),
    (50, "B", 6.0),
    (40, "C", 5.0),
    (0, "F", 0.0),     # Fail
]

# Minimum overall percentage required to pass a subject. Kept separate from
# GRADE_SCALE because "pass/fail" and "letter grade" are technically two
# different rules, even though they use the same number by coincidence here.
PASS_PERCENTAGE = 40.0

# Decimal places used when rounding percentage, SGPA and CGPA for display
# and storage, so the whole app rounds consistently instead of each file
# picking its own precision.
ROUND_DECIMALS = 2

# The highest grade point available on the scale above (10.0). Computed
# from GRADE_SCALE instead of retyped as a literal "10.0" -- if the grading
# scale is ever edited, this stays correct automatically instead of quietly
# going out of sync. Used as the upper bound for the sgpa/cgpa CHECK
# constraints in database/db_setup.py.
MAX_GRADE_POINT = max(grade_point for _, _, grade_point in GRADE_SCALE)

# ---------------------------------------------------------------------------
# 8. MARKS & RECORD-KEEPING RULES (used by database/db_setup.py, validators.py)
# ---------------------------------------------------------------------------
# A generous, static sanity ceiling for any single mark component (internal,
# external, or practical). This is NOT the real per-subject maximum -- that
# comes from subjects.max_internal / max_external / max_practical, which can
# differ per subject and is validated dynamically in utils/validators.py
# against the database. This constant only stops obviously absurd values
# (e.g. entering 9999) at the database's CHECK-constraint level, as a second
# line of defence below the validation layer.
MAX_MARK_CEILING = 100

# The kinds of exam attempt a marks row can represent. 'backlog' and
# 'improvement' let a student have more than one marks row for the same
# subject and semester (e.g. a repeat attempt) without violating the
# UNIQUE(roll_no, subject_code, semester, exam_type) constraint on marks.
EXAM_TYPE_REGULAR = "regular"
EXAM_TYPE_BACKLOG = "backlog"
EXAM_TYPE_IMPROVEMENT = "improvement"
EXAM_TYPES = (EXAM_TYPE_REGULAR, EXAM_TYPE_BACKLOG, EXAM_TYPE_IMPROVEMENT)

# Possible values of semesters.result_status. 'pending' covers a semester
# whose SGPA has not been calculated yet (e.g. marks are still being
# entered).
RESULT_PASS = "pass"
RESULT_FAIL = "fail"
RESULT_PENDING = "pending"
RESULT_STATUSES = (RESULT_PASS, RESULT_FAIL, RESULT_PENDING)

# The three kinds of change audit_log can record. Deliberately does NOT
# include a literal 'DELETE' -- this system never hard-deletes academic
# records, so the true action taken when a record is deactivated is
# 'SOFT_DELETE' (an UPDATE that flips is_active to 0), not a real SQL
# DELETE.
AUDIT_INSERT = "INSERT"
AUDIT_UPDATE = "UPDATE"
AUDIT_SOFT_DELETE = "SOFT_DELETE"
AUDIT_ACTIONS = (AUDIT_INSERT, AUDIT_UPDATE, AUDIT_SOFT_DELETE)

# The tables modules/audit.py is allowed to record changes for. Listing
# them explicitly (rather than accepting any string) catches a typo like
# "student" instead of "students" at validation time, before it becomes a
# confusing, silently-wrong row in audit_log.
AUDITED_TABLES = ("students", "subjects", "marks", "attendance", "semesters", "users")

# ---------------------------------------------------------------------------
# 9. MACHINE LEARNING CONFIGURATION (used in ml/ and modules/ml_predictions.py)
# ---------------------------------------------------------------------------
# A fixed random seed makes train/test splits and model training
# reproducible -- running the notebook twice gives the same result, which
# matters when you need to explain a specific number in your report/viva.
RANDOM_STATE = 42

# 20% of data held out for testing, 80% used for training (a standard split
# ratio, large enough to train on, large enough to test reliably on).
TEST_SIZE = 0.2

# Number of folds for k-fold cross-validation.
CV_FOLDS = 5
