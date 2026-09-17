"""
ml/generate_data.py
====================
Generates a large synthetic dataset of past student outcomes, used to
TRAIN the machine learning models in ml/model_training.ipynb (step 10).

WHY A SEPARATE DATASET, INSTEAD OF READING THE APP'S OWN DATABASE:
This is a brand-new system -- database/db_setup.py creates an EMPTY
database. There is no real historical data yet, and supervised ML models
(the classifier in Task 1, the regressor in Task 2) need many past
examples where the final outcome is already known, to learn the pattern
between early signals and that outcome. So this script manufactures a
large, realistic "history" (450 synthetic past students, each with a
already-known final result) purely to bootstrap model training. This is a
completely standard real-world approach called a "cold start" problem.

This also explains why some of the columns below (assignments_submitted,
backlog_count, consistency, improvement_rate) do not exist anywhere in
database/db_setup.py's schema: those columns are engineered specifically
as ML FEATURES, describing properties this project's own ML tasks asked
for, that the live transactional database was never asked to track
(e.g. assignment submission might belong to a separate LMS in a real
college). This script's output is a flat CSV table, unrelated to the
students/marks/attendance tables -- it is training data, not application
data.

OUTPUT:
    ml/data/synthetic_student_data.csv -- one row per synthetic student.

HOW TO RUN (from the project root):
    python -m ml.generate_data

======================================================================
HOW THE RANDOMNESS/NOISE IS BUILT (read this before reading the code)
======================================================================
Every synthetic student starts from ONE hidden number: `latent_ability`,
drawn from a normal (bell-curve) distribution. Think of it as an
unobservable mix of intelligence, discipline, and study habits -- in real
life, nobody can measure this directly, but it is the underlying reason a
disciplined student tends to have good attendance AND good marks AND few
backlogs, all at once (they share a common cause).

Every OBSERVABLE column (attendance, internal marks, external marks,
practical marks, ...) is then generated as:

    observable = intercept + slope * latent_ability + INDEPENDENT random noise

Because every observable shares the same `latent_ability` term, they end
up correlated with each other (a high-ability student tends to score
higher on all of them) -- but because each one ALSO adds its own
independent random noise (drawn fresh, separately, for every column and
every student), no single observable is a perfect predictor of any other.
A student can have strong attendance but an unlucky exam performance, or
solid internal marks but weak external marks. This is what the assignment
means by "genuine randomness" and "not fake 99% accuracy": even the model
that GENERATED this data cannot predict external marks perfectly from
internal marks, because a real, independent random term separates them.
The ML models trained later, in ml/model_training.ipynb, will therefore
top out at a realistic accuracy/R^2 -- not a suspicious 0.99+.

The final regression target, final_percentage, is deliberately weighted
mostly by external_pct (60%), which is NEVER saved to the output CSV and
NEVER used as a feature. This avoids "target leakage": if external marks
were both a feature and 60% of the answer, a model would look artificially
perfect without having learned anything real.
"""

import numpy as np
import pandas as pd

import config
from modules.grades import get_grade, is_pass
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# GENERATION CONSTANTS
# ---------------------------------------------------------------------------
# These constants control ONLY this synthetic-data experiment -- they are
# not used anywhere else in the application, which is why they live here
# rather than in config.py (config.py holds constants the LIVE APP needs;
# this file's constants only exist to make this one script's random-data
# recipe explicit and adjustable, instead of scattering raw numbers through
# the code below).

NUM_STUDENTS = 450  # comfortably over the assignment's 400+ requirement

# The hidden "ability" variable every observable score is derived from.
LATENT_ABILITY_MEAN = 62.0
LATENT_ABILITY_STD = 16.0
LATENT_ABILITY_MIN = 15.0
LATENT_ABILITY_MAX = 100.0

# Each tuple is (intercept, slope, noise_std) for the formula:
#   observable = intercept + slope * latent_ability + Normal(0, noise_std)
ATTENDANCE_PARAMS = (30.0, 0.65, 9.0)
INTERNAL_PARAMS = (15.0, 0.60, 10.0)
PRACTICAL_PARAMS = (10.0, 0.65, 11.0)
EXTERNAL_PARAMS = (5.0, 0.75, 13.0)  # hidden: used to build the target, never saved

# The "previous semester" is modelled as a SEPARATE noisy draw around the
# same latent_ability (representing that past performance is related to,
# but not identical to, current ability -- students improve or slip over
# time), rather than reusing latent_ability directly.
PREV_ABILITY_NOISE_STD = 8.0
PREV_PERCENTAGE_PARAMS = (5.0, 0.75, 12.0)

# Assignments submitted, out of this many total assignments in a semester.
# config.ASSIGNMENT_ENGAGEMENT_SCALE, not a local constant, because the
# live app (modules/ml_predictions.py) now computes a real value on this
# same 0-10 scale from actual assignment records -- both places must
# agree on the same scale for predictions to mean what this training data
# taught the model to expect.
MAX_ASSIGNMENTS = config.ASSIGNMENT_ENGAGEMENT_SCALE
ASSIGNMENTS_SLOPE = 0.09  # applied to latent_ability, roughly 0-9 before noise
ASSIGNMENTS_NOISE_STD = 1.3

# Maximum backlog (failed/carried-over) subjects we allow in this dataset.
MAX_BACKLOG_COUNT = 5

# How final_percentage (the regression target) is weighted from the three
# visible-in-real-life components -- mirrors a typical internal/external/
# practical split, and matches the shape modules/grades.py already expects.
FINAL_PERCENTAGE_WEIGHTS = {"internal": 0.20, "external": 0.60, "practical": 0.20}

# Semester range for synthetic records: starts at 2 (not 1), because every
# record needs a "previous semester" to have existed.
MIN_RECORD_SEMESTER = config.MIN_SEMESTER + 1
MAX_RECORD_SEMESTER = config.MAX_SEMESTER

OUTPUT_DIR = config.ML_DIR / "data"
OUTPUT_PATH = OUTPUT_DIR / "synthetic_student_data.csv"


def _clip(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Clamp an array of values into [minimum, maximum]. A percentage
    generated from a random formula could technically land below 0 or
    above 100 -- this keeps every generated number in a realistic range."""
    return np.clip(values, minimum, maximum)


def generate_dataset(num_students: int = NUM_STUDENTS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Generate the full synthetic student dataset.

    Args:
        num_students: How many synthetic student rows to generate.
        seed: Random seed, so re-running this function produces the exact
            same dataset every time (reproducibility -- see config.RANDOM_STATE).

    Returns:
        A pandas DataFrame with one row per synthetic student.
    """
    # np.random.default_rng() creates a dedicated random number generator
    # object tied to our own seed, rather than relying on numpy's single,
    # shared, hidden global random state (the older np.random.seed()
    # style). Using our own Generator object means nothing else running
    # in the same program (e.g. scikit-learn shuffling data elsewhere)
    # can accidentally change the sequence of random numbers we get here.
    rng = np.random.default_rng(seed)

    # ---- Step 1: the hidden latent_ability driving everything else ----
    latent_ability = _clip(
        rng.normal(LATENT_ABILITY_MEAN, LATENT_ABILITY_STD, size=num_students),
        LATENT_ABILITY_MIN,
        LATENT_ABILITY_MAX,
    )

    # ---- Step 2: observable scores, each = intercept + slope*ability + independent noise ----
    def observable(params: tuple[float, float, float]) -> np.ndarray:
        intercept, slope, noise_std = params
        noise = rng.normal(0, noise_std, size=num_students)
        return _clip(intercept + slope * latent_ability + noise, 0.0, 100.0)

    attendance_pct = _clip(observable(ATTENDANCE_PARAMS), 30.0, 100.0)
    internal_pct = observable(INTERNAL_PARAMS)
    practical_pct = observable(PRACTICAL_PARAMS)
    external_pct = observable(EXTERNAL_PARAMS)  # NOT saved to the output CSV

    # ---- Step 3: previous semester's performance (its own noisy draw) ----
    prev_ability = _clip(
        latent_ability + rng.normal(0, PREV_ABILITY_NOISE_STD, size=num_students),
        LATENT_ABILITY_MIN,
        LATENT_ABILITY_MAX,
    )
    prev_intercept, prev_slope, prev_noise_std = PREV_PERCENTAGE_PARAMS
    prev_percentage = _clip(
        prev_intercept + prev_slope * prev_ability + rng.normal(0, prev_noise_std, size=num_students),
        0.0,
        100.0,
    )
    # previous_sgpa reuses modules/grades.py's own get_grade() -- the exact
    # same percentage-to-grade-point mapping the live app uses -- so the
    # training data's notion of "SGPA" is never out of sync with the app's.
    previous_sgpa = np.array([get_grade(p)[1] for p in prev_percentage])

    # ---- Step 4: assignments submitted (out of MAX_ASSIGNMENTS) ----
    assignments_raw = ASSIGNMENTS_SLOPE * latent_ability + rng.normal(
        0, ASSIGNMENTS_NOISE_STD, size=num_students
    )
    assignments_submitted = np.round(_clip(assignments_raw, 0, MAX_ASSIGNMENTS)).astype(int)

    # ---- Step 5: backlog count -----------------------------------------
    # Weaker students (lower latent_ability) get a higher Poisson "rate"
    # (lambda) of backlog subjects. Poisson is the standard distribution
    # for "how many times does a somewhat-rare event happen" -- here, the
    # event is "failed a subject in an earlier semester".
    backlog_lambda = _clip((58.0 - latent_ability) / 18.0, 0.01, None)
    backlog_count = np.minimum(rng.poisson(backlog_lambda), MAX_BACKLOG_COUNT)

    # ---- Step 6: derived columns (computed FROM the noisy values above) ----
    final_percentage = np.round(
        FINAL_PERCENTAGE_WEIGHTS["internal"] * internal_pct
        + FINAL_PERCENTAGE_WEIGHTS["external"] * external_pct
        + FINAL_PERCENTAGE_WEIGHTS["practical"] * practical_pct,
        config.ROUND_DECIMALS,
    )

    component_stack = np.stack([internal_pct, external_pct, practical_pct], axis=1)
    average_marks = np.round(component_stack.mean(axis=1), config.ROUND_DECIMALS)
    consistency = np.round(component_stack.std(axis=1), config.ROUND_DECIMALS)
    improvement_rate = np.round(final_percentage - prev_percentage, config.ROUND_DECIMALS)

    # ---- Step 7: classification target, using the SAME pass/fail rule the app uses ----
    # at_risk = 1 means "this student is predicted/known to be at risk of
    # failing". Reusing modules.grades.is_pass() means the training
    # data's definition of "pass" can never drift out of sync with the
    # live app's definition (both read config.PASS_PERCENTAGE).
    at_risk = np.array([0 if is_pass(p) else 1 for p in final_percentage])

    student_id = [f"SYN{i:04d}" for i in range(1, num_students + 1)]
    semester = rng.integers(MIN_RECORD_SEMESTER, MAX_RECORD_SEMESTER + 1, size=num_students)

    dataset = pd.DataFrame({
        "student_id": student_id,
        "semester": semester,
        "attendance_pct": np.round(attendance_pct, config.ROUND_DECIMALS),
        "internal_pct": np.round(internal_pct, config.ROUND_DECIMALS),
        "practical_pct": np.round(practical_pct, config.ROUND_DECIMALS),
        "assignments_submitted": assignments_submitted,
        "previous_sgpa": np.round(previous_sgpa, config.ROUND_DECIMALS),
        "backlog_count": backlog_count,
        "average_marks": average_marks,
        "consistency": consistency,
        "improvement_rate": improvement_rate,
        "final_percentage": final_percentage,
        "at_risk": at_risk,
    })

    logger.info("Generated synthetic dataset with %d rows.", len(dataset))
    return dataset


def save_dataset(dataset: pd.DataFrame) -> None:
    """
    Write the generated dataset to ml/data/synthetic_student_data.csv.

    Args:
        dataset: The DataFrame returned by generate_dataset().
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved synthetic dataset to %s", OUTPUT_PATH)


if __name__ == "__main__":
    # This block only runs when the file is executed directly
    # (python -m ml.generate_data), not when it is imported -- e.g. later,
    # from ml/model_training.ipynb, if we ever want to regenerate data
    # from inside the notebook instead of reading the saved CSV.
    generated = generate_dataset()
    save_dataset(generated)
