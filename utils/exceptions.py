"""
utils/exceptions.py
====================
Custom exception classes for domain-specific errors.

WHY CUSTOM EXCEPTIONS?
Python already has generic exceptions like ValueError and Exception. We
could raise those everywhere, but then code that CATCHES errors cannot
tell the difference between "this failed because the input was invalid"
and "this failed because the database connection dropped" -- both would
just be a ValueError or an Exception.

By defining our own exception classes, one per category of problem, the
rest of the app can catch exactly the situation it cares about:

    try:
        students.create_student(data)
    except ValidationError as e:
        st.error(f"Please fix this input: {e}")
    except DuplicateRecordError as e:
        st.error(f"That roll number is already taken: {e}")

This is plain, single-level inheritance (every custom exception below
inherits directly from AppError, which inherits from Python's built-in
Exception) -- not the "advanced OOP" you asked to avoid. Subclassing
Exception is simply how Python expects custom errors to be defined.
"""


class AppError(Exception):
    """
    Base class for every custom exception in this application.

    Never raised directly -- it exists so calling code can, if it wants,
    catch "any error this app defines" with a single `except AppError:`
    instead of listing every specific subclass.
    """
    pass


class ValidationError(AppError):
    """
    Raised by utils/validators.py when input data fails a business rule
    BEFORE it reaches the database (e.g. marks entered above the subject's
    maximum, an email address without an '@', a semester outside 1-8).
    """
    pass


class DuplicateRecordError(AppError):
    """
    Raised when an operation would violate a UNIQUE constraint, e.g.
    creating a student with a roll_no that already exists, or a user with
    a username that is already taken.
    """
    pass


class RecordNotFoundError(AppError):
    """
    Raised when a lookup finds no matching row, e.g. searching for a
    roll_no that does not exist in the students table.
    """
    pass


class AuthenticationError(AppError):
    """
    Raised when login fails: unknown username, wrong password, or an
    account whose is_active flag is False.
    """
    pass


class AuthorizationError(AppError):
    """
    Raised when a logged-in user's role does not permit the action they
    are attempting, e.g. a Student trying to open the Audit Log page,
    which is Admin-only.
    """
    pass


class DatabaseError(AppError):
    """
    Raised when a database operation fails for a reason outside the
    caller's control, e.g. the .db file is locked or a query fails to
    execute even though the input was valid.
    """
    pass


class ModelNotFoundError(AppError):
    """
    Raised by modules/ml_predictions.py when a required trained model
    (.pkl file) is missing from ml/models/, e.g. because
    ml/model_training.ipynb has not been run yet.
    """
    pass
