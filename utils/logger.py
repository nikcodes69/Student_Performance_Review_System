"""
utils/logger.py
================
Central place that configures Python's built-in `logging` module.

WHY NOT print()?
print() output only appears in the terminal that is currently open, has no
timestamp, no severity level, and disappears the moment the window closes.
The `logging` module fixes all of that: every message is timestamped,
tagged with a level (INFO, WARNING, ERROR...), written permanently to
logs/app.log, and tells us WHICH file/module it came from. That is exactly
what an examiner or a real-world developer expects for diagnosing problems
after the fact (e.g. "why did marks entry fail at 3pm yesterday?").

HOW OTHER FILES USE THIS:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Student %s created successfully", roll_no)
    logger.error("Failed to insert marks: %s", str(error))

`__name__` is a built-in Python variable that automatically holds the
current file's module name (e.g. "modules.students"). Passing it to
get_logger() means every log line records exactly which file produced it,
without us typing that name by hand.
"""

import logging
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """
    Create (or reuse) a logger configured to write to logs/app.log.

    Args:
        name: Usually the caller's __name__, so log lines show their
            source module, e.g. "modules.marks".

    Returns:
        A logging.Logger instance ready to use (.info, .warning, .error).
    """
    logger = logging.getLogger(name)

    # logging.getLogger(name) returns the SAME logger object every time it
    # is called with the same name. Without this check, calling get_logger()
    # from the same module twice (e.g. on a Streamlit page rerun) would
    # attach a second file handler, and every message would be written to
    # the log file twice, then three times, and so on. This check makes
    # get_logger() safe to call repeatedly.
    if logger.handlers:
        return logger

    logger.setLevel(config.LOG_LEVEL)

    # Make sure the logs/ folder exists before we try to write into it.
    # parents=True creates any missing parent folders too; exist_ok=True
    # stops it from raising an error if the folder is already there.
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # RotatingFileHandler writes to app.log, and once it passes
    # LOG_MAX_BYTES it starts a new file, keeping only LOG_BACKUP_COUNT
    # old copies. This prevents one giant, unmanageable log file.
    file_handler = RotatingFileHandler(
        filename=config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )

    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
