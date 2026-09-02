# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging utilities for SuperDex Physics.

Recommended usage:
    import logging
    from superdex.physics.utils.logging import configure_logger

    configure_logger()
    logger = logging.getLogger(__name__)  # Use module name for logger
    logger.info("Application started")
"""

from __future__ import annotations

import logging
import os
from copy import copy
from datetime import datetime
from logging import FileHandler, Logger
from pathlib import Path

from superdex.physics.environment import (
    get_env_var_value,
    LEGACY_LOG_LEVEL_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
)
from superdex.physics.utils.decorators import override_from

########################################################################################


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log levels for terminal output.

    This formatter enhances console log readability by applying color codes to different
    components of log messages. Colors are applied using ANSI escape sequences which work
    in most modern terminals including VS Code, Git Bash, WSL, and Linux/Mac terminals.

    Note: Color codes are NOT applied to file handlers to keep log files clean and
    easily parseable by automated tools.
    """

    # ANSI color codes for styling log components
    TIMESTAMP_STYLE = "\033[90m"  # Gray - for timestamps
    FILENAME_STYLE = "\033[90;3m"  # Gray, italic - for file locations
    LEVEL_STYLE = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[36m",  # Cyan
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset to default terminal color

    @override_from(logging.Formatter)
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format the timestamp with gray color for subtle appearance."""
        formatted_time = super().formatTime(record, datefmt)
        return f"{self.TIMESTAMP_STYLE}{formatted_time}{self.RESET}"

    @override_from(logging.Formatter)
    def format(self, record: logging.LogRecord) -> str:
        """Format the complete log record with appropriate color coding."""

        # Get color for this log level, defaulting to DEBUG color if level is unknown.
        level_color = self.LEVEL_STYLE.get(record.levelname, self.LEVEL_STYLE["DEBUG"])

        # Create a copy to avoid mutating the original record (important for
        # multi-handler setups).
        record = copy(record)

        # Apply colors to different components.
        record.filename = f"{self.FILENAME_STYLE}{record.filename}{self.RESET}"
        record.levelname = f"{level_color}{record.levelname:8}{self.RESET}"
        # pyre-ignore[8] (allow lineno to be used as str to allow color formatting)
        record.lineno = f"{self.FILENAME_STYLE}{record.lineno}{self.RESET}"

        # Let the parent formatter handle the actual message formatting.
        return super().format(record)


########################################################################################


def configure_logger(
    log_output_path: Path | None = None,
):
    """Configure the Python logger with sensible defaults for SuperDex Physics applications.

    This function sets up both console and optional file logging. Provides simple colored
    console output and unique file logging for each process. Also provides environment
    variable-based log level control via SUPERDEX_PYTHON_LOG_LEVEL.

    Args:
        log_output_path: Optional directory path for log files. If provided, creates a
            unique log file combining the current timestamp and process ID within this
            directory (e.g., /var/logs/20251127_105323_12345.log). The directory will be
            created if it doesn't exist. Each process gets its own log file to avoid
            file locking and race condition issues.

    Environment Variables:
        SUPERDEX_PYTHON_LOG_LEVEL: Sets the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO if not set or if an invalid level is specified.

    Note:
        This function is idempotent - calling it multiple times will have no effect after
        the first call, as it checks if the root logger already has handlers configured.
    """

    # Check if the root logger already has handlers. This prevents duplicate
    # configuration if configure_logger() is called multiple times or if another part of
    # the application has already set up logging.
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    # Read the desired log level from the environment variable, with INFO as the default.
    # This allows users to control verbosity without code changes.
    log_level_name = (
        get_env_var_value(LOG_LEVEL_ENV_VAR, LEGACY_LOG_LEVEL_ENV_VAR) or "INFO"
    ).upper()
    LOG_LEVEL = getattr(logging, log_level_name, logging.INFO)

    # Initialize console handler with colored formatter.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColoredFormatter(
            fmt="{asctime} {levelname} {message} ({filename}:{lineno})",
            style="{",
            datefmt="[%X]",  # [%X] formats as [HH:MM:SS] in the local timezone
        )
    )
    handlers: list[logging.Handler] = [console_handler]

    # Initialize file handler if a log output path is provided. This is optional.
    if log_output_path:
        # Ensure the output directory exists.
        log_output_path.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename using timestamp and process ID.
        # This ensures each process gets its own log file, preventing file locking
        # issues and race conditions in multi-process environments.
        # Format: YYYYMMDD_HHMMSS_PID.log (e.g., 20251127_105323_12345.log)
        pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_output_path / f"{timestamp}_{pid}.log"

        # Create a file handler that writes to the unique log file.
        # Use a standard (non-colored) formatter for file output.
        # File logs should be plain text for easy parsing by log analysis tools.
        # Note the additional [{name}] field which includes the logger name,
        # making it easier to filter logs by module in log files.
        file_handler = FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="{asctime} {levelname} [{name}] {message} ({filename}:{lineno})",
                style="{",
                datefmt="[%X]",
            )
        )
        handlers.append(file_handler)

    # Configure the root logger with our handlers.
    logging.basicConfig(level=LOG_LEVEL, handlers=handlers)


def forward_mochi_logs_to_logger(logger: Logger) -> None:
    """Forward SuperDex Physics C++ logs to a Python logger for unified logging.

    This function bridges the SuperDex Physics C++ logging system with Python's logging framework
    by setting up a callback that intercepts SuperDex Physics log messages and redirects them
    to the specified Python logger. This allows you to:

    - See SuperDex Physics C++ logs in the same output stream as Python logs
    - Apply Python logging filters and formatters to C++ logs
    - Write both Python and C++ logs to the same log files
    - Control C++ log verbosity using Python's log level system

    The mapping between SuperDex Physics log channels and Python log levels is:
        - LogChannel.INFO    → logging.INFO    (general information)
        - LogChannel.WARNING → logging.WARNING (warnings and potential issues)
        - LogChannel.ERROR   → logging.ERROR   (errors and failures)
        - Other channels     → logging.DEBUG   (detailed diagnostic info)

    Args:
        logger: The Python logger instance to receive forwarded SuperDex Physics logs.
            Typically created with ``logging.getLogger(__name__)``.

    Raises:
        RuntimeError: If called before initializing a SuperDex Physics context. You must create
            and enter a SuperDex Physics context before calling this function, for example with
            ``with superdex.physics.utils.context.MochiContext():``.

    Note:
        Each forwarded log record includes extra fields with C++ source location:
            - mochi_filename: C++ source file where the log originated.
            - mochi_lineno: Line number in the C++ source file.

        These can be used in custom formatters to show C++ source locations.
    """

    # Import SuperDex Physics bindings inside this function - This allows importing the logging
    # utils module without initializing SuperDex Physics.
    import superdex.physics as sdp
    from superdex.physics import LogChannel

    # Verify that a SuperDex Physics context has been initialized. The log callback system
    # requires an active SuperDex Physics context to function properly.
    if not sdp.is_initialized():
        raise RuntimeError(
            "You must initialize a SuperDex Physics context before attempting to redirect logs."
        )

    # Define the mapping from SuperDex Physics C++ log channels to Python logging levels.
    # This determines the severity level that will be used when forwarding logs.
    LOG_CHANNEL_TO_LOGGING_LEVEL = {
        LogChannel.INFO: logging.INFO,
        LogChannel.WARNING: logging.WARNING,
        LogChannel.ERROR: logging.ERROR,
    }

    def log_callback_impl(channel: LogChannel, message: str, file: str, line: int):
        """Internal callback invoked by SuperDex Physics for each log message.

        This function is called by the SuperDex Physics C++ logging system whenever a log
        message is generated. It translates the SuperDex Physics log format to Python's format.

        Args:
            channel: The SuperDex Physics log channel (INFO, WARNING, ERROR, etc.)
            message: The log message text from C++
            file: Source file path where the log originated (C++ file)
            line: Line number in the source file (C++ line number)
        """
        # Map the SuperDex Physics channel to a Python logging level.
        # If the channel isn't recognized, default to DEBUG level.
        level = LOG_CHANNEL_TO_LOGGING_LEVEL.get(channel, logging.DEBUG)

        # Clean up the message by removing leading/trailing whitespace and newlines.
        # C++ logs include trailing newlines that we don't want in Python logs.
        message = message.strip()

        # Forward the log message to the Python logger with extra metadata.
        # The 'extra' dict adds custom fields that can be used by formatters:
        # - mochi_filename: Shows which C++ file generated the log
        # - mochi_lineno: Shows which line in the C++ file generated the log
        logger.log(level, message, extra={"mochi_filename": file, "mochi_lineno": line})

    # Register our callback with the SuperDex Physics logging system.
    # From this point on, all SuperDex Physics C++ logs will be forwarded to the Python logger.
    sdp.set_log_callback(log_callback_impl)
