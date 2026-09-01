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

from __future__ import annotations

import datetime
import enum
import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

########################################################################################


class LogLevel(enum.Flag):
    """Log level flags for the logger window."""

    NONE = 0
    """No log messages are displayed."""
    DEBUG = 1 << 0
    """Debug log messages are displayed."""
    INFO = 1 << 1
    """Info log messages are displayed."""
    WARNING = 1 << 2
    """Warning log messages are displayed."""
    ERROR = 1 << 3
    """Error log messages are displayed."""
    CRITICAL = 1 << 4
    """Critical log messages are displayed."""
    ALL = int(DEBUG) | int(INFO) | int(WARNING) | int(ERROR) | int(CRITICAL)
    """All log messages are displayed."""


@dataclass
class LogMessage:
    """Represents a single log message with metadata."""

    timestamp: str
    """Timestamp of the log message in HH:MM:SS format."""
    level: LogLevel
    """Log level of the message."""
    module: str
    """Python/C++ module that generated the log message."""
    message: str
    """Log message."""
    pathname: str
    """Full path to the source file that generated the log message."""
    filename: str
    """Filename of the source file that generated the log message."""
    lineno: int
    """Line number in the source file that generated the log message."""


########################################################################################


class LoggingHandler(logging.Handler):
    """
    A logging handler that stores log messages in a circular buffer.

    This handler captures log messages and stores them in a bounded circular buffer
    (deque with maxlen). It is thread-safe and designed to be attached to Python's
    logging system to capture logs for display in the SuperDex Physics Viewer UI.
    """

    def __init__(self, max_messages: int = 100) -> None:
        """
        Initialize the circular buffer logging handler.

        Args:
            max_messages: Maximum number of messages to store. Older messages are
                automatically discarded when the buffer is full.
        """
        super().__init__()
        self._messages: deque[LogMessage] = deque(maxlen=max_messages)
        self._lock = threading.Lock()
        self._max_messages = max_messages
        self._has_new_messages = False

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by storing it in the circular buffer.

        This method is called by the logging system when a log message is generated.
        It is thread-safe and will not raise exceptions.

        Args:
            record: The log record to store.
        """
        try:
            dt = datetime.datetime.fromtimestamp(record.created)
            timestamp = dt.strftime("%H:%M:%S")

            # Convert logging level to LogLevel enum
            if record.levelno >= logging.CRITICAL:
                level = LogLevel.CRITICAL
            elif record.levelno >= logging.ERROR:
                level = LogLevel.ERROR
            elif record.levelno >= logging.WARNING:
                level = LogLevel.WARNING
            elif record.levelno >= logging.INFO:
                level = LogLevel.INFO
            else:
                level = LogLevel.DEBUG

            # Override the filename and lineno if the record originates from SuperDex Physics
            module = record.name
            if "mochi_filename" in record.__dict__:
                pathname = record.__dict__["mochi_filename"]
                lineno = record.__dict__["mochi_lineno"]
            else:
                pathname = record.pathname
                lineno = record.lineno

            # Store the new message into the circular buffer.
            message = LogMessage(
                timestamp=timestamp,
                level=level,
                module=module,
                pathname=pathname,
                filename=str(Path(pathname).name),
                lineno=lineno,
                message=self.format(record),
            )
            with self._lock:
                self._messages.append(message)
                self._has_new_messages = True
        except Exception:
            self.handleError(record)

    @property
    def messages(self) -> Sequence[LogMessage]:
        """
        Get a snapshot of all stored log messages.

        Returns a copy of the message list to ensure thread safety.
        Clears the has_new_messages flag when messages are retrieved.

        Returns:
            A list of LogMessage objects in chronological order (oldest first).
        """
        with self._lock:
            self._has_new_messages = False
            return list(self._messages)

    @property
    def has_new_messages(self) -> bool:
        """
        Check if new messages have been added since the last retrieval.

        This flag is set to True when a new message is emitted and cleared
        when the messages property is accessed. This can be used by the logging
        window to determine if autoscroll should be triggered.

        Returns:
            True if new messages have been added, False otherwise.
        """
        with self._lock:
            return self._has_new_messages

    def clear(self) -> None:
        """Clear all stored log messages."""
        with self._lock:
            self._messages.clear()

    @property
    def max_messages(self) -> int:
        """Get the maximum number of messages that can be stored."""
        return self._max_messages

    @max_messages.setter
    def max_messages(self, max_messages: int):
        """
        Set the maximum number of messages to store.

        If the new maximum is smaller than the current number of messages,
        the oldest messages will be discarded.

        Args:
            max_messages: New maximum number of messages.
        """
        if max_messages < 1:
            raise ValueError("max_messages must be greater than 0")
        with self._lock:
            messages = list(self._messages)
            self._max_messages = max_messages
            self._messages = deque(messages, maxlen=max_messages)
