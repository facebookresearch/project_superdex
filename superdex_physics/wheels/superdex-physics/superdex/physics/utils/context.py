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

"""
Context managers for SuperDex Physics engine lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

import superdex.physics as mochi
from superdex.physics.utils.logging import (
    configure_logger,
    forward_mochi_logs_to_logger,
)

########################################################################################


class MochiContext:
    """Context manager for SuperDex Physics engine initialization and cleanup.

    This context manager ensures that the SuperDex Physics engine is properly
    initialized before use and cleanly shut down afterwards, preventing
    resource leaks and ensuring proper cleanup of the simulation environment.

    Args:
        num_worker_threads: Number of worker threads for parallel simulation.
            Defaults to -1, which lets SuperDex Physics auto-detect the optimal number
            based on available CPU cores.

    Example:
        >>> with MochiContext():
        ...     scene = mochi.create_scene("")
        ...     # Perform simulation operations...
        >>> # SuperDex Physics is automatically shut down here

    Note:
        All SuperDex Physics scene creation and simulation operations must be performed
        within this context. Attempting to use SuperDex Physics outside of this context
        will result in undefined behavior.
    """

    def __init__(
        self, num_worker_threads: int = -1, logger: logging.Logger | None = None
    ):
        self.num_worker_threads = num_worker_threads
        self.logger = logger

    def __enter__(self) -> MochiContext:
        """Initialize SuperDex Physics and configure logging."""
        mochi.initialize(num_worker_threads=self.num_worker_threads)
        logger = self.logger
        if logger is not None:
            configure_logger()
            forward_mochi_logs_to_logger(logger)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shut down SuperDex Physics and release all resources."""
        mochi.shutdown()
