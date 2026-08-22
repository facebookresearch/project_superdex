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

import unittest
from functools import wraps
from typing import Any, Callable

from superdex.physics.utils.testing.environment import IS_INTERNAL_CI

########################################################################################


def skip(reason: str) -> Callable[[Any], Any]:
    """
    Unconditionally skip a test with the given reason. This is a convenience wrapper
    around skip_if that always skips the test.
    """
    return skip_if(condition=True, reason=reason)


def skip_if(condition: bool, reason: str) -> Callable[[Any], Any]:
    """
    Conditionally skip a test based on a boolean condition. This decorator provides
    different behavior depending on the environment. If the test is running in
    internal CI, it will print a skip message and return early, but will not set the
    skipped flag. This avoids flagging intentionally skipped tests as warnings in the
    internal CI UI. In other environments, it falls back to unittest.skip() for proper
    test reporting.
    """

    def factory(item: Callable[[Any], Any]) -> Callable[[Any], Any]:
        # If condition is False, return the original item unchanged.
        if not condition:
            return item

        # Use unittest's built-in skip mechanism for types and for methods, as long
        # as the test is not running in internal CI.
        if isinstance(item, type) or not IS_INTERNAL_CI:
            return unittest.skip(reason)(item)

        # For methods running in internal CI, create a wrapper that prints skip message.
        @wraps(item)
        def wrapper(*args, **kwargs):
            print(f"Skipping test {item.__name__} on internal CI because {reason}.")

        return wrapper

    return factory
