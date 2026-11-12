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

from typing import Any, Callable

########################################################################################


def skip(reason: str) -> Callable[[Any], Any]:
    """Unconditionally omit a test from discovery with the given reason."""
    return skip_if(condition=True, reason=reason)


def skip_if(condition: bool, reason: str) -> Callable[[Any], Any]:
    """
    Omit a test method or class from discovery when the condition is true.

    Excluded methods and classes become None, which unittest does not discover.
    The reason documents why the test does not apply to this configuration.
    """

    def factory(item: Any) -> Any:
        return None if condition else item

    return factory
