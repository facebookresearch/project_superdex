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

"""FP64 native payload for :mod:`superdex.physics`.

Not an API: it exists so that ``superdex.physics.loader`` can locate the ``_native/``
directory beside this file through the import system, without assuming where pip placed it.

Use the facade instead::

    SUPERDEX_PRECISION=fp64 python -c "import superdex.physics"
"""

from __future__ import annotations
