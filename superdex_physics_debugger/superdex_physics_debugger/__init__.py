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

"""Native payload and launcher for the SuperDex physics debugger.

Not an API: it carries the debugger executable and its shared libraries in ``_native/``,
and gives the ``superdex-physics-debugger`` command something to point at. See
:mod:`superdex_physics_debugger.__main__`.

Run it with::

    uvx superdex-physics-debugger
"""

from __future__ import annotations
