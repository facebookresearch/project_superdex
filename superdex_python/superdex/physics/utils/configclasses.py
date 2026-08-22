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

import dataclasses
from typing import dataclass_transform


########################################################################################


@dataclass_transform(kw_only_default=True)
def configclass(cls):
    """
    A decorator for specifying configuration classes. Essentially a wrapper around
    dataclass enforcing the requirement that all fields must be initialized using
    keyword arguments. In the future, this decorator may also enforce additional
    requirements, such as run-time type checking and other validation checks.
    """

    # Specify the settings for the dataclass decorator.
    settings = {
        "kw_only": True,  # All fields must be initialized with keyword arguments.
    }

    # Apply the dataclass decorator.
    return dataclasses.dataclass(cls, **settings)
