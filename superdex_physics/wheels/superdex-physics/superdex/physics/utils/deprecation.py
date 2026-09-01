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

import warnings
from typing import Any, cast, Generic, Type, TypeVar

import typing_extensions

########################################################################################

# NOTE: Here we use an alias to the built-in typing library. However, it could be
# convenient to override it in the future to provide additional information (e.g.
# name of the class/function that replaces the deprecated one) or change the behavior
# (raise errors instead of warnings).

deprecated = typing_extensions.deprecated
"""A decorator for marking classes and functions as deprecated"""

########################################################################################

T = TypeVar("T")


class DeprecatedField(Generic[T]):
    """
    A descriptor for deprecated dataclass fields that issues warnings when the field
    is accessed (get) or modified (set). A deprecation warning is also issued whenever
    the field is initialized with a different value from the default.
    """

    ####################################################################################
    # Structures
    ####################################################################################

    class DefaultSentinel:
        """Used to determine whenever a field is initialized with default values."""

        pass

    #####################################################################################
    # Constructor
    #####################################################################################

    def __init__(
        self,
        default: T,
        message: str | None = None,
        /,
        *,
        category: Type[Warning] = DeprecationWarning,
        stacklevel: int = 2,
    ):
        """Initializes the deprecated field descriptor"""
        self.name = None
        self.internal_name = None
        self.message = message
        self.default = default
        self.category = category
        self.stacklevel = stacklevel

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """Sets the name of the field"""
        self.name = name
        self.internal_name = f"__deprecated_{name}"
        self.message = f"Field '{name}' is deprecated" + (
            f". {self.message}" if self.message is not None else ""
        )

    def __get__(self, instance: Any, owner: Type[Any]) -> T:
        """Gets the field value"""
        # Upon instantiation of the dataclass, the __get__ method is called with a None
        # instance. We can use this to signal that the next __set__ value (which will be
        # called with whatever we return here) is going to be the default value.
        if instance is None:
            return cast(T, DeprecatedField.DefaultSentinel)
        # For any other get calls, warn access to the field.
        warnings.warn(self.message or "", self.category, stacklevel=self.stacklevel)
        return getattr(instance, self.internal_name, self.default)

    def __set__(self, instance: Any, value: T) -> None:
        """Sets the field value"""
        # If the default value is being set, silently set the attribute.
        if value is DeprecatedField.DefaultSentinel:
            setattr(instance, self.internal_name, self.default)
        # Otherwise, warn about the deprecated field, and set the attribute.
        else:
            # Increase the stack level if this is being set during the dataclass'
            # init. Otherwise, the deprecation message will not point to the correct
            # location.
            stacklevel = self.stacklevel + (not hasattr(instance, self.internal_name))
            warnings.warn(self.message or "", self.category, stacklevel=stacklevel)
            setattr(instance, self.internal_name, value)
