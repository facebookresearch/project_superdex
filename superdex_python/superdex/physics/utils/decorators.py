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

from typing import Any, Callable, cast, Type

from typing_extensions import override

########################################################################################


def override_from(
    parent_class: Type,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that marks a method as overriding a method from a specific parent class.

    Unlike the typing.override decorator, this decorator also does a checks at runtime
    if the method exists in the parent class, it is indeed callable, and that the
    overridden method is bound to an instance that inherits from the parent class. If
    none of these hold true, a TypeError is raised.

    This decorator also applies typing.override for static type checkers.

    Args:
        parent_class: The class that defines the method being overridden.

    Returns:
        A decorator function that validates the override relationship.

    Raises:
        TypeError: If the method doesn't exist in the parent class or if the
                   instance is not of the expected type.

    Example:
        .. code-block:: python

            class Parent:
                def method(self) -> str:
                    return "parent"

            class Child(Parent):
                @override_from(Parent)
                def method(self) -> str:
                    return "child"

            class NotChild:
                @override_from(Parent) # Raises TypeError
                def method(self) -> str:
                    return "not child"
    """

    class OverrideChecker:
        def __init__(self, func: Callable[..., Any], parent_class: Type) -> None:
            self.func = func
            self.parent_class = parent_class

        def __set_name__(self, owner: Type, name: str) -> None:
            if not issubclass(owner, self.parent_class):
                raise TypeError(
                    f"Instance of type '{owner.__name__}' is not an instance of "
                    f"expected parent class '{parent_class.__name__}'."
                )
            setattr(owner, name, self.func)

    def factory(func: Callable[..., Any]) -> Callable[..., Any]:
        # Validate that the method exists in the parent class, and that it is indeed
        # callable.
        method_name = func.__name__
        if not hasattr(parent_class, method_name):
            raise TypeError(
                f"Method '{method_name}' does not exist in parent class "
                f"'{parent_class.__name__}'"
            )

        # Ensure the overridden object is a method.
        parent_method = getattr(parent_class, method_name)
        if not callable(parent_method):
            raise TypeError(
                f"Attribute '{method_name}' in parent class '{parent_class.__name__}' "
                "is not callable"
            )

        # Wrap function with an object that checks if the parent class type at class
        # construction time. This is needed because the decorator is applied to the
        # method, not the class.
        return cast(Callable[..., Any], OverrideChecker(override(func), parent_class))

    return factory
