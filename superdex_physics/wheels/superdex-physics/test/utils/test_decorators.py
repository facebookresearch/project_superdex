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

from superdex.physics.utils.decorators import override_from

########################################################################################


class TestOverrideFrom(unittest.TestCase):
    """Test class for override_from decorator functionality."""

    def test_valid_override(self):
        """Test that override_from works correctly for valid method overrides."""

        class Parent:
            def method(self) -> str:
                return "parent"

            def another_method(self, x: int) -> int:
                return x * 2

        class Child(Parent):
            @override_from(Parent)
            def method(self) -> str:
                return "child"

            @override_from(Parent)
            def another_method(self, x: int) -> int:
                return x * 3

        # Test that the child class can be instantiated successfully
        child = Child()
        self.assertEqual(child.method(), "child")
        self.assertEqual(child.another_method(5), 15)

    def test_method_not_in_parent_class(self):
        """Test that TypeError is raised when method doesn't exist in parent class."""

        class Parent:
            def existing_method(self) -> str:
                return "parent"

        with self.assertRaises(TypeError) as context:

            class Child(Parent):
                @override_from(Parent)
                def non_existing_method(self) -> str:
                    return "child"

        self.assertIn(
            "Method 'non_existing_method' does not exist in parent class 'Parent'",
            str(context.exception),
        )

    def test_parent_attribute_not_callable(self):
        """Test that TypeError is raised when parent class attribute is not callable."""

        class Parent:
            not_a_method = "just a string"

        with self.assertRaises(TypeError) as context:

            class Child(Parent):
                @override_from(Parent)
                def not_a_method(self) -> str:
                    return "child"

        self.assertIn(
            "Attribute 'not_a_method' in parent class 'Parent' is not callable",
            str(context.exception),
        )

    def test_class_not_subclass_of_parent(self):
        """Test that TypeError is raised when class is not a subclass of parent class."""

        class Parent:
            def method(self) -> str:
                return "parent"

        # NOTE: Some Python versions catch the TypeError up the stack and raise a
        # RuntimeError instead. Handle both cases.
        with self.assertRaises((TypeError, RuntimeError)) as context:

            class NotChild:
                @override_from(Parent)
                def method(self) -> str:
                    return "not child"

        exception = context.exception
        if isinstance(exception, RuntimeError):
            exception = exception.__cause__

        self.assertIn(
            "Instance of type 'NotChild' is not an instance of expected parent class 'Parent'",
            str(exception),
        )

    def test_multiple_inheritance(self):
        """Test override_from works correctly with multiple inheritance."""

        class Parent1:
            def method1(self) -> str:
                return "parent1"

        class Parent2:
            def method2(self) -> str:
                return "parent2"

        class Child(Parent1, Parent2):
            @override_from(Parent1)
            def method1(self) -> str:
                return "child1"

            @override_from(Parent2)
            def method2(self) -> str:
                return "child2"

        child = Child()
        self.assertEqual(child.method1(), "child1")
        self.assertEqual(child.method2(), "child2")


########################################################################################

if __name__ == "__main__":
    unittest.main()
