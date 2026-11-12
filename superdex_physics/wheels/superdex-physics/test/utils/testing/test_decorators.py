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
from types import ModuleType

from superdex.physics.utils.testing.decorators import skip, skip_if

########################################################################################


class TestDecorators(unittest.TestCase):
    def test_excluded_methods_are_not_discovered(self) -> None:
        for decorator in (skip("Unsupported"), skip_if(True, "Unsupported")):
            with self.subTest(decorator=decorator):
                executed: list[str] = []

                class Example(unittest.TestCase):
                    executions = executed

                    @decorator
                    def test_excluded(self) -> None:
                        self.fail("Excluded test ran")

                    @skip_if(False, "Supported")
                    def test_included(self) -> None:
                        self.assertEqual(self.executions, [])
                        self.executions.append("included")

                loader = unittest.TestLoader()
                self.assertEqual(loader.getTestCaseNames(Example), ["test_included"])
                result = unittest.TestResult()
                loader.loadTestsFromTestCase(Example).run(result)
                self.assertTrue(result.wasSuccessful())
                self.assertEqual(result.testsRun, 1)
                self.assertEqual(result.skipped, [])
                self.assertEqual(executed, ["included"])

    def test_excluded_classes_are_not_discovered(self) -> None:
        for decorator in (skip("Unsupported"), skip_if(True, "Unsupported")):
            with self.subTest(decorator=decorator):

                @decorator
                class Excluded(unittest.TestCase):
                    def test_excluded(self) -> None:
                        self.fail("Excluded class ran")

                module = ModuleType("example_tests")
                module.__dict__["Excluded"] = Excluded
                suite = unittest.TestLoader().loadTestsFromModule(module)
                self.assertEqual(suite.countTestCases(), 0)

    def test_false_condition_preserves_methods_and_classes(self) -> None:
        def method() -> str:
            return "enabled"

        for item in (method, unittest.TestCase):
            with self.subTest(item=item):
                self.assertIs(skip_if(False, "Supported")(item), item)
        self.assertEqual(skip_if(False, "Supported")(method)(), "enabled")

    def test_excluded_override_does_not_discover_inherited_method(self) -> None:
        class Base(unittest.TestCase):
            def test_excluded(self) -> None:
                self.fail("Inherited test ran")

        class Derived(Base):
            @skip_if(True, "Unsupported")
            def test_excluded(self) -> None:
                self.fail("Excluded override ran")

        self.assertEqual(unittest.TestLoader().getTestCaseNames(Derived), [])
