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

import contextlib
import io
import unittest
from unittest.mock import patch

from superdex.physics.utils.profiling import Profiler, ProfilerSection

########################################################################################


class EncodingCheckingStringIO(io.StringIO):
    def __init__(self, encoding: str):
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def errors(self) -> str:
        return "strict"

    def write(self, value: str) -> int:
        value.encode(self.encoding, self.errors)
        return super().write(value)


class TestProfiler(unittest.TestCase):
    """Test class for profiler functionality."""

    def test_profiler_initialization(self):
        """Test that Profiler can be initialized properly."""
        # Test default initialization
        profiler = Profiler()
        assert profiler.enabled is True
        assert profiler.record_individual_samples is False
        assert profiler.sections == {}
        assert profiler.stack == []

        # Test with individual samples recording enabled
        profiler = Profiler(record_individual_samples=True)
        assert profiler.record_individual_samples is True

    def test_profiler_basic_timing(self):
        """Test basic profiling functionality."""
        profiler = Profiler()

        # Mock timer to control timing.
        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            duration = profiler.exit()

        # Duration should be 1ms converted to milliseconds (1.0)
        assert duration == 1.0

        # Check section was created and statistics updated
        assert "test_section" in profiler.sections
        section = profiler.sections["test_section"]
        assert section.name == "test_section"
        assert section.last == 1.0
        assert section.total == 1.0
        assert section.count == 1
        assert section.mean == 1.0

    def test_profiler_multiple_executions(self):
        """Test profiling the same section multiple times."""
        profiler = Profiler()

        # First execution: 1ms
        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            assert profiler.current_section.name == "test_section"
            profiler.exit()

        # Second execution: 2ms
        with patch(
            "time.perf_counter_ns", side_effect=[0, 2000000]
        ):  # 2ms in nanoseconds
            profiler.enter("test_section")
            assert profiler.current_section.name == "test_section"
            profiler.exit()

        section = profiler.sections["test_section"]
        assert section.count == 2
        assert section.last == 2.0  # Last execution was 2ms
        assert section.total == 3.0  # 1ms + 2ms
        assert section.mean == 1.5  # (1 + 2) / 2

    def test_profiler_nested_sections(self):
        """Test nested profiling sections."""
        profiler = Profiler()

        # Outer section
        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000, 2000000, 5000000]
        ):  # 0, 1ms, 2ms, 5ms in nanoseconds
            profiler.enter("outer")
            assert profiler.current_section.name == "outer"
            profiler.enter("inner")
            assert profiler.current_section.name == "inner"
            profiler.exit()  # inner takes 1ms
            profiler.exit()  # outer takes 5ms total
            assert profiler.current_section is None

        # Check outer section
        assert "outer" in profiler.sections
        outer_section = profiler.sections["outer"]
        assert outer_section.last == 5.0

        # Check inner section is nested
        assert "inner" in outer_section.nested_sections
        inner_section = outer_section.nested_sections["inner"]
        assert inner_section.last == 1.0

    def test_profiler_context_manager(self):
        """Test using profiler as a context manager."""
        profiler = Profiler()

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            with profiler:
                pass  # Context manager should call exit()

        # Should have exited and recorded timing
        section = profiler.sections["test_section"]
        assert section.count == 1
        assert section.last == 1.0

    def test_profiler_reset(self):
        """Test resetting profiler state."""
        profiler = Profiler()

        # Add some data
        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            profiler.exit()

        assert len(profiler.sections) == 1

        # Reset should clear everything
        profiler.reset()
        assert len(profiler.sections) == 0
        assert len(profiler.stack) == 0

    def test_profiler_to_dict_basic(self):
        """Test converting profiler data to dictionary."""
        profiler = Profiler()

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            profiler.current_section.info["hello"] = "world"
            profiler.exit()

        # Basic dictionary (just last times)
        result = profiler.to_dict()
        assert result == {"test_section": 1.0}

        # Extended dictionary (all statistics)
        result = profiler.to_dict(extended=True)
        expected = {
            "test_section": {
                "last": 1.0,
                "total": 1.0,
                "count": 1,
                "mean": 1.0,
                "stddev": 0.0,
                "info": {"hello": "world"},
                "pct": 100.0,
            }
        }
        assert result == expected

    def test_profiler_to_dict_nested(self):
        """Test dictionary conversion with nested sections."""
        profiler = Profiler()

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000, 2000000, 5000000]
        ):  # 0, 1ms, 2ms, 5ms in nanoseconds
            profiler.enter("outer")
            profiler.enter("inner")
            profiler.current_section.info["oh"] = "hai"
            profiler.exit()
            profiler.current_section.info["howdy"] = "fellas"
            profiler.exit()

        # Basic dictionary
        result = profiler.to_dict()
        expected = {"outer": 5.0, "outer/inner": 1.0}
        assert result == expected

        # Extended dictionary (all statistics)
        result = profiler.to_dict(extended=True)
        expected = {
            "outer": {
                "last": 5.0,
                "total": 5.0,
                "count": 1,
                "mean": 5.0,
                "stddev": 0.0,
                "info": {"howdy": "fellas"},
                "pct": 100.0,
            },
            "outer/inner": {
                "last": 1.0,
                "total": 1.0,
                "count": 1,
                "mean": 1.0,
                "stddev": 0.0,
                "info": {"oh": "hai"},
                "pct": 20.0,
            },
        }
        assert result == expected

    def test_profiler_exit_without_enter(self):
        """Test that exiting without entering raises an assertion error."""
        profiler = Profiler()

        try:
            profiler.exit()
            raise AssertionError("Should have raised AssertionError")
        except AssertionError as e:
            assert "No active profiling section to exit" in str(e)

    def test_profiler_print_summary(self):
        """Test print_summary method (basic functionality test)."""
        profiler = Profiler()

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            profiler.exit()

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            profiler.print_summary()
            assert len(buf.getvalue()) > 0

    def test_profiler_print_summary_falls_back_for_cp1252_stdout(self):
        """Test print_summary with a Windows-style stdout encoding."""
        profiler = Profiler()

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            profiler.exit()

        with EncodingCheckingStringIO("cp1252") as buf:
            with contextlib.redirect_stdout(buf):
                profiler.print_summary()
            assert "test_section" in buf.getvalue()

    def test_profiler_disabled(self):
        """Test profiler behavior when disabled."""
        profiler = Profiler()
        profiler.enabled = False

        # No sections should be created when disabled.
        with profiler.enter("test_section"):
            pass
        assert len(profiler.sections) == 0

        # A section should be created when enabled.
        profiler.enabled = True
        with profiler.enter("test_section"):
            pass
        assert len(profiler.sections) == 1

    def test_profiler_variance_calculation(self):
        """Test that variance and standard deviation are calculated correctly."""
        profiler = Profiler()

        # Execute section multiple times with known values
        times = [2000000, 4000000, 6000000]  # 2ms, 4ms, 6ms in nanoseconds

        for duration in times:
            with patch("time.perf_counter_ns", side_effect=[0, duration]):
                profiler.enter("test_section")
                profiler.exit()

        section = profiler.sections["test_section"]

        # Mean should be 4ms
        assert section.mean == 4

        # Variance calculation using Welford's algorithm
        # For values [2, 4, 6], variance = 4.0, stddev = 2.0
        assert abs(section.var - 4.0) < 0.001
        assert abs(section.stddev - 2.0) < 0.001

    def test_profiler_to_dict_with_prefix(self):
        """Test dictionary conversion with prefix."""
        profiler = Profiler()

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            profiler.exit()

        result = profiler.to_dict(prefix="prefix", separator=".")
        assert result == {"prefix.test_section": 1.0}

    def test_profiler_section_to_dict(self):
        """Test ProfilerSection to_dict method."""
        section = ProfilerSection(name="test")
        section.last = 1.0
        section.total = 5.0
        section.count = 3
        section.mean = 1.67
        section.m2 = 2.0

        # Basic dictionary
        result = section.to_dict()
        assert result == {"test": 1.0}

        # Extended dictionary
        result = section.to_dict(extended=True)
        expected = {
            "test": {
                "last": 1.0,
                "total": 5.0,
                "count": 3,
                "mean": 1.67,
                "stddev": section.stddev,
                "info": {},
                "pct": 100.0,
            }
        }
        assert result == expected

    def test_profiler_section_to_dict_nested(self):
        """Test ProfilerSection to_dict with nested sections."""
        parent = ProfilerSection(name="parent")
        parent.last = 5.0

        child = ProfilerSection(name="child")
        child.last = 2.0
        parent.nested_sections["child"] = child

        result = parent.to_dict()
        expected = {"parent": 5.0, "parent/child": 2.0}
        assert result == expected

    def test_profiler_section_to_dict_with_prefix(self):
        """Test ProfilerSection to_dict with prefix and separator."""
        section = ProfilerSection(name="test")
        section.last = 1.0

        result = section.to_dict(prefix="prefix", separator=".")
        assert result == {"prefix.test": 1.0}

    def test_profiler_individual_samples_disabled(self):
        """Test that history is not stored when individual samples recording is
        disabled."""
        profiler = Profiler(record_individual_samples=False)

        with patch(
            "time.perf_counter_ns", side_effect=[0, 1000000]
        ):  # 1ms in nanoseconds
            profiler.enter("test_section")
            profiler.exit()

        section = profiler.sections["test_section"]
        assert section.history is None

    def test_profiler_context_manager_returns_self(self):
        """Test that __enter__ returns the profiler instance for context manager
        usage."""
        profiler = Profiler()

        result = profiler.__enter__()
        assert result is profiler


########################################################################################

if __name__ == "__main__":
    unittest.main()
