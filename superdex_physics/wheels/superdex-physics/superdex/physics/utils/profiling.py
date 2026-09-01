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

import dataclasses
import fnmatch
import sys
import time
from typing import Any, Iterable, NamedTuple

from tabulate import tabulate

########################################################################################


def _stdout_supports_fancy_outline() -> bool:
    """Returns whether stdout can encode tabulate's fancy outline characters."""
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding is None:
        return False

    try:
        tabulate([[""]], [""], tablefmt="fancy_outline").encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False

    return True


@dataclasses.dataclass
class ProfilerSection:
    """
    Statistics container for a profiled code section.

    Tracks timing statistics including duration, count, mean, and variance
    for performance analysis. Supports nested profiling sections for
    hierarchical performance measurement.
    """

    ####################################################################################
    # Members
    ####################################################################################

    name: str = ""
    """Name of the section."""
    last: float = 0.0
    """Duration of the most recent execution"""
    total: float = 0.0
    """Cumulative duration across all executions"""
    count: int = 0
    """Number of times this section has been executed"""
    mean: float = 0.0
    """Average execution duration"""
    m2: float = 0.0
    """Welford's running M2, used to compute the execution variance"""
    history: list[float] | None = None
    """Optional history of all durations"""
    nested_sections: dict[str, ProfilerSection] = dataclasses.field(
        default_factory=dict
    )
    """Child profiling sections for hierarchical profiling"""
    info: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Optional dictionary for storing additional information"""

    ####################################################################################
    # Properties
    ####################################################################################

    @property
    def var(self) -> float:
        """Execution variance."""
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        """Execution standard deviation."""
        return self.var**0.5

    ####################################################################################
    # Methods
    ####################################################################################

    def to_dict(
        self, prefix: str = "", separator: str = "/", extended: bool = False
    ) -> dict[str, Any]:
        """
        Generate a flat dictionary summary of all profiling statistics.

        Recursively traverses nested sections to create a comprehensive
        summary with hierarchical path names.
        """

        # If a prefix was provided, include a first separator.
        if prefix:
            prefix += separator

        # Generate flattened dictionary of timings for the section and its predecesors.
        timings = {}

        def recurse(section: ProfilerSection, path: str):
            path += section.name
            if not extended:
                timings[path] = section.last
            else:
                pct = 100.0 * section.total / self.total
                timings[path] = {
                    "last": section.last,
                    "total": section.total,
                    "count": section.count,
                    "mean": section.mean,
                    "stddev": section.stddev,
                    "info": section.info,
                    "pct": pct,
                }
            path += separator
            for nested_section in section.nested_sections.values():
                recurse(nested_section, path)

        recurse(self, prefix)
        return timings

    ####################################################################################
    # Other operators
    ####################################################################################

    def __getitem__(self, key: str) -> ProfilerSection:
        """Get a nested section by name."""
        section = self.nested_sections.get(key, None)
        if section is None:
            raise KeyError(f"Section '{key}' not found")
        return section

    def __contains__(self, key: str) -> bool:
        """Check if a nested section exists by name."""
        return key in self.nested_sections


class ProfilerStackEntry(NamedTuple):
    """
    Stack entry for tracking active profiling sections.

    Represents a single level in the profiling call stack, containing
    the section being profiled and when it started.
    """

    section: ProfilerSection
    """The profiling section being tracked"""
    start_time: float
    """Timestamp when this section started"""


class Profiler:
    """
    Hierarchical performance profiler for measuring code execution times.

    Provides stack-based profiling with support for nested sections, statistical
    analysis, and configurable timing mechanisms. Tracks execution statistics
    including mean, variance, and total time.
    """

    ####################################################################################
    # Members
    ####################################################################################

    # Public members
    enabled: bool
    """Whether the profiler is enabled"""
    record_individual_samples: bool
    """Whether to record individual sample timings"""
    sections: dict[str, ProfilerSection]
    """Top-level profiling sections"""
    stack: list[ProfilerStackEntry]
    """Active profiling section stack"""

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(
        self,
        record_individual_samples: bool = False,
    ):
        """
        Initialize the profiler with high performance timer.
        """

        self.enabled = True
        self.record_individual_samples = record_individual_samples

        # Initialize profiler state
        self.sections = {}
        self.stack = []

    ####################################################################################
    # Methods
    ####################################################################################

    def enter(self, name: str) -> Profiler:
        """
        Enter a profiling section and start timing. Creates a new section if it doesn't
        exist, or reuses an existing one. Supports nested profiling by maintaining a
        section hierarchy.
        """

        if not self.enabled:
            return self

        # Determine where to store this section (top-level or nested)
        section_container = (
            self.sections if not self.stack else self.stack[-1].section.nested_sections
        )

        # Get or create the profiling section
        section = section_container.get(name, None)
        if section is None:
            section_container[name] = section = ProfilerSection(name)

        # Record start time and push to stack
        start_time = time.perf_counter_ns()
        self.stack.append(ProfilerStackEntry(section, start_time))
        return self

    def exit(self) -> float | None:
        """
        Exit the current profiling section and update statistics. Calculates elapsed
        time and updates running statistics using Welford's algorithm for numerically
        stable variance computation.
        """

        if not self.enabled:
            return None

        # Check if there's an active section to exit.
        assert self.stack, "No active profiling section to exit"

        # Calculate elapsed time.
        end_time = time.perf_counter_ns()
        section_entry = self.stack.pop()
        duration = 1e-6 * (end_time - section_entry.start_time)  # ns -> ms

        # Update section statistics.
        self._add_measurement(section_entry.section, duration)

        # Done!
        return duration

    def record(self, name: str, duration_ms: float) -> ProfilerSection | None:
        """
        Record a custom duration for a profiling section. This is useful when
        profiling code that doesn't use the profiler's enter/exit methods.
        """

        if not self.enabled:
            return None

        # Determine where to store this section (top-level or nested)
        section_container = (
            self.sections if not self.stack else self.stack[-1].section.nested_sections
        )

        # Get or create the profiling section
        section = section_container.get(name, None)
        if section is None:
            section_container[name] = section = ProfilerSection(name)

        # Update section statistics.
        self._add_measurement(section, duration_ms)

        # Done!
        return section

    def _add_measurement(self, section: ProfilerSection, duration_ms: float) -> None:
        """
        Update the given section statistics with a new measurement.
        """

        section.last = duration_ms
        section.total += duration_ms
        section.count += 1

        # Update running mean and variance using Welford's algorithm
        # This provides numerically stable computation of variance.
        delta = duration_ms - section.mean
        section.mean += delta / section.count
        delta2 = duration_ms - section.mean
        section.m2 += delta * delta2

        # Store execution history if bookkeeping is enabled.
        if self.record_individual_samples and section.history is not None:
            section.history.append(duration_ms)

    def reset(self) -> None:
        """
        Reset all profiling statistics and clear the active stack. Removes all collected
        timing data and resets the profiler to its initial state.
        """
        self.stack.clear()
        self.sections.clear()

    @property
    def current_section(self) -> ProfilerSection | None:
        """
        Get the current active profiling section. Returns None if no section is active.
        """
        return self.stack[-1].section if self.stack else None

    def to_dict(
        self, prefix: str = "", separator: str = "/", extended: bool = False
    ) -> dict[str, Any]:
        """
        Generate a flat dictionary summary of all profiling statistics.
        """
        timings = {}
        for section in self.sections.values():
            section_timings = section.to_dict(
                prefix=prefix,
                separator=separator,
                extended=extended,
            )
            timings.update(section_timings)
        return timings

    def print_summary(self, pattern: str | Iterable[str] | None = None) -> None:
        """
        Prints a table summarizing all profiling statistics. If a pattern is provided,
        only sections matching the pattern will be included in the summary. The pattern
        can use glob-style wildcards (* and ?).
        """

        summary = self.to_dict(extended=True)

        # Filter sections by pattern, if provided.
        if pattern is not None:
            if isinstance(pattern, str):
                pattern = (pattern,)
            summary = {
                k: v
                for k, v in summary.items()
                if any(fnmatch.fnmatchcase(k, p) for p in pattern)
            }

        # Print the summary table.
        headers = [
            "Section",
            "Last (ms)",
            "Total (ms)",
            "Total (%)",
            "Count",
            "Mean (ms)",
            "Std.Dev (ms)",
            "Info",
        ]
        rows = []
        for name, timings in summary.items():
            rows.append(
                [
                    name,
                    timings["last"],
                    timings["total"],
                    timings["pct"],
                    timings["count"],
                    timings["mean"],
                    timings["stddev"],
                    ", ".join(
                        [f"{key}={value}" for key, value in timings["info"].items()]
                    ),
                ]
            )
        tablefmt = "fancy_outline" if _stdout_supports_fancy_outline() else "simple"
        print(tabulate(rows, headers, tablefmt=tablefmt))

    ####################################################################################
    # Other operators
    ####################################################################################

    def __getitem__(self, key: str) -> ProfilerSection:
        """Get a profiling section by name."""
        section = self.sections.get(key, None)
        if section is None:
            raise KeyError(f"Section '{key}' not found")
        return section

    def __contains__(self, key: str) -> bool:
        """Check if a profiling section exists by name."""
        return key in self.sections

    def __enter__(self) -> Profiler:
        """Context manager entry, returns self."""
        return self

    def __exit__(self, *ignored) -> bool:
        """Context manager exit, equivalent to calling exit()."""
        self.exit()
        return False  # Propagate exceptions
