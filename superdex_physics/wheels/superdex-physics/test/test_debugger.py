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

import os
import signal
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any
from unittest.mock import patch

from .module_stand_ins import load_physics_module

# Loading debugger.py by path keeps this test runnable without the native extensions.
load_physics_module("environment")
load_physics_module("loader")
debugger = load_physics_module("debugger")


def _debugger_attr(name: str) -> Any:
    return getattr(debugger, name)


class _FakeServer:
    def __init__(
        self,
        *,
        has_connection_values: list[bool] | None = None,
        port: int = 1234,
        started: bool = False,
    ) -> None:
        self.has_connection_values = (
            [False, True] if has_connection_values is None else has_connection_values
        )
        self.port = port
        self.started = started
        self.started_ports: list[int] = []
        self.stop_count = 0

    def has_connection(self) -> bool:
        return self.has_connection_values.pop(0)

    def has_started(self) -> bool:
        return self.started

    def start(self, preferred_port: int) -> None:
        self.started = True
        self.started_ports.append(preferred_port)

    def get_port(self) -> int:
        return self.port

    def stop(self) -> None:
        self.started = False
        self.stop_count += 1


class _FakePhysics:
    """Stands in for the `superdex.physics` facade the debugger imports lazily."""

    def __init__(self, server: _FakeServer) -> None:
        self._server = server
        # A bare ModuleType carries no __file__, so it contributes no candidate
        # directory to the executable search.
        self._extension = ModuleType("mochi_physics")

    def get_debug_server(self) -> _FakeServer:
        return self._server


class DebuggerTest(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot process-global signal state so no test can leak it to another.
        self._saved_sigint_handler = signal.getsignal(signal.SIGINT)
        self._saved_wakeup_fd = signal.set_wakeup_fd(-1)

    def tearDown(self) -> None:
        _debugger_attr("_stop_sigint_stopper")()
        signal.set_wakeup_fd(self._saved_wakeup_fd)
        signal.signal(signal.SIGINT, self._saved_sigint_handler)

    def test_attach_installs_ctrl_c_handler_that_exits_cleanly(self) -> None:
        server = _FakeServer(has_connection_values=[False, True], started=True)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            with patch.object(
                _debugger_attr("importlib"),
                "import_module",
                return_value=_FakePhysics(server),
            ):
                with patch.object(
                    debugger, "_find_debugger_executable", return_value=Path("debugger")
                ):
                    with patch.object(_debugger_attr("subprocess"), "Popen"):
                        self.assertTrue(debugger.attach(timeout_seconds=0.2))

                with self.assertRaises(SystemExit) as exit_context:
                    signal.raise_signal(signal.SIGINT)
                _debugger_attr("_stop_sigint_stopper")()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        self.assertEqual(130, exit_context.exception.code)
        self.assertEqual(1, server.stop_count)

    def test_sigint_handler_returns_when_previous_handler_ignored(self) -> None:
        server = _FakeServer(started=True)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            with patch.object(
                _debugger_attr("importlib"),
                "import_module",
                return_value=_FakePhysics(server),
            ):
                stopper = _debugger_attr("_SigintDebugServerStopper")()
                try:
                    stopper._handle_sigint(signal.SIGINT, None)
                finally:
                    stopper.stop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        self.assertEqual(1, server.stop_count)

    def test_sigint_handler_chains_to_custom_previous_handler(self) -> None:
        server = _FakeServer(started=True)
        received: list[int] = []

        def previous_handler(signum: int, frame: FrameType | None) -> None:
            received.append(signum)

        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, previous_handler)

        try:
            with patch.object(
                _debugger_attr("importlib"),
                "import_module",
                return_value=_FakePhysics(server),
            ):
                stopper = _debugger_attr("_SigintDebugServerStopper")()
                try:
                    stopper._handle_sigint(signal.SIGINT, None)
                finally:
                    stopper.stop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        self.assertEqual(1, server.stop_count)
        self.assertEqual([signal.SIGINT], received)

    def test_sigint_handler_exits_cleanly_when_previous_handler_is_none(self) -> None:
        # signal.getsignal returns None when the previous handler was installed
        # from non-Python code. Ctrl+C must still stop the server and exit
        # cleanly rather than silently swallowing the interrupt.
        server = _FakeServer(started=True)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            with patch.object(
                _debugger_attr("importlib"),
                "import_module",
                return_value=_FakePhysics(server),
            ):
                stopper = _debugger_attr("_SigintDebugServerStopper")()
                try:
                    stopper._previous_sigint_handler = None
                    with self.assertRaises(SystemExit) as exit_context:
                        stopper._handle_sigint(signal.SIGINT, None)
                finally:
                    stopper.stop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        self.assertEqual(130, exit_context.exception.code)
        self.assertEqual(1, server.stop_count)

    def test_server_is_stopped_only_once(self) -> None:
        # The watcher thread and the main-thread handler share one stop path, so
        # concurrent SIGINT handling must never stop the server twice.
        server = _FakeServer(started=True)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            with patch.object(
                _debugger_attr("importlib"),
                "import_module",
                return_value=_FakePhysics(server),
            ):
                stopper = _debugger_attr("_SigintDebugServerStopper")()
                try:
                    stopper._stop_server()
                    stopper._stop_server()
                finally:
                    stopper.stop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        self.assertEqual(1, server.stop_count)

    def test_watcher_thread_stops_server_on_wakeup_byte(self) -> None:
        # Simulates the interpreter reporting SIGINT through the wakeup fd while
        # the main thread is blocked (e.g. inside a paused native step). The
        # daemon watcher thread must stop the server on its own, without the
        # main-thread handler running.
        server = _FakeServer(started=True)
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            with patch.object(
                _debugger_attr("importlib"),
                "import_module",
                return_value=_FakePhysics(server),
            ):
                stopper = _debugger_attr("_SigintDebugServerStopper")()
                try:
                    stopper._write_socket.send(bytes([signal.SIGINT]))
                    deadline = time.monotonic() + 2.0
                    while server.stop_count == 0 and time.monotonic() < deadline:
                        time.sleep(0.01)
                finally:
                    stopper.stop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        self.assertEqual(1, server.stop_count)

    def test_install_bails_when_wakeup_fd_already_owned(self) -> None:
        # If another component already owns the signal wakeup fd, the stopper must
        # refuse to install and leave the pre-existing handler and fd untouched.
        read_socket, write_socket = socket.socketpair()
        write_socket.setblocking(False)
        self.addCleanup(read_socket.close)
        self.addCleanup(write_socket.close)

        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        existing_wakeup_fd = signal.set_wakeup_fd(write_socket.fileno())

        try:
            with self.assertRaises(RuntimeError):
                _debugger_attr("_SigintDebugServerStopper")()

            self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)
            self.assertEqual(write_socket.fileno(), signal.set_wakeup_fd(-1))
        finally:
            signal.set_wakeup_fd(existing_wakeup_fd)
            signal.signal(signal.SIGINT, previous_sigint_handler)

    def test_debugger_exe_name_tracks_selected_precision(self) -> None:
        with patch.object(
            debugger,
            "precision_variant_for_module",
            return_value="mochi_debugger_double",
        ):
            expected_name = (
                "mochi_debugger_double.exe"
                if debugger.platform.system() == "Windows"
                else "mochi_debugger_double"
            )
            self.assertEqual(expected_name, debugger._debugger_exe_name())

    def test_candidate_paths_prefer_superdex_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as canonical_dir:
            with tempfile.TemporaryDirectory() as legacy_dir:
                canonical_path = Path(canonical_dir) / debugger._debugger_exe_name()
                legacy_path = Path(legacy_dir) / debugger._debugger_exe_name()
                canonical_path.write_text("")
                legacy_path.write_text("")

                with patch.dict(
                    os.environ,
                    {
                        debugger.DEBUGGER_PATH_ENV_VAR: str(canonical_path),
                        debugger.LEGACY_DEBUGGER_PATH_ENV_VAR: str(legacy_path),
                    },
                    clear=True,
                ):
                    with patch.object(
                        debugger,
                        "_physics_extension",
                        return_value=ModuleType("mochi_physics"),
                    ):
                        candidates = debugger._candidate_paths()

                self.assertEqual(canonical_path, candidates[0])

    def test_find_debugger_executable_checks_package_local_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "superdex" / "physics"
            bin_dir = package_dir / "bin"
            bin_dir.mkdir(parents=True)
            debugger_path = bin_dir / debugger._debugger_exe_name()
            debugger_path.write_text("")
            debugger_path.chmod(0o755)

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    debugger, "__file__", str(package_dir / "debugger.py")
                ):
                    # A bare ModuleType has no __file__, so the extension contributes no
                    # candidate directory.
                    with patch.object(
                        debugger,
                        "_physics_extension",
                        return_value=ModuleType("mochi_physics"),
                    ):
                        self.assertEqual(
                            debugger_path.resolve(),
                            debugger._find_debugger_executable(),
                        )

    def test_find_debugger_executable_checks_buck_resource_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_root = Path(temp_dir)
            package_dir = bundle_root / "superdex" / "physics"
            package_dir.mkdir(parents=True)
            resource_dir = bundle_root / "superdex_physics_bin_resource"
            resource_dir.mkdir()
            debugger_path = resource_dir / debugger._debugger_exe_name()
            debugger_path.write_text("")
            debugger_path.chmod(0o755)

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    debugger, "__file__", str(package_dir / "debugger.py")
                ):
                    # A bare ModuleType has no __file__, so the extension contributes no
                    # candidate directory.
                    with patch.object(
                        debugger,
                        "_physics_extension",
                        return_value=ModuleType("mochi_physics"),
                    ):
                        self.assertEqual(
                            debugger_path.resolve(),
                            debugger._find_debugger_executable(),
                        )

    def test_loaded_extension_directory_beats_package_local_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extension_dir = temp_path / "extension"
            extension_dir.mkdir()
            extension_path = extension_dir / "mochi_physics.so"
            extension_path.write_text("")
            extension_debugger = extension_dir / debugger._debugger_exe_name()
            extension_debugger.write_text("")
            extension_debugger.chmod(0o755)

            package_dir = temp_path / "superdex" / "physics"
            package_bin_dir = package_dir / "bin"
            package_bin_dir.mkdir(parents=True)
            package_debugger = package_bin_dir / debugger._debugger_exe_name()
            package_debugger.write_text("")
            package_debugger.chmod(0o755)

            fake_extension = ModuleType("mochi_physics")
            fake_extension.__file__ = str(extension_path)

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(
                    debugger, "__file__", str(package_dir / "debugger.py")
                ):
                    with patch.object(
                        debugger,
                        "_physics_extension",
                        return_value=fake_extension,
                    ):
                        self.assertEqual(
                            extension_debugger.resolve(),
                            debugger._find_debugger_executable(),
                        )
