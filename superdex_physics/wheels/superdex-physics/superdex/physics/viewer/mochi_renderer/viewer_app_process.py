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

"""
Discovery and lifecycle management for the ``mochi_viewer_app`` server process.

Lets :class:`MochiRendererViewer` bring up its own renderer instead of requiring
the user to start one by hand, mirroring how ``mochi.debugger.attach()`` spawns
``mochi_debugger``.
"""

from __future__ import annotations

import importlib
import logging
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

from superdex.physics.environment import (
    get_env_var_value,
    LEGACY_VIEWER_APP_PATH_ENV_VAR,
    VIEWER_APP_PATH_ENV_VAR,
)
from superdex.physics.loader import selected_native_module_name

logger: logging.Logger = logging.getLogger(__name__)

########################################################################################

_PORT_FILE_POLL_INTERVAL_SEC = 0.05
"""Delay between polls while waiting for the child to publish its port."""

_TERMINATE_TIMEOUT_SEC = 5.0
"""Grace period given to a terminating child before it is killed."""


def _viewer_app_exe_name() -> str:
    return (
        "mochi_viewer_app.exe" if platform.system() == "Windows" else "mochi_viewer_app"
    )


def _loaded_mochi_extension_module() -> ModuleType | None:
    """Return the loaded precision-correct ``mochi_physics`` native extension, if any.

    A CMake build emits ``mochi_viewer_app`` into the same ``bin/`` directory as the
    extension module, so the extension's on-disk location anchors the executable
    search without relying on any environment variable.
    """
    return sys.modules.get(selected_native_module_name("mochi_physics"))


def _superdex_physics_package_root() -> Path | None:
    """Directory of the installed ``superdex.physics`` package, which ships ``bin/`` resources."""
    package_file = getattr(
        importlib.import_module("superdex.physics"), "__file__", None
    )
    return Path(package_file).parent if package_file else None


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []

    def add_path_or_directory(path: Path) -> None:
        exe_name = _viewer_app_exe_name()
        paths.extend([path, path / exe_name, path / "bin" / exe_name])

    # Explicit override always wins.
    override = get_env_var_value(
        VIEWER_APP_PATH_ENV_VAR, LEGACY_VIEWER_APP_PATH_ENV_VAR
    )
    if override:
        paths.append(Path(override))

    # Next to the loaded native extension (covers CMake: the executable and the
    # mochi_physics extension share one bin/ output directory).
    module_file = getattr(_loaded_mochi_extension_module(), "__file__", None)
    if module_file:
        add_path_or_directory(Path(module_file).parent)

    # Next to the installed superdex.physics package (covers Buck/packaged builds,
    # which ship mochi_viewer_app as a superdex/physics/bin resource).
    package_root = _superdex_physics_package_root()
    if package_root is not None:
        add_path_or_directory(package_root)

        # Buck link-trees may expose the resource as a top-level directory named
        # after the target instead of materializing it under the package.
        bundle_root = package_root.parent.parent
        add_path_or_directory(
            bundle_root / "superdex_physics_bin_resource" / "superdex" / "physics"
        )
        add_path_or_directory(bundle_root / "superdex_physics_bin_resource")

    # Fall back to the current working directory.
    add_path_or_directory(Path.cwd())

    return paths


def find_viewer_app_executable() -> Path | None:
    """Locate the ``mochi_viewer_app`` executable, or None if it is not installed."""
    seen: set[Path] = set()
    for path in _candidate_paths():
        try:
            resolved = path.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


########################################################################################


class ViewerAppProcess:
    """A spawned ``mochi_viewer_app`` server process.

    The child is started on an OS-chosen port and publishes the port it actually
    bound to, so concurrent viewers never collide.
    """

    _process: subprocess.Popen[bytes]
    _port: int
    _work_dir: Path
    _log_path: Path

    def __init__(
        self,
        executable: Path,
        size: tuple[int, int],
        windowed: bool,
        launch_timeout: float,
    ) -> None:
        """Spawn the viewer app and wait until it reports its listening port.

        Args:
            executable: Path to the ``mochi_viewer_app`` binary.
            size: Window width and height in pixels.
            windowed: Whether to open an interactive window.
            launch_timeout: Seconds to wait for the child to publish its port.

        Raises:
            RuntimeError: If the child exits or fails to report a port in time.
        """
        width, height = size
        self._work_dir = Path(tempfile.mkdtemp(prefix="mochi_viewer_app_"))
        port_path = self._work_dir / "port"
        self._log_path = self._work_dir / "log.txt"

        command = [
            str(executable),
            "--port",
            "0",
            "--port-file",
            str(port_path),
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        if windowed:
            command.append("--window")

        logger.info(f"Launching mochi_viewer_app: {' '.join(command)}")
        self._process = self._popen_detached(command)
        try:
            self._port = self._wait_for_port(port_path, launch_timeout)
        except BaseException:
            self.terminate()
            raise
        logger.info(f"mochi_viewer_app listening on port {self._port}.")

    ####################################################################################

    @property
    def port(self) -> int:
        """The port the viewer app is listening on."""
        return self._port

    def is_running(self) -> bool:
        """Whether the child process is still alive."""
        return self._process.poll() is None

    def terminate(self) -> None:
        """Stop the viewer app and remove its work directory.

        Escalates to a kill if the child does not exit on its own."""
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=_TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                logger.warning("mochi_viewer_app did not exit; killing it.")
                self._process.kill()
                self._process.wait()
        shutil.rmtree(self._work_dir, ignore_errors=True)

    ####################################################################################

    def _popen_detached(self, command: list[str]) -> subprocess.Popen[bytes]:
        """Start the child detached from this process's console and signals.

        Output goes to a log file rather than DEVNULL so startup failures stay
        diagnosable, and to a file rather than a pipe so the child can never block
        on a full pipe buffer.
        """
        log_file = open(self._log_path, "wb")
        try:
            if sys.platform == "win32":
                return subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                    ),
                )
            return subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()

    def _wait_for_port(self, port_path: Path, timeout: float) -> int:
        """Poll for the port file the child renames into place once bound."""
        deadline = time.monotonic() + timeout
        while True:
            if port_path.is_file():
                text = port_path.read_text().strip()
                if text:
                    try:
                        return int(text)
                    except ValueError as e:
                        raise RuntimeError(
                            f"mochi_viewer_app wrote a malformed port file: {text!r}"
                        ) from e

            exit_code = self._process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"mochi_viewer_app exited with code {exit_code} before "
                    f"reporting a port.{self._log_tail()}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"mochi_viewer_app did not report a port within {timeout}s."
                    f"{self._log_tail()}"
                )
            time.sleep(_PORT_FILE_POLL_INTERVAL_SEC)

    def _log_tail(self) -> str:
        """Child output, for inclusion in a failure message."""
        try:
            output = self._log_path.read_text(errors="replace").strip()
        except OSError:
            return ""
        return f" Output:\n{output}" if output else ""
