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

import atexit
import importlib
import logging
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType, ModuleType

from superdex.physics.environment import (
    DEBUGGER_PATH_ENV_VAR,
    get_env_var_value,
    LEGACY_DEBUGGER_PATH_ENV_VAR,
)
from superdex.physics.loader import native_roots, precision_variant_for_module

logger: logging.Logger = logging.getLogger(__name__)

# The debugger ships as its own distribution because it carries a desktop GUI. `superdex`
# depends on it, so a full install always has it.
_DEBUGGER_PACKAGE = "superdex_physics_debugger"


class _SigintDebugServerStopper:
    """Stops the Mochi debug server on Ctrl+C, even during a paused native step.

    ``Scene.step`` releases the GIL while the debugger holds a scene paused, and
    the native pause loop has no signal awareness, so the Python-level SIGINT
    handler cannot run until that call returns. The only way to break the pause
    is to stop the debug server from another thread, which disconnects the client
    and unpauses every scene so ``Scene.step`` returns.

    To notice SIGINT while the main thread is blocked, this installs a SIGINT
    handler plus ``signal.set_wakeup_fd`` on a socketpair and runs a daemon
    thread: the interpreter writes the signal number to the socket, and the
    watcher thread stops the server without waiting for the main thread. Once the
    step returns, the main-thread handler runs and performs the usual exit.

    The server is stopped through a single guarded path (:meth:`_stop_server`), so
    the watcher thread and the main-thread handler never stop it twice for one
    interrupt, while a later interrupt can still stop a restarted server.
    """

    def __init__(self) -> None:
        self._read_socket, self._write_socket = socket.socketpair()
        self._write_socket.setblocking(False)
        self._stopped = threading.Event()
        self._server_stop_lock = threading.Lock()
        self._previous_wakeup_fd = -1
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
        # Bind the handler once; a fresh bound method is created on every
        # attribute access, so restoring relies on comparing this same object.
        self._sigint_handler = self._handle_sigint
        self._thread: threading.Thread | None = None

        try:
            self._previous_wakeup_fd = signal.set_wakeup_fd(
                self._write_socket.fileno(), warn_on_full_buffer=False
            )
            if self._previous_wakeup_fd != -1:
                # Another component (e.g. an asyncio loop) already owns the wakeup
                # fd. There is no peek API, so we briefly held it; restore it in
                # the handler below and leave its signal delivery intact.
                raise RuntimeError("A signal wakeup fd is already installed.")
            signal.signal(signal.SIGINT, self._sigint_handler)
        except (AttributeError, OSError, RuntimeError, ValueError):
            self._restore_wakeup_fd()
            self._close_sockets()
            raise

        self._thread = threading.Thread(
            target=self._watch,
            name="MochiDebuggerSigintStopper",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Uninstall the handler and wakeup fd and join the watcher thread.

        Only ever called on the main thread (via the atexit hook or tests).
        """

        if self._stopped.is_set():
            return
        # Restore prior signal state before signalling the watcher to exit, so a
        # SIGINT racing teardown is handled by the previous handler, not ours.
        self._restore_sigint_handler()
        self._restore_wakeup_fd()
        self._stopped.set()
        # Wake the watcher's blocking recv so it can observe ``_stopped``.
        try:
            self._write_socket.send(b"\0")
        except OSError:
            pass
        thread = self._thread
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=1.0)
        self._close_sockets()

    def _watch(self) -> None:
        while not self._stopped.is_set():
            try:
                data = self._read_socket.recv(4096)
            except OSError:
                return
            if not data or self._stopped.is_set():
                return
            if signal.SIGINT in data:
                self._stop_server()

    def _handle_sigint(self, signum: int, frame: FrameType | None) -> None:
        # Runs on the main thread once it is free to service signals. The watcher
        # thread covers the case where the main thread is blocked in a paused step.
        self._stop_server()
        previous_handler = self._previous_sigint_handler
        if previous_handler is signal.SIG_IGN:
            return
        if (
            callable(previous_handler)
            and previous_handler is not signal.default_int_handler
        ):
            previous_handler(signum, frame)
            return
        # SIG_DFL, default_int_handler, or a handler installed from non-Python
        # code (reported as None): exit with the conventional 128 + signum status.
        raise SystemExit(128 + signum)

    def _stop_server(self) -> None:
        """Stop the debug server once per interrupt, from whichever thread wins.

        Uses a non-blocking lock so the watcher thread and the main-thread
        handler never both stop the server for one signal, and so a reentrant
        SIGINT on the main thread cannot self-deadlock. The ``has_started()``
        guard in :func:`_stop_debug_server` makes a later interrupt able to stop
        a freshly restarted server.
        """

        if not self._server_stop_lock.acquire(blocking=False):
            return
        try:
            _stop_debug_server()
        finally:
            self._server_stop_lock.release()

    def _restore_sigint_handler(self) -> None:
        try:
            if signal.getsignal(signal.SIGINT) is not self._sigint_handler:
                # Someone replaced our handler after us; leave theirs in place.
                return
            previous_handler = self._previous_sigint_handler
            if previous_handler is None:
                # The prior handler was installed from non-Python code and cannot
                # be reinstalled; fall back to the default so ours is removed.
                previous_handler = signal.SIG_DFL
            signal.signal(signal.SIGINT, previous_handler)
        except (OSError, ValueError):
            pass

    def _restore_wakeup_fd(self) -> None:
        try:
            signal.set_wakeup_fd(self._previous_wakeup_fd, warn_on_full_buffer=False)
        except (AttributeError, OSError, ValueError):
            pass

    def _close_sockets(self) -> None:
        for sock in (self._read_socket, self._write_socket):
            try:
                sock.close()
            except OSError:
                pass


_sigint_stopper_lock = threading.Lock()
_sigint_stopper: _SigintDebugServerStopper | None = None
_sigint_stopper_atexit_registered = False


def _stop_debug_server() -> None:
    try:
        physics = importlib.import_module("superdex.physics")
        server = physics.get_debug_server()
        if server.has_started():
            server.stop()
    except Exception:
        # Runs on a signal path / watcher thread where propagating would be worse
        # than logging; there is no meaningful recovery beyond reporting.
        logger.exception("Could not stop Mochi debug server after Ctrl+C.")


def _ensure_sigint_stops_debug_server() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        return

    global _sigint_stopper, _sigint_stopper_atexit_registered
    with _sigint_stopper_lock:
        if _sigint_stopper is not None:
            return
        try:
            _sigint_stopper = _SigintDebugServerStopper()
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            logger.debug("Could not install Mochi Ctrl+C watcher: %s", error)
            return
        if not _sigint_stopper_atexit_registered:
            atexit.register(_stop_sigint_stopper)
            _sigint_stopper_atexit_registered = True


def _stop_sigint_stopper() -> None:
    global _sigint_stopper
    with _sigint_stopper_lock:
        stopper = _sigint_stopper
        _sigint_stopper = None
    if stopper is not None:
        stopper.stop()


def _executable_name(stem: str) -> str:
    return f"{stem}.exe" if platform.system() == "Windows" else stem


def _debugger_exe_name() -> str:
    """Name a CMake build emits for the selected precision."""
    return _executable_name(precision_variant_for_module("mochi_debugger"))


def _debugger_exe_names() -> tuple[str, ...]:
    """Executable names to look for, most specific first.

    CMake suffixes the FP64 build so one wheel can carry both variants. A Buck
    link tree only ever stages one debugger, under the unsuffixed name.
    """
    precision_name = _debugger_exe_name()
    plain_name = _executable_name("mochi_debugger")
    if precision_name == plain_name:
        return (precision_name,)
    return (precision_name, plain_name)


def _candidate_dirs() -> list[Path]:
    """Locations that may hold the debugger, most preferred first."""
    dirs: list[Path] = []

    # Explicit override always wins.
    debugger_path = get_env_var_value(
        DEBUGGER_PATH_ENV_VAR, LEGACY_DEBUGGER_PATH_ENV_VAR
    )
    if debugger_path:
        dirs.append(Path(debugger_path))

    # Next to the loaded native extension (covers CMake: the executable and the
    # mochi_physics extension share one bin/ output directory).
    module_file = getattr(_physics_extension(), "__file__", None)
    if module_file:
        dirs.append(Path(module_file).parent)

    # The installed `superdex-physics-debugger` distribution. Without this, a wheel install
    # reports "Could not find mochi_debugger" even though the distribution is always there.
    dirs.extend(native_roots(_DEBUGGER_PACKAGE))

    # Next to the installed mochi package (covers Buck/packaged builds, which ship
    # mochi_debugger as a mochi/bin resource beside this module).
    package_dir = Path(__file__).parent
    dirs.append(package_dir)

    # Buck link-trees may still stage the debugger under the legacy mochi/bin layout.
    bundle_root = package_dir.parent.parent
    dirs.append(bundle_root / "mochi")
    dirs.append(bundle_root / "mochi_debugger_resource" / "mochi")
    dirs.append(bundle_root / "superdex_physics_bin_resource" / "superdex" / "physics")

    # Buck link-trees may expose resources as top-level symlinks named after the target
    # instead of materializing them under the package directory.
    dirs.append(bundle_root / "superdex_physics_bin_resource")
    dirs.append(bundle_root / "mochi_debugger_resource")

    # Fall back to the current working directory.
    dirs.append(Path.cwd())

    return dirs


def _physics_extension() -> ModuleType:
    """Return the native extension the ``superdex.physics`` facade imported.

    A CMake build emits ``mochi_debugger`` into the extension module's ``bin/`` directory,
    so the extension's location anchors the executable search. Imported lazily because
    ``superdex.physics`` reaches this module through its own lazy-import table.
    """
    return importlib.import_module("superdex.physics")._extension


def _candidate_paths() -> list[Path]:
    dirs = _candidate_dirs()

    # A candidate may already be the executable itself; the env override usually is.
    paths: list[Path] = list(dirs)

    # Exhaust the precision-specific name everywhere before the unsuffixed fallback, so a
    # location staging only the plain binary cannot beat a later one with the right build.
    for exe_name in _debugger_exe_names():
        for path in dirs:
            paths.extend([path / exe_name, path / "bin" / exe_name])

    return paths


def _find_debugger_executable() -> Path | None:
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


def is_attached() -> bool:
    """Return whether a Mochi Debugger client is connected."""

    physics = importlib.import_module("superdex.physics")
    return bool(physics.get_debug_server().has_connection())


def _launch_debugger_detached(
    debugger_path: Path, port: int
) -> subprocess.Popen[bytes]:
    """Launch mochi_debugger so it runs independently of this Python process.

    Detaches the child from the parent's console and process group so a Ctrl+C in
    the parent shell is not propagated to the debugger, and silences the child's
    output so its logs do not interleave with the Python program's output. The
    debugger is allowed to outlive the Python process once a connection is made.
    """

    command = [str(debugger_path), "--connect", str(port), "--singleton"]
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            ),
        )
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def attach(timeout_seconds: float = 5.0) -> bool:
    """Launch or focus Mochi Debugger and wait for it to connect to this process.

    Args:
        timeout_seconds: Maximum seconds to wait for the debugger to connect.

    Returns:
        True if the debugger is already attached or connects before the timeout.
    """

    if is_attached():
        _ensure_sigint_stops_debug_server()
        return True

    # Attempt to locate the debugger executable.
    debugger_path = _find_debugger_executable()
    if debugger_path is None:
        logger.error("Could not find %s", _debugger_exe_name())
        return False

    # If a debug server is not already running, then start it now.
    physics = importlib.import_module("superdex.physics")
    server = physics.get_debug_server()
    started_server = False
    if not server.has_started():
        server.start(0)  # Zero means "pick a free port"
        started_server = True

    # Only stop the server on failure if we were the ones that started it.
    def _stop_if_started() -> None:
        if started_server:
            server.stop()

    port = server.get_port()
    if port == 0:
        logger.error("Could not start Mochi debug server on a TCP port.")
        _stop_if_started()
        return False

    # Launch or focus mochi_debugger and tell it to connect to our server.
    try:
        process = _launch_debugger_detached(debugger_path, port)
    except OSError as error:
        logger.error("Could not launch Mochi debugger: %s", error)
        _stop_if_started()
        return False

    # Track the launched process until a debugger connects. With --singleton the
    # process may exit early after forwarding its arguments to an already-running
    # instance, which then connects instead; that early exit is expected.
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_attached():
            # Connected. Stop tracking the process; it may outlive this process.
            _ensure_sigint_stops_debug_server()
            return True
        time.sleep(0.05)

    logger.error("Timed out waiting for Mochi debugger to connect.")
    # Clean up the process we started if it is still running. It may have already
    # terminated itself, which is fine.
    if process.poll() is None:
        process.terminate()
    _stop_if_started()
    return False
