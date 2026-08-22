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
TCP client for communicating with the mochi_renderer server.

Implements the binary wire protocol for sending batched commands and
receiving text/image responses over a persistent TCP connection.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

########################################################################################


@dataclass
class CommandEntry:
    """A single command in a batch request."""

    text: str
    binary_data: bytes = b""


@dataclass
class ResponseEntry:
    """A single response from a batch request."""

    type: int  # 0 = text, 1 = image
    data: bytes = b""


########################################################################################


class MochiRendererClient:
    """
    TCP client for the mochi_renderer server.

    Implements the binary wire protocol:
      Request:  [uint32 payload_len] [payload]
        payload: [uint32 num_commands]
                 per command: [uint32 text_len] [text] [uint32 data_len] [data]
      Response: [uint32 payload_len] [payload]
        payload: [uint32 num_responses]
                 per response: [uint8 type] [uint32 data_len] [data]

    Thread-safe: all socket operations are serialized through a lock.
    """

    def __init__(self, host: str = "localhost", port: int = 9000) -> None:
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Connection management
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 5.0, retry_attempts: int = 3) -> bool:
        """
        Connect to the mochi_renderer server.

        Args:
            timeout: Socket timeout in seconds for the connection attempt.
            retry_attempts: Number of times to retry on failure.

        Returns:
            True if the connection was established, False otherwise.
        """
        if self._socket is not None:
            self.disconnect()

        for attempt in range(retry_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((self._host, self._port))
                # After connecting, use blocking mode with no timeout for I/O
                sock.settimeout(None)
                with self._lock:
                    self._socket = sock
                logger.info(f"Connected to mochi_renderer at {self._host}:{self._port}")
                return True
            except OSError as e:
                logger.warning(
                    f"Connection attempt {attempt + 1}/{retry_attempts} failed: {e}"
                )
                try:
                    sock.close()
                except Exception:
                    pass
                if attempt < retry_attempts - 1:
                    backoff = min(2**attempt, 8)
                    logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
        logger.error(
            f"Failed to connect to mochi_renderer at "
            f"{self._host}:{self._port} after {retry_attempts} attempts"
        )
        return False

    def disconnect(self) -> None:
        """Disconnect from the server. Safe to call if already disconnected."""
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
                logger.info("Disconnected from mochi_renderer")

    def is_connected(self) -> bool:
        """Return whether the client currently holds an open connection."""
        with self._lock:
            return self._socket is not None

    # ──────────────────────────────────────────────────────────────────────────
    # Public request API
    # ──────────────────────────────────────────────────────────────────────────

    def request(self, command: str, binary_data: bytes | None = None) -> bytes:
        """
        Send a single command and return the raw response data.

        This is a convenience wrapper around :meth:`request_batch` for the
        common single-command case.

        Args:
            command: The command text (e.g. ``"vset /object/box/show"``).
            binary_data: Optional binary payload attached to the command.

        Returns:
            The ``data`` field of the first :class:`ResponseEntry`.

        Raises:
            ConnectionError: If not connected or the server closes the
                connection unexpectedly.
            RuntimeError: If the server returns an unexpected number of
                responses.
        """
        entry = CommandEntry(text=command, binary_data=binary_data or b"")
        responses = self.request_batch([entry])
        if not responses:
            raise RuntimeError("Server returned zero responses for a single command")
        return responses[0].data

    def request_batch(self, commands: list[CommandEntry]) -> list[ResponseEntry]:
        """
        Send a batch of commands and return all responses.

        Args:
            commands: List of :class:`CommandEntry` objects to send.

        Returns:
            List of :class:`ResponseEntry` objects, one per server response.

        Raises:
            ConnectionError: If not connected or the connection is lost.
        """
        with self._lock:
            sock = self._socket
            if sock is None:
                raise ConnectionError("Not connected to mochi_renderer")
            payload = self._encode_request(commands)
            self._send_framed(sock, payload)
            response_payload = self._recv_framed(sock)
            return self._decode_response(response_payload)

    # ──────────────────────────────────────────────────────────────────────────
    # Wire protocol encoding / decoding
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_request(commands: list[CommandEntry]) -> bytes:
        """
        Encode a list of commands into the request payload.

        Layout:
            [uint32 num_commands]
            per command:
                [uint32 text_length] [UTF-8 text]
                [uint32 data_length] [binary data]
        """
        parts: list[bytes] = []
        parts.append(struct.pack("<I", len(commands)))
        for cmd in commands:
            text_bytes = cmd.text.encode("utf-8")
            parts.append(struct.pack("<I", len(text_bytes)))
            parts.append(text_bytes)
            parts.append(struct.pack("<I", len(cmd.binary_data)))
            if cmd.binary_data:
                parts.append(cmd.binary_data)
        return b"".join(parts)

    @staticmethod
    def _decode_response(payload: bytes) -> list[ResponseEntry]:
        """
        Decode a response payload into a list of ResponseEntry objects.

        Layout:
            [uint32 num_responses]
            per response:
                [uint8 type]
                [uint32 data_length] [data]
        """
        offset = 0
        (num_responses,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        entries: list[ResponseEntry] = []
        for _ in range(num_responses):
            (resp_type,) = struct.unpack_from("<B", payload, offset)
            offset += 1
            (data_len,) = struct.unpack_from("<I", payload, offset)
            offset += 4
            data = payload[offset : offset + data_len]
            offset += data_len
            entries.append(ResponseEntry(type=resp_type, data=data))
        return entries

    # ──────────────────────────────────────────────────────────────────────────
    # Low-level framed I/O
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _send_framed(sock: socket.socket, payload: bytes) -> None:
        """Send a length-prefixed payload: [uint32_le length] [bytes]."""
        header = struct.pack("<I", len(payload))
        sock.sendall(header + payload)

    @staticmethod
    def _recv_framed(sock: socket.socket) -> bytes:
        """Receive a length-prefixed payload from the socket."""
        header = MochiRendererClient._recv_exact(sock, 4)
        (length,) = struct.unpack("<I", header)
        return MochiRendererClient._recv_exact(sock, length)

    @staticmethod
    def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
        """Read exactly *num_bytes* from *sock*, raising on premature EOF."""
        chunks: list[bytes] = []
        remaining = num_bytes
        while remaining > 0:
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                raise ConnectionError(
                    "Connection closed while reading from mochi_renderer"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
