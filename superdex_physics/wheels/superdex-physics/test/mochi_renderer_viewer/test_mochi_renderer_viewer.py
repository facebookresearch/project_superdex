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
Tests for the MochiRendererClient wire protocol encoding and decoding.

Uses mock sockets to test the binary protocol without a running server.
"""

from __future__ import annotations

import struct
import unittest
from unittest.mock import MagicMock, patch

from superdex.physics.viewer.mochi_renderer.mochi_renderer_client import (
    CommandEntry,
    MochiRendererClient,
    ResponseEntry,
)

########################################################################################
# Dataclass tests
########################################################################################


class TestCommandEntry:
    """Tests for the CommandEntry dataclass."""

    def test_default_binary_data(self) -> None:
        entry = CommandEntry(text="vset /object/box/show")
        assert entry.text == "vset /object/box/show"
        assert entry.binary_data == b""

    def test_with_binary_data(self) -> None:
        data = b"\x01\x02\x03\x04"
        entry = CommandEntry(text="vset /object/box/mesh 1 3 0", binary_data=data)
        assert entry.text == "vset /object/box/mesh 1 3 0"
        assert entry.binary_data == data


class TestResponseEntry:
    """Tests for the ResponseEntry dataclass."""

    def test_text_response(self) -> None:
        entry = ResponseEntry(type=0, data=b"ok")
        assert entry.type == 0
        assert entry.data == b"ok"

    def test_image_response(self) -> None:
        pixel_data = b"\xff" * 16
        entry = ResponseEntry(type=1, data=pixel_data)
        assert entry.type == 1
        assert entry.data == pixel_data

    def test_default_data(self) -> None:
        entry = ResponseEntry(type=0)
        assert entry.data == b""


########################################################################################
# Wire protocol encoding tests
########################################################################################


class TestEncodeRequest:
    """Tests for MochiRendererClient._encode_request."""

    def test_single_command_no_binary(self) -> None:
        commands = [CommandEntry(text="vset /object/box/show")]
        payload = MochiRendererClient._encode_request(commands)

        offset = 0
        # num_commands
        (num_cmds,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert num_cmds == 1

        # text_length + text
        text_bytes = b"vset /object/box/show"
        (text_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert text_len == len(text_bytes)
        assert payload[offset : offset + text_len] == text_bytes
        offset += text_len

        # data_length (should be 0)
        (data_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert data_len == 0

        assert offset == len(payload)

    def test_single_command_with_binary(self) -> None:
        binary = struct.pack("<3f", 1.0, 2.0, 3.0)
        commands = [
            CommandEntry(text="vset /object/box/mesh 1 3 0", binary_data=binary)
        ]
        payload = MochiRendererClient._encode_request(commands)

        offset = 0
        (num_cmds,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert num_cmds == 1

        # text
        (text_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        text = payload[offset : offset + text_len]
        offset += text_len
        assert text == b"vset /object/box/mesh 1 3 0"

        # binary data
        (data_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert data_len == len(binary)
        assert payload[offset : offset + data_len] == binary
        offset += data_len

        assert offset == len(payload)

    def test_multiple_commands(self) -> None:
        commands = [
            CommandEntry(text="vset /object/a/show"),
            CommandEntry(text="vset /object/b/hide"),
            CommandEntry(text="vget /camera/default/lit"),
        ]
        payload = MochiRendererClient._encode_request(commands)

        offset = 0
        (num_cmds,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert num_cmds == 3

        # Parse each command
        for cmd in commands:
            text_bytes = cmd.text.encode("utf-8")
            (text_len,) = struct.unpack_from("<I", payload, offset)
            offset += 4
            assert text_len == len(text_bytes)
            assert payload[offset : offset + text_len] == text_bytes
            offset += text_len

            (data_len,) = struct.unpack_from("<I", payload, offset)
            offset += 4
            assert data_len == 0

        assert offset == len(payload)

    def test_empty_command_list(self) -> None:
        payload = MochiRendererClient._encode_request([])
        assert payload == struct.pack("<I", 0)

    def test_unicode_command_text(self) -> None:
        commands = [CommandEntry(text="vset /object/cafe\u0301/show")]
        payload = MochiRendererClient._encode_request(commands)

        offset = 0
        (num_cmds,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        assert num_cmds == 1

        (text_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        text = payload[offset : offset + text_len].decode("utf-8")
        assert text == "vset /object/cafe\u0301/show"


########################################################################################
# Wire protocol decoding tests
########################################################################################


class TestDecodeResponse:
    """Tests for MochiRendererClient._decode_response."""

    def test_single_text_response(self) -> None:
        text = b"ok"
        payload = struct.pack("<I", 1)  # num_responses
        payload += struct.pack("<B", 0)  # type = text
        payload += struct.pack("<I", len(text))
        payload += text

        entries = MochiRendererClient._decode_response(payload)
        assert len(entries) == 1
        assert entries[0].type == 0
        assert entries[0].data == b"ok"

    def test_single_image_response(self) -> None:
        # Simulate a 2x2 RGBA image header + pixels
        width, height, channels = 2, 2, 4
        image_header = struct.pack("<III", width, height, channels)
        pixels = bytes(range(width * height * channels))
        image_data = image_header + pixels

        payload = struct.pack("<I", 1)  # num_responses
        payload += struct.pack("<B", 1)  # type = image
        payload += struct.pack("<I", len(image_data))
        payload += image_data

        entries = MochiRendererClient._decode_response(payload)
        assert len(entries) == 1
        assert entries[0].type == 1
        assert entries[0].data == image_data

    def test_multiple_responses(self) -> None:
        payload = struct.pack("<I", 3)  # num_responses

        # Response 1: text "ok"
        payload += struct.pack("<B", 0)
        payload += struct.pack("<I", 2)
        payload += b"ok"

        # Response 2: text "ok"
        payload += struct.pack("<B", 0)
        payload += struct.pack("<I", 2)
        payload += b"ok"

        # Response 3: image (minimal)
        img = struct.pack("<III", 1, 1, 4) + b"\xff\x00\x00\xff"
        payload += struct.pack("<B", 1)
        payload += struct.pack("<I", len(img))
        payload += img

        entries = MochiRendererClient._decode_response(payload)
        assert len(entries) == 3
        assert entries[0].type == 0
        assert entries[0].data == b"ok"
        assert entries[1].type == 0
        assert entries[1].data == b"ok"
        assert entries[2].type == 1
        assert entries[2].data == img

    def test_error_text_response(self) -> None:
        error_msg = b"error: unknown command"
        payload = struct.pack("<I", 1)
        payload += struct.pack("<B", 0)
        payload += struct.pack("<I", len(error_msg))
        payload += error_msg

        entries = MochiRendererClient._decode_response(payload)
        assert len(entries) == 1
        assert entries[0].type == 0
        assert entries[0].data == error_msg

    def test_empty_response_list(self) -> None:
        payload = struct.pack("<I", 0)
        entries = MochiRendererClient._decode_response(payload)
        assert len(entries) == 0


########################################################################################
# Roundtrip tests
########################################################################################


class TestEncodeDecodeRoundtrip:
    """Test that encode and decode are consistent."""

    def test_encode_produces_valid_framing(self) -> None:
        """Verify the encoded payload can be wrapped in a frame header."""
        commands = [
            CommandEntry(text="vset /object/box/show"),
            CommandEntry(
                text="vset /object/box/mesh 4 6 0",
                binary_data=b"\x00" * 48,
            ),
        ]
        payload = MochiRendererClient._encode_request(commands)

        # Frame it
        frame = struct.pack("<I", len(payload)) + payload

        # Read back the frame
        (frame_len,) = struct.unpack_from("<I", frame, 0)
        assert frame_len == len(payload)
        assert frame[4:] == payload


########################################################################################
# Connection / mock socket tests
########################################################################################


class TestConnection:
    """Tests for connect / disconnect using mock sockets."""

    @patch("superdex.physics.viewer.mochi_renderer.mochi_renderer_client.socket.socket")
    def test_connect_success(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock

        client = MochiRendererClient("localhost", 9000)
        result = client.connect(timeout=1.0, retry_attempts=1)

        assert result is True
        assert client.is_connected()
        mock_sock.settimeout.assert_any_call(1.0)
        mock_sock.connect.assert_called_once_with(("localhost", 9000))

    @patch("superdex.physics.viewer.mochi_renderer.mochi_renderer_client.socket.socket")
    def test_connect_failure(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        mock_sock.connect.side_effect = ConnectionRefusedError("refused")

        client = MochiRendererClient("localhost", 9000)
        result = client.connect(timeout=1.0, retry_attempts=2)

        assert result is False
        assert not client.is_connected()
        assert mock_sock.connect.call_count == 2

    def test_disconnect_when_not_connected(self) -> None:
        client = MochiRendererClient()
        # Should not raise
        client.disconnect()
        assert not client.is_connected()

    @patch("superdex.physics.viewer.mochi_renderer.mochi_renderer_client.socket.socket")
    def test_disconnect_after_connect(self, mock_socket_class: MagicMock) -> None:
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock

        client = MochiRendererClient()
        client.connect(timeout=1.0, retry_attempts=1)
        assert client.is_connected()

        client.disconnect()
        assert not client.is_connected()
        mock_sock.close.assert_called_once()


########################################################################################
# request / request_batch with mock socket I/O
########################################################################################


class TestRequestBatch(unittest.TestCase):
    """Tests for request_batch using mock socket I/O."""

    def _make_connected_client(self) -> tuple[MochiRendererClient, MagicMock]:
        """Create a client with a mocked socket injected."""
        client = MochiRendererClient("localhost", 9000)
        mock_sock = MagicMock()
        client._socket = mock_sock
        return client, mock_sock

    def test_request_batch_not_connected(self) -> None:
        client = MochiRendererClient()
        with self.assertRaises(ConnectionError):
            client.request_batch([CommandEntry(text="vget /objects")])

    def test_request_single_text(self) -> None:
        client, mock_sock = self._make_connected_client()

        # Build the expected response payload
        response_text = b"ok"
        response_payload = struct.pack("<I", 1)  # num_responses
        response_payload += struct.pack("<B", 0)  # type = text
        response_payload += struct.pack("<I", len(response_text))
        response_payload += response_text

        # Frame it
        response_frame = struct.pack("<I", len(response_payload)) + response_payload

        # Mock recv to return the framed response
        mock_sock.recv.side_effect = [response_frame[:4], response_frame[4:]]

        commands = [CommandEntry(text="vget /objects")]
        results = client.request_batch(commands)

        assert len(results) == 1
        assert results[0].type == 0
        assert results[0].data == b"ok"
        mock_sock.sendall.assert_called_once()

    def test_request_convenience_method(self) -> None:
        client, mock_sock = self._make_connected_client()

        response_text = b"ok"
        response_payload = struct.pack("<I", 1)
        response_payload += struct.pack("<B", 0)
        response_payload += struct.pack("<I", len(response_text))
        response_payload += response_text
        response_frame = struct.pack("<I", len(response_payload)) + response_payload

        mock_sock.recv.side_effect = [response_frame[:4], response_frame[4:]]

        data = client.request("vset /object/box/show")
        assert data == b"ok"


########################################################################################
# Low-level I/O tests
########################################################################################


class TestRecvExact(unittest.TestCase):
    """Tests for _recv_exact."""

    def test_recv_exact_single_chunk(self) -> None:
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b"hello"
        result = MochiRendererClient._recv_exact(mock_sock, 5)
        assert result == b"hello"

    def test_recv_exact_multiple_chunks(self) -> None:
        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [b"he", b"ll", b"o"]
        result = MochiRendererClient._recv_exact(mock_sock, 5)
        assert result == b"hello"

    def test_recv_exact_connection_closed(self) -> None:
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b""
        with self.assertRaises(ConnectionError):
            MochiRendererClient._recv_exact(mock_sock, 5)

    def test_send_framed(self) -> None:
        mock_sock = MagicMock()
        payload = b"test_payload"
        MochiRendererClient._send_framed(mock_sock, payload)

        expected = struct.pack("<I", len(payload)) + payload
        mock_sock.sendall.assert_called_once_with(expected)
