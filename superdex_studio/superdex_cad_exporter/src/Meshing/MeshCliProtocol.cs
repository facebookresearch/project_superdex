/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Meshing
{
    /// <summary>
    /// Operations the superdex_mesh_cli helper can dispatch.
    /// </summary>
    internal enum GeometryOp : uint
    {
        Ping = 0,

        /// <summary>
        /// Write a STEP out as one or more render-ready mesh files. The STEP is loaded and
        /// tessellated once for the whole request.
        /// </summary>
        ExportStepVisual = 10,
    }

    /// <summary>
    /// Thrown when the helper's response cannot be parsed, or when it reports a failure.
    /// </summary>
    public class MeshCliException : Exception
    {
        public MeshCliException(string message) : base(message) { }
        public MeshCliException(string message, Exception inner) : base(message, inner) { }
    }

    /// <summary>
    /// Binary framing and payload encoding for the superdex_mesh_cli wire protocol.
    /// </summary>
    internal static class MeshCliProtocol
    {
        /// <summary>Frame magic, "SDX2" as a little-endian u32.</summary>
        public const uint FrameMagic = 0x53445832u;

        /// <summary>Wire protocol version. Requests carry it; responses do not.</summary>
        public const uint ProtocolVersion = 3u;

        /// <summary>
        /// Wraps <paramref name="payload"/> in a request frame:
        /// [u32 magic][u32 version][u32 opcode][u64 payloadLen][payload].
        /// </summary>
        public static byte[] EncodeRequestFrame(GeometryOp op, byte[] payload)
        {
            // MemoryStream.ToArray works after the stream is closed, so the writer can own it
            // outright rather than nesting two disposals over the same object.
            var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                writer.Write(FrameMagic);
                writer.Write(ProtocolVersion);
                writer.Write((uint)op);
                writer.Write((ulong)payload.Length);
                writer.Write(payload);
            }
            return stream.ToArray();
        }

        /// <summary>
        /// Parses a response frame: [u32 magic][u32 status][u64 payloadLen][payload].
        /// A status of 0 means success and the payload is the result; anything else means failure
        /// and the payload is the whole UTF-8 error message.
        /// </summary>
        public static void DecodeResponseFrame(byte[] frame, out uint status, out byte[] payload)
        {
            const int headerSize = sizeof(uint) + sizeof(uint) + sizeof(ulong);
            if (frame == null || frame.Length < headerSize)
            {
                throw new MeshCliException(
                    "superdex_mesh_cli returned a truncated or empty response.");
            }

            using (var reader =
                new BinaryReader(new MemoryStream(frame, writable: false), Encoding.UTF8))
            {
                uint magic = reader.ReadUInt32();
                if (magic != FrameMagic)
                {
                    throw new MeshCliException(
                        $"superdex_mesh_cli response has bad frame magic 0x{magic:X8}.");
                }
                status = reader.ReadUInt32();
                ulong payloadLen = reader.ReadUInt64();

                // Exact-size comparison rejects both truncation and trailing bytes.
                if (payloadLen != (ulong)(frame.Length - headerSize))
                {
                    throw new MeshCliException(
                        "superdex_mesh_cli response payload length does not match the frame.");
                }
                payload = reader.ReadBytes((int)payloadLen);
            }
        }

        /// <summary>Serializes payload values. Mirrors the C++ PayloadWriter field for field.</summary>
        public sealed class PayloadWriter : IDisposable
        {
            private readonly MemoryStream _stream = new MemoryStream();
            private readonly BinaryWriter _writer;

            public PayloadWriter()
            {
                _writer = new BinaryWriter(_stream, Encoding.UTF8, leaveOpen: true);
            }

            public void WriteU32(uint value) => _writer.Write(value);

            public void WriteU64(ulong value) => _writer.Write(value);

            public void WriteInt32(int value) => _writer.Write(value);

            /// <summary>A bool travels as a u32 of exactly 0 or 1 -- four bytes, not one.</summary>
            public void WriteBool(bool value) => _writer.Write(value ? 1u : 0u);

            public void WriteDouble(double value) => _writer.Write(value);

            /// <summary>[u64 byteCount][bytes]. No NUL terminator.</summary>
            public void WriteByteArray(byte[] bytes)
            {
                _writer.Write((ulong)bytes.Length);
                _writer.Write(bytes);
            }

            /// <summary>
            /// Strings travel as UTF-8 byte arrays. The helper hands the bytes straight to
            /// OpenCascade, so this is what makes non-ASCII paths work.
            /// </summary>
            public void WriteString(string value) =>
                WriteByteArray(Encoding.UTF8.GetBytes(value ?? string.Empty));

            public byte[] ToArray()
            {
                _writer.Flush();
                return _stream.ToArray();
            }

            public void Dispose()
            {
                _writer.Dispose();
                _stream.Dispose();
            }
        }

        /// <summary>Reads payload values. Every read is bounds-checked.</summary>
        public sealed class PayloadReader : IDisposable
        {
            private readonly MemoryStream _stream;
            private readonly BinaryReader _reader;

            public PayloadReader(byte[] payload)
            {
                _stream = new MemoryStream(payload, writable: false);
                _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);
            }

            public bool AtEnd => _stream.Position == _stream.Length;

            public uint ReadU32()
            {
                Require(sizeof(uint));
                return _reader.ReadUInt32();
            }

            public ulong ReadU64()
            {
                Require(sizeof(ulong));
                return _reader.ReadUInt64();
            }

            public bool ReadBool()
            {
                uint raw = ReadU32();
                if (raw > 1u)
                {
                    throw new MeshCliException(
                        $"superdex_mesh_cli response contains {raw} where a bool was expected.");
                }
                return raw != 0u;
            }

            public double ReadDouble()
            {
                Require(sizeof(double));
                return _reader.ReadDouble();
            }

            /// <summary>Reads [u64 count][count x u32].</summary>
            public List<uint> ReadU32Array()
            {
                ulong count = ReadU64();
                long remaining = _stream.Length - _stream.Position;
                if (count > (ulong)(remaining / sizeof(uint)))
                {
                    throw new MeshCliException(
                        "superdex_mesh_cli response declares more array elements than it contains.");
                }
                var values = new List<uint>((int)count);
                for (ulong i = 0; i < count; ++i)
                {
                    values.Add(_reader.ReadUInt32());
                }
                return values;
            }

            /// <summary>
            /// Confirms the payload is fully consumed. The C++ side requires this of every decode
            /// path, so trailing bytes are a protocol mismatch rather than something to ignore.
            /// </summary>
            public void ExpectAtEnd()
            {
                if (!AtEnd)
                {
                    throw new MeshCliException(
                        "superdex_mesh_cli response has unexpected trailing bytes.");
                }
            }

            private void Require(int byteCount)
            {
                if (_stream.Length - _stream.Position < byteCount)
                {
                    throw new MeshCliException(
                        "superdex_mesh_cli response ended before all expected values were read.");
                }
            }

            public void Dispose()
            {
                _reader.Dispose();
                _stream.Dispose();
            }
        }
    }
}
