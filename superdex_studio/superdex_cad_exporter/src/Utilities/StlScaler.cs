/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.IO;

namespace CADRobotExporter.Utilities
{
    public static class StlScaler
    {
        private const int HeaderSize = 80;
        private const int TriangleRecordSize = 50; // 12 floats (48 bytes) + 2 byte attribute

        public static void ScaleInPlace(string filePath, float scale)
        {
            byte[] data = File.ReadAllBytes(filePath);

            if (data.Length < HeaderSize + 4)
                throw new InvalidDataException("File too small to be a valid binary STL");

            uint triangleCount = BitConverter.ToUInt32(data, HeaderSize);
            int expectedSize = HeaderSize + 4 + (int)triangleCount * TriangleRecordSize;

            if (data.Length != expectedSize)
                throw new InvalidDataException(
                    $"STL file size mismatch: expected {expectedSize} bytes for {triangleCount} triangles, got {data.Length}");

            int offset = HeaderSize + 4;
            for (uint i = 0; i < triangleCount; i++)
            {
                // Skip normal (3 floats = 12 bytes), scale 3 vertices (9 floats)
                int vertexStart = offset + 12;
                for (int v = 0; v < 9; v++)
                {
                    int pos = vertexStart + v * 4;
                    float val = BitConverter.ToSingle(data, pos);
                    byte[] scaled = BitConverter.GetBytes(val * scale);
                    Buffer.BlockCopy(scaled, 0, data, pos, 4);
                }

                offset += TriangleRecordSize;
            }

            File.WriteAllBytes(filePath, data);
        }
    }
}
