/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// NOTE: Do not include headers from any mochi libraries in this file.
//       It is shared with superdex_mesh_cli, which does not depend on mochi libraries.
//       mochi_mesh_cli_types.h is the sibling public protocol header; it is likewise free of
//       mochi dependencies.

#include <mochi_mesh/mochi_mesh_cli_types.h>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

// Private wire-protocol encoding shared by the superdex_mesh_cli helper executable and its client,
// mochi_mesh. Defines the binary framing and the payload (de)serialization used to ship geometry
// across a process boundary over pipes.
//
// Every value is written field by field as a fixed-width primitive: u32, u64 or IEEE-754 double.
// Nothing is written as a struct memory image, so the wire format does not depend on field order,
// padding, alignment, enum underlying type or bool representation, and no padding byte is ever
// transmitted. This matters because the two processes are separately compiled and, under CMake on
// Windows, do not even share a C runtime.
//
// Decoding is strict: every Read fails rather than reading out of bounds, enums and bools are
// range-checked, and aggregate reads validate their own structural invariants. Callers must also
// confirm @ref PayloadReader::AtEnd once they have read every value they expect, so a payload with
// trailing bytes is rejected instead of silently ignored.

namespace mochi::mesh::cli {

// Operations dispatched to the helper executable.
enum class GeometryOp : uint32_t {
  Ping = 0, ///< Echo the request payload back unchanged (used for transport tests).
  RemeshSurface = 1, ///< Surface remeshing (alpha-wrap / ACVD / Surface-Delaunay).
  ReconstructSurfaceFromSdf = 2, ///< Marching-cubes isosurface from a Cartesian SDF grid.
  ApproximateHausdorffDistance =
      3, ///< Approximate one-sided Hausdorff distance between two meshes.
  TessellateStep = 4, ///< Tessellate a STEP CAD file (.step/.stp) into a triangle mesh (OCCT).
  MeshStepBody = 5, ///< Parameterized STEP body tessellation: combine + normalize + custom CGAL
                    ///< per-face mesher.
  CleanupMesh = 6, ///< Stage 2: weld + remove internal/overlapping faces + manifold repair.
  CloseMesh = 7, ///< Stage 3: fill holes / shrink-wrap / none.
  EdgeSwapMesh = 8, ///< Stage 4: re-fit triangle diagonals toward a reference surface cloud.
  DecimateMesh = 9, ///< Stage 5: collapse short edges (CGAL Surface Mesh Simplification).
  ExportStepVisual = 10, ///< Write a STEP file out as one or more render-ready GLB/glTF/OBJ/STL
                         ///< files, keeping the CAD colors and the surfaces' analytic normals. The
                         ///< STEP is loaded and tessellated once for the whole request.
};

inline constexpr uint32_t kFrameMagic = 0x53445832u; ///< Frame magic ("SDX2").
inline constexpr uint32_t kProtocolVersion = 3u; ///< Wire protocol version.

// Plain mesh data transmitted over the wire.
struct MeshData {
  uint32_t nodesPerElement = 3; // 3 for triangles, 4 for tetrahedrons
  std::vector<double> coordinates; // 3 per vertex (x,y,z)
  std::vector<int32_t> connectivity; // 3 per triangle, or 4 per tetrahedron
};

// Plain Cartesian scalar field transmitted over the wire (e.g. an SDF sampled on a grid).
struct ScalarField3d {
  std::array<int32_t, 3> dims{}; // grid dimensions (x, y, z)
  std::vector<double> values; // dims[0]*dims[1]*dims[2], x-slowest / z-fastest
  std::array<double, 3> boundsMin{};
  std::array<double, 3> boundsMax{};
  std::array<double, 3> negativeValueBoundsMin{};
  std::array<double, 3> negativeValueBoundsMax{};
};

// Frame layout (native endian; client and helper must use the same byte order because values are
// copied directly into the pipe payload without byte swapping):
//   Request:  [u32 magic][u32 version][u32 opcode][u64 payloadLen][payload]
//   Response: [u32 magic][u32 status ][u64 payloadLen][payload]
// A response status of 0 means success (payload is the result); any other value means failure
// (payload is a UTF-8 error message).

// Builds a complete request/response frame around @p payload.
[[nodiscard]] std::vector<char> EncodeRequestFrame(GeometryOp op, std::span<char const> payload);
[[nodiscard]] std::vector<char> EncodeResponseFrame(uint32_t status, std::span<char const> payload);

// Parses a complete frame (the entire stream content, read to EOF). Returns false if the frame is
// truncated, has trailing bytes, or its magic/version is wrong.
[[nodiscard]] bool
DecodeRequestFrame(std::span<char const> frame, GeometryOp& outOp, std::vector<char>& outPayload);
[[nodiscard]] bool DecodeResponseFrame(
    std::span<char const> frame,
    uint32_t& outStatus,
    std::vector<char>& outPayload);

// Serializes typed values into a payload byte buffer. Geometry coordinates and grid samples are
// always written as `double`, and indices/enums as fixed-width integer types regardless of the
// caller's Mochi `real` precision.
class PayloadWriter {
 public:
  void WriteU32(uint32_t value);
  void WriteU64(uint64_t value);
  void WriteInt32(int32_t value);
  void WriteBool(bool value); ///< [u32 0 or 1]
  void WriteDouble(double value);
  void WriteDoubleArray(std::span<double const> values); ///< [u64 count][count doubles]
  void WriteInt32Array(std::span<int32_t const> values); ///< [u64 count][count int32s]
  void WriteByteArray(std::span<char const> bytes); ///< [u64 count][count bytes]
  void WriteMeshData(MeshData const& mesh); ///< [u32 nodesPerElement][coordinates][connectivity]
  /// [u64 count][count x ([u32 format][byte-array path])]
  void WriteVisualExportOutputs(std::span<VisualExportOutput const> outputs);
  /// [u64 count][count x u32 status]
  void WriteVisualExportStatuses(std::span<VisualExportStatus const> statuses);
  void WriteScalarField3d(ScalarField3d const& field); ///< [u32 dims x3][values][4 x double bounds
                                                       ///< triple]

  // Operation parameters. Each writes its fields in declaration order; see the matching
  // PayloadReader::Read* for the exact layout.
  void WriteSurfaceRemeshingParams(SurfaceRemeshingParams const& params);
  void WriteStepTessellationParams(StepTessellationParams const& params);
  void WriteStepMeshBodyParams(StepMeshBodyParams const& params);
  void WriteStepVisualExportParams(StepVisualExportParams const& params);
  void WriteMeshClosureParams(MeshClosureParams const& params);
  void WriteMeshEdgeSwapParams(MeshEdgeSwapParams const& params);
  void WriteMeshDecimateParams(MeshDecimateParams const& params);

  [[nodiscard]] std::vector<char> const& Bytes() const {
    return _bytes;
  }

 private:
  std::vector<char> _bytes;
};

// Reads typed values from a payload byte buffer. Each Read returns false (rather than reading out
// of bounds) when the buffer holds too few bytes, so malformed payloads are rejected cleanly.
class PayloadReader {
 public:
  explicit PayloadReader(std::span<char const> data) : _data(data) {}

  [[nodiscard]] bool ReadU32(uint32_t& outValue);
  [[nodiscard]] bool ReadU64(uint64_t& outValue);
  [[nodiscard]] bool ReadInt32(int32_t& outValue);
  [[nodiscard]] bool ReadBool(bool& outValue); ///< Rejects anything other than 0 or 1.
  [[nodiscard]] bool ReadDouble(double& outValue);
  [[nodiscard]] bool ReadDoubleArray(std::vector<double>& outValues);
  [[nodiscard]] bool ReadInt32Array(std::vector<int32_t>& outValues);
  [[nodiscard]] bool ReadByteArray(std::vector<char>& outBytes);
  [[nodiscard]] bool ReadMeshData(MeshData& outMesh);
  /// Rejects an out-of-range format and a count larger than the remaining bytes could describe.
  [[nodiscard]] bool ReadVisualExportOutputs(std::vector<VisualExportOutput>& outOutputs);
  [[nodiscard]] bool ReadVisualExportStatuses(std::vector<VisualExportStatus>& outStatuses);
  [[nodiscard]] bool ReadScalarField3d(ScalarField3d& outField);

  // Operation parameters. Each rejects an out-of-range enumerator or a bool that is not 0 or 1.
  [[nodiscard]] bool ReadSurfaceRemeshingParams(SurfaceRemeshingParams& outParams);
  [[nodiscard]] bool ReadStepTessellationParams(StepTessellationParams& outParams);
  [[nodiscard]] bool ReadStepMeshBodyParams(StepMeshBodyParams& outParams);
  [[nodiscard]] bool ReadStepVisualExportParams(StepVisualExportParams& outParams);
  [[nodiscard]] bool ReadMeshClosureParams(MeshClosureParams& outParams);
  [[nodiscard]] bool ReadMeshEdgeSwapParams(MeshEdgeSwapParams& outParams);
  [[nodiscard]] bool ReadMeshDecimateParams(MeshDecimateParams& outParams);

  /// True once every byte has been consumed. Callers must check this after reading the last value
  /// they expect; a payload with trailing bytes is malformed.
  [[nodiscard]] bool AtEnd() const {
    return _pos == _data.size();
  }

 private:
  std::span<char const> _data;
  size_t _pos = 0;
};

} // namespace mochi::mesh::cli
