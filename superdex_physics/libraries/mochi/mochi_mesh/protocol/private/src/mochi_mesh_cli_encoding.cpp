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

#include <mochi_mesh/mochi_mesh_cli_encoding.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace mochi::mesh::cli;

namespace {

template <class T>
void AppendPod(std::vector<char>& bytes, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  auto const* const src = reinterpret_cast<char const*>(&value);
  bytes.insert(bytes.end(), src, src + sizeof(T));
}

template <class T>
[[nodiscard]] bool ReadPod(std::span<char const> data, size_t& pos, T& outValue) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (pos + sizeof(T) > data.size()) {
    return false;
  }
  auto const* const src = data.data() + pos;
  std::copy(src, src + sizeof(T), reinterpret_cast<char*>(&outValue));
  pos += sizeof(T);
  return true;
}

void AppendBytes(std::vector<char>& bytes, std::span<char const> payload) {
  bytes.insert(bytes.end(), payload.data(), payload.data() + payload.size());
}

// Enums travel as u32 and are range-checked here, so a corrupt or newer peer cannot hand us an
// enumerator that no branch handles and that would otherwise be silently treated as a default.
template <class Enum>
[[nodiscard]] bool ReadEnum(PayloadReader& reader, Enum& outValue, uint32_t validCount) {
  static_assert(std::is_enum_v<Enum>);
  uint32_t raw = 0;
  if (!reader.ReadU32(raw) || raw >= validCount) {
    return false;
  }
  outValue = static_cast<Enum>(raw);
  return true;
}

// Valid enumerator ranges for ReadEnum. Unlike RemeshMethod, these two enums have no Count
// sentinel, so the static_asserts are what fail the build if an enumerator is ever added.
constexpr uint32_t kMeshClosureModeCount = 4;
static_assert(
    static_cast<uint32_t>(mochi::mesh::MeshClosureMode::ConvexHull) + 1 == kMeshClosureModeCount,
    "MeshClosureMode gained or lost a value; update the wire range check.");

constexpr uint32_t kEdgeSamplingCount = 2;
static_assert(
    static_cast<uint32_t>(mochi::mesh::StepMeshBodyParams::EdgeSampling::Adaptive) + 1 ==
        kEdgeSamplingCount,
    "StepMeshBodyParams::EdgeSampling gained or lost a value; update the wire range check.");

} // namespace

std::vector<char> mochi::mesh::cli::EncodeRequestFrame(
    GeometryOp op,
    std::span<char const> payload) {
  std::vector<char> frame;
  frame.reserve(3 * sizeof(uint32_t) + sizeof(uint64_t) + payload.size());
  AppendPod(frame, kFrameMagic);
  AppendPod(frame, kProtocolVersion);
  AppendPod(frame, static_cast<uint32_t>(op));
  AppendPod(frame, static_cast<uint64_t>(payload.size()));
  AppendBytes(frame, payload);
  return frame;
}

std::vector<char> mochi::mesh::cli::EncodeResponseFrame(
    uint32_t status,
    std::span<char const> payload) {
  std::vector<char> frame;
  frame.reserve(2 * sizeof(uint32_t) + sizeof(uint64_t) + payload.size());
  AppendPod(frame, kFrameMagic);
  AppendPod(frame, status);
  AppendPod(frame, static_cast<uint64_t>(payload.size()));
  AppendBytes(frame, payload);
  return frame;
}

bool mochi::mesh::cli::DecodeRequestFrame(
    std::span<char const> frame,
    GeometryOp& outOp,
    std::vector<char>& outPayload) {
  size_t pos = 0;
  uint32_t magic = 0;
  uint32_t version = 0;
  uint32_t opcode = 0;
  uint64_t payloadLen = 0;
  if (!ReadPod(frame, pos, magic) || magic != kFrameMagic) {
    return false;
  }
  if (!ReadPod(frame, pos, version) || version != kProtocolVersion) {
    return false;
  }
  if (!ReadPod(frame, pos, opcode) || !ReadPod(frame, pos, payloadLen)) {
    return false;
  }
  // Exact-size comparison rejects both truncation and trailing bytes without risking overflow.
  if (payloadLen != frame.size() - pos) {
    return false;
  }
  outOp = static_cast<GeometryOp>(opcode);
  outPayload.assign(frame.data() + pos, frame.data() + pos + payloadLen);
  return true;
}

bool mochi::mesh::cli::DecodeResponseFrame(
    std::span<char const> frame,
    uint32_t& outStatus,
    std::vector<char>& outPayload) {
  size_t pos = 0;
  uint32_t magic = 0;
  uint32_t status = 0;
  uint64_t payloadLen = 0;
  if (!ReadPod(frame, pos, magic) || magic != kFrameMagic) {
    return false;
  }
  if (!ReadPod(frame, pos, status) || !ReadPod(frame, pos, payloadLen)) {
    return false;
  }
  // Exact-size comparison rejects both truncation and trailing bytes without risking overflow.
  if (payloadLen != frame.size() - pos) {
    return false;
  }
  outStatus = status;
  outPayload.assign(frame.data() + pos, frame.data() + pos + payloadLen);
  return true;
}

void PayloadWriter::WriteU32(uint32_t value) {
  AppendPod(_bytes, value);
}

void PayloadWriter::WriteU64(uint64_t value) {
  AppendPod(_bytes, value);
}

void PayloadWriter::WriteInt32(int32_t value) {
  AppendPod(_bytes, value);
}

void PayloadWriter::WriteBool(bool value) {
  WriteU32(value ? 1u : 0u);
}

void PayloadWriter::WriteDouble(double value) {
  AppendPod(_bytes, value);
}

void PayloadWriter::WriteDoubleArray(std::span<double const> values) {
  AppendPod(_bytes, static_cast<uint64_t>(values.size()));
  AppendBytes(_bytes, std::span{reinterpret_cast<char const*>(values.data()), values.size_bytes()});
}

void PayloadWriter::WriteInt32Array(std::span<int32_t const> values) {
  AppendPod(_bytes, static_cast<uint64_t>(values.size()));
  AppendBytes(_bytes, std::span{reinterpret_cast<char const*>(values.data()), values.size_bytes()});
}

void PayloadWriter::WriteByteArray(std::span<char const> bytes) {
  AppendPod(_bytes, static_cast<uint64_t>(bytes.size()));
  AppendBytes(_bytes, bytes);
}

void PayloadWriter::WriteMeshData(MeshData const& mesh) {
  WriteU32(mesh.nodesPerElement);
  WriteDoubleArray(mesh.coordinates);
  WriteInt32Array(mesh.connectivity);
}

// Mirror of ReadVisualExportOutputs.
void PayloadWriter::WriteVisualExportOutputs(std::span<VisualExportOutput const> outputs) {
  WriteU64(static_cast<uint64_t>(outputs.size()));
  for (VisualExportOutput const& output : outputs) {
    WriteU32(static_cast<uint32_t>(output.format));
    WriteByteArray(output.path);
  }
}

// Mirror of ReadVisualExportStatuses.
void PayloadWriter::WriteVisualExportStatuses(std::span<VisualExportStatus const> statuses) {
  WriteU64(static_cast<uint64_t>(statuses.size()));
  for (VisualExportStatus const status : statuses) {
    WriteU32(static_cast<uint32_t>(status));
  }
}

// Mirror of ReadScalarField3d: dims (as u32) come first, then the value array, then the four
// bounds triples. Kept colocated so encode and decode cannot drift.
void PayloadWriter::WriteScalarField3d(ScalarField3d const& field) {
  for (int32_t const dim : field.dims) {
    WriteU32(static_cast<uint32_t>(dim));
  }
  WriteDoubleArray(field.values);
  for (std::array<double, 3> const& bounds :
       {field.boundsMin,
        field.boundsMax,
        field.negativeValueBoundsMin,
        field.negativeValueBoundsMax}) {
    for (double const value : bounds) {
      WriteDouble(value);
    }
  }
}

// Each params struct is decomposed with a structured binding before it is written or read. That is
// a tripwire, not a wire contract: the payload is field by field and does not depend on struct
// layout, but the decomposition pins the field count, so adding a member to a params struct fails
// to compile here instead of being silently dropped from every request.
void PayloadWriter::WriteSurfaceRemeshingParams(SurfaceRemeshingParams const& params) {
  auto const& [method, edgeSize, detectFeatures, relativeToMeshSize, alphaWrapRelativeAlpha, alphaWrapRelativeOffset, smoothingIterations, relaxationStepsPerIteration, tangentialRelaxationIterations, angleSmoothingIterations, sharpFeatureAngle, protectConstraints, relaxConstraints, useAdaptiveSizing, adaptiveSizingTolerance, minEdgeSizeFactor, maxEdgeSizeFactor, targetVertexCount, acvdGradationFactor, facetAngleBound, facetDistanceBound, repairMesh] =
      params;
  WriteU32(static_cast<uint32_t>(method));
  WriteDouble(edgeSize);
  WriteBool(detectFeatures);
  WriteBool(relativeToMeshSize);
  WriteDouble(alphaWrapRelativeAlpha);
  WriteDouble(alphaWrapRelativeOffset);
  WriteInt32(smoothingIterations);
  WriteInt32(relaxationStepsPerIteration);
  WriteInt32(tangentialRelaxationIterations);
  WriteInt32(angleSmoothingIterations);
  WriteDouble(sharpFeatureAngle);
  WriteBool(protectConstraints);
  WriteBool(relaxConstraints);
  WriteBool(useAdaptiveSizing);
  WriteDouble(adaptiveSizingTolerance);
  WriteDouble(minEdgeSizeFactor);
  WriteDouble(maxEdgeSizeFactor);
  WriteInt32(targetVertexCount);
  WriteDouble(acvdGradationFactor);
  WriteDouble(facetAngleBound);
  WriteDouble(facetDistanceBound);
  WriteBool(repairMesh);
}

void PayloadWriter::WriteStepTessellationParams(StepTessellationParams const& params) {
  auto const& [linearDeflection, angularDeflection] = params;
  WriteDouble(linearDeflection);
  WriteDouble(angularDeflection);
}

void PayloadWriter::WriteStepMeshBodyParams(StepMeshBodyParams const& params) {
  auto const& [linearDeflection, angularDeflection, targetEdgeLength, targetEdgeLengthFraction, edgeSampling, allowPartialFailure, combineTouchingSolids] =
      params;
  WriteDouble(linearDeflection);
  WriteDouble(angularDeflection);
  WriteDouble(targetEdgeLength);
  WriteDouble(targetEdgeLengthFraction);
  WriteU32(static_cast<uint32_t>(edgeSampling));
  WriteBool(allowPartialFailure);
  WriteBool(combineTouchingSolids);
}

void PayloadWriter::WriteStepVisualExportParams(StepVisualExportParams const& params) {
  auto const& [backend, linearDeflection, angularDeflection, targetEdgeLength, targetEdgeLengthFraction, edgeSampling, scale, allowPartialFailure, rgbaMaterialNames] =
      params;
  WriteU32(static_cast<uint32_t>(backend));
  WriteDouble(linearDeflection);
  WriteDouble(angularDeflection);
  WriteDouble(targetEdgeLength);
  WriteDouble(targetEdgeLengthFraction);
  WriteU32(static_cast<uint32_t>(edgeSampling));
  WriteDouble(scale);
  WriteBool(allowPartialFailure);
  WriteBool(rgbaMaterialNames);
}

void PayloadWriter::WriteMeshClosureParams(MeshClosureParams const& params) {
  auto const& [mode, shrinkWrapTightness, shrinkWrapSnap, shrinkWrapTargetEdgeLength, shrinkWrapTargetEdgeLengthFraction] =
      params;
  WriteU32(static_cast<uint32_t>(mode));
  WriteDouble(shrinkWrapTightness);
  WriteBool(shrinkWrapSnap);
  WriteDouble(shrinkWrapTargetEdgeLength);
  WriteDouble(shrinkWrapTargetEdgeLengthFraction);
}

void PayloadWriter::WriteMeshEdgeSwapParams(MeshEdgeSwapParams const& params) {
  auto const& [relativeThreshold, maxPasses] = params;
  WriteDouble(relativeThreshold);
  WriteInt32(maxPasses);
}

void PayloadWriter::WriteMeshDecimateParams(MeshDecimateParams const& params) {
  auto const& [collapseDistance] = params;
  WriteDouble(collapseDistance);
}

bool PayloadReader::ReadU32(uint32_t& outValue) {
  return ReadPod(_data, _pos, outValue);
}

bool PayloadReader::ReadU64(uint64_t& outValue) {
  return ReadPod(_data, _pos, outValue);
}

bool PayloadReader::ReadInt32(int32_t& outValue) {
  return ReadPod(_data, _pos, outValue);
}

bool PayloadReader::ReadBool(bool& outValue) {
  uint32_t raw = 0;
  if (!ReadU32(raw) || raw > 1u) {
    return false;
  }
  outValue = raw != 0u;
  return true;
}

bool PayloadReader::ReadDouble(double& outValue) {
  return ReadPod(_data, _pos, outValue);
}

bool PayloadReader::ReadDoubleArray(std::vector<double>& outValues) {
  uint64_t count = 0;
  if (!ReadPod(_data, _pos, count)) {
    return false;
  }
  // Overflow-safe: divide instead of multiplying so a huge count cannot wrap past the guard.
  if (count > (_data.size() - _pos) / sizeof(double)) {
    return false;
  }
  outValues.resize(count);
  if (count > 0) {
    std::memcpy(outValues.data(), _data.data() + _pos, count * sizeof(double));
    _pos += count * sizeof(double);
  }
  return true;
}

bool PayloadReader::ReadInt32Array(std::vector<int32_t>& outValues) {
  uint64_t count = 0;
  if (!ReadPod(_data, _pos, count)) {
    return false;
  }
  // Overflow-safe: divide instead of multiplying so a huge count cannot wrap past the guard.
  if (count > (_data.size() - _pos) / sizeof(int32_t)) {
    return false;
  }
  outValues.resize(count);
  if (count > 0) {
    std::memcpy(outValues.data(), _data.data() + _pos, count * sizeof(int32_t));
    _pos += count * sizeof(int32_t);
  }
  return true;
}

bool PayloadReader::ReadByteArray(std::vector<char>& outBytes) {
  uint64_t count = 0;
  if (!ReadPod(_data, _pos, count)) {
    return false;
  }
  // Overflow-safe: compare against remaining bytes so a huge count cannot wrap past the guard.
  if (count > _data.size() - _pos) {
    return false;
  }
  outBytes.assign(_data.data() + _pos, _data.data() + _pos + count);
  _pos += count;
  return true;
}

bool PayloadReader::ReadMeshData(MeshData& outMesh) {
  outMesh = {};
  return ReadU32(outMesh.nodesPerElement) && ReadDoubleArray(outMesh.coordinates) &&
      ReadInt32Array(outMesh.connectivity);
}

bool PayloadReader::ReadVisualExportOutputs(std::vector<VisualExportOutput>& outOutputs) {
  outOutputs.clear();
  uint64_t count = 0;
  if (!ReadU64(count)) {
    return false;
  }
  // Overflow-safe: every output costs at least a u32 format plus an array length prefix, so a count
  // beyond that bound cannot be satisfied and must not be used to size an allocation.
  uint64_t constexpr kMinBytesPerOutput = sizeof(uint32_t) + sizeof(uint64_t);
  if (count > (_data.size() - _pos) / kMinBytesPerOutput) {
    return false;
  }
  outOutputs.resize(count);
  for (VisualExportOutput& output : outOutputs) {
    std::vector<char> pathBytes;
    if (!ReadEnum(*this, output.format, static_cast<uint32_t>(VisualMeshFormat::Count)) ||
        !ReadByteArray(pathBytes)) {
      return false;
    }
    output.path.assign(pathBytes.begin(), pathBytes.end());
  }
  return true;
}

bool PayloadReader::ReadVisualExportStatuses(std::vector<VisualExportStatus>& outStatuses) {
  outStatuses.clear();
  uint64_t count = 0;
  if (!ReadU64(count)) {
    return false;
  }
  if (count > (_data.size() - _pos) / sizeof(uint32_t)) {
    return false;
  }
  outStatuses.resize(count);
  for (VisualExportStatus& status : outStatuses) {
    if (!ReadEnum(*this, status, static_cast<uint32_t>(VisualExportStatus::Count))) {
      return false;
    }
  }
  return true;
}

bool PayloadReader::ReadScalarField3d(ScalarField3d& outField) {
  outField = {};
  uint64_t sampleCount = 1;
  for (int32_t& dim : outField.dims) {
    uint32_t value = 0;
    if (!ReadU32(value) || value == 0u ||
        value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    // Overflow-safe: divide instead of multiplying so huge dimensions cannot wrap past the guard.
    if (sampleCount > std::numeric_limits<uint64_t>::max() / value) {
      return false;
    }
    sampleCount *= value;
    dim = static_cast<int32_t>(value);
  }
  if (!ReadDoubleArray(outField.values) || sampleCount != outField.values.size()) {
    return false;
  }
  for (std::array<double, 3>* const bounds :
       {&outField.boundsMin,
        &outField.boundsMax,
        &outField.negativeValueBoundsMin,
        &outField.negativeValueBoundsMax}) {
    for (double& value : *bounds) {
      if (!ReadDouble(value)) {
        return false;
      }
    }
  }
  return true;
}

bool PayloadReader::ReadSurfaceRemeshingParams(SurfaceRemeshingParams& outParams) {
  outParams = {};
  auto& [method, edgeSize, detectFeatures, relativeToMeshSize, alphaWrapRelativeAlpha, alphaWrapRelativeOffset, smoothingIterations, relaxationStepsPerIteration, tangentialRelaxationIterations, angleSmoothingIterations, sharpFeatureAngle, protectConstraints, relaxConstraints, useAdaptiveSizing, adaptiveSizingTolerance, minEdgeSizeFactor, maxEdgeSizeFactor, targetVertexCount, acvdGradationFactor, facetAngleBound, facetDistanceBound, repairMesh] =
      outParams;
  return ReadEnum(*this, method, static_cast<uint32_t>(RemeshMethod::Count)) &&
      ReadDouble(edgeSize) && ReadBool(detectFeatures) && ReadBool(relativeToMeshSize) &&
      ReadDouble(alphaWrapRelativeAlpha) && ReadDouble(alphaWrapRelativeOffset) &&
      ReadInt32(smoothingIterations) && ReadInt32(relaxationStepsPerIteration) &&
      ReadInt32(tangentialRelaxationIterations) && ReadInt32(angleSmoothingIterations) &&
      ReadDouble(sharpFeatureAngle) && ReadBool(protectConstraints) && ReadBool(relaxConstraints) &&
      ReadBool(useAdaptiveSizing) && ReadDouble(adaptiveSizingTolerance) &&
      ReadDouble(minEdgeSizeFactor) && ReadDouble(maxEdgeSizeFactor) &&
      ReadInt32(targetVertexCount) && ReadDouble(acvdGradationFactor) &&
      ReadDouble(facetAngleBound) && ReadDouble(facetDistanceBound) && ReadBool(repairMesh);
}

bool PayloadReader::ReadStepTessellationParams(StepTessellationParams& outParams) {
  outParams = {};
  auto& [linearDeflection, angularDeflection] = outParams;
  return ReadDouble(linearDeflection) && ReadDouble(angularDeflection);
}

bool PayloadReader::ReadStepMeshBodyParams(StepMeshBodyParams& outParams) {
  outParams = {};
  auto& [linearDeflection, angularDeflection, targetEdgeLength, targetEdgeLengthFraction, edgeSampling, allowPartialFailure, combineTouchingSolids] =
      outParams;
  return ReadDouble(linearDeflection) && ReadDouble(angularDeflection) &&
      ReadDouble(targetEdgeLength) && ReadDouble(targetEdgeLengthFraction) &&
      ReadEnum(*this, edgeSampling, kEdgeSamplingCount) && ReadBool(allowPartialFailure) &&
      ReadBool(combineTouchingSolids);
}

bool PayloadReader::ReadStepVisualExportParams(StepVisualExportParams& outParams) {
  outParams = {};
  auto& [backend, linearDeflection, angularDeflection, targetEdgeLength, targetEdgeLengthFraction, edgeSampling, scale, allowPartialFailure, rgbaMaterialNames] =
      outParams;
  return ReadEnum(*this, backend, static_cast<uint32_t>(CadMeshingBackend::Count)) &&
      ReadDouble(linearDeflection) && ReadDouble(angularDeflection) &&
      ReadDouble(targetEdgeLength) && ReadDouble(targetEdgeLengthFraction) &&
      ReadEnum(*this, edgeSampling, kEdgeSamplingCount) && ReadDouble(scale) &&
      ReadBool(allowPartialFailure) && ReadBool(rgbaMaterialNames);
}

bool PayloadReader::ReadMeshClosureParams(MeshClosureParams& outParams) {
  outParams = {};
  auto& [mode, shrinkWrapTightness, shrinkWrapSnap, shrinkWrapTargetEdgeLength, shrinkWrapTargetEdgeLengthFraction] =
      outParams;
  return ReadEnum(*this, mode, kMeshClosureModeCount) && ReadDouble(shrinkWrapTightness) &&
      ReadBool(shrinkWrapSnap) && ReadDouble(shrinkWrapTargetEdgeLength) &&
      ReadDouble(shrinkWrapTargetEdgeLengthFraction);
}

bool PayloadReader::ReadMeshEdgeSwapParams(MeshEdgeSwapParams& outParams) {
  outParams = {};
  auto& [relativeThreshold, maxPasses] = outParams;
  return ReadDouble(relativeThreshold) && ReadInt32(maxPasses);
}

bool PayloadReader::ReadMeshDecimateParams(MeshDecimateParams& outParams) {
  outParams = {};
  auto& [collapseDistance] = outParams;
  return ReadDouble(collapseDistance);
}
