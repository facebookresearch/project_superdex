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

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace mochi::mesh;
using namespace mochi::mesh::cli;

namespace {

// Byte offsets of the request frame header fields: [u32 magic][u32 version][u32 op][u64 len].
constexpr size_t kRequestVersionOffset = sizeof(uint32_t);
constexpr size_t kRequestPayloadLenOffset = 3 * sizeof(uint32_t);
// Response frame header: [u32 magic][u32 status][u64 len].
constexpr size_t kResponsePayloadLenOffset = 2 * sizeof(uint32_t);

MeshData Triangle() {
  return {3, {0, 0, 0, 1, 0, 0, 0, 1, 0}, {0, 1, 2}};
}

// Deliberately mixed formats and differing path lengths, so a writer that assumed a fixed stride
// per output would fail the round trip.
std::vector<VisualExportOutput> DistinctVisualExportOutputs() {
  return {
      {VisualMeshFormat::Obj, "C:/out/part.obj"},
      {VisualMeshFormat::Stl, "C:/out/a much longer name.stl"},
      {VisualMeshFormat::Glb, ""},
  };
}

MeshData Tetrahedron() {
  return {4, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 1, 2, 3}};
}

ScalarField3d ScalarField() {
  return {
      {2, 2, 2},
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
      {-1.0, 0.0, 0.5},
      {1.0, 2.0, 3.0},
      {-0.5, 0.25, 0.75},
      {0.5, 1.5, 2.5}};
}

// Every field set to a distinct non-default value, so a dropped, duplicated or reordered field
// changes the decoded result.
SurfaceRemeshingParams DistinctSurfaceRemeshingParams() {
  SurfaceRemeshingParams params;
  params.method = RemeshMethod::SurfaceDelaunay;
  params.edgeSize = 1.5;
  params.detectFeatures = false;
  params.relativeToMeshSize = false;
  params.alphaWrapRelativeAlpha = 2.5;
  params.alphaWrapRelativeOffset = 3.5;
  params.smoothingIterations = 11;
  params.relaxationStepsPerIteration = 12;
  params.tangentialRelaxationIterations = 13;
  params.angleSmoothingIterations = 14;
  params.sharpFeatureAngle = 4.5;
  params.protectConstraints = true;
  params.relaxConstraints = true;
  params.useAdaptiveSizing = true;
  params.adaptiveSizingTolerance = 5.5;
  params.minEdgeSizeFactor = 6.5;
  params.maxEdgeSizeFactor = 7.5;
  params.targetVertexCount = 15;
  params.acvdGradationFactor = 8.5;
  params.facetAngleBound = 9.5;
  params.facetDistanceBound = 10.5;
  params.repairMesh = false;
  return params;
}

StepMeshBodyParams DistinctStepMeshBodyParams() {
  StepMeshBodyParams params;
  params.linearDeflection = 1.25;
  params.angularDeflection = 2.25;
  params.targetEdgeLength = 3.25;
  params.targetEdgeLengthFraction = 4.25;
  params.edgeSampling = StepMeshBodyParams::EdgeSampling::Adaptive;
  params.allowPartialFailure = false;
  params.combineTouchingSolids = false;
  return params;
}

StepVisualExportParams DistinctStepVisualExportParams() {
  StepVisualExportParams params;
  // Non-default so the round trip catches a field that is never written.
  params.backend = CadMeshingBackend::Delabella;
  params.linearDeflection = 1.5;
  params.angularDeflection = 2.5;
  params.targetEdgeLength = 3.5;
  params.targetEdgeLengthFraction = 4.5;
  params.edgeSampling = StepMeshBodyParams::EdgeSampling::Uniform;
  params.scale = 5.5;
  params.allowPartialFailure = false;
  params.rgbaMaterialNames = true;
  return params;
}

MeshClosureParams DistinctMeshClosureParams() {
  MeshClosureParams params;
  params.mode = MeshClosureMode::ConvexHull;
  params.shrinkWrapTightness = 1.75;
  params.shrinkWrapSnap = false;
  params.shrinkWrapTargetEdgeLength = 2.75;
  params.shrinkWrapTargetEdgeLengthFraction = 3.75;
  return params;
}

// Encodes a valid payload, then overwrites the u32 at @p offset so a decoder sees a corrupt value.
std::vector<char> WithU32At(std::vector<char> bytes, size_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
  return bytes;
}

} // namespace

TEST(MochiMeshCliEncoding, RequestFrameRoundTrip) {
  std::vector<char> const payload{'a', 'b'};
  GeometryOp op = GeometryOp::Ping;
  std::vector<char> decoded;
  EXPECT_TRUE(
      DecodeRequestFrame(EncodeRequestFrame(GeometryOp::CleanupMesh, payload), op, decoded));
  EXPECT_EQ(op, GeometryOp::CleanupMesh);
  EXPECT_EQ(decoded, payload);
}

TEST(MochiMeshCliEncoding, ResponseFrameRoundTrip) {
  std::vector<char> const payload{'o', 'k'};
  uint32_t status = 0;
  std::vector<char> decoded;
  EXPECT_TRUE(DecodeResponseFrame(EncodeResponseFrame(7, payload), status, decoded));
  EXPECT_EQ(status, 7);
  EXPECT_EQ(decoded, payload);
}

TEST(MochiMeshCliEncoding, EmptyPayloadFramesRoundTrip) {
  GeometryOp op = GeometryOp::CleanupMesh;
  std::vector<char> payload{'x'};
  EXPECT_TRUE(DecodeRequestFrame(EncodeRequestFrame(GeometryOp::Ping, {}), op, payload));
  EXPECT_EQ(op, GeometryOp::Ping);
  EXPECT_TRUE(payload.empty());

  uint32_t status = 99;
  payload = {'x'};
  EXPECT_TRUE(DecodeResponseFrame(EncodeResponseFrame(0, {}), status, payload));
  EXPECT_EQ(status, 0);
  EXPECT_TRUE(payload.empty());
}

TEST(MochiMeshCliEncoding, FramesRejectWrongMagic) {
  GeometryOp op = GeometryOp::Ping;
  uint32_t status = 0;
  std::vector<char> payload;
  EXPECT_FALSE(
      DecodeRequestFrame(WithU32At(EncodeRequestFrame(GeometryOp::Ping, {}), 0, 0), op, payload));
  EXPECT_FALSE(DecodeResponseFrame(WithU32At(EncodeResponseFrame(0, {}), 0, 0), status, payload));
}

// The version gate is what protects a new client from a stale helper binary, so it needs its own
// coverage independent of the magic.
TEST(MochiMeshCliEncoding, RequestFrameRejectsWrongVersion) {
  GeometryOp op = GeometryOp::Ping;
  std::vector<char> payload;
  auto const frame = WithU32At(
      EncodeRequestFrame(GeometryOp::Ping, {}), kRequestVersionOffset, kProtocolVersion + 1);
  EXPECT_FALSE(DecodeRequestFrame(frame, op, payload));
}

TEST(MochiMeshCliEncoding, FramesRejectPayloadLengthMismatch) {
  GeometryOp op = GeometryOp::Ping;
  uint32_t status = 0;
  std::vector<char> payload;

  // Declared length exceeds the bytes actually present.
  auto request = EncodeRequestFrame(GeometryOp::Ping, {});
  uint64_t const oversizedLength = 1;
  std::memcpy(request.data() + kRequestPayloadLenOffset, &oversizedLength, sizeof(oversizedLength));
  EXPECT_FALSE(DecodeRequestFrame(request, op, payload));

  auto response = EncodeResponseFrame(0, {});
  std::memcpy(
      response.data() + kResponsePayloadLenOffset, &oversizedLength, sizeof(oversizedLength));
  EXPECT_FALSE(DecodeResponseFrame(response, status, payload));

  // Trailing bytes past the declared length are equally invalid, on both frame kinds.
  request = EncodeRequestFrame(GeometryOp::Ping, {});
  request.push_back(0);
  EXPECT_FALSE(DecodeRequestFrame(request, op, payload));

  response = EncodeResponseFrame(0, {});
  response.push_back(0);
  EXPECT_FALSE(DecodeResponseFrame(response, status, payload));
}

TEST(MochiMeshCliEncoding, FramesRejectTruncatedHeader) {
  GeometryOp op = GeometryOp::Ping;
  uint32_t status = 0;
  std::vector<char> payload;

  auto request = EncodeRequestFrame(GeometryOp::Ping, {});
  request.pop_back();
  EXPECT_FALSE(DecodeRequestFrame(request, op, payload));
  EXPECT_FALSE(DecodeRequestFrame({}, op, payload));

  auto response = EncodeResponseFrame(0, {});
  response.pop_back();
  EXPECT_FALSE(DecodeResponseFrame(response, status, payload));
  EXPECT_FALSE(DecodeResponseFrame({}, status, payload));
}

TEST(MochiMeshCliEncoding, ScalarPayloadRoundTrip) {
  PayloadWriter writer;
  writer.WriteU32(12);
  writer.WriteU64(56);
  writer.WriteInt32(-34);
  writer.WriteBool(true);
  writer.WriteBool(false);
  writer.WriteDouble(7.25);

  PayloadReader reader(writer.Bytes());
  uint32_t u32 = 0;
  uint64_t u64 = 0;
  int32_t i32 = 0;
  bool trueValue = false;
  bool falseValue = true;
  double d = 0.0;
  EXPECT_TRUE(reader.ReadU32(u32));
  EXPECT_TRUE(reader.ReadU64(u64));
  EXPECT_TRUE(reader.ReadInt32(i32));
  EXPECT_TRUE(reader.ReadBool(trueValue));
  EXPECT_TRUE(reader.ReadBool(falseValue));
  EXPECT_TRUE(reader.ReadDouble(d));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(u32, 12);
  EXPECT_EQ(u64, 56);
  EXPECT_EQ(i32, -34);
  EXPECT_TRUE(trueValue);
  EXPECT_FALSE(falseValue);
  EXPECT_EQ(d, 7.25);
}

TEST(MochiMeshCliEncoding, ReaderRejectsBoolOutsideZeroOne) {
  PayloadWriter writer;
  writer.WriteU32(2);
  PayloadReader reader(writer.Bytes());
  bool value = false;
  EXPECT_FALSE(reader.ReadBool(value));
}

TEST(MochiMeshCliEncoding, ArrayPayloadRoundTrip) {
  std::vector<double> const doubles{1.0, 2.0, 3.0};
  std::vector<int32_t> const ints{4, 5, 6};
  std::vector<char> const bytes{'b', 'y', 't', 'e', 's'};

  PayloadWriter writer;
  writer.WriteDoubleArray(doubles);
  writer.WriteInt32Array(ints);
  writer.WriteByteArray(bytes);

  PayloadReader reader(writer.Bytes());
  std::vector<double> decodedDoubles;
  std::vector<int32_t> decodedInts;
  std::vector<char> decodedBytes;
  EXPECT_TRUE(reader.ReadDoubleArray(decodedDoubles));
  EXPECT_TRUE(reader.ReadInt32Array(decodedInts));
  EXPECT_TRUE(reader.ReadByteArray(decodedBytes));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decodedDoubles, doubles);
  EXPECT_EQ(decodedInts, ints);
  EXPECT_EQ(decodedBytes, bytes);
}

TEST(MochiMeshCliEncoding, TriMeshRoundTrip) {
  MeshData const mesh = Triangle();

  PayloadWriter writer;
  writer.WriteMeshData(mesh);

  PayloadReader reader(writer.Bytes());
  MeshData decodedMesh;
  EXPECT_TRUE(reader.ReadMeshData(decodedMesh));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decodedMesh.nodesPerElement, 3);
  EXPECT_EQ(decodedMesh.coordinates, mesh.coordinates);
  EXPECT_EQ(decodedMesh.connectivity, mesh.connectivity);
}

TEST(MochiMeshCliEncoding, TetMeshRoundTrip) {
  MeshData const mesh = Tetrahedron();

  PayloadWriter writer;
  writer.WriteMeshData(mesh);

  PayloadReader reader(writer.Bytes());
  MeshData decodedMesh;
  EXPECT_TRUE(reader.ReadMeshData(decodedMesh));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decodedMesh.nodesPerElement, 4);
  EXPECT_EQ(decodedMesh.coordinates, mesh.coordinates);
  EXPECT_EQ(decodedMesh.connectivity, mesh.connectivity);
}

TEST(MochiMeshCliEncoding, EmptyMeshRoundTrips) {
  PayloadWriter writer;
  writer.WriteMeshData(MeshData{});

  PayloadReader reader(writer.Bytes());
  MeshData decodedMesh;
  EXPECT_TRUE(reader.ReadMeshData(decodedMesh));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_TRUE(decodedMesh.coordinates.empty());
  EXPECT_TRUE(decodedMesh.connectivity.empty());
}

TEST(MochiMeshCliEncoding, ReaderRejectsTruncatedMesh) {
  PayloadWriter writer;
  writer.WriteMeshData(Triangle());
  std::vector<char> bytes = writer.Bytes();

  // Drop part of the trailing connectivity array.
  bytes.resize(bytes.size() - sizeof(int32_t));
  PayloadReader reader(bytes);
  MeshData decodedMesh;
  EXPECT_FALSE(reader.ReadMeshData(decodedMesh));
}

TEST(MochiMeshCliEncoding, ScalarField3dRoundTrip) {
  ScalarField3d const field = ScalarField();

  PayloadWriter writer;
  writer.WriteScalarField3d(field);

  PayloadReader reader(writer.Bytes());
  ScalarField3d decoded;
  EXPECT_TRUE(reader.ReadScalarField3d(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.dims, field.dims);
  EXPECT_EQ(decoded.values, field.values);
  EXPECT_EQ(decoded.boundsMin, field.boundsMin);
  EXPECT_EQ(decoded.boundsMax, field.boundsMax);
  EXPECT_EQ(decoded.negativeValueBoundsMin, field.negativeValueBoundsMin);
  EXPECT_EQ(decoded.negativeValueBoundsMax, field.negativeValueBoundsMax);
}

// ReadScalarField3d has a failure branch at every stage; cut the payload at each one.
TEST(MochiMeshCliEncoding, ReaderRejectsTruncatedScalarField) {
  PayloadWriter writer;
  writer.WriteScalarField3d(ScalarField());
  std::vector<char> const full = writer.Bytes();

  for (size_t truncatedSize :
       {size_t{0},
        sizeof(uint32_t), // mid-dims
        3 * sizeof(uint32_t), // after dims, before the value array
        full.size() - 6 * sizeof(double), // mid-bounds
        full.size() - 1}) {
    std::vector<char> const bytes(full.begin(), full.begin() + truncatedSize);
    PayloadReader reader(bytes);
    ScalarField3d decoded;
    EXPECT_FALSE(reader.ReadScalarField3d(decoded)) << "truncated to " << truncatedSize << " bytes";
  }
}

TEST(MochiMeshCliEncoding, ReaderRejectsStructurallyInvalidScalarField) {
  ScalarField3d field = ScalarField();
  field.dims = {2, 2, 3}; // values still holds 8 samples, not 12
  PayloadWriter mismatched;
  mismatched.WriteScalarField3d(field);
  PayloadReader mismatchedReader(mismatched.Bytes());
  ScalarField3d decoded;
  EXPECT_FALSE(mismatchedReader.ReadScalarField3d(decoded));

  field = ScalarField();
  field.dims = {0, 2, 2};
  PayloadWriter zeroDim;
  zeroDim.WriteScalarField3d(field);
  PayloadReader zeroDimReader(zeroDim.Bytes());
  EXPECT_FALSE(zeroDimReader.ReadScalarField3d(decoded));
}

TEST(MochiMeshCliEncoding, SurfaceRemeshingParamsRoundTrip) {
  SurfaceRemeshingParams const params = DistinctSurfaceRemeshingParams();

  PayloadWriter writer;
  writer.WriteSurfaceRemeshingParams(params);

  PayloadReader reader(writer.Bytes());
  SurfaceRemeshingParams decoded;
  EXPECT_TRUE(reader.ReadSurfaceRemeshingParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.method, params.method);
  EXPECT_EQ(decoded.edgeSize, params.edgeSize);
  EXPECT_EQ(decoded.detectFeatures, params.detectFeatures);
  EXPECT_EQ(decoded.relativeToMeshSize, params.relativeToMeshSize);
  EXPECT_EQ(decoded.alphaWrapRelativeAlpha, params.alphaWrapRelativeAlpha);
  EXPECT_EQ(decoded.alphaWrapRelativeOffset, params.alphaWrapRelativeOffset);
  EXPECT_EQ(decoded.smoothingIterations, params.smoothingIterations);
  EXPECT_EQ(decoded.relaxationStepsPerIteration, params.relaxationStepsPerIteration);
  EXPECT_EQ(decoded.tangentialRelaxationIterations, params.tangentialRelaxationIterations);
  EXPECT_EQ(decoded.angleSmoothingIterations, params.angleSmoothingIterations);
  EXPECT_EQ(decoded.sharpFeatureAngle, params.sharpFeatureAngle);
  EXPECT_EQ(decoded.protectConstraints, params.protectConstraints);
  EXPECT_EQ(decoded.relaxConstraints, params.relaxConstraints);
  EXPECT_EQ(decoded.useAdaptiveSizing, params.useAdaptiveSizing);
  EXPECT_EQ(decoded.adaptiveSizingTolerance, params.adaptiveSizingTolerance);
  EXPECT_EQ(decoded.minEdgeSizeFactor, params.minEdgeSizeFactor);
  EXPECT_EQ(decoded.maxEdgeSizeFactor, params.maxEdgeSizeFactor);
  EXPECT_EQ(decoded.targetVertexCount, params.targetVertexCount);
  EXPECT_EQ(decoded.acvdGradationFactor, params.acvdGradationFactor);
  EXPECT_EQ(decoded.facetAngleBound, params.facetAngleBound);
  EXPECT_EQ(decoded.facetDistanceBound, params.facetDistanceBound);
  EXPECT_EQ(decoded.repairMesh, params.repairMesh);
}

TEST(MochiMeshCliEncoding, StepTessellationParamsRoundTrip) {
  StepTessellationParams params;
  params.linearDeflection = 1.5;
  params.angularDeflection = 2.5;

  PayloadWriter writer;
  writer.WriteStepTessellationParams(params);

  PayloadReader reader(writer.Bytes());
  StepTessellationParams decoded;
  EXPECT_TRUE(reader.ReadStepTessellationParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.linearDeflection, params.linearDeflection);
  EXPECT_EQ(decoded.angularDeflection, params.angularDeflection);
}

TEST(MochiMeshCliEncoding, StepMeshBodyParamsRoundTrip) {
  StepMeshBodyParams const params = DistinctStepMeshBodyParams();

  PayloadWriter writer;
  writer.WriteStepMeshBodyParams(params);

  PayloadReader reader(writer.Bytes());
  StepMeshBodyParams decoded;
  EXPECT_TRUE(reader.ReadStepMeshBodyParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.linearDeflection, params.linearDeflection);
  EXPECT_EQ(decoded.angularDeflection, params.angularDeflection);
  EXPECT_EQ(decoded.targetEdgeLength, params.targetEdgeLength);
  EXPECT_EQ(decoded.targetEdgeLengthFraction, params.targetEdgeLengthFraction);
  EXPECT_EQ(decoded.edgeSampling, params.edgeSampling);
  EXPECT_EQ(decoded.allowPartialFailure, params.allowPartialFailure);
  EXPECT_EQ(decoded.combineTouchingSolids, params.combineTouchingSolids);
}

TEST(MochiMeshCliEncoding, StepVisualExportParamsRoundTrip) {
  StepVisualExportParams const params = DistinctStepVisualExportParams();

  PayloadWriter writer;
  writer.WriteStepVisualExportParams(params);

  PayloadReader reader(writer.Bytes());
  StepVisualExportParams decoded;
  EXPECT_TRUE(reader.ReadStepVisualExportParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.backend, params.backend);
  EXPECT_EQ(decoded.linearDeflection, params.linearDeflection);
  EXPECT_EQ(decoded.angularDeflection, params.angularDeflection);
  EXPECT_EQ(decoded.targetEdgeLength, params.targetEdgeLength);
  EXPECT_EQ(decoded.targetEdgeLengthFraction, params.targetEdgeLengthFraction);
  EXPECT_EQ(decoded.edgeSampling, params.edgeSampling);
  EXPECT_EQ(decoded.scale, params.scale);
  EXPECT_EQ(decoded.allowPartialFailure, params.allowPartialFailure);
  EXPECT_EQ(decoded.rgbaMaterialNames, params.rgbaMaterialNames);
}

TEST(MochiMeshCliEncoding, VisualExportOutputsRoundTrip) {
  std::vector<VisualExportOutput> const outputs = DistinctVisualExportOutputs();

  PayloadWriter writer;
  writer.WriteVisualExportOutputs(outputs);

  PayloadReader reader(writer.Bytes());
  std::vector<VisualExportOutput> decoded;
  EXPECT_TRUE(reader.ReadVisualExportOutputs(decoded));
  EXPECT_TRUE(reader.AtEnd());
  ASSERT_EQ(decoded.size(), outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(decoded[i].format, outputs[i].format);
    EXPECT_EQ(decoded[i].path, outputs[i].path);
  }
}

TEST(MochiMeshCliEncoding, VisualExportOutputsRejectOutOfRangeFormat) {
  PayloadWriter writer;
  writer.WriteVisualExportOutputs(DistinctVisualExportOutputs());
  // The first output's format sits immediately after the u64 count.
  auto const bytes =
      WithU32At(writer.Bytes(), sizeof(uint64_t), static_cast<uint32_t>(VisualMeshFormat::Count));

  PayloadReader reader(bytes);
  std::vector<VisualExportOutput> decoded;
  EXPECT_FALSE(reader.ReadVisualExportOutputs(decoded));
}

// A count far larger than the payload could describe must be rejected before it is used to size an
// allocation.
TEST(MochiMeshCliEncoding, VisualExportOutputsRejectOversizedCount) {
  PayloadWriter writer;
  writer.WriteVisualExportOutputs(DistinctVisualExportOutputs());
  std::vector<char> bytes = writer.Bytes();
  uint64_t const count = std::numeric_limits<uint64_t>::max();
  std::memcpy(bytes.data(), &count, sizeof(count));

  PayloadReader reader(bytes);
  std::vector<VisualExportOutput> decoded;
  EXPECT_FALSE(reader.ReadVisualExportOutputs(decoded));
}

TEST(MochiMeshCliEncoding, VisualExportStatusesRoundTrip) {
  std::vector<VisualExportStatus> const statuses = {
      VisualExportStatus::Written, VisualExportStatus::WrittenPartial, VisualExportStatus::Failed};

  PayloadWriter writer;
  writer.WriteVisualExportStatuses(statuses);

  PayloadReader reader(writer.Bytes());
  std::vector<VisualExportStatus> decoded;
  EXPECT_TRUE(reader.ReadVisualExportStatuses(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded, statuses);
}

TEST(MochiMeshCliEncoding, VisualExportStatusesRejectOutOfRangeStatus) {
  std::vector<VisualExportStatus> const statuses = {VisualExportStatus::Written};
  PayloadWriter writer;
  writer.WriteVisualExportStatuses(statuses);
  auto const bytes =
      WithU32At(writer.Bytes(), sizeof(uint64_t), static_cast<uint32_t>(VisualExportStatus::Count));

  PayloadReader reader(bytes);
  std::vector<VisualExportStatus> decoded;
  EXPECT_FALSE(reader.ReadVisualExportStatuses(decoded));
}

TEST(MochiMeshCliEncoding, MeshClosureParamsRoundTrip) {
  MeshClosureParams const params = DistinctMeshClosureParams();

  PayloadWriter writer;
  writer.WriteMeshClosureParams(params);

  PayloadReader reader(writer.Bytes());
  MeshClosureParams decoded;
  EXPECT_TRUE(reader.ReadMeshClosureParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.mode, params.mode);
  EXPECT_EQ(decoded.shrinkWrapTightness, params.shrinkWrapTightness);
  EXPECT_EQ(decoded.shrinkWrapSnap, params.shrinkWrapSnap);
  EXPECT_EQ(decoded.shrinkWrapTargetEdgeLength, params.shrinkWrapTargetEdgeLength);
  EXPECT_EQ(decoded.shrinkWrapTargetEdgeLengthFraction, params.shrinkWrapTargetEdgeLengthFraction);
}

TEST(MochiMeshCliEncoding, MeshEdgeSwapParamsRoundTrip) {
  MeshEdgeSwapParams params;
  params.relativeThreshold = 0.375;
  params.maxPasses = 42;

  PayloadWriter writer;
  writer.WriteMeshEdgeSwapParams(params);

  PayloadReader reader(writer.Bytes());
  MeshEdgeSwapParams decoded;
  EXPECT_TRUE(reader.ReadMeshEdgeSwapParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.relativeThreshold, params.relativeThreshold);
  EXPECT_EQ(decoded.maxPasses, params.maxPasses);
}

TEST(MochiMeshCliEncoding, MeshDecimateParamsRoundTrip) {
  MeshDecimateParams params;
  params.collapseDistance = 0.125;

  PayloadWriter writer;
  writer.WriteMeshDecimateParams(params);

  PayloadReader reader(writer.Bytes());
  MeshDecimateParams decoded;
  EXPECT_TRUE(reader.ReadMeshDecimateParams(decoded));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(decoded.collapseDistance, params.collapseDistance);
}

// Switching on an out-of-range enumerator is undefined behaviour, so every enum on the wire is
// range-checked rather than blindly cast.
TEST(MochiMeshCliEncoding, ParamsRejectOutOfRangeEnums) {
  PayloadWriter remeshWriter;
  remeshWriter.WriteSurfaceRemeshingParams(DistinctSurfaceRemeshingParams());
  auto const remeshBytes =
      WithU32At(remeshWriter.Bytes(), 0, static_cast<uint32_t>(RemeshMethod::Count));
  PayloadReader remeshReader(remeshBytes);
  SurfaceRemeshingParams remeshParams;
  EXPECT_FALSE(remeshReader.ReadSurfaceRemeshingParams(remeshParams));

  PayloadWriter closureWriter;
  closureWriter.WriteMeshClosureParams(DistinctMeshClosureParams());
  auto const closureBytes = WithU32At(closureWriter.Bytes(), 0, 4);
  PayloadReader closureReader(closureBytes);
  MeshClosureParams closureParams;
  EXPECT_FALSE(closureReader.ReadMeshClosureParams(closureParams));

  // EdgeSampling sits after four doubles in StepMeshBodyParams.
  PayloadWriter bodyWriter;
  bodyWriter.WriteStepMeshBodyParams(DistinctStepMeshBodyParams());
  auto const bodyBytes = WithU32At(bodyWriter.Bytes(), 4 * sizeof(double), 2);
  PayloadReader bodyReader(bodyBytes);
  StepMeshBodyParams bodyParams;
  EXPECT_FALSE(bodyReader.ReadStepMeshBodyParams(bodyParams));

  // CadMeshingBackend leads StepVisualExportParams.
  PayloadWriter visualWriter;
  visualWriter.WriteStepVisualExportParams(DistinctStepVisualExportParams());
  auto const backendBytes =
      WithU32At(visualWriter.Bytes(), 0, static_cast<uint32_t>(CadMeshingBackend::Count));
  PayloadReader backendReader(backendBytes);
  StepVisualExportParams backendParams;
  EXPECT_FALSE(backendReader.ReadStepVisualExportParams(backendParams));
}

TEST(MochiMeshCliEncoding, ParamsRejectTruncatedPayload) {
  PayloadWriter writer;
  writer.WriteSurfaceRemeshingParams(DistinctSurfaceRemeshingParams());
  std::vector<char> bytes = writer.Bytes();
  bytes.pop_back();

  PayloadReader reader(bytes);
  SurfaceRemeshingParams decoded;
  EXPECT_FALSE(reader.ReadSurfaceRemeshingParams(decoded));
}

TEST(MochiMeshCliEncoding, ReaderRejectsTruncatedScalar) {
  std::vector<char> const bytes;
  PayloadReader emptyReader(bytes);
  uint32_t scalar = 0;
  EXPECT_FALSE(emptyReader.ReadU32(scalar));
}

// Each array reader guards its length prefix differently (ReadByteArray compares bytes directly,
// the typed readers divide by the element size), so all three need coverage.
TEST(MochiMeshCliEncoding, ReadersRejectOversizedArrayCounts) {
  std::vector<char> bytes(sizeof(uint64_t));
  uint64_t const count = std::numeric_limits<uint64_t>::max();
  std::memcpy(bytes.data(), &count, sizeof(count));

  std::vector<double> doubles;
  PayloadReader doubleReader(bytes);
  EXPECT_FALSE(doubleReader.ReadDoubleArray(doubles));

  std::vector<int32_t> ints;
  PayloadReader intReader(bytes);
  EXPECT_FALSE(intReader.ReadInt32Array(ints));

  std::vector<char> rawBytes;
  PayloadReader byteReader(bytes);
  EXPECT_FALSE(byteReader.ReadByteArray(rawBytes));
}

TEST(MochiMeshCliEncoding, ReadersRejectTruncatedArrayPayloads) {
  PayloadWriter doubleWriter;
  doubleWriter.WriteDoubleArray(std::vector<double>{1.0, 2.0});
  std::vector<char> doubleBytes = doubleWriter.Bytes();
  doubleBytes.pop_back();
  std::vector<double> doubles;
  PayloadReader doubleReader(doubleBytes);
  EXPECT_FALSE(doubleReader.ReadDoubleArray(doubles));

  PayloadWriter intWriter;
  intWriter.WriteInt32Array(std::vector<int32_t>{1, 2});
  std::vector<char> intBytes = intWriter.Bytes();
  intBytes.pop_back();
  std::vector<int32_t> ints;
  PayloadReader intReader(intBytes);
  EXPECT_FALSE(intReader.ReadInt32Array(ints));

  PayloadWriter byteWriter;
  byteWriter.WriteByteArray(std::vector<char>{'a', 'b'});
  std::vector<char> byteBytes = byteWriter.Bytes();
  byteBytes.pop_back();
  std::vector<char> rawBytes;
  PayloadReader byteReader(byteBytes);
  EXPECT_FALSE(byteReader.ReadByteArray(rawBytes));
}

// A payload that decodes cleanly but carries extra bytes indicates a client/helper mismatch, so
// callers rely on AtEnd to catch it.
TEST(MochiMeshCliEncoding, AtEndDetectsTrailingBytes) {
  PayloadWriter writer;
  writer.WriteMeshData(Triangle());
  std::vector<char> bytes = writer.Bytes();
  bytes.push_back(0);

  PayloadReader reader(bytes);
  MeshData decoded;
  EXPECT_TRUE(reader.ReadMeshData(decoded));
  EXPECT_FALSE(reader.AtEnd());
}
