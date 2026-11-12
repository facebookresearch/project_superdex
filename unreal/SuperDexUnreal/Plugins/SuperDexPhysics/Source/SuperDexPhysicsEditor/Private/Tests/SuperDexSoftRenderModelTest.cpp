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

#include "SuperDexSoftRenderModel.h"

#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "MochiModelData.h"

#if WITH_DEV_AUTOMATION_TESTS

#define SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

namespace {

using SuperDexSoftRenderModel::CheckSoftRenderModelLink;
using SuperDexSoftRenderModel::FRenderModelAttributes;
using SuperDexSoftRenderModel::ReadRenderModelAttributes;

// Two triangles over four vertices, listed out of vertex order.
const TArray<float> kPositions = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
const TArray<uint32> kIndices = {0, 2, 1, 0, 3, 2};
const TArray<float> kUVs = {0, 0, 1, 0, 1, 1, 0, 1};
const TArray<float> kColors = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0.5f};

TArray<FVector2f> ExpectedUVs() {
  return {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
}

TArray<FLinearColor> ExpectedColors() {
  return {{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 0.5f}};
}

// The visual mesh a Studio Model Editor would embed for the test GLB.
FMochiMeshData MakeVisualMesh() {
  FMochiMeshData mesh;
  mesh.NodesPerElement = 3;
  mesh.Coordinates = kPositions;
  for (const uint32 index : kIndices) {
    mesh.Connectivity.Add(static_cast<int32>(index));
  }
  mesh.Skinning.Emplace();
  return mesh;
}

template <typename T>
void AppendBytes(TArray<uint8>& Out, const TArray<T>& Values) {
  Out.Append(reinterpret_cast<const uint8*>(Values.GetData()), Values.Num() * sizeof(T));
}

void AppendUint32(TArray<uint8>& Out, uint32 Value) {
  Out.Append(reinterpret_cast<const uint8*>(&Value), sizeof(Value));
}

// A binary glTF container holding Json and Bin, each chunk padded to four bytes.
TArray<uint8> AssembleGlb(const FString& Json, TArray<uint8> Bin) {
  const FTCHARToUTF8 utf8(*Json);
  TArray<uint8> json(reinterpret_cast<const uint8*>(utf8.Get()), utf8.Length());
  while (json.Num() % 4 != 0) {
    json.Add(' ');
  }
  while (Bin.Num() % 4 != 0) {
    Bin.Add(0);
  }
  TArray<uint8> glb;
  AppendUint32(glb, 0x46546C67); // "glTF"
  AppendUint32(glb, 2);
  AppendUint32(glb, 12 + 8 + json.Num() + 8 + Bin.Num());
  AppendUint32(glb, json.Num());
  AppendUint32(glb, 0x4E4F534A); // "JSON"
  glb.Append(json);
  AppendUint32(glb, Bin.Num());
  AppendUint32(glb, 0x004E4942); // "BIN"
  glb.Append(Bin);
  return glb;
}

// Writes a GLB of the test triangles, with a constant material, to a file named Name and returns
// its path. Attributes lists the primitive's attributes other than POSITION, from TEXCOORD_0 (2)
// and COLOR_0 (3); Extensions is prepended to the top-level JSON members.
FString WriteTestGlb(
    const TCHAR* Name,
    const TCHAR* Attributes,
    int32 NumPrimitives = 1,
    const TCHAR* Extensions = TEXT("")) {
  TArray<uint8> bin;
  AppendBytes(bin, kPositions);
  AppendBytes(bin, kIndices);
  AppendBytes(bin, kUVs);
  AppendBytes(bin, kColors);

  const FString primitive = FString::Printf(
      TEXT(R"({"attributes":{"POSITION":0%s},"indices":1,"material":0})"), Attributes);
  TArray<FString> primitives;
  primitives.Init(primitive, NumPrimitives);
  const FString json = FString::Printf(
      TEXT(
          R"({"asset":{"version":"2.0"},%s"scene":0,"scenes":[{"nodes":[0]}],)"
          R"("nodes":[{"mesh":0}],"meshes":[{"primitives":[%s]}],)"
          R"("materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.2,0.4,0.6,1]}}],)"
          R"("accessors":[)"
          R"({"bufferView":0,"componentType":5126,"count":4,"type":"VEC3",)"
          R"("min":[0,0,0],"max":[1,1,0]},)"
          R"({"bufferView":1,"componentType":5125,"count":6,"type":"SCALAR"},)"
          R"({"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},)"
          R"({"bufferView":3,"componentType":5126,"count":4,"type":"VEC4"}],)"
          R"("bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},)"
          R"({"buffer":0,"byteOffset":48,"byteLength":24},)"
          R"({"buffer":0,"byteOffset":72,"byteLength":32},)"
          R"({"buffer":0,"byteOffset":104,"byteLength":64}],)"
          R"("buffers":[{"byteLength":%d}]})"),
      Extensions,
      *FString::Join(primitives, TEXT(",")),
      bin.Num());

  const FString path =
      FPaths::Combine(FPaths::AutomationTransientDir(), TEXT("SoftRenderModel"), Name) +
      TEXT(".glb");
  FFileHelper::SaveArrayToFile(AssembleGlb(json, MoveTemp(bin)), *path);
  return path;
}

// The attributes of a test GLB, which must be readable. Deletes the GLB.
FRenderModelAttributes ReadTestGlb(FAutomationTestBase& Test, const FString& Path) {
  FRenderModelAttributes attributes;
  FString reason;
  const bool bRead = ReadRenderModelAttributes(Path, attributes, reason);
  IFileManager::Get().Delete(*Path);
  Test.TestTrue(*FString::Printf(TEXT("Read (%s)"), *reason), bRead);
  return attributes;
}

// The reason a test GLB can't be read. Deletes the GLB.
FString ReadFailureReason(FAutomationTestBase& Test, const FString& Path) {
  FRenderModelAttributes attributes;
  FString reason;
  const bool bRead = ReadRenderModelAttributes(Path, attributes, reason);
  IFileManager::Get().Delete(*Path);
  Test.TestFalse(TEXT("Read"), bRead);
  return reason;
}

FRenderModelAttributes MakeAttributes() {
  FRenderModelAttributes attributes;
  attributes.NumVertices = 4;
  attributes.TriangleIndices = kIndices;
  return attributes;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelReadsAccessorOrderTest,
    "SuperDexPhysics.SoftRenderModel.ReadsAttributesInAccessorOrder",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelReadsAccessorOrderTest::RunTest(const FString& Parameters) {
  const FRenderModelAttributes attributes = ReadTestGlb(
      *this, WriteTestGlb(TEXT("UVsAndColors"), TEXT(R"(,"TEXCOORD_0":2,"COLOR_0":3)")));
  TestEqual(TEXT("Vertex count"), attributes.NumVertices, 4);
  TestTrue(TEXT("Triangle indices"), attributes.TriangleIndices == kIndices);
  TestTrue(TEXT("UVs"), attributes.UVs == ExpectedUVs());
  TestTrue(TEXT("Colors"), attributes.Colors == ExpectedColors());
  const FMochiMeshData visualMesh = MakeVisualMesh();
  TestEqual(TEXT("Reason"), CheckSoftRenderModelLink(attributes, &visualMesh), FString());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelNoUVsTest,
    "SuperDexPhysics.SoftRenderModel.AConstantMaterialWithoutUVsIsLinked",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelNoUVsTest::RunTest(const FString& Parameters) {
  const FRenderModelAttributes attributes =
      ReadTestGlb(*this, WriteTestGlb(TEXT("NoUVs"), TEXT("")));
  TestTrue(TEXT("No UVs"), attributes.UVs.IsEmpty());
  TestTrue(TEXT("No colors"), attributes.Colors.IsEmpty());
  const FMochiMeshData visualMesh = MakeVisualMesh();
  TestEqual(TEXT("Reason"), CheckSoftRenderModelLink(attributes, &visualMesh), FString());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelColorsWithoutUVsTest,
    "SuperDexPhysics.SoftRenderModel.ReadsColorsWithoutUVs",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelColorsWithoutUVsTest::RunTest(const FString& Parameters) {
  const FRenderModelAttributes attributes =
      ReadTestGlb(*this, WriteTestGlb(TEXT("ColorsOnly"), TEXT(R"(,"COLOR_0":3)")));
  TestTrue(TEXT("No UVs"), attributes.UVs.IsEmpty());
  TestTrue(TEXT("Colors"), attributes.Colors == ExpectedColors());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelMultiplePrimitivesTest,
    "SuperDexPhysics.SoftRenderModel.MultiplePrimitivesAreNotRead",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelMultiplePrimitivesTest::RunTest(const FString& Parameters) {
  const FString reason = ReadFailureReason(*this, WriteTestGlb(TEXT("TwoPrimitives"), TEXT(""), 2));
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *reason),
      reason.Contains(TEXT("exactly one mesh primitive")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelSecondUVSetTest,
    "SuperDexPhysics.SoftRenderModel.ASecondUVSetIsNotRead",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelSecondUVSetTest::RunTest(const FString& Parameters) {
  const FString reason = ReadFailureReason(
      *this, WriteTestGlb(TEXT("SecondUVSet"), TEXT(R"(,"TEXCOORD_0":2,"TEXCOORD_1":2)")));
  TestTrue(*FString::Printf(TEXT("Reason: %s"), *reason), reason.Contains(TEXT("TEXCOORD_0")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelMeshoptTest,
    "SuperDexPhysics.SoftRenderModel.RequiredMeshoptIsNotRead",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelMeshoptTest::RunTest(const FString& Parameters) {
  const FString reason = ReadFailureReason(
      *this,
      WriteTestGlb(
          TEXT("RequiredMeshopt"),
          TEXT(""),
          1,
          TEXT(
              R"("extensionsUsed":["EXT_meshopt_compression"],)"
              R"("extensionsRequired":["EXT_meshopt_compression"],)")));
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *reason),
      reason.Contains(TEXT("EXT_meshopt_compression")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelNoVisualMeshTest,
    "SuperDexPhysics.SoftRenderModel.NoVisualMeshIsNotLinked",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelNoVisualMeshTest::RunTest(const FString& Parameters) {
  const FString reason = CheckSoftRenderModelLink(MakeAttributes(), nullptr);
  TestTrue(*FString::Printf(TEXT("Reason: %s"), *reason), reason.Contains(TEXT("no visual mesh")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelNoSkinningTest,
    "SuperDexPhysics.SoftRenderModel.AVisualMeshWithoutSkinningIsNotLinked",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelNoSkinningTest::RunTest(const FString& Parameters) {
  FMochiMeshData visualMesh = MakeVisualMesh();
  visualMesh.Skinning.Reset();
  const FString reason = CheckSoftRenderModelLink(MakeAttributes(), &visualMesh);
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *reason),
      reason.Contains(TEXT("no deformation embedding")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelVertexCountTest,
    "SuperDexPhysics.SoftRenderModel.ADifferentVertexCountIsNotLinked",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelVertexCountTest::RunTest(const FString& Parameters) {
  FRenderModelAttributes attributes = MakeAttributes();
  attributes.NumVertices = 3;
  const FMochiMeshData visualMesh = MakeVisualMesh();
  const FString reason = CheckSoftRenderModelLink(attributes, &visualMesh);
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *reason),
      reason.Contains(TEXT("has 3 vertices, but the visual mesh has 4")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelAttributeCountTest,
    "SuperDexPhysics.SoftRenderModel.ADifferentUVOrColorCountIsNotLinked",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelAttributeCountTest::RunTest(const FString& Parameters) {
  const FMochiMeshData visualMesh = MakeVisualMesh();
  FRenderModelAttributes shortUVs = MakeAttributes();
  shortUVs.UVs = ExpectedUVs();
  shortUVs.UVs.Pop();
  FRenderModelAttributes shortColors = MakeAttributes();
  shortColors.Colors = ExpectedColors();
  shortColors.Colors.Pop();
  for (const FRenderModelAttributes& attributes : {shortUVs, shortColors}) {
    const FString reason = CheckSoftRenderModelLink(attributes, &visualMesh);
    TestTrue(
        *FString::Printf(TEXT("Reason: %s"), *reason), reason.Contains(TEXT("UV or color count")));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexSoftRenderModelTrianglesTest,
    "SuperDexPhysics.SoftRenderModel.ADifferentTriangleIsNotLinked",
    SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexSoftRenderModelTrianglesTest::RunTest(const FString& Parameters) {
  FRenderModelAttributes attributes = MakeAttributes();
  attributes.TriangleIndices[5] = 1;
  const FMochiMeshData visualMesh = MakeVisualMesh();
  const FString reason = CheckSoftRenderModelLink(attributes, &visualMesh);
  TestTrue(*FString::Printf(TEXT("Reason: %s"), *reason), reason.Contains(TEXT("triangles")));
  return true;
}

#undef SUPERDEX_SOFT_RENDER_MODEL_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
