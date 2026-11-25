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

#include "SuperDexDeformableRenderModel.h"

#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "MochiGenTypes.h"

#if WITH_DEV_AUTOMATION_TESTS

#define SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

namespace {

using SuperDexDeformableRenderModel::CheckRenderModelLink;
using SuperDexDeformableRenderModel::FRenderModelAttributes;
using SuperDexDeformableRenderModel::ReadRenderModelAttributes;

// Two triangles over four vertices, listed out of vertex order.
const TArray<float> Positions = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
const TArray<uint32> Indices = {0, 2, 1, 0, 3, 2};
const TArray<float> UVs = {0, 0, 1, 0, 1, 1, 0, 1};
const TArray<float> Colors = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0.5f};

TArray<FVector2f> ExpectedUVs() {
  return {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
}

TArray<FLinearColor> ExpectedColors() {
  return {{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 0.5f}};
}

// The visual mesh a Studio Model Editor would embed for the test GLB.
FMochiMeshData MakeVisualMesh() {
  FMochiMeshData Mesh;
  Mesh.NodesPerElement = 3;
  Mesh.Coordinates = Positions;
  for (const uint32 Index : Indices) {
    Mesh.Connectivity.Add(static_cast<int32>(Index));
  }
  Mesh.Skinning.Emplace();
  return Mesh;
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
  const FTCHARToUTF8 Utf8(*Json);
  TArray<uint8> JsonBytes(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
  while (JsonBytes.Num() % 4 != 0) {
    JsonBytes.Add(' ');
  }
  while (Bin.Num() % 4 != 0) {
    Bin.Add(0);
  }
  TArray<uint8> Glb;
  AppendUint32(Glb, 0x46546C67); // "glTF"
  AppendUint32(Glb, 2);
  AppendUint32(Glb, 12 + 8 + JsonBytes.Num() + 8 + Bin.Num());
  AppendUint32(Glb, JsonBytes.Num());
  AppendUint32(Glb, 0x4E4F534A); // "JSON"
  Glb.Append(JsonBytes);
  AppendUint32(Glb, Bin.Num());
  AppendUint32(Glb, 0x004E4942); // "BIN"
  Glb.Append(Bin);
  return Glb;
}

// Writes a GLB of the test triangles, with a constant material, to a file named Name and returns
// its path. Attributes lists the primitive's attributes other than POSITION, from TEXCOORD_0 (2)
// and COLOR_0 (3); Extensions is prepended to the top-level JSON members.
FString WriteTestGlb(
    const TCHAR* Name,
    const TCHAR* Attributes,
    int32 NumPrimitives = 1,
    const TCHAR* Extensions = TEXT("")) {
  TArray<uint8> Bin;
  AppendBytes(Bin, Positions);
  AppendBytes(Bin, Indices);
  AppendBytes(Bin, UVs);
  AppendBytes(Bin, Colors);

  const FString Primitive = FString::Printf(
      TEXT(R"({"attributes":{"POSITION":0%s},"indices":1,"material":0})"), Attributes);
  TArray<FString> Primitives;
  Primitives.Init(Primitive, NumPrimitives);
  const FString Json = FString::Printf(
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
      *FString::Join(Primitives, TEXT(",")),
      Bin.Num());

  const FString Path =
      FPaths::Combine(FPaths::AutomationTransientDir(), TEXT("DeformableRenderModel"), Name) +
      TEXT(".glb");
  FFileHelper::SaveArrayToFile(AssembleGlb(Json, MoveTemp(Bin)), *Path);
  return Path;
}

// The attributes of a test GLB, which must be readable. Deletes the GLB.
FRenderModelAttributes ReadTestGlb(FAutomationTestBase& Test, const FString& Path) {
  FRenderModelAttributes Attributes;
  FString Reason;
  const bool bRead = ReadRenderModelAttributes(Path, Attributes, Reason);
  IFileManager::Get().Delete(*Path);
  Test.TestTrue(*FString::Printf(TEXT("Read (%s)"), *Reason), bRead);
  return Attributes;
}

// The reason a test GLB can't be read. Deletes the GLB.
FString ReadFailureReason(FAutomationTestBase& Test, const FString& Path) {
  FRenderModelAttributes Attributes;
  FString Reason;
  const bool bRead = ReadRenderModelAttributes(Path, Attributes, Reason);
  IFileManager::Get().Delete(*Path);
  Test.TestFalse(TEXT("Read"), bRead);
  return Reason;
}

FRenderModelAttributes MakeAttributes() {
  FRenderModelAttributes Attributes;
  Attributes.NumVertices = 4;
  Attributes.TriangleIndices = Indices;
  return Attributes;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelReadsAccessorOrderTest,
    "SuperDexPhysics.DeformableRenderModel.ReadsAttributesInAccessorOrder",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelReadsAccessorOrderTest::RunTest(const FString& Parameters) {
  const FRenderModelAttributes Attributes = ReadTestGlb(
      *this, WriteTestGlb(TEXT("UVsAndColors"), TEXT(R"(,"TEXCOORD_0":2,"COLOR_0":3)")));
  TestEqual(TEXT("Vertex count"), Attributes.NumVertices, 4);
  TestTrue(TEXT("Triangle indices"), Attributes.TriangleIndices == Indices);
  TestTrue(TEXT("UVs"), Attributes.UVs == ExpectedUVs());
  TestTrue(TEXT("Colors"), Attributes.Colors == ExpectedColors());
  const FMochiMeshData VisualMesh = MakeVisualMesh();
  TestEqual(TEXT("Reason"), CheckRenderModelLink(Attributes, &VisualMesh), FString());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelNoUVsTest,
    "SuperDexPhysics.DeformableRenderModel.AConstantMaterialWithoutUVsIsLinked",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelNoUVsTest::RunTest(const FString& Parameters) {
  const FRenderModelAttributes Attributes =
      ReadTestGlb(*this, WriteTestGlb(TEXT("NoUVs"), TEXT("")));
  TestTrue(TEXT("No UVs"), Attributes.UVs.IsEmpty());
  TestTrue(TEXT("No colors"), Attributes.Colors.IsEmpty());
  const FMochiMeshData VisualMesh = MakeVisualMesh();
  TestEqual(TEXT("Reason"), CheckRenderModelLink(Attributes, &VisualMesh), FString());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelColorsWithoutUVsTest,
    "SuperDexPhysics.DeformableRenderModel.ReadsColorsWithoutUVs",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelColorsWithoutUVsTest::RunTest(const FString& Parameters) {
  const FRenderModelAttributes Attributes =
      ReadTestGlb(*this, WriteTestGlb(TEXT("ColorsOnly"), TEXT(R"(,"COLOR_0":3)")));
  TestTrue(TEXT("No UVs"), Attributes.UVs.IsEmpty());
  TestTrue(TEXT("Colors"), Attributes.Colors == ExpectedColors());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelMultiplePrimitivesTest,
    "SuperDexPhysics.DeformableRenderModel.MultiplePrimitivesAreNotRead",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelMultiplePrimitivesTest::RunTest(const FString& Parameters) {
  const FString Reason = ReadFailureReason(*this, WriteTestGlb(TEXT("TwoPrimitives"), TEXT(""), 2));
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *Reason),
      Reason.Contains(TEXT("exactly one mesh primitive")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelSecondUVSetTest,
    "SuperDexPhysics.DeformableRenderModel.ASecondUVSetIsNotRead",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelSecondUVSetTest::RunTest(const FString& Parameters) {
  const FString Reason = ReadFailureReason(
      *this, WriteTestGlb(TEXT("SecondUVSet"), TEXT(R"(,"TEXCOORD_0":2,"TEXCOORD_1":2)")));
  TestTrue(*FString::Printf(TEXT("Reason: %s"), *Reason), Reason.Contains(TEXT("TEXCOORD_0")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelMeshoptTest,
    "SuperDexPhysics.DeformableRenderModel.RequiredMeshoptIsNotRead",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelMeshoptTest::RunTest(const FString& Parameters) {
  const FString Reason = ReadFailureReason(
      *this,
      WriteTestGlb(
          TEXT("RequiredMeshopt"),
          TEXT(""),
          1,
          TEXT(
              R"("extensionsUsed":["EXT_meshopt_compression"],)"
              R"("extensionsRequired":["EXT_meshopt_compression"],)")));
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *Reason),
      Reason.Contains(TEXT("EXT_meshopt_compression")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelNoVisualMeshTest,
    "SuperDexPhysics.DeformableRenderModel.NoVisualMeshIsNotLinked",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelNoVisualMeshTest::RunTest(const FString& Parameters) {
  const FString Reason = CheckRenderModelLink(MakeAttributes(), nullptr);
  TestTrue(*FString::Printf(TEXT("Reason: %s"), *Reason), Reason.Contains(TEXT("no visual mesh")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelNoSkinningTest,
    "SuperDexPhysics.DeformableRenderModel.AVisualMeshWithoutSkinningIsNotLinked",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelNoSkinningTest::RunTest(const FString& Parameters) {
  FMochiMeshData VisualMesh = MakeVisualMesh();
  VisualMesh.Skinning.Reset();
  const FString Reason = CheckRenderModelLink(MakeAttributes(), &VisualMesh);
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *Reason),
      Reason.Contains(TEXT("no deformation embedding")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelVertexCountTest,
    "SuperDexPhysics.DeformableRenderModel.ADifferentVertexCountIsNotLinked",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelVertexCountTest::RunTest(const FString& Parameters) {
  FRenderModelAttributes Attributes = MakeAttributes();
  Attributes.NumVertices = 3;
  const FMochiMeshData VisualMesh = MakeVisualMesh();
  const FString Reason = CheckRenderModelLink(Attributes, &VisualMesh);
  TestTrue(
      *FString::Printf(TEXT("Reason: %s"), *Reason),
      Reason.Contains(TEXT("has 3 vertices, but the visual mesh has 4")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelAttributeCountTest,
    "SuperDexPhysics.DeformableRenderModel.ADifferentUVOrColorCountIsNotLinked",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelAttributeCountTest::RunTest(const FString& Parameters) {
  const FMochiMeshData VisualMesh = MakeVisualMesh();
  FRenderModelAttributes ShortUVs = MakeAttributes();
  ShortUVs.UVs = ExpectedUVs();
  ShortUVs.UVs.Pop();
  FRenderModelAttributes ShortColors = MakeAttributes();
  ShortColors.Colors = ExpectedColors();
  ShortColors.Colors.Pop();
  for (const FRenderModelAttributes& Attributes : {ShortUVs, ShortColors}) {
    const FString Reason = CheckRenderModelLink(Attributes, &VisualMesh);
    TestTrue(
        *FString::Printf(TEXT("Reason: %s"), *Reason), Reason.Contains(TEXT("UV or color count")));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexDeformableRenderModelTrianglesTest,
    "SuperDexPhysics.DeformableRenderModel.ADifferentTriangleIsNotLinked",
    SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS)

bool FSuperDexDeformableRenderModelTrianglesTest::RunTest(const FString& Parameters) {
  FRenderModelAttributes Attributes = MakeAttributes();
  Attributes.TriangleIndices[5] = 1;
  FMochiMeshData VisualMesh = MakeVisualMesh();
  TestTrue(
      TEXT("Different indexed triangle"),
      CheckRenderModelLink(Attributes, &VisualMesh).Contains(TEXT("triangles")));

  TArray<int32>& Connectivity = VisualMesh.Connectivity;
  for (int32 i = 0; i < Connectivity.Num(); i += 3) {
    Swap(Connectivity[i], Connectivity[i + 1]);
  }
  TestEqual(
      TEXT("Global winding reversal"),
      CheckRenderModelLink(MakeAttributes(), &VisualMesh),
      FString());
  Swap(Connectivity[0], Connectivity[1]);
  TestTrue(
      TEXT("Mixed winding changes"),
      CheckRenderModelLink(MakeAttributes(), &VisualMesh).Contains(TEXT("triangles")));

  Attributes = MakeAttributes();
  Attributes.TriangleIndices.Pop();
  VisualMesh = MakeVisualMesh();
  VisualMesh.Connectivity.Pop();
  TestTrue(
      TEXT("Incomplete triangle"),
      CheckRenderModelLink(Attributes, &VisualMesh).Contains(TEXT("triangles")));
  return true;
}

#undef SUPERDEX_DEFORMABLE_RENDER_MODEL_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
