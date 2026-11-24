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

#include "MochiConversions.h"
#include "MochiDynamicMeshUtil.h"

#include "DynamicMesh/DynamicMesh3.h"
#include "DynamicMesh/DynamicMeshAttributeSet.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#define MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

namespace {

using UE::Geometry::FDynamicMesh3;

FDynamicMesh3 MakeTwoTriangleMesh() {
  FDynamicMesh3 Mesh;
  Mesh.AppendVertex(FVector3d(0, 0, 0));
  Mesh.AppendVertex(FVector3d(100, 0, 0));
  Mesh.AppendVertex(FVector3d(100, 100, 0));
  Mesh.AppendVertex(FVector3d(0, 100, 0));
  Mesh.AppendTriangle(0, 1, 2);
  Mesh.AppendTriangle(0, 2, 3);
  Mesh.EnableVertexNormals(FVector3f::UnitZ());
  return Mesh;
}

const TArray<FVector2f> UVs = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
const TArray<FLinearColor> Colors = {
    {1.f, 0.f, 0.f, 1.f},
    {0.f, 1.f, 0.f, 1.f},
    {0.f, 0.f, 1.f, 1.f},
    {1.f, 1.f, 1.f, 0.5f}};
// Mochi query data for the four nodes, in Mochi units and axes.
const TArray<mochi::real> NodePositions = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
const TArray<mochi::real> NodeNormals = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1};

FVector3f ExpectedNormal(int32 Node) {
  return FMochiToUnrealConverter_float().DirectionToUnreal(
      {NodeNormals[3 * Node], NodeNormals[3 * Node + 1], NodeNormals[3 * Node + 2]});
}

FVector ExpectedPosition(int32 Node) {
  return FVector(
      FMochiToUnrealConverter_float().TranslationToUnreal(
          {NodePositions[3 * Node], NodePositions[3 * Node + 1], NodePositions[3 * Node + 2]}));
}

// Whether each overlay triangle of Overlay has the corners of its mesh triangle.
template <typename OverlayType>
bool CornersAreVertices(const FDynamicMesh3& Mesh, const OverlayType& Overlay) {
  for (const int TriIdx : Mesh.TriangleIndicesItr()) {
    if (!(Overlay.GetTriangle(TriIdx) == Mesh.GetTriangle(TriIdx))) {
      return false;
    }
  }
  return true;
}

bool HasColorsOfNodes(const FDynamicMesh3& Mesh) {
  const UE::Geometry::FDynamicMeshColorOverlay* ColorOverlay = Mesh.Attributes()->PrimaryColors();
  if (!ColorOverlay || !CornersAreVertices(Mesh, *ColorOverlay)) {
    return false;
  }
  for (int32 VertIdx = 0; VertIdx < Colors.Num(); ++VertIdx) {
    if (ColorOverlay->GetElement(VertIdx) != FVector4f(Colors[VertIdx])) {
      return false;
    }
  }
  return true;
}

bool HasUVsOfNodes(const FDynamicMesh3& Mesh) {
  const UE::Geometry::FDynamicMeshUVOverlay* UvOverlay = Mesh.Attributes()->PrimaryUV();
  if (!CornersAreVertices(Mesh, *UvOverlay)) {
    return false;
  }
  for (int32 VertIdx = 0; VertIdx < UVs.Num(); ++VertIdx) {
    if (UvOverlay->GetElement(VertIdx) != UVs[VertIdx]) {
      return false;
    }
  }
  return true;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureAttributesTest,
    "SuperDexPhysics.SoftMeshTexture.OverlayElementsAreVertices",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureAttributesTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 Mesh = MakeTwoTriangleMesh();

  EnableRenderModelAttributes(Mesh, UVs, Colors);

  const UE::Geometry::FDynamicMeshNormalOverlay* NormalOverlay =
      Mesh.Attributes()->PrimaryNormals();
  TestTrue(TEXT("Normal corners are vertices"), CornersAreVertices(Mesh, *NormalOverlay));
  for (int32 VertIdx = 0; VertIdx < Mesh.VertexCount(); ++VertIdx) {
    TestEqual(TEXT("Initial normal"), NormalOverlay->GetElement(VertIdx), FVector3f::UnitZ());
  }
  TestTrue(TEXT("UVs of the nodes"), HasUVsOfNodes(Mesh));
  TestTrue(TEXT("Colors of the nodes"), HasColorsOfNodes(Mesh));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureNoColorsTest,
    "SuperDexPhysics.SoftMeshTexture.UVsWithoutColorsAddNoColorOverlay",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureNoColorsTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 Mesh = MakeTwoTriangleMesh();

  EnableRenderModelAttributes(Mesh, UVs, {});

  TestTrue(TEXT("UVs of the nodes"), HasUVsOfNodes(Mesh));
  TestFalse(TEXT("Has a color overlay"), Mesh.Attributes()->HasPrimaryColors());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureColorsWithoutUVsTest,
    "SuperDexPhysics.SoftMeshTexture.ColorsWithoutUVsLeaveUVsUnset",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureColorsWithoutUVsTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 Mesh = MakeTwoTriangleMesh();

  EnableRenderModelAttributes(Mesh, {}, Colors);

  TestTrue(TEXT("Colors of the nodes"), HasColorsOfNodes(Mesh));
  TestTrue(
      TEXT("Normal corners are vertices"),
      CornersAreVertices(Mesh, *Mesh.Attributes()->PrimaryNormals()));
  const UE::Geometry::FDynamicMeshUVOverlay* UvOverlay = Mesh.Attributes()->PrimaryUV();
  TestEqual(TEXT("UV elements"), UvOverlay->ElementCount(), 0);
  for (const int TriIdx : Mesh.TriangleIndicesItr()) {
    TestFalse(TEXT("UV triangle is set"), UvOverlay->IsSetTriangle(TriIdx));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureNormalUpdateTest,
    "SuperDexPhysics.SoftMeshTexture.QueryUpdatesOverlayNormalsOnly",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureNormalUpdateTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 Mesh = MakeTwoTriangleMesh();
  EnableRenderModelAttributes(Mesh, UVs, Colors);

  UpdateDynamicMeshFromMochiQuery(NodePositions, NodeNormals, true, true, FVector::OneVector, Mesh);

  const UE::Geometry::FDynamicMeshNormalOverlay* NormalOverlay =
      Mesh.Attributes()->PrimaryNormals();
  for (int32 VertIdx = 0; VertIdx < Mesh.VertexCount(); ++VertIdx) {
    TestEqual(TEXT("Overlay normal"), NormalOverlay->GetElement(VertIdx), ExpectedNormal(VertIdx));
  }
  TestTrue(TEXT("UVs of the nodes"), HasUVsOfNodes(Mesh));
  TestTrue(TEXT("Colors of the nodes"), HasColorsOfNodes(Mesh));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureEmptyNormalOverlayTest,
    "SuperDexPhysics.SoftMeshTexture.QueryIgnoresEmptyNormalOverlay",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureEmptyNormalOverlayTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 Mesh = MakeTwoTriangleMesh();
  Mesh.EnableAttributes();
  UE::Geometry::FDynamicMeshNormalOverlay* NormalOverlay = Mesh.Attributes()->PrimaryNormals();
  TestEqual(TEXT("Initial overlay elements"), NormalOverlay->ElementCount(), 0);

  UpdateDynamicMeshFromMochiQuery(
      NodePositions, NodeNormals, true, false, FVector::OneVector, Mesh);

  TestEqual(TEXT("Overlay elements"), NormalOverlay->ElementCount(), 0);
  for (const int TriIdx : Mesh.TriangleIndicesItr()) {
    TestFalse(TEXT("Overlay triangle is set"), NormalOverlay->IsSetTriangle(TriIdx));
  }
  for (int32 VertIdx = 0; VertIdx < Mesh.VertexCount(); ++VertIdx) {
    TestTrue(TEXT("Position"), Mesh.GetVertex(VertIdx).Equals(ExpectedPosition(VertIdx)));
    TestEqual(TEXT("Vertex normal"), Mesh.GetVertexNormal(VertIdx), ExpectedNormal(VertIdx));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureUntexturedUpdateTest,
    "SuperDexPhysics.SoftMeshTexture.QueryLeavesUntexturedMeshUntextured",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureUntexturedUpdateTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 Mesh = MakeTwoTriangleMesh();

  UpdateDynamicMeshFromMochiQuery(
      NodePositions, NodeNormals, true, false, FVector::OneVector, Mesh);

  TestFalse(TEXT("Has attributes"), Mesh.HasAttributes());
  TestEqual(TEXT("Triangle count"), Mesh.TriangleCount(), 2);
  for (int32 VertIdx = 0; VertIdx < Mesh.VertexCount(); ++VertIdx) {
    TestTrue(TEXT("Position"), Mesh.GetVertex(VertIdx).Equals(ExpectedPosition(VertIdx)));
    TestEqual(TEXT("Vertex normal"), Mesh.GetVertexNormal(VertIdx), ExpectedNormal(VertIdx));
  }
  return true;
}

#undef MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
