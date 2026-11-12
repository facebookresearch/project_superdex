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
  FDynamicMesh3 mesh;
  mesh.AppendVertex(FVector3d(0, 0, 0));
  mesh.AppendVertex(FVector3d(100, 0, 0));
  mesh.AppendVertex(FVector3d(100, 100, 0));
  mesh.AppendVertex(FVector3d(0, 100, 0));
  mesh.AppendTriangle(0, 1, 2);
  mesh.AppendTriangle(0, 2, 3);
  mesh.EnableVertexNormals(FVector3f::UnitZ());
  return mesh;
}

const TArray<FVector2f> kUVs = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
const TArray<FLinearColor> kColors = {
    {1.f, 0.f, 0.f, 1.f},
    {0.f, 1.f, 0.f, 1.f},
    {0.f, 0.f, 1.f, 1.f},
    {1.f, 1.f, 1.f, 0.5f}};
// Mochi query data for the four nodes, in Mochi units and axes.
const TArray<mochi::real> kNodePositions = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
const TArray<mochi::real> kNodeNormals = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1};

FVector3f ExpectedNormal(int32 Node) {
  return FMochiToUnrealConverter_float().DirectionToUnreal(
      {kNodeNormals[3 * Node], kNodeNormals[3 * Node + 1], kNodeNormals[3 * Node + 2]});
}

FVector ExpectedPosition(int32 Node) {
  return FVector(
      FMochiToUnrealConverter_float().TranslationToUnreal(
          {kNodePositions[3 * Node], kNodePositions[3 * Node + 1], kNodePositions[3 * Node + 2]}));
}

// Whether each overlay triangle of Overlay has the corners of its mesh triangle.
template <typename OverlayType>
bool CornersAreVertices(const FDynamicMesh3& Mesh, const OverlayType& Overlay) {
  for (const int triIdx : Mesh.TriangleIndicesItr()) {
    if (!(Overlay.GetTriangle(triIdx) == Mesh.GetTriangle(triIdx))) {
      return false;
    }
  }
  return true;
}

bool HasColorsOfNodes(const FDynamicMesh3& Mesh) {
  const UE::Geometry::FDynamicMeshColorOverlay* colorOverlay = Mesh.Attributes()->PrimaryColors();
  if (!colorOverlay || !CornersAreVertices(Mesh, *colorOverlay)) {
    return false;
  }
  for (int32 vertIdx = 0; vertIdx < kColors.Num(); ++vertIdx) {
    if (colorOverlay->GetElement(vertIdx) != FVector4f(kColors[vertIdx])) {
      return false;
    }
  }
  return true;
}

bool HasUVsOfNodes(const FDynamicMesh3& Mesh) {
  const UE::Geometry::FDynamicMeshUVOverlay* uvOverlay = Mesh.Attributes()->PrimaryUV();
  if (!CornersAreVertices(Mesh, *uvOverlay)) {
    return false;
  }
  for (int32 vertIdx = 0; vertIdx < kUVs.Num(); ++vertIdx) {
    if (uvOverlay->GetElement(vertIdx) != kUVs[vertIdx]) {
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
  FDynamicMesh3 mesh = MakeTwoTriangleMesh();

  EnableSoftMeshRenderModelAttributes(mesh, kUVs, kColors);

  const UE::Geometry::FDynamicMeshNormalOverlay* normalOverlay =
      mesh.Attributes()->PrimaryNormals();
  TestTrue(TEXT("Normal corners are vertices"), CornersAreVertices(mesh, *normalOverlay));
  for (int32 vertIdx = 0; vertIdx < mesh.VertexCount(); ++vertIdx) {
    TestEqual(TEXT("Initial normal"), normalOverlay->GetElement(vertIdx), FVector3f::UnitZ());
  }
  TestTrue(TEXT("UVs of the nodes"), HasUVsOfNodes(mesh));
  TestTrue(TEXT("Colors of the nodes"), HasColorsOfNodes(mesh));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureNoColorsTest,
    "SuperDexPhysics.SoftMeshTexture.UVsWithoutColorsAddNoColorOverlay",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureNoColorsTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 mesh = MakeTwoTriangleMesh();

  EnableSoftMeshRenderModelAttributes(mesh, kUVs, {});

  TestTrue(TEXT("UVs of the nodes"), HasUVsOfNodes(mesh));
  TestFalse(TEXT("Has a color overlay"), mesh.Attributes()->HasPrimaryColors());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureColorsWithoutUVsTest,
    "SuperDexPhysics.SoftMeshTexture.ColorsWithoutUVsLeaveUVsUnset",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureColorsWithoutUVsTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 mesh = MakeTwoTriangleMesh();

  EnableSoftMeshRenderModelAttributes(mesh, {}, kColors);

  TestTrue(TEXT("Colors of the nodes"), HasColorsOfNodes(mesh));
  TestTrue(
      TEXT("Normal corners are vertices"),
      CornersAreVertices(mesh, *mesh.Attributes()->PrimaryNormals()));
  const UE::Geometry::FDynamicMeshUVOverlay* uvOverlay = mesh.Attributes()->PrimaryUV();
  TestEqual(TEXT("UV elements"), uvOverlay->ElementCount(), 0);
  for (const int triIdx : mesh.TriangleIndicesItr()) {
    TestFalse(TEXT("UV triangle is set"), uvOverlay->IsSetTriangle(triIdx));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureNormalUpdateTest,
    "SuperDexPhysics.SoftMeshTexture.QueryUpdatesOverlayNormalsOnly",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureNormalUpdateTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 mesh = MakeTwoTriangleMesh();
  EnableSoftMeshRenderModelAttributes(mesh, kUVs, kColors);

  UpdateDynamicMeshFromMochiQuery(kNodePositions, kNodeNormals, true, FVector::OneVector, mesh);

  const UE::Geometry::FDynamicMeshNormalOverlay* normalOverlay =
      mesh.Attributes()->PrimaryNormals();
  for (int32 vertIdx = 0; vertIdx < mesh.VertexCount(); ++vertIdx) {
    TestEqual(TEXT("Overlay normal"), normalOverlay->GetElement(vertIdx), ExpectedNormal(vertIdx));
  }
  TestTrue(TEXT("UVs of the nodes"), HasUVsOfNodes(mesh));
  TestTrue(TEXT("Colors of the nodes"), HasColorsOfNodes(mesh));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSoftMeshTextureUntexturedUpdateTest,
    "SuperDexPhysics.SoftMeshTexture.QueryLeavesUntexturedMeshUntextured",
    MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS)

bool FMochiSoftMeshTextureUntexturedUpdateTest::RunTest(const FString& Parameters) {
  FDynamicMesh3 mesh = MakeTwoTriangleMesh();

  UpdateDynamicMeshFromMochiQuery(kNodePositions, kNodeNormals, true, FVector::OneVector, mesh);

  TestFalse(TEXT("Has attributes"), mesh.HasAttributes());
  TestEqual(TEXT("Triangle count"), mesh.TriangleCount(), 2);
  for (int32 vertIdx = 0; vertIdx < mesh.VertexCount(); ++vertIdx) {
    TestTrue(TEXT("Position"), mesh.GetVertex(vertIdx).Equals(ExpectedPosition(vertIdx)));
    TestEqual(TEXT("Vertex normal"), mesh.GetVertexNormal(vertIdx), ExpectedNormal(vertIdx));
  }
  return true;
}

#undef MOCHI_SOFT_MESH_TEXTURE_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
