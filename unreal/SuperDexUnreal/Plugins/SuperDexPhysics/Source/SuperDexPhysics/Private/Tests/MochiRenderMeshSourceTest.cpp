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

#include "MochiDynamicMeshUtil.h"

#include "Misc/AutomationTest.h"
#include "Misc/ScopeExit.h"
#include "MochiPhysicsModule.h"

#include <mochi_physics/mochi_physics.h>

#if WITH_DEV_AUTOMATION_TESTS

#define MOCHI_RENDER_MESH_SOURCE_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

namespace {

const mochi::real TriangleCoordinates[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
const int TriangleConnectivity[] = {0, 1, 2};

mochi::MeshDataView MakeTriangle(bool bSkinned) {
  mochi::MeshDataView Mesh;
  Mesh.nodesPerElement = 3;
  Mesh.coordinates = mochi::Span<mochi::real const>(TriangleCoordinates, 9);
  Mesh.connectivity = mochi::Span<int const>(TriangleConnectivity, 3);
  if (bSkinned) {
    Mesh.skinning = mochi::SkinningDataView{};
  }
  return Mesh;
}

const mochi::MeshDataView NoMesh{};
const mochi::MeshDataView UnskinnedTriangle = MakeTriangle(false);
const mochi::MeshDataView SkinnedTriangle = MakeTriangle(true);

constexpr bool bNodeQueryPosed = true;
constexpr bool bRigidlyPosed = false;

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiRenderMeshSourceTest,
    "SuperDexPhysics.RenderMeshSource.Choice",
    MOCHI_RENDER_MESH_SOURCE_TEST_FLAGS)

bool FMochiRenderMeshSourceTest::RunTest(const FString& Parameters) {
  const auto Expect =
      [this](const TCHAR* What, EMochiRenderMeshSource Actual, EMochiRenderMeshSource Expected) {
        TestEqual(What, static_cast<int32>(Actual), static_cast<int32>(Expected));
      };
  Expect(
      TEXT("Skinned visual mesh preferred"),
      ChooseRenderMeshSource(SkinnedTriangle, SkinnedTriangle, bNodeQueryPosed),
      EMochiRenderMeshSource::Visual);
  Expect(
      TEXT("No visual mesh"),
      ChooseRenderMeshSource(NoMesh, SkinnedTriangle, bNodeQueryPosed),
      EMochiRenderMeshSource::ContactSkin);
  Expect(
      TEXT("Unskinned visual mesh does not deform"),
      ChooseRenderMeshSource(UnskinnedTriangle, SkinnedTriangle, bNodeQueryPosed),
      EMochiRenderMeshSource::ContactSkin);
  Expect(
      TEXT("Unskinned contact skin does not deform"),
      ChooseRenderMeshSource(NoMesh, UnskinnedTriangle, bNodeQueryPosed),
      EMochiRenderMeshSource::Surface);
  Expect(
      TEXT("Rigidly posed mesh draws an unskinned visual mesh"),
      ChooseRenderMeshSource(UnskinnedTriangle, NoMesh, bRigidlyPosed),
      EMochiRenderMeshSource::Visual);
  Expect(
      TEXT("Rigidly posed mesh draws an unskinned contact skin"),
      ChooseRenderMeshSource(NoMesh, UnskinnedTriangle, bRigidlyPosed),
      EMochiRenderMeshSource::ContactSkin);
  Expect(
      TEXT("Nothing but the surface"),
      ChooseRenderMeshSource(NoMesh, NoMesh, bNodeQueryPosed),
      EMochiRenderMeshSource::Surface);
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiDeformableActorTypeTest,
    "SuperDexPhysics.RenderMeshSource.DeformableActorTypes",
    MOCHI_RENDER_MESH_SOURCE_TEST_FLAGS)

bool FMochiDeformableActorTypeTest::RunTest(const FString& Parameters) {
  TestTrue(TEXT("Soft"), IsDeformableActorType(mochi::ActorType::Soft));
  TestTrue(TEXT("Shell"), IsDeformableActorType(mochi::ActorType::Shell));
  TestTrue(TEXT("Rod"), IsDeformableActorType(mochi::ActorType::Rod));
  TestFalse(TEXT("Rigid"), IsDeformableActorType(mochi::ActorType::Rigid));
  TestFalse(TEXT("Articulated"), IsDeformableActorType(mochi::ActorType::Articulated));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiActorRenderMeshSourceTest,
    "SuperDexPhysics.RenderMeshSource.ActorQueryAvailability",
    (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter))

bool FMochiActorRenderMeshSourceTest::RunTest(const FString& Parameters) {
  mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  if (!TestNotNull(TEXT("Mochi context"), MochiContext)) {
    return false;
  }
  mochi::Scene* Scene = MochiContext->CreateScene("RenderMeshSourceTest");
  ON_SCOPE_EXIT {
    MochiContext->DestroyScene(Scene);
  };

  mochi::ModelData Model;
  Model.mesh.emplace();
  Model.mesh->nodesPerElement = 4;
  Model.mesh->coordinates = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  Model.mesh->connectivity = {0, 1, 2, 3};
  Model.contactSkinMesh.emplace();
  Model.contactSkinMesh->nodesPerElement = 3;
  Model.contactSkinMesh->coordinates = Model.mesh->coordinates;
  Model.contactSkinMesh->connectivity = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
  Model.contactSkinMesh->skinning.emplace();
  Model.contactSkinMesh->skinning->weightsPerNode = 4;
  Model.contactSkinMesh->skinning->indices = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  Model.contactSkinMesh->skinning->weights = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  mochi::Error Error;
  const mochi::ShapeHandle Shape = MochiContext->CreateModelShape(Model, Error);
  if (!TestTrue(TEXT("Contact-skin shape created"), Error.IsOK() && Shape.IsValid())) {
    return false;
  }
  mochi::RigidActorParams Params;
  Params.shape = Shape;
  Params.density = mochi::real(1);
  Params.colliderType = mochi::ColliderType::None;
  mochi::Actor* Actor = Scene->CreateRigidActor(Params, Error);
  if (!TestTrue(TEXT("Rigid actor created"), Error.IsOK() && Actor)) {
    return false;
  }
  // Contact skin node queries are documented as deformable-only.
  const FMochiRenderMesh RigidMesh = GetDrawnMochiMesh(*Actor, true, Error);
  TestTrue(TEXT("Rigid mesh resolved"), Error.IsOK() && RigidMesh.Shape.IsValid());
  TestEqual(TEXT("Rigid mesh uses surface"), RigidMesh.Source, EMochiRenderMeshSource::Surface);
  TestEqual(TEXT("Rigid mesh keeps vertex indexing"), RigidMesh.Data.GetNumNodes(), 4);

  mochi::SoftActorParams SoftParams;
  SoftParams.shape = Shape;
  SoftParams.useContactSkin = false;
  mochi::Actor* SoftActor = Scene->CreateSoftActor(SoftParams, Error);
  if (!TestTrue(TEXT("Soft actor created"), Error.IsOK() && SoftActor)) {
    return false;
  }
  const FMochiRenderMesh SoftMesh = GetDrawnMochiMesh(*SoftActor, true, Error);
  TestTrue(TEXT("Soft mesh resolved"), Error.IsOK());
  TestEqual(
      TEXT("Deformable queries its skin even when not used for contact"),
      SoftMesh.Source,
      EMochiRenderMeshSource::ContactSkin);
  return true;
}

#undef MOCHI_RENDER_MESH_SOURCE_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
