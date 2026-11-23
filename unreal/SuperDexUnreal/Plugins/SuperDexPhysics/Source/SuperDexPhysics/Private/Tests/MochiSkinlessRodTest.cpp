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

#include "MochiActorObject.h"
#include "MochiDynamicMeshUtil.h"
#include "MochiPhysicsModule.h"
#include "MochiSceneObject.h"

#include "Misc/AutomationTest.h"
#include "Misc/ScopeExit.h"

#include <mochi_physics/mochi_physics_experimental.h>

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiSkinlessRodTest,
    "SuperDexPhysics.Rods.SkinlessRendering",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)

bool FMochiSkinlessRodTest::RunTest(const FString& Parameters) {
  mochi::Context* Context = IMochiPhysicsModule::Get()->GetMochiContext();
  if (!TestNotNull(TEXT("Mochi context"), Context)) {
    return false;
  }
  UMochiSceneObject* SceneObject = UMochiSceneObject::CreateScene(
      GetTransientPackage(), EMochiSceneType::Sync, TEXT("SkinlessRodTest"), {}, false);
  ON_SCOPE_EXIT {
    SceneObject->DestroyScene();
  };

  AddExpectedError(
      TEXT("has no drawable visual mesh or contact skin"),
      EAutomationExpectedErrorFlags::Contains,
      2);
  const mochi::Real3 Nodes[] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f}};
  for (bool bHasSkin : {false, true}) {
    mochi::Error Error;
    mochi::ModelData Model = mochi::experimental::GenerateTubularRodModelData(
        mochi::Span<mochi::Real3 const>(Nodes, 3), {}, mochi::real(0.01), 8, false, Error);
    if (!bHasSkin) {
      Model.contactSkinMesh.reset();
    }
    const mochi::ShapeHandle Shape = Context->CreateModelShape(Model, Error);
    if (!TestTrue(TEXT("Rod shape created"), Error.IsOK() && Shape.IsValid())) {
      return false;
    }

    UE::Geometry::FDynamicMesh3 Mesh;
    PopulateMeshFromMochiShape(&Mesh, Shape, false, true);
    TestEqual(TEXT("Only authored skin produces triangles"), Mesh.VertexCount() > 0, bHasSkin);

    if (!bHasSkin) {
      SceneObject->QueueOrExecuteCommand([&](mochi::Scene* Scene) {
        mochi::RodActorParams Params;
        Params.shape = Shape;
        Params.useContactSkin = true;
        mochi::Error MissingSkinError;
        TestNull(
            TEXT("Contact sampling still requires a skin"),
            Scene->CreateRodActor(Params, MissingSkinError));
        TestFalse(TEXT("Missing contact skin is a configuration error"), MissingSkinError.IsOK());
      });
    }

    for (bool bBorrowed : {false, true}) {
      UMochiActorObject* ActorObject = nullptr;
      if (bBorrowed) {
        mochi::ActorHandle Handle;
        SceneObject->QueueOrExecuteCommand([&](mochi::Scene* Scene) {
          mochi::RodActorParams Params;
          Params.name = "BorrowedRod";
          Params.shape = Shape;
          Params.hasGravity = false;
          mochi::Actor* Actor = Scene->CreateRodActor(Params, Error);
          if (Actor) {
            Handle = Actor->GetHandle();
          }
        });
        ActorObject = UMochiActorObject::CreateFromExistingActor(
            GetTransientPackage(), Handle, true, false, true, true, SceneObject, true);
      } else {
        ActorObject = UMochiActorObject::CreateRod(
            GetTransientPackage(),
            TEXT("OwnedRod"),
            FTransform::Identity,
            Shape,
            nullptr,
            nullptr,
            EMochiColliderType::None,
            FMochiLayer::GenericObject,
            false,
            nullptr,
            false,
            EMochiActorSegmentElementType::Default,
            EMochiActorBoundaryElementType::Default,
            true,
            true,
            true,
            SceneObject);
      }
      if (!TestNotNull(TEXT("Rod wrapper created"), ActorObject)) {
        return false;
      }
      ActorObject->bDestroyMochiActorOnObjectDestruction = !bBorrowed;
      const mochi::ActorHandle Handle = ActorObject->GetMochiActorHandle();
      ON_SCOPE_EXIT {
        ActorObject->DestroyMochiResources();
        if (bBorrowed) {
          SceneObject->QueueOrExecuteCommand(
              [Handle](mochi::Scene* Scene) { Scene->DestroyActor(Handle); });
        }
      };

      SceneObject->QueueOrExecuteActorCommand(Handle, [&](mochi::Actor* Actor) {
        TestTrue(
            TEXT("Contact user data initialized"), Actor->GetUserData() == ActorObject->GetOuter());
      });
      Mesh.Clear();
      PopulateMeshFromMochiActor(&Mesh, Handle, SceneObject, true, true);
      TestEqual(TEXT("Archive mesh matches authored geometry"), Mesh.VertexCount() > 0, bHasSkin);

      for (int Step = 0; Step < 3; ++Step) {
        SceneObject->StepPause();
        bool bGotQueryData = false;
        ActorObject->TryGetQueryData([&](const FMochiActorQueryData& Data) {
          bGotQueryData = true;
          TestEqual(
              TEXT("Render positions queried only for skin"),
              !Data.SurfaceNodePositions.IsEmpty(),
              bHasSkin);
          TestEqual(
              TEXT("Render normals queried only for skin"), !Data.NodeNormals.IsEmpty(), bHasSkin);
        });
        TestTrue(TEXT("Invisible rod still publishes query data"), bGotQueryData);
        SceneObject->QueueOrExecuteActorCommand(Handle, [&](mochi::Actor* Actor) {
          mochi::Error ContactError;
          static_cast<void>(Actor->GetContactPointsWorld(ContactError));
          TestTrue(TEXT("Contact query remains available"), ContactError.IsOK());
        });
      }
    }
  }
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
