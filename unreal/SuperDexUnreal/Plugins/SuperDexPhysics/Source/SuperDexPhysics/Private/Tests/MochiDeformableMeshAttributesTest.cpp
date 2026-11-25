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

#include "MochiDeformableMeshComponent.h"
#include "MochiModel.h"
#include "MochiPhysicsModule.h"
#include "MochiRodMeshComponent.h"
#include "MochiShellMeshComponent.h"
#include "MochiSoftMeshComponent.h"

#include "DynamicMesh/DynamicMeshAttributeSet.h"
#include "Misc/AutomationTest.h"
#include "UDynamicMesh.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"

#include <mochi_physics/mochi_physics_experimental.h>

#if WITH_EDITOR && WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiDeformableMeshAttributesTest,
    "SuperDexPhysics.DeformableMeshAttributes.OnlyVisualMeshReceivesAttributes",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)

bool FMochiDeformableMeshAttributesTest::RunTest(const FString& Parameters) {
  if (!TestNotNull(TEXT("Mochi context"), IMochiPhysicsModule::Get()->GetMochiContext())) {
    return false;
  }
  const mochi::Real3 Nodes[] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f}};
  mochi::Error Error;
  const mochi::ModelData ContactModel = mochi::experimental::GenerateTubularRodModelData(
      mochi::Span<mochi::Real3 const>(Nodes, 3), {}, mochi::real(0.01), 8, false, Error);
  const mochi::ModelData VisualModel = mochi::experimental::GenerateTubularRodModelData(
      mochi::Span<mochi::Real3 const>(Nodes, 3), {}, mochi::real(0.02), 8, false, Error);
  if (!TestTrue(TEXT("Authored meshes generated"), Error.IsOK())) {
    return false;
  }
  const int32 NumNodes = ContactModel.contactSkinMesh->GetNumNodes();
  TestEqual(
      TEXT("Fallback and visual meshes have equal node counts"),
      VisualModel.contactSkinMesh->GetNumNodes(),
      NumNodes);
  FProperty* ModelProperty = FindFProperty<FProperty>(
      UMochiDeformableMeshComponent::StaticClass(),
      GET_MEMBER_NAME_CHECKED(UMochiDeformableMeshComponent, Model));
  if (!TestNotNull(TEXT("Model property"), ModelProperty)) {
    return false;
  }

  AddExpectedError(
      TEXT("not the matching visual mesh"), EAutomationExpectedErrorFlags::Contains, 6);
  for (UClass* ComponentClass :
       {UMochiSoftMeshComponent::StaticClass(),
        UMochiShellMeshComponent::StaticClass(),
        UMochiRodMeshComponent::StaticClass()}) {
    TStrongObjectPtr<UMochiDeformableMeshComponent> Component(
        NewObject<UMochiDeformableMeshComponent>(GetTransientPackage(), ComponentClass));
    TStrongObjectPtr<UMochiModel> Model(NewObject<UMochiModel>());
    Component->Model = Model.Get();
    for (int32 Node = 0; Node < NumNodes; ++Node) {
      const float U = static_cast<float>(Node) / NumNodes;
      Component->RenderModelUVs.Add(FVector2f(U, 1.f - U));
      Component->RenderModelColors.Add(FLinearColor(U, 0.25f, 1.f - U, 1.f));
    }

    enum class EVisualMesh { Skinned, Unskinned, Missing };
    for (EVisualMesh Visual :
         {EVisualMesh::Skinned, EVisualMesh::Unskinned, EVisualMesh::Missing}) {
      mochi::ModelData Data = ContactModel;
      if (Visual != EVisualMesh::Missing) {
        Data.visualMesh = VisualModel.contactSkinMesh;
        if (Visual == EVisualMesh::Unskinned) {
          Data.visualMesh->skinning.reset();
        }
      }
      Model->ModelData = ToUnreal(Data);
      FPropertyChangedEvent Event(ModelProperty);
      static_cast<UObject*>(Component.Get())->PostEditChangeProperty(Event);
      Component->GetDynamicMesh()->ProcessMesh([&](const UE::Geometry::FDynamicMesh3& Mesh) {
        TestEqual(TEXT("Preview uses the expected node count"), Mesh.VertexCount(), NumNodes);
        const bool bVisual = Visual == EVisualMesh::Skinned;
        const UE::Geometry::FDynamicMeshUVOverlay* UvOverlay =
            Mesh.HasAttributes() ? Mesh.Attributes()->PrimaryUV() : nullptr;
        const UE::Geometry::FDynamicMeshColorOverlay* ColorOverlay =
            Mesh.HasAttributes() ? Mesh.Attributes()->PrimaryColors() : nullptr;
        TestEqual(
            TEXT("Only visual nodes receive UVs"),
            UvOverlay ? UvOverlay->ElementCount() : 0,
            bVisual ? NumNodes : 0);
        TestEqual(
            TEXT("Only visual nodes receive colors"),
            ColorOverlay ? ColorOverlay->ElementCount() : 0,
            bVisual ? NumNodes : 0);
        for (int32 Triangle : Mesh.TriangleIndicesItr()) {
          TestEqual(
              TEXT("Only visual triangles receive UVs"),
              UvOverlay && UvOverlay->IsSetTriangle(Triangle),
              bVisual);
          TestEqual(
              TEXT("Only visual triangles receive colors"),
              ColorOverlay && ColorOverlay->IsSetTriangle(Triangle),
              bVisual);
        }
        if (!bVisual || !UvOverlay || !ColorOverlay) {
          return;
        }
        for (int32 Node = 0; Node < NumNodes; ++Node) {
          TestEqual(
              TEXT("Visual UV"), UvOverlay->GetElement(Node), Component->RenderModelUVs[Node]);
          TestEqual(
              TEXT("Visual color"),
              ColorOverlay->GetElement(Node),
              FVector4f(Component->RenderModelColors[Node]));
        }
      });
    }

    // An empty model clears the prior mesh without blaming its saved attributes for a mismatch.
    Model->ModelData = FMochiModelData{};
    FPropertyChangedEvent Event(ModelProperty);
    static_cast<UObject*>(Component.Get())->PostEditChangeProperty(Event);
    Component->GetDynamicMesh()->ProcessMesh([&](const UE::Geometry::FDynamicMesh3& Mesh) {
      TestEqual(TEXT("Empty model clears the preview"), Mesh.VertexCount(), 0);
      const UE::Geometry::FDynamicMeshUVOverlay* UvOverlay =
          Mesh.HasAttributes() ? Mesh.Attributes()->PrimaryUV() : nullptr;
      TestEqual(
          TEXT("Empty model leaves no UV elements"), UvOverlay ? UvOverlay->ElementCount() : 0, 0);
    });
  }
  return true;
}

#endif // WITH_EDITOR && WITH_DEV_AUTOMATION_TESTS
