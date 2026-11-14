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

#include "MochiConversions.h"
#include "MochiPhysicsModule.h"

#include "DynamicMesh/DynamicMeshAttributeSet.h"
#include "DynamicMesh/MeshNormals.h"

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/utils/profile.h>

static void PopulateMeshFromMochiMeshData(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::MeshDataView const& meshData) {
  MOCHI_PROFILE_SCOPE();
  FMochiToUnrealConverter m2u;
  int const numNodes = meshData.GetNumNodes();
  for (int NodeIdx = 0; NodeIdx < numNodes; ++NodeIdx) {
    mochi::Real3 vertex = mochi::Real3{
        meshData.coordinates[NodeIdx * 3],
        meshData.coordinates[NodeIdx * 3 + 1],
        meshData.coordinates[NodeIdx * 3 + 2],
    };
    // NOTE: Assumes Mochi position data is stored in Mochi units (meters)
    OutMesh->AppendVertex(m2u.TranslationToUnreal(vertex));
  }

  // Initialize faces
  auto const& TriIndices = meshData.connectivity;
  if (m2u.FlipsHandedness()) {
    for (int i = 0; i < TriIndices.size(); i += 3) {
      UE::Geometry::FIndex3i Tri{
          TriIndices[i + 0],
          TriIndices[i + 1],
          TriIndices[i + 2],
      };
      OutMesh->AppendTriangle(Tri);
      // NOTE: result < 0 could be Invalid, NonManifold, or Duplicate; the latter two are probably
      // OK, and some generated meshes trigger this check, so it is off for now.
      // check(result >= 0);
    }
  } else {
    for (int i = 0; i < TriIndices.size(); i += 3) {
      UE::Geometry::FIndex3i Tri{
          TriIndices[i + 2],
          TriIndices[i + 1],
          TriIndices[i + 0],
      };
      OutMesh->AppendTriangle(Tri);
    }
  }
}

static mochi::MeshDataView GetMeshDataForDynamicMesh(
    mochi::Context const& mochiContext,
    mochi::ShapeHandle shape,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  mochi::MeshDataView meshData{};
  if (bUseVisualMeshIfPresent) {
    meshData = mochiContext.GetShapeVisualMesh(shape, error);
    MOCHI_ERROR_RETURN(error, {});
  }
  if (bUseContactSkinIfPresent && meshData.IsEmpty()) {
    mochi::MeshDataView const contactSkin = mochiContext.GetShapeContactSkinMesh(shape, error);
    MOCHI_ERROR_RETURN(error, {});
    if (contactSkin.skinning.has_value()) {
      meshData = contactSkin;
    }
  }
  if (meshData.IsEmpty()) {
    meshData = mochiContext.GetShapeSurfaceMesh(shape, error);
    MOCHI_ERROR_RETURN(error, {});
  }
  MOCHI_ERROR_IF(
      meshData.IsEmpty(),
      error,
      "Cannot populate mesh: shape has no surface mesh data, and no requested visual mesh or "
      "contact skin.");
  MOCHI_ERROR_RETURN(error, {});

  return meshData;
}

void PopulateMeshFromMochiModel(
    UE::Geometry::FDynamicMesh3* OutMesh,
    UMochiModel* MochiModel,
    const FTransform& ShapeTransform,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent) {
  MOCHI_PROFILE_SCOPE();
  if (!MochiModel || !MochiModel->IsValid()) {
    return;
  }
  mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  if (mochiContext) {
    mochi::ErrorLog e{};
    mochi::ShapeHandle shape = MochiModel->GetMochiShape(FVector::OneVector, ShapeTransform);
    MOCHI_ERROR_RETURN(e);

    mochi::MeshDataView meshData = GetMeshDataForDynamicMesh(
        *mochiContext, shape, bUseVisualMeshIfPresent, bUseContactSkinIfPresent, e);
    MOCHI_ERROR_RETURN(e);
    PopulateMeshFromMochiMeshData(OutMesh, meshData);
  }
}

void PopulateMeshFromMochiShape(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ShapeHandle shape,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent) {
  MOCHI_PROFILE_SCOPE();
  if (!shape.IsValid()) {
    return;
  }
  mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  if (mochiContext) {
    mochi::ErrorLog e{};
    mochi::MeshDataView meshData = GetMeshDataForDynamicMesh(
        *mochiContext, shape, bUseVisualMeshIfPresent, bUseContactSkinIfPresent, e);
    MOCHI_ERROR_RETURN(e);
    PopulateMeshFromMochiMeshData(OutMesh, meshData);
  }
}

void SUPERDEXPHYSICS_API PopulateMeshFromMochiActor(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ActorHandle const& MochiActor,
    UMochiSceneObject* MochiScene,
    bool bUseVisualMeshIfPresent,
    bool bUseContactSkinIfPresent) {
  MOCHI_PROFILE_SCOPE();
  if (!MochiActor.IsValid() || !MochiScene->IsValid()) {
    return;
  }
  mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  if (mochiContext) {
    mochi::ShapeHandle shape;
    MochiScene->QueueOrExecuteActorCommand(MochiActor, [&shape](mochi::Actor* actor) {
      mochi::ErrorLog e{};
      shape = actor->GetReferenceShape(e);
    });
    MochiScene->WaitForQueuedCommands();

    mochi::ErrorLog e{};
    mochi::MeshDataView meshData = GetMeshDataForDynamicMesh(
        *mochiContext, shape, bUseVisualMeshIfPresent, bUseContactSkinIfPresent, e);
    MOCHI_ERROR_RETURN(e);
    PopulateMeshFromMochiMeshData(OutMesh, meshData);
  }
}

void ConvertNodePositionsToUnreal(
    const mochi::Span<const mochi::real>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D) {
  check(InMochi.size() % 3 == 0) OutUnreal.SetNum(InMochi.size() / 3);
  FMochiToUnrealConverter m2u;
  for (int i = 0; i < InMochi.size(); i += 3) {
    OutUnreal[i / 3] =
        m2u.TranslationToUnreal({InMochi[i], InMochi[i + 1], InMochi[i + 2]}) / Scale3D;
  }
}

void ConvertNodePositionsToUnreal(
    const TArray<mochi::real>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D) {
  check(InMochi.Num() % 3 == 0) OutUnreal.SetNum(InMochi.Num() / 3);
  FMochiToUnrealConverter m2u;
  for (int i = 0; i < InMochi.Num(); i += 3) {
    OutUnreal[i / 3] =
        m2u.TranslationToUnreal({InMochi[i], InMochi[i + 1], InMochi[i + 2]}) / Scale3D;
  }
}

void ConvertVolumeNodeReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal) {
  mochi::ErrorLog e{};
  mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  const auto shape = MochiModel->GetMochiShape(Scale3D, FTransform::Identity);
  auto const& mesh = mochiContext->GetShapeMesh(shape, e);
  ConvertNodePositionsToUnreal(mesh.coordinates, OutReferencePositionsUnreal, Scale3D);
}

void ConvertElementBarycenterReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal) {
  if (MochiModel) {
    mochi::ErrorLog e{};
    mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
    const auto shape = MochiModel->GetMochiShape(Scale3D, FTransform::Identity);
    auto const mesh = mochiContext->GetShapeMesh(shape, e);
    auto barycenters = mochi::ComputeElementBarycenters(mesh);
    ConvertNodePositionsToUnreal(
        mochi::MakeConstSpan(barycenters), OutReferencePositionsUnreal, Scale3D);
  }
}

void EnableRenderModelAttributes(
    UE::Geometry::FDynamicMesh3& Mesh,
    TConstArrayView<FVector2f> UVs,
    TConstArrayView<FLinearColor> Colors) {
  MOCHI_PROFILE_SCOPE();
  const int32 vertexCount = Mesh.MaxVertexID();
  // Element IDs must start at 0 to match vertex IDs.
  check(!Mesh.HasAttributes());
  check(UVs.IsEmpty() || UVs.Num() == vertexCount);
  check(Colors.IsEmpty() || Colors.Num() == vertexCount);
  // Enabling attributes creates one UV layer and one normal layer. Without UVs, the UV layer's
  // triangles stay unset, as in a mesh with no UVs.
  Mesh.EnableAttributes();
  UE::Geometry::FDynamicMeshAttributeSet* attributes = Mesh.Attributes();
  UE::Geometry::FDynamicMeshNormalOverlay* normalOverlay = attributes->PrimaryNormals();
  UE::Geometry::FDynamicMeshUVOverlay* uvOverlay =
      UVs.IsEmpty() ? nullptr : attributes->PrimaryUV();
  UE::Geometry::FDynamicMeshColorOverlay* colorOverlay = nullptr;
  if (!Colors.IsEmpty()) {
    attributes->EnablePrimaryColors();
    colorOverlay = attributes->PrimaryColors();
  }
  for (int vertIdx = 0; vertIdx < vertexCount; ++vertIdx) {
    normalOverlay->AppendElement(Mesh.GetVertexNormal(vertIdx));
    if (uvOverlay) {
      uvOverlay->AppendElement(UVs[vertIdx]);
    }
    if (colorOverlay) {
      colorOverlay->AppendElement(FVector4f(Colors[vertIdx]));
    }
  }
  for (const int triIdx : Mesh.TriangleIndicesItr()) {
    const UE::Geometry::FIndex3i tri = Mesh.GetTriangle(triIdx);
    normalOverlay->SetTriangle(triIdx, tri);
    if (uvOverlay) {
      uvOverlay->SetTriangle(triIdx, tri);
    }
    if (colorOverlay) {
      colorOverlay->SetTriangle(triIdx, tri);
    }
  }
}

void UpdateDynamicMeshFromMochiQuery(
    const TArray<mochi::real>& NodePositions,
    const TArray<mochi::real>& NodeNormals,
    bool bUseMochiNodeNormals,
    bool bUpdateRenderModelNormalOverlay,
    FVector InvScale3D,
    UE::Geometry::FDynamicMesh3& Mesh) {
  MOCHI_PROFILE_SCOPE();
  const FVector3f invScale3d = static_cast<FVector3f>(InvScale3D);
  UE::Geometry::FDynamicMeshNormalOverlay* normalOverlay =
      bUseMochiNodeNormals && bUpdateRenderModelNormalOverlay ? Mesh.Attributes()->PrimaryNormals()
                                                              : nullptr;
  check(!normalOverlay || normalOverlay->ElementCount() == Mesh.MaxVertexID());
  // single precision converter for speed
  FMochiToUnrealConverter_float m2uf;
  for (int i = 0; i < NodePositions.Num(); i += 3) {
    const int VertIdx = i / 3;
    FVector3f VertPos =
        m2uf.TranslationToUnreal({NodePositions[i], NodePositions[i + 1], NodePositions[i + 2]});
    Mesh.SetVertex(VertIdx, static_cast<FVector>(VertPos * invScale3d));
    if (bUseMochiNodeNormals) {
      FVector3f VertNormal =
          m2uf.DirectionToUnreal({NodeNormals[i], NodeNormals[i + 1], NodeNormals[i + 2]});
      Mesh.SetVertexNormal(VertIdx, VertNormal);
      if (normalOverlay) {
        normalOverlay->SetElement(VertIdx, VertNormal);
      }
    }
  }
}

void PrepareDynamicMeshForRuntime(FVector Scale3D, UE::Geometry::FDynamicMesh3& Mesh) {
  MOCHI_PROFILE_SCOPE();
  // Guard against degenerate scales: any zero axis would produce inf/NaN
  // vertex positions from the inverse-scale divide. Skip the bake entirely
  // in that case so the component renders the untransformed (degenerate but
  // finite) geometry instead of garbage.
  if (Scale3D.X == 0.0 || Scale3D.Y == 0.0 || Scale3D.Z == 0.0) {
    return;
  }
  // Bake the component's inverse scale into the mesh vertex data.
  if (Scale3D != FVector::OneVector) {
    for (int i = 0; i < Mesh.VertexCount(); ++i) {
      FVector pos = Mesh.GetVertex(i) / Scale3D;
      Mesh.SetVertex(i, pos);
    }
  }
}

void ComputeOrDiscardVertexNormals(UE::Geometry::FDynamicMesh3& Mesh, bool bComputeNormals) {
  if (bComputeNormals) {
    // NOTE: If the mesh has its attribute set enabled, and it has a NormalOverlay, those will
    // take precedence over these non-attribute vertex normals.
    Mesh.EnableVertexNormals(FVector3f{0, 0, 1});
    UE::Geometry::FMeshNormals meshNormals(&Mesh);
    meshNormals.ComputeVertexNormals();
    meshNormals.CopyToVertexNormals(&Mesh);
  } else {
    Mesh.DiscardVertexNormals();
  }
}
