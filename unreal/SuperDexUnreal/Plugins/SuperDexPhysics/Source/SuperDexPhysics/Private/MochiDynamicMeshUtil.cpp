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
    mochi::MeshDataView const& MeshData) {
  MOCHI_PROFILE_SCOPE();
  FMochiToUnrealConverter M2u;
  int const NumNodes = MeshData.GetNumNodes();
  for (int NodeIdx = 0; NodeIdx < NumNodes; ++NodeIdx) {
    mochi::Real3 Vertex = mochi::Real3{
        MeshData.coordinates[NodeIdx * 3],
        MeshData.coordinates[NodeIdx * 3 + 1],
        MeshData.coordinates[NodeIdx * 3 + 2],
    };
    // NOTE: Assumes Mochi position data is stored in Mochi units (meters)
    OutMesh->AppendVertex(M2u.TranslationToUnreal(Vertex));
  }

  // Initialize faces
  const mochi::Span<int const>& TriIndices = MeshData.connectivity;
  if (M2u.FlipsHandedness()) {
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

bool IsDeformableActorType(mochi::ActorType Type) {
  return Type == mochi::ActorType::Soft || Type == mochi::ActorType::Shell ||
      Type == mochi::ActorType::Rod;
}

EMochiRenderMeshSource ChooseRenderMeshSource(
    mochi::MeshDataView const& VisualMesh,
    mochi::MeshDataView const& ContactSkinMesh,
    bool bPosedByNodeQueries) {
  const auto Usable = [bPosedByNodeQueries](mochi::MeshDataView const& Mesh) {
    return !Mesh.IsEmpty() && (!bPosedByNodeQueries || Mesh.skinning.has_value());
  };
  if (Usable(VisualMesh)) {
    return EMochiRenderMeshSource::Visual;
  }
  if (Usable(ContactSkinMesh)) {
    return EMochiRenderMeshSource::ContactSkin;
  }
  return EMochiRenderMeshSource::Surface;
}

static mochi::MeshDataView SelectDrawnMochiMesh(
    mochi::Context const& MochiContext,
    mochi::ShapeHandle Shape,
    mochi::MeshDataView const& VisualMesh,
    mochi::MeshDataView const& ContactSkin,
    bool bPosedByNodeQueries,
    EMochiRenderMeshSource& OutSource,
    mochi::Error& Error) {
  MOCHI_ERROR_RETURN(Error, {});

  OutSource = ChooseRenderMeshSource(VisualMesh, ContactSkin, bPosedByNodeQueries);
  switch (OutSource) {
    case EMochiRenderMeshSource::Visual:
      return VisualMesh;
    case EMochiRenderMeshSource::ContactSkin:
      return ContactSkin;
    case EMochiRenderMeshSource::Surface:
      break;
  }
  const mochi::MeshDataView MeshData = MochiContext.GetShapeSurfaceMesh(Shape, Error);
  MOCHI_ERROR_RETURN(Error, {});
  if (MeshData.IsEmpty()) {
    const mochi::MeshDataView Mesh = MochiContext.GetShapeMesh(Shape, Error);
    MOCHI_ERROR_RETURN(Error, {});
    // A rod centerline with no usable embedding has nothing to draw.
    if (Mesh.nodesPerElement == 2) {
      return {};
    }
  }
  MOCHI_ERROR_IF(
      MeshData.IsEmpty(),
      Error,
      "Cannot populate mesh: shape has no surface mesh data, and no usable visual mesh or "
      "contact skin.");
  MOCHI_ERROR_RETURN(Error, {});

  return MeshData;
}

// The mesh of Shape that PopulateMeshFromMochiShape draws for the same flags.
static mochi::MeshDataView GetDrawnMochiMesh(
    mochi::Context const& MochiContext,
    mochi::ShapeHandle Shape,
    bool bUseContactSkinIfPresent,
    bool bPosedByNodeQueries,
    EMochiRenderMeshSource& OutSource,
    mochi::Error& Error) {
  MOCHI_ERROR_RETURN(Error, {});
  mochi::MeshDataView const VisualMesh = MochiContext.GetShapeVisualMesh(Shape, Error);
  mochi::MeshDataView const ContactSkin = bUseContactSkinIfPresent
      ? MochiContext.GetShapeContactSkinMesh(Shape, Error)
      : mochi::MeshDataView{};
  return SelectDrawnMochiMesh(
      MochiContext, Shape, VisualMesh, ContactSkin, bPosedByNodeQueries, OutSource, Error);
}

FMochiRenderMesh
GetDrawnMochiMesh(mochi::Actor const& Actor, bool bUseContactSkinIfPresent, mochi::Error& Error) {
  MOCHI_ERROR_RETURN(Error, {});
  FMochiRenderMesh Mesh;
  Mesh.Shape = Actor.GetReferenceShape(Error);
  MOCHI_ERROR_RETURN(Error, {});
  // A mesh's node normals query is supported exactly when its positions query is (see
  // mochi::QueryType).
  const mochi::Context& MochiContext = *Actor.GetContext();
  const mochi::MeshDataView VisualMesh =
      Actor.IsQuerySupported(mochi::QueryType::VisualNodePositions)
      ? MochiContext.GetShapeVisualMesh(Mesh.Shape, Error)
      : mochi::MeshDataView{};
  const mochi::MeshDataView ContactSkin =
      bUseContactSkinIfPresent && Actor.IsQuerySupported(mochi::QueryType::ContactSkinNodePositions)
      ? MochiContext.GetShapeContactSkinMesh(Mesh.Shape, Error)
      : mochi::MeshDataView{};
  Mesh.Data = SelectDrawnMochiMesh(
      MochiContext,
      Mesh.Shape,
      VisualMesh,
      ContactSkin,
      IsDeformableActorType(Actor.GetType()),
      Mesh.Source,
      Error);
  MOCHI_ERROR_RETURN(Error, {});
  if (Actor.GetType() == mochi::ActorType::Rod && Mesh.Data.IsEmpty()) {
    return Mesh;
  }
  MOCHI_ERROR_IF(
      Mesh.Source == EMochiRenderMeshSource::Surface &&
          !Actor.IsQuerySupported(mochi::QueryType::SurfaceNodePositions),
      Error,
      "The actor supports node queries on none of its meshes that may be drawn.");
  MOCHI_ERROR_RETURN(Error, {});
  return Mesh;
}

void PopulateMeshFromMochiShape(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ShapeHandle Shape,
    bool bUseContactSkinIfPresent,
    bool bPosedByNodeQueries,
    EMochiRenderMeshSource* OutSource) {
  MOCHI_PROFILE_SCOPE();
  if (OutSource) {
    *OutSource = EMochiRenderMeshSource::Surface;
  }
  if (!Shape.IsValid()) {
    return;
  }
  mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  if (MochiContext) {
    mochi::ErrorLog E{};
    EMochiRenderMeshSource Source = EMochiRenderMeshSource::Surface;
    mochi::MeshDataView MeshData = GetDrawnMochiMesh(
        *MochiContext, Shape, bUseContactSkinIfPresent, bPosedByNodeQueries, Source, E);
    MOCHI_ERROR_RETURN(E);
    PopulateMeshFromMochiMeshData(OutMesh, MeshData);
    if (OutSource) {
      *OutSource = Source;
    }
  }
}

void SUPERDEXPHYSICS_API PopulateMeshFromMochiActor(
    UE::Geometry::FDynamicMesh3* OutMesh,
    mochi::ActorHandle const& MochiActor,
    UMochiSceneObject* MochiScene,
    bool bUseContactSkinIfPresent,
    bool bPosedByNodeQueries,
    EMochiRenderMeshSource* OutSource) {
  MOCHI_PROFILE_SCOPE();
  if (OutSource) {
    *OutSource = EMochiRenderMeshSource::Surface;
  }
  if (!MochiActor.IsValid() || !MochiScene || !MochiScene->IsValid()) {
    return;
  }
  FMochiRenderMesh Mesh;
  mochi::ErrorLog E{};
  MochiScene->QueueOrExecuteActorCommand(MochiActor, [&](mochi::Actor* Actor) {
    if (bPosedByNodeQueries) {
      Mesh = GetDrawnMochiMesh(*Actor, bUseContactSkinIfPresent, E);
      return;
    }
    Mesh.Shape = Actor->GetReferenceShape(E);
    Mesh.Data = GetDrawnMochiMesh(
        *Actor->GetContext(), Mesh.Shape, bUseContactSkinIfPresent, false, Mesh.Source, E);
  });
  MochiScene->WaitForQueuedCommands();
  MOCHI_ERROR_RETURN(E);
  PopulateMeshFromMochiMeshData(OutMesh, Mesh.Data);
  if (OutSource) {
    *OutSource = Mesh.Source;
  }
}

void ConvertNodePositionsToUnreal(
    const mochi::Span<const mochi::real>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D) {
  check(InMochi.size() % 3 == 0) OutUnreal.SetNum(InMochi.size() / 3);
  FMochiToUnrealConverter M2u;
  for (int i = 0; i < InMochi.size(); i += 3) {
    OutUnreal[i / 3] =
        M2u.TranslationToUnreal({InMochi[i], InMochi[i + 1], InMochi[i + 2]}) / Scale3D;
  }
}

void ConvertNodePositionsToUnreal(
    const TArray<mochi::real>& InMochi,
    TArray<FVector>& OutUnreal,
    FVector Scale3D) {
  check(InMochi.Num() % 3 == 0) OutUnreal.SetNum(InMochi.Num() / 3);
  FMochiToUnrealConverter M2u;
  for (int i = 0; i < InMochi.Num(); i += 3) {
    OutUnreal[i / 3] =
        M2u.TranslationToUnreal({InMochi[i], InMochi[i + 1], InMochi[i + 2]}) / Scale3D;
  }
}

void ConvertVolumeNodeReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal) {
  mochi::ErrorLog E{};
  mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  const mochi::ShapeHandle Shape = MochiModel->GetMochiShape(Scale3D, FTransform::Identity);
  const mochi::MeshDataView& Mesh = MochiContext->GetShapeMesh(Shape, E);
  ConvertNodePositionsToUnreal(Mesh.coordinates, OutReferencePositionsUnreal, Scale3D);
}

void ConvertElementBarycenterReferencePositions(
    const UMochiModel* MochiModel,
    FVector Scale3D,
    TArray<FVector>& OutReferencePositionsUnreal) {
  if (MochiModel) {
    mochi::ErrorLog E{};
    mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
    const mochi::ShapeHandle Shape = MochiModel->GetMochiShape(Scale3D, FTransform::Identity);
    const mochi::MeshDataView Mesh = MochiContext->GetShapeMesh(Shape, E);
    mochi::DynamicArray<mochi::real> Barycenters = mochi::ComputeElementBarycenters(Mesh);
    ConvertNodePositionsToUnreal(
        mochi::MakeConstSpan(Barycenters), OutReferencePositionsUnreal, Scale3D);
  }
}

void EnableRenderModelAttributes(
    UE::Geometry::FDynamicMesh3& Mesh,
    TConstArrayView<FVector2f> UVs,
    TConstArrayView<FLinearColor> Colors) {
  MOCHI_PROFILE_SCOPE();
  const int32 VertexCount = Mesh.MaxVertexID();
  // Element IDs must start at 0 to match vertex IDs.
  check(!Mesh.HasAttributes());
  check(UVs.IsEmpty() || UVs.Num() == VertexCount);
  check(Colors.IsEmpty() || Colors.Num() == VertexCount);
  // Enabling attributes creates one UV layer and one normal layer. Without UVs, the UV layer's
  // triangles stay unset, as in a mesh with no UVs.
  Mesh.EnableAttributes();
  UE::Geometry::FDynamicMeshAttributeSet* Attributes = Mesh.Attributes();
  UE::Geometry::FDynamicMeshNormalOverlay* NormalOverlay = Attributes->PrimaryNormals();
  UE::Geometry::FDynamicMeshUVOverlay* UvOverlay =
      UVs.IsEmpty() ? nullptr : Attributes->PrimaryUV();
  UE::Geometry::FDynamicMeshColorOverlay* ColorOverlay = nullptr;
  if (!Colors.IsEmpty()) {
    Attributes->EnablePrimaryColors();
    ColorOverlay = Attributes->PrimaryColors();
  }
  for (int VertIdx = 0; VertIdx < VertexCount; ++VertIdx) {
    NormalOverlay->AppendElement(Mesh.GetVertexNormal(VertIdx));
    if (UvOverlay) {
      UvOverlay->AppendElement(UVs[VertIdx]);
    }
    if (ColorOverlay) {
      ColorOverlay->AppendElement(FVector4f(Colors[VertIdx]));
    }
  }
  for (const int TriIdx : Mesh.TriangleIndicesItr()) {
    const UE::Geometry::FIndex3i Tri = Mesh.GetTriangle(TriIdx);
    NormalOverlay->SetTriangle(TriIdx, Tri);
    if (UvOverlay) {
      UvOverlay->SetTriangle(TriIdx, Tri);
    }
    if (ColorOverlay) {
      ColorOverlay->SetTriangle(TriIdx, Tri);
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
  const FVector3f InvScale3d = static_cast<FVector3f>(InvScale3D);
  UE::Geometry::FDynamicMeshNormalOverlay* NormalOverlay =
      bUseMochiNodeNormals && bUpdateRenderModelNormalOverlay ? Mesh.Attributes()->PrimaryNormals()
                                                              : nullptr;
  check(!NormalOverlay || NormalOverlay->ElementCount() == Mesh.MaxVertexID());
  // single precision converter for speed
  FMochiToUnrealConverter_float M2uf;
  for (int i = 0; i < NodePositions.Num(); i += 3) {
    const int VertIdx = i / 3;
    FVector3f VertPos =
        M2uf.TranslationToUnreal({NodePositions[i], NodePositions[i + 1], NodePositions[i + 2]});
    Mesh.SetVertex(VertIdx, static_cast<FVector>(VertPos * InvScale3d));
    if (bUseMochiNodeNormals) {
      FVector3f VertNormal =
          M2uf.DirectionToUnreal({NodeNormals[i], NodeNormals[i + 1], NodeNormals[i + 2]});
      Mesh.SetVertexNormal(VertIdx, VertNormal);
      if (NormalOverlay) {
        NormalOverlay->SetElement(VertIdx, VertNormal);
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
      FVector Pos = Mesh.GetVertex(i) / Scale3D;
      Mesh.SetVertex(i, Pos);
    }
  }
}

void ComputeOrDiscardVertexNormals(UE::Geometry::FDynamicMesh3& Mesh, bool bComputeNormals) {
  if (bComputeNormals) {
    // NOTE: If the mesh has its attribute set enabled, and it has a NormalOverlay, those will
    // take precedence over these non-attribute vertex normals.
    Mesh.EnableVertexNormals(FVector3f{0, 0, 1});
    UE::Geometry::FMeshNormals MeshNormals(&Mesh);
    MeshNormals.ComputeVertexNormals();
    MeshNormals.CopyToVertexNormals(&Mesh);
  } else {
    Mesh.DiscardVertexNormals();
  }
}
