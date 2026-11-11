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

#include "MochiModelData.h"

#include <algorithm>
#include <type_traits>

// Please keep these mirrors in sync with Mochi
static_assert(
    std::is_same_v<mochi::real, float>,
    "FMochiModelData stores mochi::real as float; rebuild single-precision or widen the mirror.");
static_assert(sizeof(int) == sizeof(int32), "FMochiModelData stores mochi's int as int32.");

namespace {

template <typename UnrealT, typename MochiT>
TArray<UnrealT> ToUnrealArray(const mochi::DynamicArray<MochiT>& Src) {
  // TArray indexes with int32, so the mirror cannot hold what Mochi can. Reachable via a
  // model file with an SDF grid over roughly 8 GB.
  checkf(
      Src.size() <= static_cast<size_t>(MAX_int32),
      TEXT("Mochi array of %llu elements is too large to mirror."),
      static_cast<uint64>(Src.size()));
  TArray<UnrealT> result;
  result.SetNumUninitialized(static_cast<int32>(Src.size()));
  std::copy(Src.begin(), Src.end(), result.GetData());
  return result;
}

template <typename MochiT, typename UnrealT>
mochi::DynamicArray<MochiT> ToMochiArray(const TArray<UnrealT>& Src) {
  mochi::DynamicArray<MochiT> result;
  result.assign(Src.GetData(), Src.GetData() + Src.Num());
  return result;
}

FVector3f ToUnrealVector(const mochi::Real3& V) {
  return FVector3f(V[0], V[1], V[2]);
}

mochi::Real3 ToMochiVector(const FVector3f& V) {
  return mochi::Real3(V.X, V.Y, V.Z);
}

FQuat4f ToUnrealQuat(const mochi::Quaternion& Q) {
  const mochi::Real4 v = Q.ToReal4();
  return FQuat4f(v[0], v[1], v[2], v[3]);
}

mochi::Quaternion ToMochiQuat(const FQuat4f& Q) {
  return mochi::Quaternion(Q.X, Q.Y, Q.Z, Q.W);
}

FMochiAabb ToUnrealAabb(const mochi::Aabb& A) {
  FMochiAabb result;
  result.Min = ToUnrealVector(A.GetMin());
  result.Max = ToUnrealVector(A.GetMax());
  return result;
}

mochi::Aabb ToMochiAabb(const FMochiAabb& A) {
  return mochi::Aabb(ToMochiVector(A.Min), ToMochiVector(A.Max));
}

void MeshFromMochi(const mochi::MeshData& Src, FMochiMeshData& Out) {
  Out.NodesPerElement = Src.nodesPerElement;
  Out.Coordinates = ToUnrealArray<float>(Src.coordinates);
  Out.Connectivity = ToUnrealArray<int32>(Src.connectivity);
  if (Src.skinning) {
    FMochiSkinningData& skinning = Out.Skinning.Emplace();
    skinning.WeightsPerNode = Src.skinning->weightsPerNode;
    skinning.Indices = ToUnrealArray<int32>(Src.skinning->indices);
    skinning.Weights = ToUnrealArray<float>(Src.skinning->weights);
  } else {
    Out.Skinning.Reset();
  }
}

mochi::MeshData MeshToMochi(const FMochiMeshData& Src) {
  mochi::MeshData result;
  result.nodesPerElement = Src.NodesPerElement;
  result.coordinates = ToMochiArray<mochi::real>(Src.Coordinates);
  result.connectivity = ToMochiArray<int>(Src.Connectivity);
  if (Src.Skinning.IsSet()) {
    mochi::SkinningData skinning;
    skinning.weightsPerNode = Src.Skinning->WeightsPerNode;
    skinning.indices = ToMochiArray<int>(Src.Skinning->Indices);
    skinning.weights = ToMochiArray<mochi::real>(Src.Skinning->Weights);
    result.skinning = std::move(skinning);
  }
  return result;
}

} // namespace

void FMochiModelData::FromMochi(const mochi::ModelData& Mochi) {
  *this = FMochiModelData{};

  if (Mochi.mesh) {
    MeshFromMochi(*Mochi.mesh, Mesh.Emplace());
  }
  if (Mochi.visualMesh) {
    MeshFromMochi(*Mochi.visualMesh, VisualMesh.Emplace());
  }
  if (Mochi.contactSkinMesh) {
    MeshFromMochi(*Mochi.contactSkinMesh, ContactSkinMesh.Emplace());
  }

  if (Mochi.blending) {
    TArray<FMochiBlendingData>& blending = Blending.Emplace().Values;
    blending.Reserve(static_cast<int32>(Mochi.blending->size()));
    for (const mochi::BlendingData& src : *Mochi.blending) {
      FMochiBlendingData& dst = blending.AddDefaulted_GetRef();
      dst.SourceShape =
          FString(static_cast<int32>(src.sourceShape.size()), src.sourceShape.c_str());
      dst.Indices = ToUnrealArray<int32>(src.indices);
      dst.Weights = ToUnrealArray<float>(src.weights);
    }
  }

  if (Mochi.constrainedNodes) {
    ConstrainedNodes.Emplace().Values = ToUnrealArray<int32>(*Mochi.constrainedNodes);
  }

  if (Mochi.elementFrameAxes) {
    ElementFrameAxes.Emplace().Values = ToUnrealArray<float>(*Mochi.elementFrameAxes);
  }

  if (Mochi.box) {
    FMochiBox& box = Box.Emplace();
    box.Center = ToUnrealVector(Mochi.box->center);
    box.HalfExtents = ToUnrealVector(Mochi.box->halfExtents);
    box.Rotation = ToUnrealQuat(Mochi.box->rotation);
  }

  if (Mochi.plane) {
    FMochiPlane& plane = Plane.Emplace();
    plane.Normal = ToUnrealVector(Mochi.plane->normal);
    plane.Distance = Mochi.plane->distance;
  }

  if (Mochi.sphere) {
    FMochiSphere& sphere = Sphere.Emplace();
    sphere.Center = ToUnrealVector(Mochi.sphere->center);
    sphere.Radius = Mochi.sphere->radius;
  }

  if (Mochi.sdf) {
    FMochiGridSdfData& sdf = Sdf.Emplace();
    sdf.Dims = FIntVector(Mochi.sdf->dims[0], Mochi.sdf->dims[1], Mochi.sdf->dims[2]);
    sdf.Values = ToUnrealArray<float>(Mochi.sdf->values);
    sdf.Bounds = ToUnrealAabb(Mochi.sdf->bounds);
    sdf.NegativeValueBounds = ToUnrealAabb(Mochi.sdf->negativeValueBounds);
    if (Mochi.sdf->scale) {
      sdf.Scale = ToUnrealVector(*Mochi.sdf->scale);
    }
    if (Mochi.sdf->rotation) {
      sdf.Rotation = ToUnrealQuat(*Mochi.sdf->rotation);
    }
    if (Mochi.sdf->translation) {
      sdf.Translation = ToUnrealVector(*Mochi.sdf->translation);
    }
  }

  if (Mochi.material) {
    FMochiPerElementSoftMaterialData& material = Material.Emplace();
    material.Type = static_cast<EMochiSoftMaterialType>(Mochi.material->type);
    material.PsdStrategy = static_cast<EMochiMaterialPsdStrategy>(Mochi.material->psdStrategy);
    material.YoungsModulus = ToUnrealArray<float>(Mochi.material->youngsModulus);
    material.PoissonRatio = ToUnrealArray<float>(Mochi.material->poissonRatio);
    material.AnisoAlpha = ToUnrealArray<float>(Mochi.material->anisoAlpha);
    material.AnisoLength = ToUnrealArray<float>(Mochi.material->anisoLength);
    material.AnisoTheta = ToUnrealArray<float>(Mochi.material->anisoTheta);
    material.AnisoPhi = ToUnrealArray<float>(Mochi.material->anisoPhi);
    material.ArapStiffness = ToUnrealArray<float>(Mochi.material->arapStiffness);
    material.ShapeTargetTensor = ToUnrealArray<float>(Mochi.material->shapeTargetTensor);
  }
}

mochi::ModelData FMochiModelData::ToMochi() const {
  mochi::ModelData result;

  if (Mesh.IsSet()) {
    result.mesh = MeshToMochi(Mesh.GetValue());
  }
  if (VisualMesh.IsSet()) {
    result.visualMesh = MeshToMochi(VisualMesh.GetValue());
  }
  if (ContactSkinMesh.IsSet()) {
    result.contactSkinMesh = MeshToMochi(ContactSkinMesh.GetValue());
  }

  if (Blending.IsSet()) {
    mochi::DynamicArray<mochi::BlendingData> blending;
    blending.reserve(Blending->Values.Num());
    for (const FMochiBlendingData& src : Blending->Values) {
      blending.push_back(mochi::BlendingData{});
      mochi::BlendingData& dst = blending.back();
      const auto sourceShape = StringCast<ANSICHAR>(*src.SourceShape);
      dst.sourceShape =
          mochi::DynamicString(sourceShape.Get(), static_cast<size_t>(sourceShape.Length()));
      dst.indices = ToMochiArray<int>(src.Indices);
      dst.weights = ToMochiArray<mochi::real>(src.Weights);
    }
    result.blending = std::move(blending);
  }

  if (ConstrainedNodes.IsSet()) {
    result.constrainedNodes = ToMochiArray<int>(ConstrainedNodes->Values);
  }
  if (ElementFrameAxes.IsSet()) {
    result.elementFrameAxes = ToMochiArray<mochi::real>(ElementFrameAxes->Values);
  }

  if (Box.IsSet()) {
    result.box = mochi::Box(
        ToMochiVector(Box->Center), ToMochiVector(Box->HalfExtents), ToMochiQuat(Box->Rotation));
  }
  if (Plane.IsSet()) {
    result.plane = mochi::Plane(ToMochiVector(Plane->Normal), Plane->Distance);
  }
  if (Sphere.IsSet()) {
    result.sphere = mochi::Sphere(ToMochiVector(Sphere->Center), Sphere->Radius);
  }

  if (Sdf.IsSet()) {
    mochi::GridSdfData sdf;
    sdf.dims = mochi::Int3(Sdf->Dims.X, Sdf->Dims.Y, Sdf->Dims.Z);
    sdf.values = ToMochiArray<mochi::real>(Sdf->Values);
    sdf.bounds = ToMochiAabb(Sdf->Bounds);
    sdf.negativeValueBounds = ToMochiAabb(Sdf->NegativeValueBounds);
    if (Sdf->Scale.IsSet()) {
      sdf.scale = ToMochiVector(Sdf->Scale.GetValue());
    }
    if (Sdf->Rotation.IsSet()) {
      sdf.rotation = ToMochiQuat(Sdf->Rotation.GetValue());
    }
    if (Sdf->Translation.IsSet()) {
      sdf.translation = ToMochiVector(Sdf->Translation.GetValue());
    }
    result.sdf = std::move(sdf);
  }

  if (Material.IsSet()) {
    mochi::PerElementSoftMaterialData material;
    material.type = static_cast<mochi::SoftMaterialType>(Material->Type);
    material.psdStrategy = static_cast<mochi::MaterialPsdStrategy>(Material->PsdStrategy);
    material.youngsModulus = ToMochiArray<mochi::real>(Material->YoungsModulus);
    material.poissonRatio = ToMochiArray<mochi::real>(Material->PoissonRatio);
    material.anisoAlpha = ToMochiArray<mochi::real>(Material->AnisoAlpha);
    material.anisoLength = ToMochiArray<mochi::real>(Material->AnisoLength);
    material.anisoTheta = ToMochiArray<mochi::real>(Material->AnisoTheta);
    material.anisoPhi = ToMochiArray<mochi::real>(Material->AnisoPhi);
    material.arapStiffness = ToMochiArray<mochi::real>(Material->ArapStiffness);
    material.shapeTargetTensor = ToMochiArray<mochi::real>(Material->ShapeTargetTensor);
    result.material = std::move(material);
  }

  return result;
}

bool FMochiModelData::IsEmpty() const {
  return !Mesh.IsSet() && !VisualMesh.IsSet() && !ContactSkinMesh.IsSet() && !Box.IsSet() &&
      !Plane.IsSet() && !Sphere.IsSet() && !Sdf.IsSet();
}
