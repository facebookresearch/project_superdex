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

#pragma once

#include "CoreMinimal.h"
#include "MochiTypes.h"

#include <mochi_core/geometry/model_data.h>

#include "MochiModelData.generated.h"

//------------------------------------------------------------------------------
// STRUCTS
//------------------------------------------------------------------------------

// These structs mirror mochi::ModelData and its nested types. Values stay as they appear
// in the model file, in meters and its own basis; ImportTransform and the simulation
// space conversion are still applied when the shape is created.
// Each std::optional in Mochi becomes a reflected TOptional. Optional arrays use a
// small USTRUCT wrapper because UHT does not allow containers directly inside
// TOptional. Bulk arrays are hidden from the details panel, which cannot usefully
// display them.

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiSkinningData {
  GENERATED_BODY()
  // Number of weights and indices per node being skinned.
  UPROPERTY(VisibleAnywhere, Category = "Skinning")
  int32 WeightsPerNode = 0;
  // Indices for each node. Size = numNodes * WeightsPerNode.
  UPROPERTY()
  TArray<int32> Indices;
  // Weights for each node. Size = numNodes * WeightsPerNode.
  UPROPERTY()
  TArray<float> Weights;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiMeshData {
  GENERATED_BODY()
  // 2 for polyline mesh, 3 for triangle mesh, 4 for tetrahedral mesh.
  UPROPERTY(VisibleAnywhere, Category = "Mesh")
  int32 NodesPerElement = 0;
  // Flat array of node coordinate values [m]. Size is a multiple of 3.
  UPROPERTY()
  TArray<float> Coordinates;
  // Flat array of node indices forming the elements.
  UPROPERTY()
  TArray<int32> Connectivity;
  UPROPERTY(VisibleAnywhere, Category = "Mesh")
  TOptional<FMochiSkinningData> Skinning;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiBlendingData {
  GENERATED_BODY()
  // Name of the soft source shape.
  UPROPERTY(VisibleAnywhere, Category = "Blending")
  FString SourceShape;
  // Source vertex index for each target vertex.
  UPROPERTY()
  TArray<int32> Indices;
  // Blend weight in [0, 1] for each target vertex, where zero is purely articulated.
  UPROPERTY()
  TArray<float> Weights;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiBox {
  GENERATED_BODY()
  // Center of the box [m]
  UPROPERTY(VisibleAnywhere, Category = "Box")
  FVector3f Center = FVector3f::ZeroVector;
  // Half extents of the box [m]
  UPROPERTY(VisibleAnywhere, Category = "Box")
  FVector3f HalfExtents = FVector3f::OneVector;
  // Rotation about the center of the box
  UPROPERTY(VisibleAnywhere, Category = "Box")
  FQuat4f Rotation = FQuat4f::Identity;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiPlane {
  GENERATED_BODY()
  // Outward-facing plane normal (unit length)
  UPROPERTY(VisibleAnywhere, Category = "Plane")
  FVector3f Normal = FVector3f(0.0f, 1.0f, 0.0f);
  // Signed distance from the origin along the normal [m]
  UPROPERTY(VisibleAnywhere, Category = "Plane")
  float Distance = 0.0f;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiSphere {
  GENERATED_BODY()
  // Center of the sphere [m]
  UPROPERTY(VisibleAnywhere, Category = "Sphere")
  FVector3f Center = FVector3f::ZeroVector;
  // Radius of the sphere [m]
  UPROPERTY(VisibleAnywhere, Category = "Sphere")
  float Radius = 0.0f;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiAabb {
  GENERATED_BODY()
  UPROPERTY(VisibleAnywhere, Category = "Bounds")
  FVector3f Min = FVector3f::ZeroVector;
  UPROPERTY(VisibleAnywhere, Category = "Bounds")
  FVector3f Max = FVector3f::ZeroVector;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiGridSdfData {
  GENERATED_BODY()
  // Dimensions of the SDF grid in X, Y, and Z
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  FIntVector Dims = FIntVector::ZeroValue;
  // Signed distance values [m]. Size = Dims.X * Dims.Y * Dims.Z.
  // Not a UPROPERTY: FPropertyTag holds a property's serialized size in an int32, which a
  // fine grid exceeds. UMochiModel::Serialize writes this as compressed bulk data
  // instead, so the property system never sees it and undo and duplication drop it.
  TArray<float> Values;
  // Spatial bounds of the grid; values are distributed uniformly within this volume
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  FMochiAabb Bounds;
  // Spatial bounds of the portion of the grid with negative values
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  FMochiAabb NegativeValueBounds;
  // Parent-from-grid transform, applied as scale, then rotation, then translation.
  // Each component is independently optional.
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  TOptional<FVector3f> Scale;
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  TOptional<FQuat4f> Rotation;
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  TOptional<FVector3f> Translation;
};

// Arrays hold one value per element, except ShapeTargetTensor which holds six per element:
// the symmetric upper triangle of the shape target offset.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiPerElementSoftMaterialData {
  GENERATED_BODY()
  UPROPERTY(VisibleAnywhere, Category = "Material")
  EMochiSoftMaterialType Type = EMochiSoftMaterialType::NeoHookean;
  UPROPERTY(VisibleAnywhere, Category = "Material")
  EMochiMaterialPsdStrategy PsdStrategy = EMochiMaterialPsdStrategy::MaterialDefault;
  // Young's modulus [Pa]
  UPROPERTY()
  TArray<float> YoungsModulus;
  // Poisson's ratio (dimensionless)
  UPROPERTY()
  TArray<float> PoissonRatio;
  // Anisotropic stiffness [Pa]
  UPROPERTY()
  TArray<float> AnisoAlpha;
  // Anisotropic reference length (dimensionless)
  UPROPERTY()
  TArray<float> AnisoLength;
  // Fiber azimuthal angle [rad]
  UPROPERTY()
  TArray<float> AnisoTheta;
  // Fiber elevation angle [rad]
  UPROPERTY()
  TArray<float> AnisoPhi;
  // ARAP stiffness [Pa]
  UPROPERTY()
  TArray<float> ArapStiffness;
  UPROPERTY()
  TArray<float> ShapeTargetTensor;
};

// UHT does not support TOptional<TArray<T>>, so optional Mochi arrays are wrapped in
// reflected structs.
USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiBlendingDataArray {
  GENERATED_BODY()
  UPROPERTY()
  TArray<FMochiBlendingData> Values;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiConstrainedNodeArray {
  GENERATED_BODY()
  UPROPERTY()
  TArray<int32> Values;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiElementFrameAxesArray {
  GENERATED_BODY()
  UPROPERTY()
  TArray<float> Values;
};

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiModelData {
  GENERATED_BODY()
  UPROPERTY(VisibleAnywhere, Category = "Mesh")
  TOptional<FMochiMeshData> Mesh;
  UPROPERTY(VisibleAnywhere, Category = "Mesh")
  TOptional<FMochiMeshData> VisualMesh;
  // Triangular mesh used for surface queries and rod contact quadrature.
  UPROPERTY(VisibleAnywhere, Category = "Mesh")
  TOptional<FMochiMeshData> ContactSkinMesh;
  UPROPERTY(VisibleAnywhere, Category = "Blending")
  TOptional<FMochiBlendingDataArray> Blending;
  // Indices of mesh nodes that are constrained.
  UPROPERTY(VisibleAnywhere, Category = "Constraints")
  TOptional<FMochiConstrainedNodeArray> ConstrainedNodes;
  // Per-element reference frame axes, 3 reals per element. Polyline meshes only.
  UPROPERTY(VisibleAnywhere, Category = "Mesh")
  TOptional<FMochiElementFrameAxesArray> ElementFrameAxes;
  UPROPERTY(VisibleAnywhere, Category = "Implicit Geometry")
  TOptional<FMochiBox> Box;
  UPROPERTY(VisibleAnywhere, Category = "Implicit Geometry")
  TOptional<FMochiPlane> Plane;
  UPROPERTY(VisibleAnywhere, Category = "Implicit Geometry")
  TOptional<FMochiSphere> Sphere;
  UPROPERTY(VisibleAnywhere, Category = "SDF")
  TOptional<FMochiGridSdfData> Sdf;
  UPROPERTY(VisibleAnywhere, Category = "Material")
  TOptional<FMochiPerElementSoftMaterialData> Material;

  // Convert to native Mochi type.
  mochi::ModelData ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::ModelData& Mochi);
  // True if no mesh, implicit shape or SDF is present. Material and blending data alone
  // do not make a loadable model.
  bool IsEmpty() const;
};
