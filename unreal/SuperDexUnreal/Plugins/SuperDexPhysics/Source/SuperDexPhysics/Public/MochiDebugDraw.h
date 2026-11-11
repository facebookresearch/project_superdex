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
#include "UObject/Object.h"

#include <mochi_physics/mochi_physics.h>

#include <mutex>

#include "MochiDebugDraw.generated.h"

USTRUCT(BlueprintType)
struct FMochiDebugDrawFlags {
  GENERATED_BODY()

  //----------------------------------------------------------------------------
  // UE SPECIFIC FLAGS
  //----------------------------------------------------------------------------

  // Debug draw all enabled entities in the foreground.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDrawForeground = false;

  //----------------------------------------------------------------------------
  // MOCHI FLAGS
  //----------------------------------------------------------------------------

  // clang-format off

  // NOTE: These flags need to be kept up to date with Mochi debug features, else we'll get warnings like:
  //
  //     [MochiDebugDraw] Attempting to enable unknown debug draw feature: "Actor Bsh Priority Values"
  //
  // Until a better solution is put forth, here's how to keep this file up to date:
  //
  // 1. Enable bGenerateAndLogMochiDebugDrawFlagsCode at the bottom of this file.
  // 2. Recompile.
  // 3. Run any Mochi scene in PIE to dump code to logs.
  // 4. Copy/paste the MochiDebugDraw.h section below and remove `Display LogMochiPhysics`.
  // 5. Copy/paste the MochiDebugDraw.cpp sections into MochiDebugDraw.cpp and remove `Display LogMochiPhysics`.
  // 6. Disable bGenerateAndLogMochiDebugDrawFlagsCode at the bottom of this file.
  // 7. arc lint, commit, and recompile.

  // BEGIN GENERATED CODE

  // Draw a wireframe box for the actor's local-space AABB.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_AABB_Local = false;
  // Draw a wireframe box for the actor's conservative step bounds (used for island culling).
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_AABB_World_conservative = false;
  // Draw a wireframe box for the actor's world-space Aabb.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_AABB_World_tight_fit = false;
  // Draw a green line line representing the velocity of each active contact point.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Active_Contact_Forces = false;
  // Draw a red line for the normal direction of each active contact point.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Active_Contact_Normals = false;
  // Draw a little sphere at each active contact point
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Active_Contact_Positions = false;
  // Draw a green line line representing the velocity of each active contact point.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Active_Contact_Velocities = false;
  // Draw Bsh Structure
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Bsh = false;
  // Draw wireframe points for distance.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Bsh_Distance_Samples = false;
  // Draw wireframe points for distance.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Bsh_Proxy_Values = false;
  // Draw a wireframe representing the current collider of the actor
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Collider = false;
  // Draw a little grey box at each potential contact sample
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 0.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_00 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 1.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_01 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 2.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_02 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 3.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_03 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 4.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_04 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 5.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_05 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 6.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_06 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 7.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_07 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 8.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_08 = false;
  // Draw the contact sample bounding sphere hierarchy at tree depth 9.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Contact_Samples_BSH_level_09 = false;
  // Draw wireframe triangles for the boundary of the actor's volume mesh.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Mesh = false;
  // Draw wireframe triangles for the actor's active boundary faces.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Mesh_Active_Surface_Elements = false;
  // Draw a wireframe mesh for the surface of the actor, but freeze the actor's "rigid pivot" wherever it is. This lets you see how the actor is deforming independent of any global rotation or translation.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Mesh_freeze_pivot = false;
  // Draw a wireframe mesh for the surface of the actor, but freeze the actor's root transform wherever it is. This lets you see how the actor is moving in local-space (if at all).
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Mesh_freeze_root = false;
  // Draw wireframe triangles for the boundary of the actor's volume mesh. Color mesh based on the elastic energy.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Mesh_Elastic_Energy = false;
  // Draw wireframe triangles for all the elements in the actor's volume mesh, including the interior.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Mesh_Interior = false;
  // Draw little boxes for each soft node with a boundary condition.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Node_BCs = false;
  // Draw a magenta line representing the contact force on each surface node.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Node_Contact_Forces = false;
  // Draw a line for the normal of each boundary node on the actor's volume mesh.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Node_Normals = false;
  // Draw a little red sphere at each quadrature point and deform according to def gradient
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Quadrature_Points = false;
  // Draw a scaled tri-axis representing the actor's principal axes and moments of inertia at the COM.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Rigid_Body_Inertia = false;
  // Draw an RGB tri-axis for the point used to evaluate a soft actor's rigid pivot. Usually near the center-of-mass.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Rigid_Pivot_Eval_Point = false;
  // Draw an RGB tri-axis for the XYZ basis vectors of the actor's local coordinate space.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Root_Transform = false;
  // Draw a point representing distance to nearest collider at a contact point.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_SDF_Distances = false;
  // Draw lines for the surface normals of an actor's collision SDF (arbitrary sampling).
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_SDF_Normals = false;
  // Draw wireframe triangles for the actor's visual mesh (high res mesh mapped to the simulation mesh).
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Visual_Mesh = false;
  // Draw a line for the normal of each vertex in the actor's visual mesh.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Actor_Visual_Normals = false;
  // Draw wireframe meshes for the links of an articulated actor at the target pose of its pose controller. This visualizes where the controller is trying to move the actor.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Articulated_Target_Pose = false;
  // Draw a wireframe box for the compound's world-space AABB.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Compound_AABB_World = false;
  // Draw a wireframe box around the conservative step bounds of all actors within an island.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Island_AABB_World_conservative = false;
  // Draw a wireframe box around the actors within an island.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Island_AABB_World_tight_fit = false;
  // Draw a purple line between pairs of actors in potential contact
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Potential_Colliders = false;
  // Draw a cyan/magenta lines representing actor's linear/angular velocity
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Rigid_Velocity = false;
  // Draw deformed polyline centerline for rod actors
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Rod_Actor_Polyline = false;
  // Draw local frame axes for rod elements to visualize twisting state
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Mochi")
  bool bDebugDraw_Rod_Element_Frame_Axes = false;

  // END GENERATED CODE
  // clang-format on

 public:
  static void SetMochiDebugDrawFlags(
      mochi::DebugDraw& DebugDraw,
      const FMochiDebugDrawFlags& Flags);
  static constexpr bool bGenerateAndLogMochiDebugDrawFlagsCode = false;
  static void GenerateAndLogMochiDebugDrawFlagsCode(const mochi::DebugDraw& DebugDraw);
};
