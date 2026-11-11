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

#include "MochiDebugDraw.h"

#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"

#include "DrawDebugHelpers.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "HAL/IConsoleManager.h"

#include <algorithm>
#include <iterator>

void FMochiDebugDrawFlags::GenerateAndLogMochiDebugDrawFlagsCode(
    const mochi::DebugDraw& DebugDraw) {
  UE_LOG(LogMochiPhysics, Error, TEXT("BEGIN MOCHI DEBUG DRAW FLAGS CODE"));
  int numFeatures = DebugDraw.GetNumFeatures();
  TArray<FString> names;
  TArray<FString> codes;
  TArray<FString> descs;
  for (int i = 0; i < numFeatures; i++) {
    FString name(std::string(DebugDraw.GetFeatureName(i)).c_str());
    FString code = name;
    code.ReplaceInline(TEXT(" "), TEXT("_"));
    code.ReplaceInline(TEXT("("), TEXT(""));
    code.ReplaceInline(TEXT(")"), TEXT(""));
    code.InsertAt(0, TEXT("bDebugDraw_"));
    FString desc(std::string(DebugDraw.GetFeatureDescription(i)).c_str());
    desc.ReplaceInline(TEXT("\n"), TEXT(" "));
    names.Add(name);
    codes.Add(code);
    descs.Add(desc);
  }
  UE_LOG(LogMochiPhysics, Warning, TEXT("MochiDebugDraw.h (1/1):"));
  FString hpp;
  for (int i = 0; i < numFeatures; i++) {
    FString line = FString::Printf(
        TEXT(
            "// %s\n"
            "UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = \"Mochi\")\n"
            "bool %s = false;\n"),
        *descs[i],
        *codes[i]);
    hpp += line;
  }
  UE_LOG(LogMochiPhysics, Display, TEXT("%s"), *hpp);
  UE_LOG(LogMochiPhysics, Warning, TEXT("MochiDebugDraw.cpp (1/2):"));
  FString cpp1;
  for (int i = 0; i < numFeatures; i++) {
    FString line =
        FString::Printf(TEXT("EnableFeature(\"%s\", Flags.%s);\n"), *names[i], *codes[i]);
    cpp1 += line;
  }
  UE_LOG(LogMochiPhysics, Display, TEXT("%s"), *cpp1);
  UE_LOG(LogMochiPhysics, Warning, TEXT("MochiDebugDraw.cpp (2/2):"));
  FString cpp2;
  for (int i = 0; i < numFeatures; i++) {
    FString line = FString::Printf(TEXT("MOCHI_DEBUG_FLAG_COMMAND(%s)\n"), *codes[i]);
    cpp2 += line;
  }
  UE_LOG(LogMochiPhysics, Display, TEXT("%s"), *cpp2);
  UE_LOG(LogMochiPhysics, Error, TEXT("END MOCHI DEBUG DRAW FLAGS CODE"));
}

void FMochiDebugDrawFlags::SetMochiDebugDrawFlags(
    mochi::DebugDraw& DebugDraw,
    const FMochiDebugDrawFlags& Flags) {
  auto EnableFeature = [&](char const* name, bool enable) {
    auto index = DebugDraw.FindFeature(name);
    if (index >= 0) {
      DebugDraw.EnableFeature(index, enable);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[MochiDebugDraw] Attempting to enable unknown debug draw feature: \"%hs\"; MochiDebugDraw.h/cpp needs to be regenerated!"),
          name);
    }
  };

  // clang-format off
  // BEGIN GENERATED CODE
  EnableFeature("Actor AABB Local", Flags.bDebugDraw_Actor_AABB_Local);
  EnableFeature("Actor AABB World (conservative)", Flags.bDebugDraw_Actor_AABB_World_conservative);
  EnableFeature("Actor AABB World (tight fit)", Flags.bDebugDraw_Actor_AABB_World_tight_fit);
  EnableFeature("Actor Active Contact Forces", Flags.bDebugDraw_Actor_Active_Contact_Forces);
  EnableFeature("Actor Active Contact Normals", Flags.bDebugDraw_Actor_Active_Contact_Normals);
  EnableFeature("Actor Active Contact Positions", Flags.bDebugDraw_Actor_Active_Contact_Positions);
  EnableFeature("Actor Active Contact Velocities", Flags.bDebugDraw_Actor_Active_Contact_Velocities);
  EnableFeature("Actor Bsh", Flags.bDebugDraw_Actor_Bsh);
  EnableFeature("Actor Bsh Distance Samples", Flags.bDebugDraw_Actor_Bsh_Distance_Samples);
  EnableFeature("Actor Bsh Proxy Values", Flags.bDebugDraw_Actor_Bsh_Proxy_Values);
  EnableFeature("Actor Collider", Flags.bDebugDraw_Actor_Collider);
  EnableFeature("Actor Contact Samples", Flags.bDebugDraw_Actor_Contact_Samples);
  EnableFeature("Actor Contact Samples BSH (level 00)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_00);
  EnableFeature("Actor Contact Samples BSH (level 01)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_01);
  EnableFeature("Actor Contact Samples BSH (level 02)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_02);
  EnableFeature("Actor Contact Samples BSH (level 03)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_03);
  EnableFeature("Actor Contact Samples BSH (level 04)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_04);
  EnableFeature("Actor Contact Samples BSH (level 05)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_05);
  EnableFeature("Actor Contact Samples BSH (level 06)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_06);
  EnableFeature("Actor Contact Samples BSH (level 07)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_07);
  EnableFeature("Actor Contact Samples BSH (level 08)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_08);
  EnableFeature("Actor Contact Samples BSH (level 09)", Flags.bDebugDraw_Actor_Contact_Samples_BSH_level_09);
  EnableFeature("Actor Mesh", Flags.bDebugDraw_Actor_Mesh);
  EnableFeature("Actor Mesh (Active Surface Elements)", Flags.bDebugDraw_Actor_Mesh_Active_Surface_Elements);
  EnableFeature("Actor Mesh (freeze pivot)", Flags.bDebugDraw_Actor_Mesh_freeze_pivot);
  EnableFeature("Actor Mesh (freeze root)", Flags.bDebugDraw_Actor_Mesh_freeze_root);
  EnableFeature("Actor Mesh Elastic Energy", Flags.bDebugDraw_Actor_Mesh_Elastic_Energy);
  EnableFeature("Actor Mesh Interior", Flags.bDebugDraw_Actor_Mesh_Interior);
  EnableFeature("Actor Node BCs", Flags.bDebugDraw_Actor_Node_BCs);
  EnableFeature("Actor Node Contact Forces", Flags.bDebugDraw_Actor_Node_Contact_Forces);
  EnableFeature("Actor Node Normals", Flags.bDebugDraw_Actor_Node_Normals);
  EnableFeature("Actor Quadrature Points", Flags.bDebugDraw_Actor_Quadrature_Points);
  EnableFeature("Actor Rigid Body Inertia", Flags.bDebugDraw_Actor_Rigid_Body_Inertia);
  EnableFeature("Actor Rigid Pivot Eval Point", Flags.bDebugDraw_Actor_Rigid_Pivot_Eval_Point);
  EnableFeature("Actor Root Transform", Flags.bDebugDraw_Actor_Root_Transform);
  EnableFeature("Actor SDF Distances", Flags.bDebugDraw_Actor_SDF_Distances);
  EnableFeature("Actor SDF Normals", Flags.bDebugDraw_Actor_SDF_Normals);
  EnableFeature("Actor Visual Mesh", Flags.bDebugDraw_Actor_Visual_Mesh);
  EnableFeature("Actor Visual Normals", Flags.bDebugDraw_Actor_Visual_Normals);
  EnableFeature("Articulated Target Pose", Flags.bDebugDraw_Articulated_Target_Pose);
  EnableFeature("Compound AABB World", Flags.bDebugDraw_Compound_AABB_World);
  EnableFeature("Island AABB World (conservative)", Flags.bDebugDraw_Island_AABB_World_conservative);
  EnableFeature("Island AABB World (tight fit)", Flags.bDebugDraw_Island_AABB_World_tight_fit);
  EnableFeature("Potential Colliders", Flags.bDebugDraw_Potential_Colliders);
  EnableFeature("Rigid Velocity", Flags.bDebugDraw_Rigid_Velocity);
  EnableFeature("Rod Actor Polyline", Flags.bDebugDraw_Rod_Actor_Polyline);
  EnableFeature("Rod Element Frame Axes", Flags.bDebugDraw_Rod_Element_Frame_Axes);
  // END GENERATED CODE
  // clang-format on
}

//------------------------------------------------------------------------------
// Debug Draw Console Commands
//------------------------------------------------------------------------------

static FMochiDebugDrawFlags GetCurrentDebugDrawFlags(const UWorld* World) {
  const auto game = World->GetGameInstance();
  const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
  if (subsys) {
    if (const auto scene = subsys->GetScene()) {
      return scene->GetDebugDrawFlags();
    }
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[MochiDebugDraw] Failed to get current scene debug draw flags"));
  return FMochiDebugDrawFlags();
}

static void UpdateDebugDrawFlags(const UWorld* World, const FMochiDebugDrawFlags& Flags) {
  const auto game = World->GetGameInstance();
  const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
  if (subsys) {
    if (const auto scene = subsys->GetScene()) {
      scene->ConfigureDebugDraw(true, Flags);
    }
  }
}

static FAutoConsoleCommandWithWorld GMochiEnableDebugDraw(
    TEXT("Mochi.EnableDebugDraw"),
    TEXT("Enable Mochi debug drawing."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](const UWorld* World) {
      const auto game = World->GetGameInstance();
      const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (subsys) {
        subsys->EnableDebugDraw();
      }
    }));

static FAutoConsoleCommandWithWorld GMochiDisableDebugDraw(
    TEXT("Mochi.DisableDebugDraw"),
    TEXT("Disable Mochi debug drawing."),
    FConsoleCommandWithWorldDelegate::CreateLambda([](const UWorld* World) {
      const auto game = World->GetGameInstance();
      const auto subsys = game->GetSubsystem<UMochiGameInstanceSubsystem>();
      if (subsys) {
        subsys->DisableDebugDraw();
      }
    }));

#define MOCHI_DEBUG_FLAG_COMMAND(FlagName)                                \
  static FAutoConsoleCommandWithWorldAndArgs GMochi##FlagName(            \
      TEXT("Mochi." #FlagName),                                           \
      TEXT("Toggle " #FlagName ". Usage: Mochi." #FlagName " [0|1]"),     \
      FConsoleCommandWithWorldAndArgsDelegate::CreateLambda(              \
          [](const TArray<FString>& Args, UWorld* World) {                \
            FMochiDebugDrawFlags Flags = GetCurrentDebugDrawFlags(World); \
            if (Args.Num() > 0) {                                         \
              bool bValue = FCString::Atoi(*Args[0]) != 0;                \
              Flags.FlagName = bValue;                                    \
              UpdateDebugDrawFlags(World, Flags);                         \
            } else {                                                      \
              Flags.FlagName = !Flags.FlagName;                           \
              UpdateDebugDrawFlags(World, Flags);                         \
            }                                                             \
            UE_LOG(                                                       \
                LogMochiPhysics,                                          \
                Log,                                                      \
                TEXT("[MochiDebugDraw] Mochi." #FlagName " set to %s"),   \
                Flags.FlagName ? TEXT("true") : TEXT("false"));           \
          }));

// clang-format off
// BEGIN GENERATED CODE
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_AABB_Local)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_AABB_World_conservative)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_AABB_World_tight_fit)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Active_Contact_Forces)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Active_Contact_Normals)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Active_Contact_Positions)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Active_Contact_Velocities)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Bsh)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Bsh_Distance_Samples)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Bsh_Proxy_Values)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Collider)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_00)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_01)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_02)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_03)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_04)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_05)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_06)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_07)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_08)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Contact_Samples_BSH_level_09)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Mesh)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Mesh_Active_Surface_Elements)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Mesh_freeze_pivot)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Mesh_freeze_root)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Mesh_Elastic_Energy)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Mesh_Interior)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Node_BCs)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Node_Contact_Forces)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Node_Normals)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Quadrature_Points)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Rigid_Body_Inertia)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Rigid_Pivot_Eval_Point)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Root_Transform)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_SDF_Distances)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_SDF_Normals)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Visual_Mesh)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Actor_Visual_Normals)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Articulated_Target_Pose)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Compound_AABB_World)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Island_AABB_World_conservative)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Island_AABB_World_tight_fit)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Potential_Colliders)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Rigid_Velocity)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Rod_Actor_Polyline)
MOCHI_DEBUG_FLAG_COMMAND(bDebugDraw_Rod_Element_Frame_Axes)
// END GENERATED CODE
// clang-format on

#undef MOCHI_DEBUG_FLAG_COMMAND
