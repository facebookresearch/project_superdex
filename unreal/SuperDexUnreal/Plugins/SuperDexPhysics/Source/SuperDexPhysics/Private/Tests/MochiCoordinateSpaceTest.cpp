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

#include "MochiSpaceConverter.h"

#include "MochiPhysicsModule.h"
#include "MochiPhysicsSettings.h"

#include <mochi_physics/mochi_physics.h>

#include "Misc/AutomationTest.h"
#include "UObject/UnrealType.h"

#if WITH_DEV_AUTOMATION_TESTS

#if ENGINE_MAJOR_VERSION >= 5
#define MOCHI_COORDINATE_SPACE_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#else
#define MOCHI_COORDINATE_SPACE_TEST_FLAGS \
  (EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#endif

namespace {

// An FRU combination's enumerator name, e.g. "pXnYpZ", spells out the axis that Forward, Right,
// and Up point along. Returns false for the named presets.
bool ParseAxesName(const FString& Name, FVector OutAxes[3]) {
  if (Name.Len() != 6) {
    return false;
  }
  for (int32 Semantic = 0; Semantic < 3; ++Semantic) {
    const TCHAR Sign = Name[2 * Semantic];
    const TCHAR Axis = Name[2 * Semantic + 1];
    if ((Sign != TEXT('p') && Sign != TEXT('n')) || Axis < TEXT('X') || Axis > TEXT('Z')) {
      return false;
    }
    FVector Direction = FVector::ZeroVector;
    Direction[Axis - TEXT('X')] = Sign == TEXT('p') ? 1.0 : -1.0;
    OutAxes[Semantic] = Direction;
  }
  return true;
}

FVector ToFVector(const mochi::Real3& Vector) {
  return FVector(Vector[0], Vector[1], Vector[2]);
}

struct FSimulationSpaceStateGuard {
  UMochiPhysicsSettings* ProjectSettings = GetMutableDefault<UMochiPhysicsSettings>();
  FMochiSpaceDefinition ProjectSpace = ProjectSettings->SimulationSpace;
  bool bHadOverride = UMochiPhysicsSettings::HasActiveSimulationSpaceOverride();
  FMochiSpaceDefinition ActiveSpace = UMochiPhysicsSettings::GetActiveSimulationSpace();

  ~FSimulationSpaceStateGuard() {
    ProjectSettings->SimulationSpace = ProjectSpace;
    if (bHadOverride) {
      UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(ActiveSpace);
    } else {
      UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
    }
  }
};

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiCoordinateSpaceNamedMappingsTest,
    "SuperDexPhysics.Coordinates.NativeMappings",
    MOCHI_COORDINATE_SPACE_TEST_FLAGS)

bool FMochiCoordinateSpaceNamedMappingsTest::RunTest(const FString& Parameters) {
  struct FExpectedMapping {
    EMochiSpaceAxes PluginAxes;
    mochi::CoordinateSpaceAxes NativeAxes;
  };
  const FExpectedMapping ExpectedMappings[] = {
      {EMochiSpaceAxes::Unreal, mochi::CoordinateSpaceAxes::FRU},
      {EMochiSpaceAxes::ROS, mochi::CoordinateSpaceAxes::FLU},
      {EMochiSpaceAxes::SuperDex, mochi::CoordinateSpaceAxes::FLU},
      {EMochiSpaceAxes::Mochi, mochi::CoordinateSpaceAxes::RUB},
      {EMochiSpaceAxes::Unity, mochi::CoordinateSpaceAxes::RUF},
  };

  for (const FExpectedMapping& Expected : ExpectedMappings) {
    const mochi::CoordinateSpace Actual = ToMochiCoordinateSpace({Expected.PluginAxes, 7.5});
    TestEqual(TEXT("Native axes"), Actual.axes, Expected.NativeAxes);
    TestEqual(TEXT("Units per meter"), Actual.unitsPerMeter, 7.5);
  }

  // New projects simulate in SuperDex space: it is the C++ default of the project setting and of
  // the per-scene override.
  const FMochiSceneSettings SceneSettings;
  TestTrue(
      TEXT("Scene override defaults to SuperDex space"),
      SceneSettings.SimulationSpace == kSuperDexSpace);
  const mochi::CoordinateSpace DefaultSpace = ToMochiCoordinateSpace(kSuperDexSpace);
  TestEqual(TEXT("Default axes"), DefaultSpace.axes, mochi::CoordinateSpaceAxes::FLU);
  TestEqual(TEXT("Default uses meters"), DefaultSpace.unitsPerMeter, 1.0);
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiCoordinateSpaceExhaustiveMappingTest,
    "SuperDexPhysics.Coordinates.ExhaustiveNativeMapping",
    MOCHI_COORDINATE_SPACE_TEST_FLAGS)

bool FMochiCoordinateSpaceExhaustiveMappingTest::RunTest(const FString& Parameters) {
  const UEnum* AxesEnum = StaticEnum<EMochiSpaceAxes>();
  TestNotNull(TEXT("EMochiSpaceAxes reflection"), AxesEnum);
  if (!AxesEnum) {
    return false;
  }

  int32 NumCombinations = 0;
  for (int32 Index = 0; Index < AxesEnum->NumEnums() - 1; ++Index) {
    const FString Name = AxesEnum->GetNameStringByIndex(Index);
    FVector Expected[3];
    if (!ParseAxesName(Name, Expected)) {
      continue;
    }
    ++NumCombinations;
    const auto PluginAxes = static_cast<EMochiSpaceAxes>(AxesEnum->GetValueByIndex(Index));
    const mochi::CoordinateSpace NativeSpace = ToMochiCoordinateSpace({PluginAxes, 1.0});

    TestEqual(
        *FString::Printf(TEXT("%s forward"), *Name),
        ToFVector(NativeSpace.GetForward()),
        Expected[0]);
    TestEqual(
        *FString::Printf(TEXT("%s right"), *Name), ToFVector(NativeSpace.GetRight()), Expected[1]);
    TestEqual(*FString::Printf(TEXT("%s up"), *Name), ToFVector(NativeSpace.GetUp()), Expected[2]);
  }
  TestEqual(TEXT("Every FRU combination is listed"), NumCombinations, 48);

  TestTrue(
      TEXT("FLU to FRU flips handedness"),
      FMochiSpaceConverter(
          FMochiSpaceDefinition{EMochiSpaceAxes::ROS, 1.0},
          FMochiSpaceDefinition{EMochiSpaceAxes::Unreal, 1.0})
          .FlipsHandedness());
  TestFalse(
      TEXT("FLU to Mochi preserves handedness"),
      FMochiSpaceConverter(
          FMochiSpaceDefinition{EMochiSpaceAxes::ROS, 1.0},
          FMochiSpaceDefinition{EMochiSpaceAxes::Mochi, 1.0})
          .FlipsHandedness());
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiCoordinateSpaceActivePolicyTest,
    "SuperDexPhysics.Coordinates.ActiveSimulationSpacePolicy",
    MOCHI_COORDINATE_SPACE_TEST_FLAGS)

bool FMochiCoordinateSpaceActivePolicyTest::RunTest(const FString& Parameters) {
  const FMochiSpaceDefinition OverrideSpace{EMochiSpaceAxes::ROS, 7.5};
  const FMochiSpaceDefinition ProjectSpace{EMochiSpaceAxes::Unity, 2.5};

#if !UE_BUILD_SHIPPING
  IMochiPhysicsModule* Module = IMochiPhysicsModule::Get();
  mochi::Context* Context = Module ? Module->GetMochiContext() : nullptr;
  if (Context && Context->GetDebugServer().GetPort() != 0) {
    AddInfo(TEXT("Skipping active-space policy mutation while the Mochi debugger is listening."));
    return true;
  }
#endif

  FSimulationSpaceStateGuard StateGuard;
  UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
  StateGuard.ProjectSettings->SimulationSpace = ProjectSpace;
  TestEqual(
      TEXT("Project space is active without an override"),
      UMochiPhysicsSettings::GetActiveSimulationSpace(),
      ProjectSpace);

  UMochiPhysicsSettings::SetActiveSimulationSpaceOverride(OverrideSpace);
  TestEqual(
      TEXT("Scene override replaces the project space"),
      UMochiPhysicsSettings::GetActiveSimulationSpace(),
      OverrideSpace);

  UMochiPhysicsSettings::ClearActiveSimulationSpaceOverride();
  TestEqual(
      TEXT("Clearing the override restores the project space"),
      UMochiPhysicsSettings::GetActiveSimulationSpace(),
      ProjectSpace);

  return true;
}

#undef MOCHI_COORDINATE_SPACE_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS
