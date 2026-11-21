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

#include "SuperDexBotComponent.h"

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "Tests/SuperDexBotComponentTestTypes.h"
#include "UObject/StrongObjectPtr.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotComponentScaleTest,
    "SuperDexRobotics.SuperDexBotComponent.BotScaleFromComponentScale",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotComponentScaleTest::RunTest(const FString& Parameters) {
  using EFit = USuperDexBotComponent::EBotScaleFit;
  struct FCase {
    FVector ComponentScale;
    EFit Fit;
    double BotScale;
  };
  const FCase Cases[] = {
      {FVector(1.0), EFit::Uniform, 1.0},
      {FVector(0.7), EFit::Uniform, 0.7},
      {FVector(1.0, 0.5, 0.6), EFit::NonUniform, 0.7},
      {FVector(-0.5), EFit::NonUniform, 0.5},
      {FVector(1.0, -1.0, 1.0), EFit::NonUniform, 1.0},
      {FVector(0.0), EFit::Degenerate, 0.0},
      // A zero axis is degenerate even when the mean magnitude is not small.
      {FVector(0.0, 0.0, 3.0), EFit::Degenerate, 1.0},
  };
  for (const FCase& Case : Cases) {
    double BotScale = -1.0;
    const EFit Fit = USuperDexBotComponent::ResolveUniformBotScale(Case.ComponentScale, BotScale);
    const FString What = Case.ComponentScale.ToString();
    TestEqual(What + TEXT(" fit"), static_cast<int>(Fit), static_cast<int>(Case.Fit));
    TestNearlyEqual(What + TEXT(" bot scale"), BotScale, Case.BotScale, 1e-9);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotComponentControllerSwapTest,
    "SuperDexRobotics.SuperDexBotComponent.ControllerSwapLetsTheControlCallbackRun",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotComponentControllerSwapTest::RunTest(const FString& Parameters) {
  // The Mochi simulation thread runs the bot's control callback every step, under the component's
  // controller lock. A controller that waits for that thread while being initialized or shut down
  // (UController_MochiArticulatedPose destroys its Mochi controller that way when its roboperator
  // is destroyed) must not do so under the lock: the thread would wait on it and never get there.
  TStrongObjectPtr<USuperDexBotComponentTestComponent> Component(
      NewObject<USuperDexBotComponentTestComponent>());
  TestTrue(
      TEXT("The controller initializes"),
      Component->SetController(USuperDexBotControllerWaitsForControlCallback::StaticClass()));
  auto* Controller =
      Cast<USuperDexBotControllerWaitsForControlCallback>(Component->GetController());
  if (!TestNotNull(TEXT("The component has the controller"), Controller)) {
    return false;
  }
  TestTrue(
      TEXT("The control callback runs while the controller initializes"),
      Controller->bCallbackRanDuringInitialize);
  TestTrue(TEXT("The controller is removed"), Component->SetController(nullptr));
  TestTrue(
      TEXT("The control callback runs while the controller shuts down"),
      Controller->bCallbackRanDuringShutdown);
  TestNull(TEXT("The component has no controller"), Component->GetController());
  Controller->Callback.Wait();
  TestFalse(TEXT("Without a controller the callback does not control"), Controller->Callback.Get());
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
