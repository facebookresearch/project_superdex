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

#include "ISuperDexRoboticsModule.h"

#include "MochiPhysicsModule.h"

#include <superdex_robotics/superdex_robotics.h>

#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "UObject/CoreRedirects.h"

// ------------------------------------------------------------------------
// FSuperDexRoboticsModule Declaration
// ------------------------------------------------------------------------
class FSuperDexRoboticsModule final : public ISuperDexRoboticsModule {
 public:
  // IModuleInterface interface
  void StartupModule() override;
  void ShutdownModule() override;

  // ISuperDexRoboticsModule interface
  superdex::robotics::RoboticsContext* GetRoboticsContext() override;

  // Fields
  void* SuperDexRoboticsDllHandle = nullptr;
  superdex::robotics::RoboticsContext* BotsCtx = nullptr;
};

// ------------------------------------------------------------------------
// FSuperDexRoboticsModule Implementation
// ------------------------------------------------------------------------

void FSuperDexRoboticsModule::StartupModule() {
  FCoreRedirects::AddRedirectList(
      {FCoreRedirect(
           ECoreRedirectFlags::Type_Struct,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Enum,
           TEXT("/Script/SuperDexRobotics.ESuperDexBotIKTargetSpace"),
           TEXT("/Script/SuperDexRobotics.ESuperDexBotIKGoalSpace")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.Targets"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.Goals")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.Bot"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBot")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBot"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBot")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBot"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBot")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BotConfiguration"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBotConfiguration")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBotConfiguration"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotConfiguration")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBotConfiguration"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotConfiguration")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BaseFromReferenceBot"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBaseFrameFromInputBaseFrame")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BaseFromTargetBot"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBaseFrameFromInputBaseFrame")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBotScale"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotScale")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBotScale"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotScale")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.BotReference"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.OutputBotReference")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.ReferenceBotReference"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotReference")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.TargetBotReference"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.InputBotReference")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceBotPath"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceOutputBotPath")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceReferenceBotPath"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceInputBotPath")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceTargetBotPath"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKConfig.SourceInputBotPath")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.Link"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.OutputBotLink")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.ReferenceBotLink"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.TargetBotLink"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.LinkRelativeFrame"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.LinkFrameFromConstraintFrame")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKTargetConfig.Transform"),
           TEXT(
               "/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.BaseOrInputLinkFrameFromConstraintFrame")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.Link"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.OutputBotLink")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.ReferenceBotLink"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.TargetBotLink"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.InputBotLink")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.LinkRelativeFrame"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.LinkFrameFromConstraintFrame")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.Transform"),
           TEXT(
               "/Script/SuperDexRobotics.SuperDexBotIKGoalConfig.BaseOrInputLinkFrameFromConstraintFrame")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Function,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.ResetTargetsToConfig"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.ResetGoalsToConfig")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Function,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetTargetTransform"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetGoalTransform")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Function,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetTargetEnabled"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetGoalEnabled")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Function,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetTargetStiffness"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SetGoalStiffness")),
       FCoreRedirect(
           ECoreRedirectFlags::Type_Property,
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SolveIK.OutTargetsReached"),
           TEXT("/Script/SuperDexRobotics.SuperDexBotIKComponent.SolveIK.OutGoalsReached"))},
      TEXT("SuperDexRobotics.BotIKInputOutputNomenclature"));

  // Force SuperDexPhysics to start up (and load mochi_physics.dll) before we load
  // superdex_robotics.dll below. superdex_robotics.dll is a shared library that links
  // mochi_core/mochi_physics, so it has a real OS-loader-level import dependency on
  // mochi_physics.dll -- unlike this UE module's own PublicDelayLoadDLLs imports, that native
  // import is resolved eagerly the moment superdex_robotics.dll is loaded. The .uplugin "Plugins"
  // dependency on SuperDexPhysics only guarantees it is *enabled*; it does not guarantee its
  // PreDefault module actually starts up before ours. LoadModulePtr loads SuperDexPhysics now if
  // it hasn't already, and (per FModuleManager's LoadOrder tracking) guarantees it is shut down
  // after us, matching the required teardown order.
  IMochiPhysicsModule::Get();

  // Load superdex_robotics.dll. mochi_physics.dll is loaded by the sibling SuperDexPhysics
  // plugin's own module (FMochiPhysicsModule::Initialize); this module only owns its own native
  // library.
  const FString baseDir = IPluginManager::Get().FindPlugin(TEXT("SuperDexRobotics"))->GetBaseDir();
#if PLATFORM_WINDOWS
  FString BinariesDirectory =
      FPaths::Combine(baseDir, TEXT("/ThirdParty/superdex_robotics/lib/Win64"));
#elif PLATFORM_LINUX
  FString BinariesDirectory =
      FPaths::Combine(baseDir, TEXT("/ThirdParty/superdex_robotics/lib/Linux"));
#else
  FString BinariesDirectory =
      FPaths::Combine(baseDir, TEXT("/ThirdParty/superdex_robotics/lib/Android"));
#endif

  FPlatformProcess::PushDllDirectory(*BinariesDirectory);
#if PLATFORM_WINDOWS
  FString DllName = "superdex_robotics.dll";
#else
  FString DllName = "libsuperdex_robotics.so";
#endif
  SuperDexRoboticsDllHandle = FPlatformProcess::GetDllHandle(*DllName);
  FPlatformProcess::PopDllDirectory(*BinariesDirectory);

  if (!SuperDexRoboticsDllHandle) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("FSuperDexRoboticsModule::StartupModule failed to load %s"),
        *DllName);
  }
}

void FSuperDexRoboticsModule::ShutdownModule() {
  UE_LOG(LogMochiPhysics, Display, TEXT("FSuperDexRoboticsModule::ShutdownModule"));

  /* Destroy the RoboticsContext before unloading the DLL it was allocated in. Mirrors the
   * ordering constraint that used to live in FMochiPhysicsModule::ShutDown(). */
  if (BotsCtx) {
    superdex::robotics::DestroyRoboticsContext(BotsCtx);
    BotsCtx = nullptr;
  }

  if (SuperDexRoboticsDllHandle) {
    FPlatformProcess::FreeDllHandle(SuperDexRoboticsDllHandle);
    SuperDexRoboticsDllHandle = nullptr;
  }
}

superdex::robotics::RoboticsContext* FSuperDexRoboticsModule::GetRoboticsContext() {
  if (!BotsCtx) {
    BotsCtx = superdex::robotics::CreateRoboticsContext();
  }
  return BotsCtx;
}

IMPLEMENT_MODULE(FSuperDexRoboticsModule, SuperDexRobotics)
