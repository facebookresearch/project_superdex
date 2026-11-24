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
  const FString BaseDir = IPluginManager::Get().FindPlugin(TEXT("SuperDexRobotics"))->GetBaseDir();
#if PLATFORM_WINDOWS
  FString BinariesDirectory =
      FPaths::Combine(BaseDir, TEXT("/ThirdParty/superdex_robotics/lib/Win64"));
#elif PLATFORM_LINUX
  FString BinariesDirectory =
      FPaths::Combine(BaseDir, TEXT("/ThirdParty/superdex_robotics/lib/Linux"));
#else
  FString BinariesDirectory =
      FPaths::Combine(BaseDir, TEXT("/ThirdParty/superdex_robotics/lib/Android"));
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
