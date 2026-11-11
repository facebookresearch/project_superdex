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

#include "MochiPhysicsModule.h"

#include "MochiPhysicsSettings.h"
#include "MochiSpaceConverter.h"

#include <mochi_physics/mochi_physics.h>

#include "Interfaces/IPluginManager.h"

#include <thread>

#ifdef _MSC_VER
#define RT_DISABLE_OPTIMIZE 0
#if RT_DISABLE_OPTIMIZE
#pragma optimize("", off)
#endif
#endif

DEFINE_LOG_CATEGORY(LogMochiPhysics);

// ------------------------------------------------------------------------
// FMochiPhysicsModule Declaration
// ------------------------------------------------------------------------
class FMochiPhysicsModule final : public IMochiPhysicsModule {
 public:
  // IModuleInterface interface
  void StartupModule() override;
  void ShutdownModule() override;

  // IMochiPhysicsModule interface
  mochi::Context* GetMochiContext() override;

  // Methods
  void Initialize();
  void ShutDown();

  // Fields
  void* MochiPhysicsDllHandle = nullptr;
  mochi::Context* MochiContext = nullptr;
};

// ------------------------------------------------------------------------
// FMochiPhysicsModule Implementation
// ------------------------------------------------------------------------

void FMochiPhysicsModule::StartupModule() {
  Initialize();
}

void FMochiPhysicsModule::ShutdownModule() {
  ShutDown();
}

mochi::Context* FMochiPhysicsModule::GetMochiContext() {
  return MochiContext;
}

void FMochiPhysicsModule::Initialize() {
  if (MochiPhysicsDllHandle || MochiContext) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("FMochiPhysicsModule::Initialize called when already initialized"));
    return;
  }

  // Load library
  // TODO: figure out Android
  const FString baseDir = IPluginManager::Get().FindPlugin(TEXT("SuperDexPhysics"))->GetBaseDir();
#if PLATFORM_WINDOWS
  FString BinariesDirectory = FPaths::Combine(baseDir, TEXT("/ThirdParty/mochi_physics/lib/Win64"));
#elif PLATFORM_LINUX
  FString BinariesDirectory = FPaths::Combine(baseDir, TEXT("/ThirdParty/mochi_physics/lib/Linux"));
#else
  FString BinariesDirectory =
      FPaths::Combine(baseDir, TEXT("/ThirdParty/mochi_physics/lib/Android"));
#endif

  FPlatformProcess::PushDllDirectory(*BinariesDirectory);
#if PLATFORM_WINDOWS
  FString DllName = "mochi_physics.dll"; // TODO: figure out Android. libmochi_physics.so?
  MochiPhysicsDllHandle = FPlatformProcess::GetDllHandle(*DllName);
#else
  FString DllName = "libmochi_physics.so";
  MochiPhysicsDllHandle = FPlatformProcess::GetDllHandle(*DllName);
#endif
  FPlatformProcess::PopDllDirectory(*BinariesDirectory);
  // The superdex_robotics library is loaded by the SuperDexRobotics plugin's own module.

  // Create MochiContext
  const auto settings = GetDefault<UMochiPhysicsSettings>();
  if (settings && (settings->NumWorkerThreads > 0)) {
    MochiContext = mochi::CreateContext(settings->NumWorkerThreads);
  } else {
    // Let Mochi decide how many worker threads to use on this machine
    MochiContext = mochi::CreateContext();
  }

  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT("FMochiPhysicsModule::Initialize initialized MochiPhysics with %d worker threads."),
      MochiContext->GetNumThreads());

#if !UE_BUILD_SHIPPING
  if (settings && settings->bEnablePhysicsDebuggerConnection) {
    auto& DebugServer = MochiContext->GetDebugServer();
    DebugServer.SetCoordinateSpace(
        ToMochiCoordinateSpace(UMochiPhysicsSettings::GetActiveSimulationSpace()));
    DebugServer.Start();

    const uint16 DebugServerPort = DebugServer.GetPort();
    if (DebugServerPort != 0) {
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT("Mochi physics debugger server started on port %u."),
          DebugServerPort);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT("Mochi physics debugger server could not open a TCP listening port."));
    }
  }
#endif

  MochiContext->SetLogCallback(
      [](mochi::LogChannel channel, const char* msg, const char* file, int line) {
        // Note that the ANSI_TO_TCHAR macro uses "a class instance declared in local scope and
        // allocate space on the stack, so it is very important you do not retain pointers to them!
        // They are intended only for passing strings to function calls."
        if (channel == mochi::LogChannel::Verbose) {
          UE_LOG(LogMochiPhysics, Verbose, TEXT("%s"), ANSI_TO_TCHAR(msg));
        } else if (channel == mochi::LogChannel::Info) {
          UE_LOG(LogMochiPhysics, Display, TEXT("%s"), ANSI_TO_TCHAR(msg));
        } else if (channel == mochi::LogChannel::Warning) {
          UE_LOG(LogMochiPhysics, Warning, TEXT("%s"), ANSI_TO_TCHAR(msg));
        } else if (channel == mochi::LogChannel::Error) {
          UE_LOG(LogMochiPhysics, Error, TEXT("%s"), ANSI_TO_TCHAR(msg));
        } else {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT("Unexpected mochi log level [%d]"),
              static_cast<int>(channel));
          UE_LOG(LogMochiPhysics, Error, TEXT("%s"), ANSI_TO_TCHAR(msg));
        }
      });
}

void FMochiPhysicsModule::ShutDown() {
  UE_LOG(LogMochiPhysics, Display, TEXT("FMochiPhysicsModule::ShutDown"));

  // Destroy world
  if (MochiContext) {
    mochi::DestroyContext(MochiContext);
    MochiContext = nullptr;
  }

  if (MochiPhysicsDllHandle) {
    FPlatformProcess::FreeDllHandle(MochiPhysicsDllHandle);
    MochiPhysicsDllHandle = nullptr;
  }
}

IMPLEMENT_MODULE(FMochiPhysicsModule, SuperDexPhysics)

#ifdef _MSC_VER
#if RT_DISABLE_OPTIMIZE
#pragma optimize("", on)
#endif
#undef RT_DISABLE_OPTIMIZE
#endif
