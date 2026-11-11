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

#include "SuperDexRecorderModule.h"

#include "ISuperDexRoboticsModule.h"
#include "MochiPhysicsModule.h"

#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"

#include <mochi_physics/mochi_physics.h>
#include <superdex_recorder/recorder.h>

DEFINE_LOG_CATEGORY(LogSuperDexRecorder);

void FSuperDexRecorderModule::StartupModule() {
  IMochiPhysicsModule::Get();
  ISuperDexRoboticsModule::Get();

  const FString BaseDir = IPluginManager::Get().FindPlugin(TEXT("SuperDexRobotics"))->GetBaseDir();
  const FString LibraryDirectory =
      FPaths::Combine(BaseDir, TEXT("ThirdParty/superdex_recorder/lib/Win64"));
  const FString LibraryName = TEXT("superdex_recorder_shared.dll");

  FPlatformProcess::PushDllDirectory(*LibraryDirectory);
  RecorderDllHandle = FPlatformProcess::GetDllHandle(*LibraryName);
  FPlatformProcess::PopDllDirectory(*LibraryDirectory);
  if (RecorderDllHandle == nullptr) {
    UE_LOG(LogSuperDexRecorder, Error, TEXT("Failed to load %s"), *LibraryName);
    return;
  }
  // SuperDexPhysics routes Mochi's log to UE_LOG, but the recorder library has its own callback.
  superdex::recorder::SetLogCallback(mochi::Context::GetLogCallback());
}

void FSuperDexRecorderModule::ShutdownModule() {
  if (RecorderDllHandle != nullptr) {
    superdex::recorder::SetLogCallback({});
    FPlatformProcess::FreeDllHandle(RecorderDllHandle);
    RecorderDllHandle = nullptr;
  }
}

IMPLEMENT_MODULE(FSuperDexRecorderModule, SuperDexRecorder)
