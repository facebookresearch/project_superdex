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
#include "HAL/CriticalSection.h"
#include "InputCoreTypes.h"
#include "Modules/ModuleManager.h"

DECLARE_LOG_CATEGORY_EXTERN(LogAndroidInput, Log, All);

struct ANDROIDINPUT_API FAndroidInputKeys {
  static const FKey AndroidPrintScreen;
  static void RegisterKeys();
};

struct FAxisState {
  float LeftStickX = 0.0f;
  float LeftStickY = 0.0f;
  float RightStickX = 0.0f;
  float RightStickY = 0.0f;
  float LeftTrigger = 0.0f;
  float RightTrigger = 0.0f;
};

class FAndroidInputModule : public IModuleInterface {
 public:
  static FAndroidInputModule* Get() {
    return FModuleManager::LoadModulePtr<FAndroidInputModule>("AndroidInput");
  }

  virtual void StartupModule() override;
  virtual void ShutdownModule() override;

  FAxisState AxisState;
  FCriticalSection AxisStateLock;
};
