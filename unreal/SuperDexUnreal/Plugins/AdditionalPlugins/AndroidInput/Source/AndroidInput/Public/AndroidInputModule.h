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

/** Additional Android input keys and their registration. */
struct ANDROIDINPUT_API FAndroidInputKeys {
  /** Key for the Android SysRq/Print Screen button. */
  static const FKey AndroidPrintScreen;

  /** Adds the plugin's key and input-menu category to the engine key registry. */
  static void RegisterKeys();
};

/** Stored Android gamepad axis values, protected by the module's axis-state lock. */
struct FAxisState {
  /** Stored left thumbstick X value. */
  float LeftStickX = 0.0f;

  /** Stored left thumbstick Y value. */
  float LeftStickY = 0.0f;

  /** Stored right thumbstick X value. */
  float RightStickX = 0.0f;

  /** Stored right thumbstick Y value. */
  float RightStickY = 0.0f;

  /** Stored left trigger value. */
  float LeftTrigger = 0.0f;

  /** Stored right trigger value. */
  float RightTrigger = 0.0f;
};

/** Registers Android input keys and installs the native input-queue hook on Android. */
class FAndroidInputModule : public IModuleInterface {
 public:
  /**
   * Loads and retrieves the Android input module.
   *
   * @return The module pointer, or nullptr if the module cannot be loaded.
   */
  static FAndroidInputModule* Get() {
    return FModuleManager::LoadModulePtr<FAndroidInputModule>("AndroidInput");
  }

  virtual void StartupModule() override;
  virtual void ShutdownModule() override;

  /** Axis values retained when the native input hook submits an axis event for dispatch. */
  FAxisState AxisState;

  /** Protects axis-state updates while processing native motion events. */
  FCriticalSection AxisStateLock;
};
