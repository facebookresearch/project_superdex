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
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"

// forward declarations
namespace superdex::robotics {
class RoboticsContext;
} // namespace superdex::robotics

/// Interface for the SuperDexRobotics module. Owns the superdex_robotics native library load and
/// the superdex::robotics::RoboticsContext singleton; the Mochi context (GetMochiContext()) and
/// the mochi_physics library load live on IMochiPhysicsModule.
class ISuperDexRoboticsModule : public IModuleInterface {
 public:
  static inline ISuperDexRoboticsModule* Get() {
    return FModuleManager::LoadModulePtr<ISuperDexRoboticsModule>("SuperDexRobotics");
  }

  static inline bool IsLoaded() {
    return FModuleManager::Get().IsModuleLoaded("SuperDexRobotics");
  }

  // ISuperDexRoboticsModule Interface

  /* Return the RoboticsContext, lazily creating it inside the superdex_robotics DLL memory region
   * on first call. Safe to call repeatedly -- subsequent calls return the same pointer. */
  virtual superdex::robotics::RoboticsContext* GetRoboticsContext() = 0;
};
