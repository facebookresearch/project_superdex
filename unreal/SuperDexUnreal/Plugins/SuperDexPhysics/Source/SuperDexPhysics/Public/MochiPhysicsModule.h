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

// Exported (not just extern) so sibling plugins (e.g. SuperDexRobotics) that were split out of this
// module can keep logging to the shared "Mochi" category without needing their own.
SUPERDEXPHYSICS_API DECLARE_LOG_CATEGORY_EXTERN(LogMochiPhysics, Log, All);

// forward declarations
namespace mochi {
class Context;
} // namespace mochi

class IMochiPhysicsModule : public IModuleInterface {
 public:
  static inline IMochiPhysicsModule* Get() {
    return FModuleManager::LoadModulePtr<IMochiPhysicsModule>("SuperDexPhysics");
  }

  static inline bool IsLoaded() {
    return FModuleManager::Get().IsModuleLoaded("SuperDexPhysics");
  }

  // IMochiPhysicsModule Interface
  virtual mochi::Context* GetMochiContext() = 0;
};
