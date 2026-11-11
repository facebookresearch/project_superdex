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

#include "Modules/ModuleManager.h"

class FZenohSession;

TELEOPNODEZENOH_API DECLARE_LOG_CATEGORY_EXTERN(LogZenoh, Log, All);

class TELEOPNODEZENOH_API FTeleopNodeZenohModule : public IModuleInterface {
 public:
  static FTeleopNodeZenohModule& Get() {
    return FModuleManager::LoadModuleChecked<FTeleopNodeZenohModule>("TeleopNodeZenoh");
  }

  // The process-wide session. Opened on first use from UTeleopNodeZenohSettings and command-line
  // overrides, and is closed when the last holder lets go. Null if it could not open; the next call
  // tries again. Game thread only.
  TSharedPtr<FZenohSession> AcquireSession();

  virtual void ShutdownModule() override;

 private:
  TWeakPtr<FZenohSession> Session;
};
