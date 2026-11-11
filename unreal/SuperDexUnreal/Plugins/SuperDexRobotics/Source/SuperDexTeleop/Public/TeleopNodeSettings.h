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
#include "Engine/DeveloperSettings.h"

#include "TeleopNodeSettings.generated.h"

// Project-level teleop node settings, shown under Project Settings > Plugins > SuperDex Teleop.
UCLASS(Config = Game, defaultconfig, DisplayName = "SuperDex Teleop")
class SUPERDEXTELEOP_API UTeleopNodeSettings : public UDeveloperSettings {
  GENERATED_BODY()

 public:
  // The UTeleopNodeImplBase class that teleop nodes with an empty CommunicationClassName use.
  // When empty, they use UTeleopNodeImplDirect.
  UPROPERTY(
      Config,
      EditAnywhere,
      Category = "Teleop Node",
      meta = (GetOptions = "GetCommunicationClassOptions"))
  FString DefaultCommunicationClassName;

  virtual FName GetCategoryName() const override;

#if WITH_EDITOR
  UFUNCTION(Category = "Teleop Node")
  TArray<FString> GetCommunicationClassOptions() const;
#endif
};
