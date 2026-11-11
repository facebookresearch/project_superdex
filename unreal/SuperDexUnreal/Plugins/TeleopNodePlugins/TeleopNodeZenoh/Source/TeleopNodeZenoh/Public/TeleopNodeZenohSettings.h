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
#include "ZenohTypes.h"

#include "TeleopNodeZenohSettings.generated.h"

// Stored in Config/DefaultGameUserSettings.ini
UCLASS(Config = GameUserSettings, DefaultConfig, meta = (DisplayName = "Teleop Node Zenoh"))
class TELEOPNODEZENOH_API UTeleopNodeZenohSettings : public UDeveloperSettings {
  GENERATED_BODY()

 public:
  /**
   * Client: connect to a zenoh router (zenohd) at Connect Endpoints, or find one by multicast
   * scouting.
   *
   * Peer: no router. Fully local (e.g. operator and robot in the same executable): set Listen
   * Endpoint to tcp/127.0.0.1:7447 and leave Connect Endpoints empty.
   */
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Session")
  EZenohSessionMode Mode = EZenohSessionMode::Client;

  // Client: the router, e.g. tcp/192.168.1.20:7447 (empty = find one via multicast scouting).
  // Peer: other peers to connect to directly.
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Session")
  TArray<FString> ConnectEndpoints;

  // Peer only: e.g. tcp/127.0.0.1:7447 for this device only, tcp/0.0.0.0:7447 to accept other
  // machines.
  UPROPERTY(
      Config,
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Session",
      meta = (EditCondition = "Mode == EZenohSessionMode::Peer", EditConditionHides))
  FString ListenEndpoint;

  // Look for routers/peers via UDP multicast. Prefer explicit endpoints; scouting can pick the
  // wrong network adapter on machines with virtual adapters (WSL, Hyper-V).
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Session")
  bool bMulticastScouting = true;

  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Publishers")
  EZenohPriority Priority = EZenohPriority::InteractiveHigh;

  // Drop keeps the game thread from waiting on a congested link
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Publishers")
  EZenohCongestionControl CongestionControl = EZenohCongestionControl::Drop;

  // Send each message right away instead of batching it with others
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Publishers")
  bool bExpress = true;

  // Samples kept per subscriber between TeleopNode pumps
  UPROPERTY(
      Config,
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Subscribers",
      meta = (ClampMin = 1))
  int32 SubscriberQueueDepth = 1;

  // The Session settings with any command-line overrides applied.
  FZenohSessionConfig GetSessionConfig() const;

  FZenohPublisherQos GetPublisherQos() const {
    return {Priority, CongestionControl, bExpress};
  }

  virtual FName GetCategoryName() const override {
    return FName(TEXT("Plugins"));
  }
};
