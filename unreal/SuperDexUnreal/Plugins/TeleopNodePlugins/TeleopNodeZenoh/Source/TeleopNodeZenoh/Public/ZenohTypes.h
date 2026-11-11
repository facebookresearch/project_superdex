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

#include "ZenohTypes.generated.h"

UENUM(BlueprintType)
enum class EZenohSessionMode : uint8 {
  // Connect to a zenoh router (zenohd).
  Client,
  // Talk directly to other peers, no router.
  Peer,
};

// Zenoh message priorities, highest first.
UENUM(BlueprintType)
enum class EZenohPriority : uint8 {
  RealTime,
  InteractiveHigh,
  InteractiveLow,
  DataHigh,
  Data,
  DataLow,
  Background,
};

UENUM(BlueprintType)
enum class EZenohCongestionControl : uint8 {
  // Drop the message when the link is congested.
  Drop,
  // Wait for the link (blocks the publishing thread).
  Block,
};

struct FZenohSessionConfig {
  EZenohSessionMode Mode = EZenohSessionMode::Client;
  TArray<FString> ConnectEndpoints;
  // Peer only; empty = don't listen.
  FString ListenEndpoint;
  bool bMulticastScouting = true;

  FString Describe() const;
};

struct FZenohPublisherQos {
  EZenohPriority Priority = EZenohPriority::Data;
  EZenohCongestionControl CongestionControl = EZenohCongestionControl::Drop;
  // Send right away instead of batching with other messages.
  bool bExpress = false;
};
