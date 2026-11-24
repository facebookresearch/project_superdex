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

/** Selects whether the session connects to a router or talks directly to peers. */
UENUM(BlueprintType)
enum class EZenohSessionMode : uint8 {
  /** Connect to a zenoh router (zenohd). */
  Client,

  /** Talk directly to other peers, no router. */
  Peer,
};

/** Zenoh message priorities, highest first. */
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

/** Selects whether a congested publisher drops messages or waits for the link. */
UENUM(BlueprintType)
enum class EZenohCongestionControl : uint8 {
  /** Drop the message when the link is congested. */
  Drop,
  /** Wait for the link (blocks the publishing thread). */
  Block,
};

/** Settings passed to the native session-opening operation. */
struct FZenohSessionConfig {
  /** Whether to connect through a router or directly to peers. */
  EZenohSessionMode Mode = EZenohSessionMode::Client;

  /** Endpoints supplied to the native connection configuration. */
  TArray<FString> ConnectEndpoints;

  /** Peer only; empty = don't listen. */
  FString ListenEndpoint;

  /** Whether to enable multicast scouting. */
  bool bMulticastScouting = true;

  /**
   * Formats the session settings for diagnostics.
   *
   * @return The mode, endpoints and scouting setting as a diagnostic string.
   */
  FString Describe() const;
};

/** Quality-of-service settings supplied when declaring a publisher. */
struct FZenohPublisherQos {
  /** The outgoing message priority. */
  EZenohPriority Priority = EZenohPriority::Data;

  /** How to handle a congested link. */
  EZenohCongestionControl CongestionControl = EZenohCongestionControl::Drop;

  /** Send right away instead of batching with other messages. */
  bool bExpress = false;
};
