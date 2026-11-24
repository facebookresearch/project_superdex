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
#include "ZenohTypes.h"

class FZenohPublisher;
class FZenohSubscriber;

/**
 * Converts "/teleop/left/arm_joint_states" -> "teleop/left/arm_joint_states".
 * Returns false for a topic with no chunks, or containing zenoh's wildcard and
 * verbatim characters (* $ ? #).
 *
 * @param Topic The topic to split into nonempty path chunks.
 * @param OutKeyExpr Receives the joined key; left unchanged if a forbidden character is found.
 * @return True if the topic has at least one chunk and no forbidden characters.
 */
TELEOPNODEZENOH_API bool ZenohTopicToKeyExpr(const FString& Topic, FString& OutKeyExpr);

/** One zenoh session. Get the shared one from FTeleopNodeZenohModule::AcquireSession. */
class TELEOPNODEZENOH_API FZenohSession : public TSharedFromThis<FZenohSession> {
 public:
  /**
   * Null if the session could not open.
   *
   * @param Config The mode, endpoints and scouting settings used to open the session.
   * @return The opened session, or null on failure.
   */
  static TSharedPtr<FZenohSession> Open(const FZenohSessionConfig& Config);

  /** Closes the session after its publishers and subscribers release it. */
  ~FZenohSession();

  /**
   * encoding e.g. "application/protobuf;superdex.teleop.v1.DoubleArray"; empty for none.
   *
   * @param KeyExpr The key on which to publish.
   * @param Encoding The sample encoding, or empty for none.
   * @param Qos The priority, congestion control and express settings.
   * @return The declared publisher, or null on failure.
   */
  TSharedPtr<FZenohPublisher>
  DeclarePublisher(const FString& KeyExpr, const FString& Encoding, const FZenohPublisherQos& Qos);

  /**
   * Keeps the newest QueueDepth samples between drains; older ones are dropped.
   *
   * @param KeyExpr The key to subscribe to.
   * @param QueueDepth The number of queued samples; values below one are treated as one.
   * @return The declared subscriber, or null on failure.
   */
  TSharedPtr<FZenohSubscriber> DeclareSubscriber(const FString& KeyExpr, int32 QueueDepth);

 private:
  friend class FZenohPublisher;
  friend class FZenohSubscriber;
  struct FImpl;

  FZenohSession();
  TUniquePtr<FImpl> Impl;
};

/** Publishes payloads on one key while keeping its session alive. */
class TELEOPNODEZENOH_API FZenohPublisher {
 public:
  /** Undeclares the publisher. */
  ~FZenohPublisher();

  /**
   * Publishes one payload.
   *
   * @param Bytes The payload bytes to copy into the zenoh message.
   * @return True if zenoh accepted the publish operation.
   */
  bool Put(TConstArrayView<uint8> Bytes) const;

  /**
   * Returns the publisher's key expression.
   *
   * @return The key stored when the publisher was declared.
   */
  const FString& GetKeyExpr() const {
    return KeyExpr;
  }

 private:
  friend class FZenohSession;
  struct FImpl;

  FZenohPublisher(TSharedRef<FZenohSession> InSession, FString InKeyExpr);
  TSharedRef<FZenohSession> Session;
  FString KeyExpr;
  TUniquePtr<FImpl> Impl;
};

/** Samples arrive on zenoh's thread and are queued; Drain hands them over on the calling thread. */
class TELEOPNODEZENOH_API FZenohSubscriber {
 public:
  /** Stops queueing samples and undeclares the subscriber. */
  ~FZenohSubscriber();

  /**
   * Calls Fn with each queued payload, oldest first. Returns how many were delivered.
   *
   * @param Fn The callback invoked synchronously for each payload removed from the queue.
   * @return The number of payloads delivered by this drain.
   */
  int32 Drain(TFunctionRef<void(TConstArrayView<uint8>)> Fn);

  /**
   * Returns the subscriber's key expression.
   *
   * @return The key stored when the subscriber was declared.
   */
  const FString& GetKeyExpr() const {
    return KeyExpr;
  }

 private:
  friend class FZenohSession;
  struct FImpl;

  FZenohSubscriber(TSharedRef<FZenohSession> InSession, FString InKeyExpr);
  TSharedRef<FZenohSession> Session;
  FString KeyExpr;
  TUniquePtr<FImpl> Impl;
};
