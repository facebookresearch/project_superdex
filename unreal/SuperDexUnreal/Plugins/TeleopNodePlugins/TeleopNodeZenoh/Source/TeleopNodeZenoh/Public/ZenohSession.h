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

// Converts "/teleop/left/arm_joint_states" -> "teleop/left/arm_joint_states".
// Returns false for a topic with no chunks, or containing zenoh's wildcard and
// verbatim characters (* $ ? #).
TELEOPNODEZENOH_API bool ZenohTopicToKeyExpr(const FString& Topic, FString& OutKeyExpr);

// One zenoh session. Get the shared one from FTeleopNodeZenohModule::AcquireSession.
class TELEOPNODEZENOH_API FZenohSession : public TSharedFromThis<FZenohSession> {
 public:
  // Null if the session could not open.
  static TSharedPtr<FZenohSession> Open(const FZenohSessionConfig& Config);
  ~FZenohSession();

  // encoding e.g. "application/protobuf;superdex.teleop.v1.DoubleArray"; empty for none.
  TSharedPtr<FZenohPublisher>
  DeclarePublisher(const FString& KeyExpr, const FString& Encoding, const FZenohPublisherQos& Qos);

  // Keeps the newest QueueDepth samples between drains; older ones are dropped.
  TSharedPtr<FZenohSubscriber> DeclareSubscriber(const FString& KeyExpr, int32 QueueDepth);

 private:
  friend class FZenohPublisher;
  friend class FZenohSubscriber;
  struct FImpl;

  FZenohSession();
  TUniquePtr<FImpl> Impl;
};

class TELEOPNODEZENOH_API FZenohPublisher {
 public:
  ~FZenohPublisher();

  bool Put(TConstArrayView<uint8> Bytes) const;

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

// Samples arrive on zenoh's thread and are queued; Drain hands them over on the calling thread.
class TELEOPNODEZENOH_API FZenohSubscriber {
 public:
  ~FZenohSubscriber();

  // Calls Fn with each queued payload, oldest first. Returns how many were delivered.
  int32 Drain(TFunctionRef<void(TConstArrayView<uint8>)> Fn);

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
