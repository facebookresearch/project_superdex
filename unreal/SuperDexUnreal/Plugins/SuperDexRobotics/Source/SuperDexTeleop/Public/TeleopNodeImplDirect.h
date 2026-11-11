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
#include "TeleopDirectSubsystem.h"
#include "TeleopNodeImplBase.h"

#include "TeleopNodeImplDirect.generated.h"

class UTeleopDirectSubsystem;

/**
 * Direct communication backend for in-process teleop communication.
 * Uses UTeleopDirectSubsystem as the message broker.
 *
 * This implementation enables communication between operator and robot
 * components within the same game instance without requiring ROS2.
 */
UCLASS()
class SUPERDEXTELEOP_API UTeleopNodeImplDirect : public UTeleopNodeImplBase {
  GENERATED_BODY()

 public:
  //----------------------------------------------------------------------------
  // Lifecycle
  //----------------------------------------------------------------------------

  virtual bool Initialize() override;
  virtual void Deinitialize() override;
  virtual void Pump() override;

  //----------------------------------------------------------------------------
  // Publisher creation
  //----------------------------------------------------------------------------

  virtual FTeleopPublisherHandle CreatePublisher_DoubleArray(const FString& TopicName) override;
  virtual FTeleopPublisherHandle CreatePublisher_TransformArray(const FString& TopicName) override;
  virtual FTeleopPublisherHandle CreatePublisher_String(const FString& TopicName) override;
  virtual FTeleopPublisherHandle CreatePublisher_ByteArray(const FString& TopicName) override;

  //----------------------------------------------------------------------------
  // Subscriber creation
  //----------------------------------------------------------------------------

  virtual FTeleopSubscriberHandle CreateSubscriber_DoubleArray(
      const FString& TopicName,
      FDoubleArrayCallback Callback) override;
  virtual FTeleopSubscriberHandle CreateSubscriber_TransformArray(
      const FString& TopicName,
      FTransformArrayCallback Callback) override;
  virtual FTeleopSubscriberHandle CreateSubscriber_String(
      const FString& TopicName,
      FStringCallback Callback) override;
  virtual FTeleopSubscriberHandle CreateSubscriber_ByteArray(
      const FString& TopicName,
      FByteArrayCallback Callback) override;

  //----------------------------------------------------------------------------
  // Publishing
  //----------------------------------------------------------------------------

  virtual void Publish_DoubleArray(FTeleopPublisherHandle Handle, const TArray<double>& Data)
      override;
  virtual void Publish_TransformArray(FTeleopPublisherHandle Handle, const TArray<FTransform>& Data)
      override;
  virtual void Publish_String(FTeleopPublisherHandle Handle, const FString& Data) override;
  virtual void Publish_ByteArray(FTeleopPublisherHandle Handle, const TArray<uint8>& Data) override;

 private:
  // Cached reference to the subsystem
  TWeakObjectPtr<UTeleopDirectSubsystem> Subsystem;

  // Map from our local handle IDs to subsystem handle IDs
  TMap<int32, int32> LocalToSubsystemPublisherHandle;
  TMap<int32, int32> LocalToSubsystemSubscriberHandle;

  // Map from our local handle IDs to topic names (for cleanup)
  TMap<int32, FString> PublisherTopics;
  TMap<int32, FString> SubscriberTopics;

  // Local handle counter
  int32 NextLocalHandleId = 0;

  // Helper to get the subsystem
  UTeleopDirectSubsystem* GetSubsystem();
};
