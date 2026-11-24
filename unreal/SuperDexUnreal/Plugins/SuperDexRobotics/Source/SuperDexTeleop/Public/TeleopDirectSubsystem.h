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
#include "Subsystems/GameInstanceSubsystem.h"

#include <functional>

#include "TeleopDirectSubsystem.generated.h"

// Callback types
using FTeleopDoubleArrayCallback = std::function<void(const TArray<double>&)>;
using FTeleopTransformArrayCallback = std::function<void(const TArray<FTransform>&)>;
using FTeleopStringCallback = std::function<void(const FString&)>;
using FTeleopByteArrayCallback = std::function<void(const TArray<uint8>&)>;

/** Message type enumeration for type safety */
enum class ETeleopMessageType : uint8 { DoubleArray, TransformArray, String, ByteArray };

/** Typed subscriber storage */
struct FTeleopSubscriberEntry {
  ETeleopMessageType Type;
  FTeleopDoubleArrayCallback DoubleArrayCallback;
  FTeleopTransformArrayCallback TransformArrayCallback;
  FTeleopStringCallback StringCallback;
  FTeleopByteArrayCallback ByteArrayCallback;
};

/** Base structure for topic data storage */
struct FTeleopTopicData {
  ETeleopMessageType Type;
  FDateTime LastUpdated;
  TArray<FTeleopSubscriberEntry> Subscribers;

  /** Typed data storage (only one will be used based on Type) */
  TArray<double> DoubleArrayData;
  TArray<FTransform> TransformArrayData;
  FString StringData;
  TArray<uint8> ByteArrayData;

  /** Dirty flag: true if new data has been published since last pump */
  bool bDirty = false;
};

/**
 * UTeleopDirectSubsystem acts as a message broker for in-process teleop
 * communication. It provides a topic-based publish-subscribe system that allows
 * multiple UPrototypeTeleopNodeImplDirect instances to communicate within the
 * same game instance.
 *
 * Thread Safety:
 * - All public methods are thread-safe through use of FRWLock
 * - When bInstantDelivery is false (default): Callbacks are invoked during Pump() on the game
 * thread
 * - When bInstantDelivery is true: Callbacks are invoked immediately during Publish_*
 */
UCLASS()
class SUPERDEXTELEOP_API UTeleopDirectSubsystem : public UGameInstanceSubsystem {
  GENERATED_BODY()

 public:
  //----------------------------------------------------------------------------
  // UGameInstanceSubsystem interface
  //----------------------------------------------------------------------------

  virtual void Initialize(FSubsystemCollectionBase& Collection) override;
  virtual void Deinitialize() override;

  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------

  /**
   * When true, Publish_* calls invoke subscriber callbacks immediately.
   * When false (default), messages are queued and delivered via PumpSubscribers().
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TeleopDirect")
  bool bInstantDelivery = false;

  //----------------------------------------------------------------------------
  // Publisher Registration
  //----------------------------------------------------------------------------

  /** Registers a publisher on TopicName, creating the topic if needed, and returns its ID. */
  int32 RegisterPublisher(const FString& TopicName, ETeleopMessageType Type);
  /** Unregisters a publisher; its topic is kept. */
  void UnregisterPublisher(int32 PublisherId, const FString& TopicName);

  //----------------------------------------------------------------------------
  // Subscriber Registration (typed)
  //----------------------------------------------------------------------------

  /**
   * Registers a double-array callback on TopicName, creating the topic if needed, and returns its
   * ID.
   */
  int32 RegisterSubscriber_DoubleArray(
      const FString& TopicName,
      FTeleopDoubleArrayCallback Callback);

  /** Like RegisterSubscriber_DoubleArray, for transform-array messages. */
  int32 RegisterSubscriber_TransformArray(
      const FString& TopicName,
      FTeleopTransformArrayCallback Callback);

  /** Like RegisterSubscriber_DoubleArray, for string messages. */
  int32 RegisterSubscriber_String(const FString& TopicName, FTeleopStringCallback Callback);

  /** Like RegisterSubscriber_DoubleArray, for byte-array messages. */
  int32 RegisterSubscriber_ByteArray(const FString& TopicName, FTeleopByteArrayCallback Callback);

  /**
   * Unregisters a subscriber by clearing its callback in the topic registry. A delivery that has
   * already copied the subscriber list may still call it.
   */
  void UnregisterSubscriber(int32 SubscriberId, const FString& TopicName);

  //----------------------------------------------------------------------------
  // Publishing (typed)
  //----------------------------------------------------------------------------

  /**
   * Publishes Data on TopicName from a registered publisher, delivering it now with
   * bInstantDelivery or on the next PumpSubscribers otherwise.
   */
  void Publish_DoubleArray(int32 PublisherId, const FString& TopicName, const TArray<double>& Data);

  /** Like Publish_DoubleArray, for transform-array messages. */
  void Publish_TransformArray(
      int32 PublisherId,
      const FString& TopicName,
      const TArray<FTransform>& Data);

  /** Like Publish_DoubleArray, for string messages. */
  void Publish_String(int32 PublisherId, const FString& TopicName, const FString& Data);

  /** Like Publish_DoubleArray, for byte-array messages. */
  void Publish_ByteArray(int32 PublisherId, const FString& TopicName, const TArray<uint8>& Data);

  //----------------------------------------------------------------------------
  // Message Delivery
  //----------------------------------------------------------------------------

  /** Delivers each topic's latest queued message to its subscribers. */
  void PumpSubscribers();

 private:
  // Thread-safe topic registry
  mutable FRWLock TopicLock;
  TMap<FString, FTeleopTopicData> Topics;

  // Handle tracking
  int32 NextHandleId = 0;

  /** Track which handles are registered to which topics (for validation) */
  TMap<int32, FString> PublisherHandleToTopic;
  TMap<int32, FString> SubscriberHandleToTopic;
  TMap<int32, int32> SubscriberHandleToIndex;
};
