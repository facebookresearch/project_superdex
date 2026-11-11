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

#include "TeleopDirectSubsystem.h"

#include "SuperDexTeleopModule.h"

void UTeleopDirectSubsystem::Initialize(FSubsystemCollectionBase& Collection) {
  Super::Initialize(Collection);
  UE_LOG(
      LogSuperDexTeleop,
      Display,
      TEXT("UTeleopDirectSubsystem::Initialize - Topic broker started"));
}

void UTeleopDirectSubsystem::Deinitialize() {
  UE_LOG(
      LogSuperDexTeleop,
      Display,
      TEXT("UTeleopDirectSubsystem::Deinitialize - Topic broker shutting down"));

  {
    FWriteScopeLock Lock(TopicLock);
    Topics.Empty();
    PublisherHandleToTopic.Empty();
    SubscriberHandleToTopic.Empty();
    SubscriberHandleToIndex.Empty();
  }

  Super::Deinitialize();
}

int32 UTeleopDirectSubsystem::RegisterPublisher(const FString& TopicName, ETeleopMessageType Type) {
  FWriteScopeLock Lock(TopicLock);

  if (!Topics.Contains(TopicName)) {
    FTeleopTopicData NewTopic;
    NewTopic.Type = Type;
    NewTopic.LastUpdated = FDateTime::MinValue();
    Topics.Add(TopicName, NewTopic);
    UE_LOG(
        LogSuperDexTeleop, Verbose, TEXT("UTeleopDirectSubsystem: Created topic '%s'"), *TopicName);
  }

  const int32 HandleId = NextHandleId++;
  PublisherHandleToTopic.Add(HandleId, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopDirectSubsystem: Registered publisher %d for topic '%s'"),
      HandleId,
      *TopicName);

  return HandleId;
}

void UTeleopDirectSubsystem::UnregisterPublisher(int32 PublisherId, const FString& TopicName) {
  FWriteScopeLock Lock(TopicLock);

  if (PublisherHandleToTopic.Contains(PublisherId)) {
    PublisherHandleToTopic.Remove(PublisherId);
    UE_LOG(
        LogSuperDexTeleop,
        Verbose,
        TEXT("UTeleopDirectSubsystem: Unregistered publisher %d from topic '%s'"),
        PublisherId,
        *TopicName);
  }
}

//------------------------------------------------------------------------------
// Subscriber Registration (typed)
//------------------------------------------------------------------------------

int32 UTeleopDirectSubsystem::RegisterSubscriber_DoubleArray(
    const FString& TopicName,
    FTeleopDoubleArrayCallback Callback) {
  FWriteScopeLock Lock(TopicLock);

  if (!Topics.Contains(TopicName)) {
    FTeleopTopicData NewTopic;
    NewTopic.Type = ETeleopMessageType::DoubleArray;
    NewTopic.LastUpdated = FDateTime::MinValue();
    Topics.Add(TopicName, NewTopic);
    UE_LOG(
        LogSuperDexTeleop,
        Verbose,
        TEXT("UTeleopDirectSubsystem: Created topic '%s' for subscriber"),
        *TopicName);
  }

  FTeleopTopicData& TopicData = Topics[TopicName];

  FTeleopSubscriberEntry Entry;
  Entry.Type = ETeleopMessageType::DoubleArray;
  Entry.DoubleArrayCallback = MoveTemp(Callback);

  const int32 SubscriberIndex = TopicData.Subscribers.Num();
  TopicData.Subscribers.Add(MoveTemp(Entry));

  const int32 HandleId = NextHandleId++;
  SubscriberHandleToTopic.Add(HandleId, TopicName);
  SubscriberHandleToIndex.Add(HandleId, SubscriberIndex);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopDirectSubsystem: Registered DoubleArray subscriber %d for topic '%s'"),
      HandleId,
      *TopicName);

  return HandleId;
}

int32 UTeleopDirectSubsystem::RegisterSubscriber_TransformArray(
    const FString& TopicName,
    FTeleopTransformArrayCallback Callback) {
  FWriteScopeLock Lock(TopicLock);

  if (!Topics.Contains(TopicName)) {
    FTeleopTopicData NewTopic;
    NewTopic.Type = ETeleopMessageType::TransformArray;
    NewTopic.LastUpdated = FDateTime::MinValue();
    Topics.Add(TopicName, NewTopic);
    UE_LOG(
        LogSuperDexTeleop,
        Verbose,
        TEXT("UTeleopDirectSubsystem: Created topic '%s' for subscriber"),
        *TopicName);
  }

  FTeleopTopicData& TopicData = Topics[TopicName];

  FTeleopSubscriberEntry Entry;
  Entry.Type = ETeleopMessageType::TransformArray;
  Entry.TransformArrayCallback = MoveTemp(Callback);

  const int32 SubscriberIndex = TopicData.Subscribers.Num();
  TopicData.Subscribers.Add(MoveTemp(Entry));

  const int32 HandleId = NextHandleId++;
  SubscriberHandleToTopic.Add(HandleId, TopicName);
  SubscriberHandleToIndex.Add(HandleId, SubscriberIndex);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopDirectSubsystem: Registered TransformArray subscriber %d for topic '%s'"),
      HandleId,
      *TopicName);

  return HandleId;
}

int32 UTeleopDirectSubsystem::RegisterSubscriber_String(
    const FString& TopicName,
    FTeleopStringCallback Callback) {
  FWriteScopeLock Lock(TopicLock);

  if (!Topics.Contains(TopicName)) {
    FTeleopTopicData NewTopic;
    NewTopic.Type = ETeleopMessageType::String;
    NewTopic.LastUpdated = FDateTime::MinValue();
    Topics.Add(TopicName, NewTopic);
    UE_LOG(
        LogSuperDexTeleop,
        Verbose,
        TEXT("UTeleopDirectSubsystem: Created topic '%s' for subscriber"),
        *TopicName);
  }

  FTeleopTopicData& TopicData = Topics[TopicName];

  FTeleopSubscriberEntry Entry;
  Entry.Type = ETeleopMessageType::String;
  Entry.StringCallback = MoveTemp(Callback);

  const int32 SubscriberIndex = TopicData.Subscribers.Num();
  TopicData.Subscribers.Add(MoveTemp(Entry));

  const int32 HandleId = NextHandleId++;
  SubscriberHandleToTopic.Add(HandleId, TopicName);
  SubscriberHandleToIndex.Add(HandleId, SubscriberIndex);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopDirectSubsystem: Registered String subscriber %d for topic '%s'"),
      HandleId,
      *TopicName);

  return HandleId;
}

int32 UTeleopDirectSubsystem::RegisterSubscriber_ByteArray(
    const FString& TopicName,
    FTeleopByteArrayCallback Callback) {
  FWriteScopeLock Lock(TopicLock);

  if (!Topics.Contains(TopicName)) {
    FTeleopTopicData NewTopic;
    NewTopic.Type = ETeleopMessageType::ByteArray;
    NewTopic.LastUpdated = FDateTime::MinValue();
    Topics.Add(TopicName, NewTopic);
    UE_LOG(
        LogSuperDexTeleop,
        Verbose,
        TEXT("UTeleopDirectSubsystem: Created topic '%s' for subscriber"),
        *TopicName);
  }

  FTeleopTopicData& TopicData = Topics[TopicName];

  FTeleopSubscriberEntry Entry;
  Entry.Type = ETeleopMessageType::ByteArray;
  Entry.ByteArrayCallback = MoveTemp(Callback);

  const int32 SubscriberIndex = TopicData.Subscribers.Num();
  TopicData.Subscribers.Add(MoveTemp(Entry));

  const int32 HandleId = NextHandleId++;
  SubscriberHandleToTopic.Add(HandleId, TopicName);
  SubscriberHandleToIndex.Add(HandleId, SubscriberIndex);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopDirectSubsystem: Registered ByteArray subscriber %d for topic '%s'"),
      HandleId,
      *TopicName);

  return HandleId;
}

void UTeleopDirectSubsystem::UnregisterSubscriber(int32 SubscriberId, const FString& TopicName) {
  FWriteScopeLock Lock(TopicLock);

  if (SubscriberHandleToTopic.Contains(SubscriberId) &&
      SubscriberHandleToIndex.Contains(SubscriberId)) {
    const int32 Index = SubscriberHandleToIndex[SubscriberId];
    if (Topics.Contains(TopicName)) {
      FTeleopTopicData& TopicData = Topics[TopicName];
      if (Index >= 0 && Index < TopicData.Subscribers.Num()) {
        TopicData.Subscribers[Index].DoubleArrayCallback = nullptr;
        TopicData.Subscribers[Index].TransformArrayCallback = nullptr;
        TopicData.Subscribers[Index].StringCallback = nullptr;
        TopicData.Subscribers[Index].ByteArrayCallback = nullptr;
      }
    }
    SubscriberHandleToTopic.Remove(SubscriberId);
    SubscriberHandleToIndex.Remove(SubscriberId);

    UE_LOG(
        LogSuperDexTeleop,
        Verbose,
        TEXT("UTeleopDirectSubsystem: Unregistered subscriber %d from topic '%s'"),
        SubscriberId,
        *TopicName);
  }
}

//------------------------------------------------------------------------------
// Publishing (typed)
//------------------------------------------------------------------------------

void UTeleopDirectSubsystem::Publish_DoubleArray(
    int32 PublisherId,
    const FString& TopicName,
    const TArray<double>& Data) {
  TArray<FTeleopSubscriberEntry> SubscribersToCall;

  {
    FWriteScopeLock Lock(TopicLock);

    if (!PublisherHandleToTopic.Contains(PublisherId)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Invalid publisher ID %d"),
          PublisherId);
      return;
    }

    if (!Topics.Contains(TopicName)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Topic '%s' does not exist"),
          *TopicName);
      return;
    }

    FTeleopTopicData& TopicData = Topics[TopicName];
    TopicData.DoubleArrayData = Data;
    TopicData.LastUpdated = FDateTime::UtcNow();
    TopicData.bDirty = true;

    if (bInstantDelivery) {
      SubscribersToCall = TopicData.Subscribers;
      TopicData.bDirty = false;
    }
  }

  if (bInstantDelivery) {
    for (const FTeleopSubscriberEntry& Entry : SubscribersToCall) {
      if (Entry.DoubleArrayCallback) {
        Entry.DoubleArrayCallback(Data);
      }
    }
  }
}

void UTeleopDirectSubsystem::Publish_TransformArray(
    int32 PublisherId,
    const FString& TopicName,
    const TArray<FTransform>& Data) {
  TArray<FTeleopSubscriberEntry> SubscribersToCall;

  {
    FWriteScopeLock Lock(TopicLock);

    if (!PublisherHandleToTopic.Contains(PublisherId)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Invalid publisher ID %d"),
          PublisherId);
      return;
    }

    if (!Topics.Contains(TopicName)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Topic '%s' does not exist"),
          *TopicName);
      return;
    }

    FTeleopTopicData& TopicData = Topics[TopicName];
    TopicData.TransformArrayData = Data;
    TopicData.LastUpdated = FDateTime::UtcNow();
    TopicData.bDirty = true;

    if (bInstantDelivery) {
      SubscribersToCall = TopicData.Subscribers;
      TopicData.bDirty = false;
    }
  }

  if (bInstantDelivery) {
    for (const FTeleopSubscriberEntry& Entry : SubscribersToCall) {
      if (Entry.TransformArrayCallback) {
        Entry.TransformArrayCallback(Data);
      }
    }
  }
}

void UTeleopDirectSubsystem::Publish_String(
    int32 PublisherId,
    const FString& TopicName,
    const FString& Data) {
  TArray<FTeleopSubscriberEntry> SubscribersToCall;

  {
    FWriteScopeLock Lock(TopicLock);

    if (!PublisherHandleToTopic.Contains(PublisherId)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Invalid publisher ID %d"),
          PublisherId);
      return;
    }

    if (!Topics.Contains(TopicName)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Topic '%s' does not exist"),
          *TopicName);
      return;
    }

    FTeleopTopicData& TopicData = Topics[TopicName];
    TopicData.StringData = Data;
    TopicData.LastUpdated = FDateTime::UtcNow();
    TopicData.bDirty = true;

    if (bInstantDelivery) {
      SubscribersToCall = TopicData.Subscribers;
      TopicData.bDirty = false;
    }
  }

  if (bInstantDelivery) {
    for (const FTeleopSubscriberEntry& Entry : SubscribersToCall) {
      if (Entry.StringCallback) {
        Entry.StringCallback(Data);
      }
    }
  }
}

void UTeleopDirectSubsystem::Publish_ByteArray(
    int32 PublisherId,
    const FString& TopicName,
    const TArray<uint8>& Data) {
  TArray<FTeleopSubscriberEntry> SubscribersToCall;

  {
    FWriteScopeLock Lock(TopicLock);

    if (!PublisherHandleToTopic.Contains(PublisherId)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Invalid publisher ID %d"),
          PublisherId);
      return;
    }

    if (!Topics.Contains(TopicName)) {
      UE_LOG(
          LogSuperDexTeleop,
          Warning,
          TEXT("UTeleopDirectSubsystem: Topic '%s' does not exist"),
          *TopicName);
      return;
    }

    FTeleopTopicData& TopicData = Topics[TopicName];
    TopicData.ByteArrayData = Data;
    TopicData.LastUpdated = FDateTime::UtcNow();
    TopicData.bDirty = true;

    if (bInstantDelivery) {
      SubscribersToCall = TopicData.Subscribers;
      TopicData.bDirty = false;
    }
  }

  if (bInstantDelivery) {
    for (const FTeleopSubscriberEntry& Entry : SubscribersToCall) {
      if (Entry.ByteArrayCallback) {
        Entry.ByteArrayCallback(Data);
      }
    }
  }
}

void UTeleopDirectSubsystem::PumpSubscribers() {
  // Collect dirty topics and their data
  struct FTopicDelivery {
    ETeleopMessageType Type;
    TArray<FTeleopSubscriberEntry> Subscribers;
    TArray<double> DoubleArrayData;
    TArray<FTransform> TransformArrayData;
    FString StringData;
    TArray<uint8> ByteArrayData;
  };

  TArray<FTopicDelivery> ToDeliver;

  {
    FWriteScopeLock Lock(TopicLock);
    for (auto& Pair : Topics) {
      FTeleopTopicData& TopicData = Pair.Value;
      if (TopicData.bDirty) {
        FTopicDelivery Delivery;
        Delivery.Type = TopicData.Type;
        Delivery.Subscribers = TopicData.Subscribers;
        switch (TopicData.Type) {
          case ETeleopMessageType::DoubleArray:
            Delivery.DoubleArrayData = TopicData.DoubleArrayData;
            break;
          case ETeleopMessageType::TransformArray:
            Delivery.TransformArrayData = TopicData.TransformArrayData;
            break;
          case ETeleopMessageType::String:
            Delivery.StringData = TopicData.StringData;
            break;
          case ETeleopMessageType::ByteArray:
            Delivery.ByteArrayData = TopicData.ByteArrayData;
            break;
        }
        ToDeliver.Add(MoveTemp(Delivery));
        TopicData.bDirty = false;
      }
    }
  }

  // Deliver outside the lock
  for (const FTopicDelivery& Delivery : ToDeliver) {
    for (const FTeleopSubscriberEntry& Entry : Delivery.Subscribers) {
      if (Entry.Type != Delivery.Type) {
        continue;
      }
      switch (Delivery.Type) {
        case ETeleopMessageType::DoubleArray:
          if (Entry.DoubleArrayCallback) {
            Entry.DoubleArrayCallback(Delivery.DoubleArrayData);
          }
          break;
        case ETeleopMessageType::TransformArray:
          if (Entry.TransformArrayCallback) {
            Entry.TransformArrayCallback(Delivery.TransformArrayData);
          }
          break;
        case ETeleopMessageType::String:
          if (Entry.StringCallback) {
            Entry.StringCallback(Delivery.StringData);
          }
          break;
        case ETeleopMessageType::ByteArray:
          if (Entry.ByteArrayCallback) {
            Entry.ByteArrayCallback(Delivery.ByteArrayData);
          }
          break;
      }
    }
  }
}
