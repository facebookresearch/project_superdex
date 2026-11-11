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

#include "TeleopNodeImplDirect.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "SuperDexTeleopModule.h"
#include "TeleopDirectSubsystem.h"

UTeleopDirectSubsystem* UTeleopNodeImplDirect::GetSubsystem() {
  if (Subsystem.IsValid()) {
    return Subsystem.Get();
  }

  UWorld* World = GetWorld();
  if (!World) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::GetSubsystem - No world available"));
    return nullptr;
  }

  UGameInstance* GameInstance = World->GetGameInstance();
  if (!GameInstance) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::GetSubsystem - No game instance available"));
    return nullptr;
  }

  Subsystem = GameInstance->GetSubsystem<UTeleopDirectSubsystem>();
  if (!Subsystem.IsValid()) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::GetSubsystem - Could not get UTeleopDirectSubsystem"));
    return nullptr;
  }

  return Subsystem.Get();
}

bool UTeleopNodeImplDirect::Initialize() {
  UE_LOG(
      LogSuperDexTeleop, Display, TEXT("UTeleopNodeImplDirect::Initialize - Node '%s'"), *NodeName);

  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::Initialize - Failed to get subsystem"));
    return false;
  }

  return true;
}

void UTeleopNodeImplDirect::Deinitialize() {
  UE_LOG(
      LogSuperDexTeleop,
      Display,
      TEXT("UTeleopNodeImplDirect::Deinitialize - Node '%s'"),
      *NodeName);

  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (Sub) {
    for (const auto& Pair : PublisherTopics) {
      const int32 LocalHandle = Pair.Key;
      const FString& TopicName = Pair.Value;
      if (LocalToSubsystemPublisherHandle.Contains(LocalHandle)) {
        const int32 SubHandle = LocalToSubsystemPublisherHandle[LocalHandle];
        Sub->UnregisterPublisher(SubHandle, TopicName);
      }
    }

    for (const auto& Pair : SubscriberTopics) {
      const int32 LocalHandle = Pair.Key;
      const FString& TopicName = Pair.Value;
      if (LocalToSubsystemSubscriberHandle.Contains(LocalHandle)) {
        const int32 SubHandle = LocalToSubsystemSubscriberHandle[LocalHandle];
        Sub->UnregisterSubscriber(SubHandle, TopicName);
      }
    }
  }

  LocalToSubsystemPublisherHandle.Empty();
  LocalToSubsystemSubscriberHandle.Empty();
  PublisherTopics.Empty();
  SubscriberTopics.Empty();
  Subsystem.Reset();
}

void UTeleopNodeImplDirect::Pump() {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (Sub) {
    Sub->PumpSubscribers();
  }
}

//------------------------------------------------------------------------------
// Publisher creation
//------------------------------------------------------------------------------

FTeleopPublisherHandle UTeleopNodeImplDirect::CreatePublisher_DoubleArray(
    const FString& TopicName) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreatePublisher_DoubleArray - No subsystem"));
    return FTeleopPublisherHandle();
  }

  const int32 SubHandle = Sub->RegisterPublisher(TopicName, ETeleopMessageType::DoubleArray);

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemPublisherHandle.Add(LocalHandle, SubHandle);
  PublisherTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created DoubleArray publisher '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopPublisherHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

FTeleopPublisherHandle UTeleopNodeImplDirect::CreatePublisher_TransformArray(
    const FString& TopicName) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreatePublisher_TransformArray - No subsystem"));
    return FTeleopPublisherHandle();
  }

  const int32 SubHandle = Sub->RegisterPublisher(TopicName, ETeleopMessageType::TransformArray);

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemPublisherHandle.Add(LocalHandle, SubHandle);
  PublisherTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created TransformArray publisher '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopPublisherHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

FTeleopPublisherHandle UTeleopNodeImplDirect::CreatePublisher_String(const FString& TopicName) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreatePublisher_String - No subsystem"));
    return FTeleopPublisherHandle();
  }

  const int32 SubHandle = Sub->RegisterPublisher(TopicName, ETeleopMessageType::String);

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemPublisherHandle.Add(LocalHandle, SubHandle);
  PublisherTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created String publisher '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopPublisherHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

FTeleopPublisherHandle UTeleopNodeImplDirect::CreatePublisher_ByteArray(const FString& TopicName) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreatePublisher_ByteArray - No subsystem"));
    return FTeleopPublisherHandle();
  }

  const int32 SubHandle = Sub->RegisterPublisher(TopicName, ETeleopMessageType::ByteArray);

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemPublisherHandle.Add(LocalHandle, SubHandle);
  PublisherTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created ByteArray publisher '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopPublisherHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

//------------------------------------------------------------------------------
// Subscriber creation
//------------------------------------------------------------------------------

FTeleopSubscriberHandle UTeleopNodeImplDirect::CreateSubscriber_DoubleArray(
    const FString& TopicName,
    FDoubleArrayCallback Callback) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreateSubscriber_DoubleArray - No subsystem"));
    return FTeleopSubscriberHandle();
  }

  const int32 SubHandle = Sub->RegisterSubscriber_DoubleArray(TopicName, MoveTemp(Callback));

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemSubscriberHandle.Add(LocalHandle, SubHandle);
  SubscriberTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created DoubleArray subscriber '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopSubscriberHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

FTeleopSubscriberHandle UTeleopNodeImplDirect::CreateSubscriber_TransformArray(
    const FString& TopicName,
    FTransformArrayCallback Callback) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreateSubscriber_TransformArray - No subsystem"));
    return FTeleopSubscriberHandle();
  }

  const int32 SubHandle = Sub->RegisterSubscriber_TransformArray(TopicName, MoveTemp(Callback));

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemSubscriberHandle.Add(LocalHandle, SubHandle);
  SubscriberTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created TransformArray subscriber '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopSubscriberHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

FTeleopSubscriberHandle UTeleopNodeImplDirect::CreateSubscriber_String(
    const FString& TopicName,
    FStringCallback Callback) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreateSubscriber_String - No subsystem"));
    return FTeleopSubscriberHandle();
  }

  const int32 SubHandle = Sub->RegisterSubscriber_String(TopicName, MoveTemp(Callback));

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemSubscriberHandle.Add(LocalHandle, SubHandle);
  SubscriberTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created String subscriber '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopSubscriberHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

FTeleopSubscriberHandle UTeleopNodeImplDirect::CreateSubscriber_ByteArray(
    const FString& TopicName,
    FByteArrayCallback Callback) {
  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    UE_LOG(
        LogSuperDexTeleop,
        Error,
        TEXT("UTeleopNodeImplDirect::CreateSubscriber_ByteArray - No subsystem"));
    return FTeleopSubscriberHandle();
  }

  const int32 SubHandle = Sub->RegisterSubscriber_ByteArray(TopicName, MoveTemp(Callback));

  const int32 LocalHandle = NextLocalHandleId++;
  LocalToSubsystemSubscriberHandle.Add(LocalHandle, SubHandle);
  SubscriberTopics.Add(LocalHandle, TopicName);

  UE_LOG(
      LogSuperDexTeleop,
      Verbose,
      TEXT("UTeleopNodeImplDirect: Created ByteArray subscriber '%s' (local=%d, sub=%d)"),
      *TopicName,
      LocalHandle,
      SubHandle);

  FTeleopSubscriberHandle Handle;
  Handle.Id = LocalHandle;
  return Handle;
}

//------------------------------------------------------------------------------
// Publishing
//------------------------------------------------------------------------------

void UTeleopNodeImplDirect::Publish_DoubleArray(
    FTeleopPublisherHandle Handle,
    const TArray<double>& Data) {
  if (!Handle.IsValid()) {
    return;
  }

  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    return;
  }

  if (!LocalToSubsystemPublisherHandle.Contains(Handle.Id)) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::Publish_DoubleArray - Unknown handle %d"),
        Handle.Id);
    return;
  }

  const int32 SubHandle = LocalToSubsystemPublisherHandle[Handle.Id];
  const FString& TopicName = PublisherTopics[Handle.Id];

  Sub->Publish_DoubleArray(SubHandle, TopicName, Data);
}

void UTeleopNodeImplDirect::Publish_TransformArray(
    FTeleopPublisherHandle Handle,
    const TArray<FTransform>& Data) {
  if (!Handle.IsValid()) {
    return;
  }

  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    return;
  }

  if (!LocalToSubsystemPublisherHandle.Contains(Handle.Id)) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::Publish_TransformArray - Unknown handle %d"),
        Handle.Id);
    return;
  }

  const int32 SubHandle = LocalToSubsystemPublisherHandle[Handle.Id];
  const FString& TopicName = PublisherTopics[Handle.Id];

  Sub->Publish_TransformArray(SubHandle, TopicName, Data);
}

void UTeleopNodeImplDirect::Publish_String(FTeleopPublisherHandle Handle, const FString& Data) {
  if (!Handle.IsValid()) {
    return;
  }

  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    return;
  }

  if (!LocalToSubsystemPublisherHandle.Contains(Handle.Id)) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::Publish_String - Unknown handle %d"),
        Handle.Id);
    return;
  }

  const int32 SubHandle = LocalToSubsystemPublisherHandle[Handle.Id];
  const FString& TopicName = PublisherTopics[Handle.Id];

  Sub->Publish_String(SubHandle, TopicName, Data);
}

void UTeleopNodeImplDirect::Publish_ByteArray(
    FTeleopPublisherHandle Handle,
    const TArray<uint8>& Data) {
  if (!Handle.IsValid()) {
    return;
  }

  UTeleopDirectSubsystem* Sub = GetSubsystem();
  if (!Sub) {
    return;
  }

  if (!LocalToSubsystemPublisherHandle.Contains(Handle.Id)) {
    UE_LOG(
        LogSuperDexTeleop,
        Warning,
        TEXT("UTeleopNodeImplDirect::Publish_ByteArray - Unknown handle %d"),
        Handle.Id);
    return;
  }

  const int32 SubHandle = LocalToSubsystemPublisherHandle[Handle.Id];
  const FString& TopicName = PublisherTopics[Handle.Id];

  Sub->Publish_ByteArray(SubHandle, TopicName, Data);
}
