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

#include "ZenohSession.h"

#include "Misc/ScopeLock.h"
#include "TeleopNodeZenohModule.h"
#include "ZenohPicoShim.h"

#include <atomic>
#include <string>
#include <vector>

namespace {

ZenohPicoShim_Priority ToShim(EZenohPriority Priority) {
  switch (Priority) {
    case EZenohPriority::RealTime:
      return ZenohPicoShim_Priority_RealTime;
    case EZenohPriority::InteractiveHigh:
      return ZenohPicoShim_Priority_InteractiveHigh;
    case EZenohPriority::InteractiveLow:
      return ZenohPicoShim_Priority_InteractiveLow;
    case EZenohPriority::DataHigh:
      return ZenohPicoShim_Priority_DataHigh;
    case EZenohPriority::Data:
      return ZenohPicoShim_Priority_Data;
    case EZenohPriority::DataLow:
      return ZenohPicoShim_Priority_DataLow;
    case EZenohPriority::Background:
      return ZenohPicoShim_Priority_Background;
  }
  return ZenohPicoShim_Priority_Data;
}

ZenohPicoShim_CongestionControl ToShim(EZenohCongestionControl CongestionControl) {
  switch (CongestionControl) {
    case EZenohCongestionControl::Drop:
      return ZenohPicoShim_CongestionControl_Drop;
    case EZenohCongestionControl::Block:
      return ZenohPicoShim_CongestionControl_Block;
  }
  return ZenohPicoShim_CongestionControl_Drop;
}

FString Describe(int Result) {
  return FString(UTF8_TO_TCHAR(ZenohPicoShim_ErrorDescription(Result)));
}

// Shared between zenoh's read thread (producer) and FZenohSubscriber::Drain (consumer).
struct FSampleQueue {
  explicit FSampleQueue(int32 InDepth) : Depth(FMath::Max(InDepth, 1)) {}

  const int32 Depth;
  std::atomic<bool> bActive{true};
  FCriticalSection Lock;
  TArray<TArray<uint8>> Samples;
};

using FSampleQueueRef = TSharedPtr<FSampleQueue, ESPMode::ThreadSafe>;

void OnSample(const ZenohPicoShim_Sample* Sample, void* Context) {
  FSampleQueue& Queue = **static_cast<FSampleQueueRef*>(Context);
  if (!Queue.bActive.load()) {
    return;
  }

  TArray<uint8> Bytes;
  Bytes.SetNumUninitialized(static_cast<int32>(ZenohPicoShim_SamplePayloadSize(Sample)));
  ZenohPicoShim_SampleReadPayload(Sample, Bytes.GetData());

  FScopeLock ScopeLock(&Queue.Lock);
  if (Queue.Samples.Num() >= Queue.Depth) {
    Queue.Samples.RemoveAt(0);
  }
  Queue.Samples.Add(MoveTemp(Bytes));
}

void DropSampleContext(void* Context) {
  delete static_cast<FSampleQueueRef*>(Context);
}

} // namespace

bool ZenohTopicToKeyExpr(const FString& Topic, FString& OutKeyExpr) {
  TArray<FString> Chunks;
  Topic.ParseIntoArray(Chunks, TEXT("/"), /*InCullEmpty=*/true);
  for (const FString& Chunk : Chunks) {
    int32 Index = INDEX_NONE;
    if (Chunk.FindChar(TEXT('*'), Index) || Chunk.FindChar(TEXT('$'), Index) ||
        Chunk.FindChar(TEXT('?'), Index) || Chunk.FindChar(TEXT('#'), Index)) {
      return false;
    }
  }
  OutKeyExpr = FString::Join(Chunks, TEXT("/"));
  return !Chunks.IsEmpty();
}

FString FZenohSessionConfig::Describe() const {
  return FString::Printf(
      TEXT("mode %s, connect [%s], listen '%s', scouting %s"),
      Mode == EZenohSessionMode::Peer ? TEXT("peer") : TEXT("client"),
      *FString::Join(ConnectEndpoints, TEXT(", ")),
      *ListenEndpoint,
      bMulticastScouting ? TEXT("on") : TEXT("off"));
}

//------------------------------------------------------------------------------
// SESSION
//------------------------------------------------------------------------------

struct FZenohSession::FImpl {
  ZenohPicoShim_Session* Session = nullptr;
};

FZenohSession::FZenohSession() : Impl(MakeUnique<FImpl>()) {}

FZenohSession::~FZenohSession() {
  // Publishers and subscribers hold a reference to the session, so all are undeclared by now.
  if (Impl->Session) {
    ZenohPicoShim_CloseSession(Impl->Session);
    UE_LOG(LogZenoh, Display, TEXT("Zenoh session closed"));
  }
}

TSharedPtr<FZenohSession> FZenohSession::Open(const FZenohSessionConfig& Config) {
  const bool bPeer = Config.Mode == EZenohSessionMode::Peer;
  // Listening only makes sense for peers; ignore a value left over from switching modes.
  const FString ListenEndpoint = bPeer ? Config.ListenEndpoint : FString();

  // zenoh-pico would fail to open without saying why.
  if (bPeer && Config.ConnectEndpoints.IsEmpty() && ListenEndpoint.IsEmpty()) {
    UE_LOG(
        LogZenoh,
        Error,
        TEXT(
            "Peer mode needs a listen endpoint (e.g. tcp/127.0.0.1:7447 for a local-only "
            "session) and/or connect endpoints"));
    return nullptr;
  }

  // The UTF-8 strings must outlive ZenohPicoShim_OpenSession. std::string keeps its buffer on the
  // heap, so the pointers stay valid when the vector grows.
  const std::string Listen(TCHAR_TO_UTF8(*ListenEndpoint));
  std::vector<std::string> ConnectStorage;
  for (const FString& Endpoint : Config.ConnectEndpoints) {
    ConnectStorage.emplace_back(TCHAR_TO_UTF8(*Endpoint));
  }
  std::vector<const char*> Connect;
  for (const std::string& Endpoint : ConnectStorage) {
    Connect.push_back(Endpoint.c_str());
  }

  ZenohPicoShim_SessionConfig Config_Shim{};
  Config_Shim.mode = bPeer ? ZenohPicoShim_Mode_Peer : ZenohPicoShim_Mode_Client;
  Config_Shim.connectEndpoints = Connect.data();
  Config_Shim.numConnectEndpoints = Connect.size();
  Config_Shim.listenEndpoint = Listen.c_str();
  Config_Shim.multicastScouting = Config.bMulticastScouting;

  TSharedPtr<FZenohSession> Session = MakeShareable(new FZenohSession());
  const int Result = ZenohPicoShim_OpenSession(&Config_Shim, &Session->Impl->Session);
  if (Result != 0) {
    UE_LOG(
        LogZenoh,
        Error,
        TEXT("Failed to open zenoh session (%s): %s (error %d)"),
        *Config.Describe(),
        *Describe(Result),
        Result);
    return nullptr;
  }
  UE_LOG(LogZenoh, Display, TEXT("Zenoh session open (%s)"), *Config.Describe());
  return Session;
}

//------------------------------------------------------------------------------
// PUBLISHER
//------------------------------------------------------------------------------

struct FZenohPublisher::FImpl {
  ZenohPicoShim_Publisher* Publisher = nullptr;
};

FZenohPublisher::FZenohPublisher(TSharedRef<FZenohSession> InSession, FString InKeyExpr)
    : Session(MoveTemp(InSession)), KeyExpr(MoveTemp(InKeyExpr)), Impl(MakeUnique<FImpl>()) {}

FZenohPublisher::~FZenohPublisher() {
  ZenohPicoShim_UndeclarePublisher(Impl->Publisher);
}

TSharedPtr<FZenohPublisher> FZenohSession::DeclarePublisher(
    const FString& KeyExpr,
    const FString& Encoding,
    const FZenohPublisherQos& Qos) {
  const FTCHARToUTF8 KeyUtf8(*KeyExpr);
  const FTCHARToUTF8 EncodingUtf8(*Encoding);
  ZenohPicoShim_PublisherOptions Options{};
  Options.encoding = EncodingUtf8.Get();
  Options.priority = ToShim(Qos.Priority);
  Options.congestionControl = ToShim(Qos.CongestionControl);
  Options.express = Qos.bExpress;

  TSharedPtr<FZenohPublisher> Publisher = MakeShareable(new FZenohPublisher(AsShared(), KeyExpr));
  const int Result = ZenohPicoShim_DeclarePublisher(
      Impl->Session, KeyUtf8.Get(), &Options, &Publisher->Impl->Publisher);
  if (Result != 0) {
    UE_LOG(
        LogZenoh,
        Error,
        TEXT("Failed to declare publisher on '%s' (encoding '%s'): %s (error %d)"),
        *KeyExpr,
        *Encoding,
        *Describe(Result),
        Result);
    return nullptr;
  }
  return Publisher;
}

bool FZenohPublisher::Put(TConstArrayView<uint8> Bytes) const {
  return ZenohPicoShim_Put(Impl->Publisher, Bytes.GetData(), static_cast<size_t>(Bytes.Num())) == 0;
}

//------------------------------------------------------------------------------
// SUBSCRIBER
//------------------------------------------------------------------------------

struct FZenohSubscriber::FImpl {
  ZenohPicoShim_Subscriber* Subscriber = nullptr;
  FSampleQueueRef Queue;
};

FZenohSubscriber::FZenohSubscriber(TSharedRef<FZenohSession> InSession, FString InKeyExpr)
    : Session(MoveTemp(InSession)), KeyExpr(MoveTemp(InKeyExpr)), Impl(MakeUnique<FImpl>()) {}

FZenohSubscriber::~FZenohSubscriber() {
  if (Impl->Queue) {
    Impl->Queue->bActive = false;
  }
  ZenohPicoShim_UndeclareSubscriber(Impl->Subscriber);
}

TSharedPtr<FZenohSubscriber> FZenohSession::DeclareSubscriber(
    const FString& KeyExpr,
    int32 QueueDepth) {
  const FTCHARToUTF8 KeyUtf8(*KeyExpr);
  TSharedPtr<FZenohSubscriber> Subscriber =
      MakeShareable(new FZenohSubscriber(AsShared(), KeyExpr));
  Subscriber->Impl->Queue = MakeShared<FSampleQueue, ESPMode::ThreadSafe>(QueueDepth);

  // The shim calls DropSampleContext exactly once, also when declaring fails.
  const int Result = ZenohPicoShim_DeclareSubscriber(
      Impl->Session,
      KeyUtf8.Get(),
      &OnSample,
      &DropSampleContext,
      new FSampleQueueRef(Subscriber->Impl->Queue),
      &Subscriber->Impl->Subscriber);
  if (Result != 0) {
    UE_LOG(
        LogZenoh,
        Error,
        TEXT("Failed to declare subscriber on '%s': %s (error %d)"),
        *KeyExpr,
        *Describe(Result),
        Result);
    return nullptr;
  }
  return Subscriber;
}

int32 FZenohSubscriber::Drain(TFunctionRef<void(TConstArrayView<uint8>)> Fn) {
  TArray<TArray<uint8>> Samples;
  {
    FScopeLock ScopeLock(&Impl->Queue->Lock);
    Samples = MoveTemp(Impl->Queue->Samples);
  }
  for (const TArray<uint8>& Sample : Samples) {
    Fn(Sample);
  }
  return Samples.Num();
}
