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

#include "TeleopNodeImplZenoh.h"

#include "Misc/DateTime.h"
#include "TeleopNodeZenohModule.h"
#include "TeleopNodeZenohSettings.h"
#include "ZenohSession.h"

THIRD_PARTY_INCLUDES_START
#include "teleop_codec.h"
THIRD_PARTY_INCLUDES_END

DEFINE_LOG_CATEGORY_STATIC(LogTeleopZenoh, Log, All);

namespace {

constexpr double kCentiToUnit = 0.01; // cm to m
constexpr double kUnitToCenti = 100.0; // m to cm

const TCHAR* const kStringEncoding = TEXT("text/plain;charset=utf-8");
const TCHAR* const kBytesEncoding = TEXT("application/octet-stream");

FString ProtobufEncoding(const char* TypeName) {
  return FString(TEXT("application/protobuf;")) + UTF8_TO_TCHAR(TypeName);
}

teleop_msgs::Header MakeHeader(uint32& Seq) {
  teleop_msgs::Header Header{};
  Header.stamp_ns =
      (FDateTime::UtcNow() - FDateTime(1970, 1, 1)).GetTicks() * ETimespan::NanosecondsPerTick;
  Header.seq = Seq++;
  return Header;
}

// FRU (Unreal) <-> FLU (x-forward, y-left, z-up)
teleop_msgs::Pose ToPose(const FTransform& In) {
  const FVector Loc = In.GetLocation();
  const FQuat Rot = In.GetRotation();
  return {
      Loc.X * kCentiToUnit,
      -Loc.Y * kCentiToUnit,
      Loc.Z * kCentiToUnit,
      Rot.X,
      -Rot.Y,
      Rot.Z,
      -Rot.W};
}

FTransform FromPose(const teleop_msgs::Pose& In) {
  const FVector Position(In.px * kUnitToCenti, -In.py * kUnitToCenti, In.pz * kUnitToCenti);
  // The peer's quaternion may be non-unit; GetNormalized() falls back to identity for a zero one.
  const FQuat Orientation = FQuat(In.qx, -In.qy, In.qz, -In.qw).GetNormalized();
  return FTransform(Orientation, Position);
}

// Encodes into Buffer, sized exactly: Encode(pb_ostream_t*) runs once to size, once to write.
template <typename TEncode>
bool EncodeInto(TArray<uint8>& Buffer, TEncode&& Encode) {
  pb_ostream_t Sizing = PB_OSTREAM_SIZING;
  if (!Encode(&Sizing)) {
    return false;
  }
  Buffer.SetNumUninitialized(static_cast<int32>(Sizing.bytes_written));
  pb_ostream_t Stream = pb_ostream_from_buffer(Buffer.GetData(), Buffer.Num());
  return Encode(&Stream);
}

} // namespace

//------------------------------------------------------------------------------
// LIFECYCLE
//------------------------------------------------------------------------------

bool UTeleopNodeImplZenoh::Initialize() {
  Session = FTeleopNodeZenohModule::Get().AcquireSession();
  if (!Session) {
    UE_LOG(
        LogTeleopZenoh,
        Error,
        TEXT("[%s] No zenoh session (see LogZenoh errors); node will not communicate."),
        *NodeName);
    return false;
  }
  UE_LOG(LogTeleopZenoh, Log, TEXT("[%s] Initialized"), *NodeName);
  return true;
}

void UTeleopNodeImplZenoh::Deinitialize() {
  Publishers.Empty();
  Subscribers.Empty();
  // The session closes once the last node lets go.
  Session.Reset();
}

void UTeleopNodeImplZenoh::Pump() {
  for (TPair<int32, FSubscriber>& Pair : Subscribers) {
    FSubscriber& Sub = Pair.Value;
    Sub.Subscriber->Drain([&Sub, this](TConstArrayView<uint8> Bytes) {
      if (!Sub.Dispatch(Bytes)) {
        UE_LOG(
            LogTeleopZenoh,
            Warning,
            TEXT("[%s] Could not decode %d bytes on %s"),
            *NodeName,
            Bytes.Num(),
            *Sub.Subscriber->GetKeyExpr());
      }
    });
  }
}

//------------------------------------------------------------------------------
// PUBLISHERS
//------------------------------------------------------------------------------

FTeleopPublisherHandle UTeleopNodeImplZenoh::AddPublisher(
    const FString& TopicName,
    const FString& Encoding) {
  FString Key;
  if (!Session || !ZenohTopicToKeyExpr(TopicName, Key)) {
    UE_LOG(
        LogTeleopZenoh,
        Error,
        TEXT("[%s] Cannot publish on %s (no session, or not a valid zenoh key)"),
        *NodeName,
        *TopicName);
    return FTeleopPublisherHandle();
  }
  TSharedPtr<FZenohPublisher> Publisher = Session->DeclarePublisher(
      Key, Encoding, GetDefault<UTeleopNodeZenohSettings>()->GetPublisherQos());
  if (!Publisher) {
    return FTeleopPublisherHandle();
  }
  const int32 HandleId = NextHandleId++;
  Publishers.Add(HandleId, FPublisher{MoveTemp(Publisher)});
  return FTeleopPublisherHandle(HandleId);
}

FTeleopPublisherHandle UTeleopNodeImplZenoh::CreatePublisher_DoubleArray(const FString& TopicName) {
  return AddPublisher(TopicName, ProtobufEncoding(teleop_msgs::kDoubleArrayTypeName));
}

FTeleopPublisherHandle UTeleopNodeImplZenoh::CreatePublisher_TransformArray(
    const FString& TopicName) {
  return AddPublisher(TopicName, ProtobufEncoding(teleop_msgs::kPoseArrayTypeName));
}

FTeleopPublisherHandle UTeleopNodeImplZenoh::CreatePublisher_String(const FString& TopicName) {
  return AddPublisher(TopicName, kStringEncoding);
}

FTeleopPublisherHandle UTeleopNodeImplZenoh::CreatePublisher_ByteArray(const FString& TopicName) {
  return AddPublisher(TopicName, kBytesEncoding);
}

UTeleopNodeImplZenoh::FPublisher* UTeleopNodeImplZenoh::FindPublisher(
    FTeleopPublisherHandle Handle,
    const TCHAR* Caller) {
  FPublisher* Pub = Handle.IsValid() ? Publishers.Find(Handle.Id) : nullptr;
  if (!Pub) {
    UE_LOG(
        LogTeleopZenoh,
        Warning,
        TEXT("[%s] %s called with invalid handle %d"),
        *NodeName,
        Caller,
        Handle.Id);
  }
  return Pub;
}

void UTeleopNodeImplZenoh::Publish_DoubleArray(
    FTeleopPublisherHandle Handle,
    const TArray<double>& Data) {
  FPublisher* Pub = FindPublisher(Handle, TEXT("Publish_DoubleArray"));
  if (!Pub) {
    return;
  }
  const teleop_msgs::Header Header = MakeHeader(Pub->Seq);
  if (EncodeInto(EncodeBuffer, [&](pb_ostream_t* Stream) {
        return teleop_msgs::EncodeDoubleArray(Stream, Header, Data.GetData(), Data.Num());
      })) {
    Pub->Publisher->Put(EncodeBuffer);
  }
}

void UTeleopNodeImplZenoh::Publish_TransformArray(
    FTeleopPublisherHandle Handle,
    const TArray<FTransform>& Data) {
  FPublisher* Pub = FindPublisher(Handle, TEXT("Publish_TransformArray"));
  if (!Pub) {
    return;
  }
  TArray<teleop_msgs::Pose, TInlineAllocator<4>> Poses;
  for (const FTransform& Transform : Data) {
    Poses.Add(ToPose(Transform));
  }
  const teleop_msgs::Header Header = MakeHeader(Pub->Seq);
  if (EncodeInto(EncodeBuffer, [&](pb_ostream_t* Stream) {
        return teleop_msgs::EncodePoseArray(Stream, Header, Poses.GetData(), Poses.Num());
      })) {
    Pub->Publisher->Put(EncodeBuffer);
  }
}

void UTeleopNodeImplZenoh::Publish_String(FTeleopPublisherHandle Handle, const FString& Data) {
  FPublisher* Pub = FindPublisher(Handle, TEXT("Publish_String"));
  if (!Pub) {
    return;
  }
  const FTCHARToUTF8 Utf8(*Data);
  Pub->Publisher->Put(
      TConstArrayView<uint8>(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length()));
}

void UTeleopNodeImplZenoh::Publish_ByteArray(
    FTeleopPublisherHandle Handle,
    const TArray<uint8>& Data) {
  FPublisher* Pub = FindPublisher(Handle, TEXT("Publish_ByteArray"));
  if (!Pub) {
    return;
  }
  Pub->Publisher->Put(Data);
}

//------------------------------------------------------------------------------
// SUBSCRIBERS
//------------------------------------------------------------------------------

FTeleopSubscriberHandle UTeleopNodeImplZenoh::AddSubscriber(
    const FString& TopicName,
    FDispatch Dispatch) {
  FString Key;
  if (!Session || !ZenohTopicToKeyExpr(TopicName, Key)) {
    UE_LOG(
        LogTeleopZenoh,
        Error,
        TEXT("[%s] Cannot subscribe to %s (no session, or not a valid zenoh key)"),
        *NodeName,
        *TopicName);
    return FTeleopSubscriberHandle();
  }
  TSharedPtr<FZenohSubscriber> Subscriber =
      Session->DeclareSubscriber(Key, GetDefault<UTeleopNodeZenohSettings>()->SubscriberQueueDepth);
  if (!Subscriber) {
    return FTeleopSubscriberHandle();
  }
  const int32 HandleId = NextHandleId++;
  Subscribers.Add(HandleId, FSubscriber{MoveTemp(Subscriber), MoveTemp(Dispatch)});
  return FTeleopSubscriberHandle(HandleId);
}

FTeleopSubscriberHandle UTeleopNodeImplZenoh::CreateSubscriber_DoubleArray(
    const FString& TopicName,
    FDoubleArrayCallback Callback) {
  return AddSubscriber(TopicName, [Callback = MoveTemp(Callback)](TConstArrayView<uint8> Bytes) {
    TArray<double> Values;
    if (!teleop_msgs::DecodeDoubleArray(
            Bytes.GetData(), static_cast<size_t>(Bytes.Num()), nullptr, [&Values](double Value) {
              Values.Add(Value);
            })) {
      return false;
    }
    Callback(Values);
    return true;
  });
}

FTeleopSubscriberHandle UTeleopNodeImplZenoh::CreateSubscriber_TransformArray(
    const FString& TopicName,
    FTransformArrayCallback Callback) {
  return AddSubscriber(TopicName, [Callback = MoveTemp(Callback)](TConstArrayView<uint8> Bytes) {
    TArray<FTransform> Transforms;
    if (!teleop_msgs::DecodePoseArray(
            Bytes.GetData(),
            static_cast<size_t>(Bytes.Num()),
            nullptr,
            [&Transforms](const teleop_msgs::Pose& Pose) { Transforms.Add(FromPose(Pose)); })) {
      return false;
    }
    Callback(Transforms);
    return true;
  });
}

FTeleopSubscriberHandle UTeleopNodeImplZenoh::CreateSubscriber_String(
    const FString& TopicName,
    FStringCallback Callback) {
  return AddSubscriber(TopicName, [Callback = MoveTemp(Callback)](TConstArrayView<uint8> Bytes) {
    const FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData()), Bytes.Num());
    Callback(FString(Text.Length(), Text.Get()));
    return true;
  });
}

FTeleopSubscriberHandle UTeleopNodeImplZenoh::CreateSubscriber_ByteArray(
    const FString& TopicName,
    FByteArrayCallback Callback) {
  return AddSubscriber(TopicName, [Callback = MoveTemp(Callback)](TConstArrayView<uint8> Bytes) {
    Callback(TArray<uint8>(Bytes.GetData(), Bytes.Num()));
    return true;
  });
}
