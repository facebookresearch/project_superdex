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

#include "SuperDexGlbAssetDigests.h"

#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/SecureHash.h"
#include "ObjectTools.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace {

// GLB layout: a 12-byte header (magic, version, length), then chunks of an 8-byte header (data
// length, type) and their data: JSON first, then an optional binary buffer.
constexpr int64 GlbHeaderSize = 12;
constexpr int64 ChunkHeaderSize = 8;
constexpr int64 ChunkTypeOffset = 4;
constexpr int64 JsonDataOffset = GlbHeaderSize + ChunkHeaderSize;
constexpr uint32 GlbMagic = 0x46546C67; // "glTF"
constexpr uint32 JsonChunk = 0x4E4F534A; // "JSON"
constexpr uint32 BinChunk = 0x004E4942; // "BIN\0"

FString Md5(const FString& Text) {
  const FTCHARToUTF8 Utf8(*Text);
  return FMD5::HashBytes(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
}

// Key order-independent text of a glTF JSON value, with the index of every texture reference (the
// {"index": i, ...} object of a key like "baseColorTexture") replaced by its image's digest, so
// equal materials compare equal across GLBs whatever their texture order.
FString CanonicalJson(
    const TSharedPtr<FJsonValue>& Value,
    const TArray<FString>& TextureDigests,
    bool bTextureReference = false) {
  if (!Value.IsValid()) {
    return TEXT("null");
  }
  switch (Value->Type) {
    case EJson::Object: {
      // The fields, sorted by key. FJsonObject's key type differs between engine versions, so the
      // keys are copied out rather than read through it.
      TArray<TPair<FString, TSharedPtr<FJsonValue>>> Fields;
      for (const auto& Field : Value->AsObject()->Values) {
        Fields.Emplace(*Field.Key, Field.Value);
      }
      Fields.Sort([](const auto& A, const auto& B) { return A.Key < B.Key; });
      FString Text = TEXT("{");
      for (const TPair<FString, TSharedPtr<FJsonValue>>& Field : Fields) {
        const FString& Key = Field.Key;
        const TSharedPtr<FJsonValue>& Child = Field.Value;
        FString ChildText;
        if (bTextureReference && Key == TEXT("index") && Child->Type == EJson::Number) {
          const int32 Index = static_cast<int32>(Child->AsNumber());
          ChildText = TextureDigests.IsValidIndex(Index) ? TextureDigests[Index] : TEXT("?");
        } else {
          ChildText = CanonicalJson(
              Child, TextureDigests, Key.EndsWith(TEXT("Texture"), ESearchCase::IgnoreCase));
        }
        Text += FString::Printf(TEXT("\"%s\":%s,"), *Key, *ChildText);
      }
      return Text + TEXT("}");
    }
    case EJson::Array: {
      FString Text = TEXT("[");
      for (const TSharedPtr<FJsonValue>& Child : Value->AsArray()) {
        Text += CanonicalJson(Child, TextureDigests) + TEXT(",");
      }
      return Text + TEXT("]");
    }
    case EJson::String:
      return TEXT("\"") + Value->AsString() + TEXT("\"");
    case EJson::Number:
      return FString::Printf(TEXT("%.9g"), Value->AsNumber());
    case EJson::Boolean:
      return Value->AsBool() ? TEXT("true") : TEXT("false");
    default:
      return TEXT("null");
  }
}

} // namespace

namespace SuperDexGlbAssetDigests {

FString AssetName(const FString& GltfName) {
  return ObjectTools::SanitizeObjectName(GltfName);
}

FString AssetObjectPath(const FString& PackagePath, const FString& AssetName) {
  return FPaths::Combine(PackagePath, AssetName + TEXT(".") + AssetName);
}

bool Read(const FString& GlbPath, TMap<FString, FString>& OutDigests) {
  TArray<uint8> Bytes;
  if (!FFileHelper::LoadFileToArray(Bytes, *GlbPath, FILEREAD_Silent) ||
      Bytes.Num() < JsonDataOffset) {
    return false;
  }
  auto ReadU32 = [&Bytes](int64 Offset) {
    return static_cast<uint32>(Bytes[Offset]) | static_cast<uint32>(Bytes[Offset + 1]) << 8 |
        static_cast<uint32>(Bytes[Offset + 2]) << 16 | static_cast<uint32>(Bytes[Offset + 3]) << 24;
  };
  const int64 JsonLength = ReadU32(GlbHeaderSize);
  if (ReadU32(0) != GlbMagic || ReadU32(GlbHeaderSize + ChunkTypeOffset) != JsonChunk ||
      JsonDataOffset + JsonLength > Bytes.Num()) {
    return false;
  }
  int64 BinOffset = -1;
  int64 BinLength = 0;
  const int64 BinHeader = JsonDataOffset + JsonLength;
  if (BinHeader + ChunkHeaderSize <= Bytes.Num() &&
      ReadU32(BinHeader + ChunkTypeOffset) == BinChunk) {
    BinOffset = BinHeader + ChunkHeaderSize;
    BinLength = FMath::Min<int64>(ReadU32(BinHeader), Bytes.Num() - BinOffset);
  }

  const FUTF8ToTCHAR JsonChars(
      reinterpret_cast<const ANSICHAR*>(Bytes.GetData() + JsonDataOffset),
      static_cast<int32>(JsonLength));
  TSharedPtr<FJsonObject> Json;
  if (!FJsonSerializer::Deserialize(
          TJsonReaderFactory<>::Create(FString(JsonChars.Length(), JsonChars.Get())), Json) ||
      !Json.IsValid()) {
    return false;
  }
  auto GetArray = [](const TSharedPtr<FJsonObject>& Object, const TCHAR* Field) {
    const TArray<TSharedPtr<FJsonValue>>* Array = nullptr;
    return Object->TryGetArrayField(Field, Array) ? *Array : TArray<TSharedPtr<FJsonValue>>();
  };
  const TArray<TSharedPtr<FJsonValue>> BufferViews = GetArray(Json, TEXT("bufferViews"));

  TArray<FString> ImageDigests;
  for (const TSharedPtr<FJsonValue>& ImageValue : GetArray(Json, TEXT("images"))) {
    const TSharedPtr<FJsonObject>& Image = ImageValue->AsObject();
    FString Digest;
    int32 BufferView = -1;
    FString Uri;
    if (Image->TryGetNumberField(TEXT("bufferView"), BufferView) &&
        BufferViews.IsValidIndex(BufferView)) {
      const TSharedPtr<FJsonObject>& View = BufferViews[BufferView]->AsObject();
      int64 Offset = 0;
      int64 Length = 0;
      View->TryGetNumberField(TEXT("byteOffset"), Offset);
      View->TryGetNumberField(TEXT("byteLength"), Length);
      if (BinOffset >= 0 && Offset >= 0 && Length >= 0 && Offset + Length <= BinLength) {
        Digest = FMD5::HashBytes(Bytes.GetData() + BinOffset + Offset, Length);
      }
    } else if (Image->TryGetStringField(TEXT("uri"), Uri)) {
      TArray<uint8> File;
      const FString Path = FPaths::Combine(FPaths::GetPath(GlbPath), Uri);
      Digest = !Uri.StartsWith(TEXT("data:")) &&
              FFileHelper::LoadFileToArray(File, *Path, FILEREAD_Silent)
          ? FMD5::HashBytes(File.GetData(), File.Num())
          : Md5(Uri);
    }
    ImageDigests.Add(Digest);
    FString Name;
    if (Image->TryGetStringField(TEXT("name"), Name) && !Name.IsEmpty() && !Digest.IsEmpty()) {
      OutDigests.Add(AssetName(Name), Digest);
    }
  }

  TArray<FString> TextureDigests;
  for (const TSharedPtr<FJsonValue>& TextureValue : GetArray(Json, TEXT("textures"))) {
    int32 Source = -1;
    TextureValue->AsObject()->TryGetNumberField(TEXT("source"), Source);
    TextureDigests.Add(ImageDigests.IsValidIndex(Source) ? ImageDigests[Source] : FString());
  }

  for (const TSharedPtr<FJsonValue>& MaterialValue : GetArray(Json, TEXT("materials"))) {
    FString Name;
    if (MaterialValue->AsObject()->TryGetStringField(TEXT("name"), Name) && !Name.IsEmpty()) {
      OutDigests.Add(AssetName(Name), Md5(CanonicalJson(MaterialValue, TextureDigests)));
    }
  }
  return true;
}

} // namespace SuperDexGlbAssetDigests
