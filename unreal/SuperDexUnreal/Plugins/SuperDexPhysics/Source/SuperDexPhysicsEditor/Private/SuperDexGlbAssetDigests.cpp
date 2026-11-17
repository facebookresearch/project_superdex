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
constexpr int64 kGlbHeaderSize = 12;
constexpr int64 kChunkHeaderSize = 8;
constexpr int64 kChunkTypeOffset = 4;
constexpr int64 kJsonDataOffset = kGlbHeaderSize + kChunkHeaderSize;
constexpr uint32 kGlbMagic = 0x46546C67; // "glTF"
constexpr uint32 kJsonChunk = 0x4E4F534A; // "JSON"
constexpr uint32 kBinChunk = 0x004E4942; // "BIN\0"

FString Md5(const FString& Text) {
  const FTCHARToUTF8 utf8(*Text);
  return FMD5::HashBytes(reinterpret_cast<const uint8*>(utf8.Get()), utf8.Length());
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
      TArray<TPair<FString, TSharedPtr<FJsonValue>>> fields;
      for (const auto& field : Value->AsObject()->Values) {
        fields.Emplace(*field.Key, field.Value);
      }
      fields.Sort([](const auto& A, const auto& B) { return A.Key < B.Key; });
      FString text = TEXT("{");
      for (const TPair<FString, TSharedPtr<FJsonValue>>& field : fields) {
        const FString& key = field.Key;
        const TSharedPtr<FJsonValue>& child = field.Value;
        FString childText;
        if (bTextureReference && key == TEXT("index") && child->Type == EJson::Number) {
          const int32 index = static_cast<int32>(child->AsNumber());
          childText = TextureDigests.IsValidIndex(index) ? TextureDigests[index] : TEXT("?");
        } else {
          childText = CanonicalJson(
              child, TextureDigests, key.EndsWith(TEXT("Texture"), ESearchCase::IgnoreCase));
        }
        text += FString::Printf(TEXT("\"%s\":%s,"), *key, *childText);
      }
      return text + TEXT("}");
    }
    case EJson::Array: {
      FString text = TEXT("[");
      for (const TSharedPtr<FJsonValue>& child : Value->AsArray()) {
        text += CanonicalJson(child, TextureDigests) + TEXT(",");
      }
      return text + TEXT("]");
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
  TArray<uint8> bytes;
  if (!FFileHelper::LoadFileToArray(bytes, *GlbPath, FILEREAD_Silent) ||
      bytes.Num() < kJsonDataOffset) {
    return false;
  }
  auto readU32 = [&bytes](int64 Offset) {
    return static_cast<uint32>(bytes[Offset]) | static_cast<uint32>(bytes[Offset + 1]) << 8 |
        static_cast<uint32>(bytes[Offset + 2]) << 16 | static_cast<uint32>(bytes[Offset + 3]) << 24;
  };
  const int64 jsonLength = readU32(kGlbHeaderSize);
  if (readU32(0) != kGlbMagic || readU32(kGlbHeaderSize + kChunkTypeOffset) != kJsonChunk ||
      kJsonDataOffset + jsonLength > bytes.Num()) {
    return false;
  }
  int64 binOffset = -1;
  int64 binLength = 0;
  const int64 binHeader = kJsonDataOffset + jsonLength;
  if (binHeader + kChunkHeaderSize <= bytes.Num() &&
      readU32(binHeader + kChunkTypeOffset) == kBinChunk) {
    binOffset = binHeader + kChunkHeaderSize;
    binLength = FMath::Min<int64>(readU32(binHeader), bytes.Num() - binOffset);
  }

  const FUTF8ToTCHAR jsonChars(
      reinterpret_cast<const ANSICHAR*>(bytes.GetData() + kJsonDataOffset),
      static_cast<int32>(jsonLength));
  TSharedPtr<FJsonObject> json;
  if (!FJsonSerializer::Deserialize(
          TJsonReaderFactory<>::Create(FString(jsonChars.Length(), jsonChars.Get())), json) ||
      !json.IsValid()) {
    return false;
  }
  auto getArray = [](const TSharedPtr<FJsonObject>& Object, const TCHAR* Field) {
    const TArray<TSharedPtr<FJsonValue>>* array = nullptr;
    return Object->TryGetArrayField(Field, array) ? *array : TArray<TSharedPtr<FJsonValue>>();
  };
  const TArray<TSharedPtr<FJsonValue>> bufferViews = getArray(json, TEXT("bufferViews"));

  TArray<FString> imageDigests;
  for (const TSharedPtr<FJsonValue>& imageValue : getArray(json, TEXT("images"))) {
    const TSharedPtr<FJsonObject>& image = imageValue->AsObject();
    FString digest;
    int32 bufferView = -1;
    FString uri;
    if (image->TryGetNumberField(TEXT("bufferView"), bufferView) &&
        bufferViews.IsValidIndex(bufferView)) {
      const TSharedPtr<FJsonObject>& view = bufferViews[bufferView]->AsObject();
      int64 offset = 0;
      int64 length = 0;
      view->TryGetNumberField(TEXT("byteOffset"), offset);
      view->TryGetNumberField(TEXT("byteLength"), length);
      if (binOffset >= 0 && offset >= 0 && length >= 0 && offset + length <= binLength) {
        digest = FMD5::HashBytes(bytes.GetData() + binOffset + offset, length);
      }
    } else if (image->TryGetStringField(TEXT("uri"), uri)) {
      TArray<uint8> file;
      const FString path = FPaths::Combine(FPaths::GetPath(GlbPath), uri);
      digest = !uri.StartsWith(TEXT("data:")) &&
              FFileHelper::LoadFileToArray(file, *path, FILEREAD_Silent)
          ? FMD5::HashBytes(file.GetData(), file.Num())
          : Md5(uri);
    }
    imageDigests.Add(digest);
    FString name;
    if (image->TryGetStringField(TEXT("name"), name) && !name.IsEmpty() && !digest.IsEmpty()) {
      OutDigests.Add(AssetName(name), digest);
    }
  }

  TArray<FString> textureDigests;
  for (const TSharedPtr<FJsonValue>& textureValue : getArray(json, TEXT("textures"))) {
    int32 source = -1;
    textureValue->AsObject()->TryGetNumberField(TEXT("source"), source);
    textureDigests.Add(imageDigests.IsValidIndex(source) ? imageDigests[source] : FString());
  }

  for (const TSharedPtr<FJsonValue>& materialValue : getArray(json, TEXT("materials"))) {
    FString name;
    if (materialValue->AsObject()->TryGetStringField(TEXT("name"), name) && !name.IsEmpty()) {
      OutDigests.Add(AssetName(name), Md5(CanonicalJson(materialValue, textureDigests)));
    }
  }
  return true;
}

} // namespace SuperDexGlbAssetDigests
