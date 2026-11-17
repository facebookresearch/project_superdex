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

#include "HAL/FileManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#if WITH_DEV_AUTOMATION_TESTS

#define SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

namespace {

// Two four-byte "images", in buffer views 0 and 1.
constexpr const TCHAR* kBufferViews = TEXT(
    R"([{"buffer":0,"byteOffset":0,"byteLength":4},{"buffer":0,"byteOffset":4,"byteLength":4}])");
constexpr const TCHAR* kImages = TEXT(
    R"([{"name":"Color","bufferView":0,"mimeType":"image/png"},)"
    R"({"name":"Normal","bufferView":1,"mimeType":"image/png"}])");
constexpr const TCHAR* kTextures = TEXT(R"([{"source":0},{"source":1}])");
constexpr const TCHAR* kMaterials = TEXT(
    R"([{"name":"Mat","pbrMetallicRoughness":{"baseColorTexture":{"index":0},)"
    R"("roughnessFactor":0.5},"normalTexture":{"index":1}}])");

void AppendUint32(TArray<uint8>& Out, uint32 Value) {
  Out.Append(reinterpret_cast<const uint8*>(&Value), sizeof(Value));
}

// Writes a GLB with the given top-level arrays and the two images' bytes to a file named Name,
// and returns its path.
FString WriteGlb(
    const TCHAR* Name,
    const TCHAR* Images = kImages,
    const TCHAR* Textures = kTextures,
    const TCHAR* Materials = kMaterials,
    const TCHAR* BufferViews = kBufferViews) {
  const FString json = FString::Printf(
      TEXT(
          R"({"asset":{"version":"2.0"},"images":%s,"textures":%s,"materials":%s,)"
          R"("bufferViews":%s,"buffers":[{"byteLength":8}]})"),
      Images,
      Textures,
      Materials,
      BufferViews);
  const FTCHARToUTF8 utf8(*json);
  TArray<uint8> jsonBytes(reinterpret_cast<const uint8*>(utf8.Get()), utf8.Length());
  while (jsonBytes.Num() % 4 != 0) {
    jsonBytes.Add(' ');
  }
  const TArray<uint8> bin = {'r', 'e', 'd', '!', 'b', 'l', 'u', 'e'};
  TArray<uint8> glb;
  AppendUint32(glb, 0x46546C67); // "glTF"
  AppendUint32(glb, 2);
  AppendUint32(glb, 12 + 8 + jsonBytes.Num() + 8 + bin.Num());
  AppendUint32(glb, jsonBytes.Num());
  AppendUint32(glb, 0x4E4F534A); // "JSON"
  glb.Append(jsonBytes);
  AppendUint32(glb, bin.Num());
  AppendUint32(glb, 0x004E4942); // "BIN"
  glb.Append(bin);

  const FString path =
      FPaths::Combine(FPaths::AutomationTransientDir(), TEXT("GlbAssetDigests"), Name) +
      TEXT(".glb");
  FFileHelper::SaveArrayToFile(glb, *path);
  return path;
}

// The digests of a test GLB, which must be readable. Deletes the GLB.
TMap<FString, FString> ReadGlb(FAutomationTestBase& Test, const FString& Path) {
  TMap<FString, FString> digests;
  Test.TestTrue(TEXT("Read"), SuperDexGlbAssetDigests::Read(Path, digests));
  IFileManager::Get().Delete(*Path);
  return digests;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsNamedEntriesTest,
    "SuperDexPhysics.GlbAssetDigests.DigestsNamedMaterialsAndImages",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsNamedEntriesTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> digests = ReadGlb(*this, WriteGlb(TEXT("Named")));
  TestEqual(TEXT("Entry count"), digests.Num(), 3);
  TestTrue(TEXT("Material"), digests.Contains(TEXT("Mat")));
  TestTrue(TEXT("Color image"), digests.Contains(TEXT("Color")));
  TestTrue(TEXT("Normal image"), digests.Contains(TEXT("Normal")));
  TestNotEqual(
      TEXT("Images differ"), digests.FindRef(TEXT("Color")), digests.FindRef(TEXT("Normal")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsOrderTest,
    "SuperDexPhysics.GlbAssetDigests.IgnoresKeyAndTextureOrder",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsOrderTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> digests = ReadGlb(*this, WriteGlb(TEXT("Ordered")));
  // The same material and images, with the images, textures and material keys listed in another
  // order.
  const TMap<FString, FString> reordered = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Reordered"),
          TEXT(
              R"([{"name":"Normal","bufferView":1,"mimeType":"image/png"},)"
              R"({"name":"Color","bufferView":0,"mimeType":"image/png"}])"),
          TEXT(R"([{"source":1},{"source":0}])"),
          TEXT(
              R"([{"normalTexture":{"index":1},"name":"Mat",)"
              R"("pbrMetallicRoughness":{"roughnessFactor":0.5,"baseColorTexture":{"index":0}}}])")));
  TestTrue(TEXT("Same digests"), digests.OrderIndependentCompareEqual(reordered));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsContentTest,
    "SuperDexPhysics.GlbAssetDigests.ChangesWithContent",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsContentTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> digests = ReadGlb(*this, WriteGlb(TEXT("Base")));
  const TMap<FString, FString> rougher = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Rougher"),
          kImages,
          kTextures,
          TEXT(
              R"([{"name":"Mat","pbrMetallicRoughness":{"baseColorTexture":{"index":0},)"
              R"("roughnessFactor":0.7},"normalTexture":{"index":1}}])")));
  TestNotEqual(
      TEXT("Material property"), rougher.FindRef(TEXT("Mat")), digests.FindRef(TEXT("Mat")));
  TestEqual(TEXT("Image"), rougher.FindRef(TEXT("Color")), digests.FindRef(TEXT("Color")));

  // The color image's bytes swapped for the normal image's.
  const TMap<FString, FString> recolored = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Recolored"),
          TEXT(
              R"([{"name":"Color","bufferView":1,"mimeType":"image/png"},)"
              R"({"name":"Normal","bufferView":1,"mimeType":"image/png"}])")));
  TestNotEqual(
      TEXT("Image bytes"), recolored.FindRef(TEXT("Color")), digests.FindRef(TEXT("Color")));
  TestNotEqual(
      TEXT("Material texture"), recolored.FindRef(TEXT("Mat")), digests.FindRef(TEXT("Mat")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsNamesTest,
    "SuperDexPhysics.GlbAssetDigests.KeysAreSanitizedAndSkipUnnamed",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsNamesTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> digests = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Names"),
          TEXT(
              R"([{"name":"Color.png","bufferView":0,"mimeType":"image/png"},)"
              R"({"bufferView":1,"mimeType":"image/png"},)"
              R"({"name":"Missing","bufferView":7,"mimeType":"image/png"}])"),
          kTextures,
          TEXT(R"([{"name":"mat.0"},{"pbrMetallicRoughness":{"roughnessFactor":0.2}}])")));
  TestEqual(TEXT("Entry count"), digests.Num(), 2);
  TestTrue(TEXT("Sanitized image name"), digests.Contains(TEXT("Color_png")));
  TestTrue(TEXT("Sanitized material name"), digests.Contains(TEXT("mat_0")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsUnreadableTest,
    "SuperDexPhysics.GlbAssetDigests.UnreadableFilesHaveNone",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsUnreadableTest::RunTest(const FString& Parameters) {
  const FString path = FPaths::Combine(
      FPaths::AutomationTransientDir(), TEXT("GlbAssetDigests"), TEXT("NotGlb.glb"));
  FFileHelper::SaveStringToFile(TEXT("not a binary glTF file"), *path);
  TMap<FString, FString> digests;
  TestFalse(TEXT("Not a GLB"), SuperDexGlbAssetDigests::Read(path, digests));
  IFileManager::Get().Delete(*path);
  TestFalse(TEXT("Missing file"), SuperDexGlbAssetDigests::Read(path, digests));
  TestEqual(TEXT("Entry count"), digests.Num(), 0);
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
