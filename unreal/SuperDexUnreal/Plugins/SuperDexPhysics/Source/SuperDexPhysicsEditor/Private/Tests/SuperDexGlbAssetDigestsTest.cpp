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
constexpr const TCHAR* DefaultBufferViews = TEXT(
    R"([{"buffer":0,"byteOffset":0,"byteLength":4},{"buffer":0,"byteOffset":4,"byteLength":4}])");
constexpr const TCHAR* DefaultImages = TEXT(
    R"([{"name":"Color","bufferView":0,"mimeType":"image/png"},)"
    R"({"name":"Normal","bufferView":1,"mimeType":"image/png"}])");
constexpr const TCHAR* DefaultTextures = TEXT(R"([{"source":0},{"source":1}])");
constexpr const TCHAR* DefaultMaterials = TEXT(
    R"([{"name":"Mat","pbrMetallicRoughness":{"baseColorTexture":{"index":0},)"
    R"("roughnessFactor":0.5},"normalTexture":{"index":1}}])");

void AppendUint32(TArray<uint8>& Out, uint32 Value) {
  Out.Append(reinterpret_cast<const uint8*>(&Value), sizeof(Value));
}

// Writes a GLB with the given top-level arrays and the two images' bytes to a file named Name,
// and returns its path.
FString WriteGlb(
    const TCHAR* Name,
    const TCHAR* Images = DefaultImages,
    const TCHAR* Textures = DefaultTextures,
    const TCHAR* Materials = DefaultMaterials,
    const TCHAR* BufferViews = DefaultBufferViews) {
  const FString Json = FString::Printf(
      TEXT(
          R"({"asset":{"version":"2.0"},"images":%s,"textures":%s,"materials":%s,)"
          R"("bufferViews":%s,"buffers":[{"byteLength":8}]})"),
      Images,
      Textures,
      Materials,
      BufferViews);
  const FTCHARToUTF8 Utf8(*Json);
  TArray<uint8> JsonBytes(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
  while (JsonBytes.Num() % 4 != 0) {
    JsonBytes.Add(' ');
  }
  const TArray<uint8> Bin = {'r', 'e', 'd', '!', 'b', 'l', 'u', 'e'};
  TArray<uint8> Glb;
  AppendUint32(Glb, 0x46546C67); // "glTF"
  AppendUint32(Glb, 2);
  AppendUint32(Glb, 12 + 8 + JsonBytes.Num() + 8 + Bin.Num());
  AppendUint32(Glb, JsonBytes.Num());
  AppendUint32(Glb, 0x4E4F534A); // "JSON"
  Glb.Append(JsonBytes);
  AppendUint32(Glb, Bin.Num());
  AppendUint32(Glb, 0x004E4942); // "BIN"
  Glb.Append(Bin);

  const FString Path =
      FPaths::Combine(FPaths::AutomationTransientDir(), TEXT("GlbAssetDigests"), Name) +
      TEXT(".glb");
  FFileHelper::SaveArrayToFile(Glb, *Path);
  return Path;
}

// The digests of a test GLB, which must be readable. Deletes the GLB.
TMap<FString, FString> ReadGlb(FAutomationTestBase& Test, const FString& Path) {
  TMap<FString, FString> Digests;
  Test.TestTrue(TEXT("Read"), SuperDexGlbAssetDigests::Read(Path, Digests));
  IFileManager::Get().Delete(*Path);
  return Digests;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsNamedEntriesTest,
    "SuperDexPhysics.GlbAssetDigests.DigestsNamedMaterialsAndImages",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsNamedEntriesTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> Digests = ReadGlb(*this, WriteGlb(TEXT("Named")));
  TestEqual(TEXT("Entry count"), Digests.Num(), 3);
  TestTrue(TEXT("Material"), Digests.Contains(TEXT("Mat")));
  TestTrue(TEXT("Color image"), Digests.Contains(TEXT("Color")));
  TestTrue(TEXT("Normal image"), Digests.Contains(TEXT("Normal")));
  TestNotEqual(
      TEXT("Images differ"), Digests.FindRef(TEXT("Color")), Digests.FindRef(TEXT("Normal")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsOrderTest,
    "SuperDexPhysics.GlbAssetDigests.IgnoresKeyAndTextureOrder",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsOrderTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> Digests = ReadGlb(*this, WriteGlb(TEXT("Ordered")));
  // The same material and images, with the images, textures and material keys listed in another
  // order.
  const TMap<FString, FString> Reordered = ReadGlb(
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
  TestTrue(TEXT("Same digests"), Digests.OrderIndependentCompareEqual(Reordered));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsContentTest,
    "SuperDexPhysics.GlbAssetDigests.ChangesWithContent",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsContentTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> Digests = ReadGlb(*this, WriteGlb(TEXT("Base")));
  const TMap<FString, FString> Rougher = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Rougher"),
          DefaultImages,
          DefaultTextures,
          TEXT(
              R"([{"name":"Mat","pbrMetallicRoughness":{"baseColorTexture":{"index":0},)"
              R"("roughnessFactor":0.7},"normalTexture":{"index":1}}])")));
  TestNotEqual(
      TEXT("Material property"), Rougher.FindRef(TEXT("Mat")), Digests.FindRef(TEXT("Mat")));
  TestEqual(TEXT("Image"), Rougher.FindRef(TEXT("Color")), Digests.FindRef(TEXT("Color")));

  // The color image's bytes swapped for the normal image's.
  const TMap<FString, FString> Recolored = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Recolored"),
          TEXT(
              R"([{"name":"Color","bufferView":1,"mimeType":"image/png"},)"
              R"({"name":"Normal","bufferView":1,"mimeType":"image/png"}])")));
  TestNotEqual(
      TEXT("Image bytes"), Recolored.FindRef(TEXT("Color")), Digests.FindRef(TEXT("Color")));
  TestNotEqual(
      TEXT("Material texture"), Recolored.FindRef(TEXT("Mat")), Digests.FindRef(TEXT("Mat")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsNamesTest,
    "SuperDexPhysics.GlbAssetDigests.KeysAreSanitizedAndSkipUnnamed",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsNamesTest::RunTest(const FString& Parameters) {
  const TMap<FString, FString> Digests = ReadGlb(
      *this,
      WriteGlb(
          TEXT("Names"),
          TEXT(
              R"([{"name":"Color.png","bufferView":0,"mimeType":"image/png"},)"
              R"({"bufferView":1,"mimeType":"image/png"},)"
              R"({"name":"Missing","bufferView":7,"mimeType":"image/png"}])"),
          DefaultTextures,
          TEXT(R"([{"name":"mat.0"},{"pbrMetallicRoughness":{"roughnessFactor":0.2}}])")));
  TestEqual(TEXT("Entry count"), Digests.Num(), 2);
  TestTrue(TEXT("Sanitized image name"), Digests.Contains(TEXT("Color_png")));
  TestTrue(TEXT("Sanitized material name"), Digests.Contains(TEXT("mat_0")));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexGlbAssetDigestsUnreadableTest,
    "SuperDexPhysics.GlbAssetDigests.UnreadableFilesHaveNone",
    SUPERDEX_GLB_ASSET_DIGESTS_TEST_FLAGS)

bool FSuperDexGlbAssetDigestsUnreadableTest::RunTest(const FString& Parameters) {
  const FString Path = FPaths::Combine(
      FPaths::AutomationTransientDir(), TEXT("GlbAssetDigests"), TEXT("NotGlb.glb"));
  FFileHelper::SaveStringToFile(TEXT("not a binary glTF file"), *Path);
  TMap<FString, FString> Digests;
  TestFalse(TEXT("Not a GLB"), SuperDexGlbAssetDigests::Read(Path, Digests));
  IFileManager::Get().Delete(*Path);
  TestFalse(TEXT("Missing file"), SuperDexGlbAssetDigests::Read(Path, Digests));
  TestEqual(TEXT("Entry count"), Digests.Num(), 0);
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
