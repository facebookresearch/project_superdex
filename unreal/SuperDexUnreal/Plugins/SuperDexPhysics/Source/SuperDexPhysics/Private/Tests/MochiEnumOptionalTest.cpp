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

#include "MochiRodMeshComponent.h"
#include "MochiShellMeshComponent.h"

#include "Misc/AutomationTest.h"
#include "Serialization/MemoryReader.h"
#include "Serialization/MemoryWriter.h"
#include "Serialization/ObjectAndNameAsStringProxyArchive.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace {

template <typename Component, typename Element>
void CheckOptionalRoundTrip(
    FAutomationTestBase& Test,
    TOptional<Element> Component::* Field,
    Element Value) {
  Component* Source = NewObject<Component>();
  Test.TestFalse(TEXT("New collider selection is unset"), (Source->*Field).IsSet());
  for (bool bSet : {true, false}) {
    Source->*Field = bSet ? TOptional<Element>(Value) : TOptional<Element>();
    Component* Duplicate = DuplicateObject<Component>(Source, GetTransientPackage());
    Test.TestTrue(
        TEXT("Duplication preserves optional selection"), Duplicate->*Field == Source->*Field);

    TArray<uint8> Bytes;
    FMemoryWriter MemoryWriter(Bytes);
    FObjectAndNameAsStringProxyArchive Writer(MemoryWriter, false);
    static_cast<UObject*>(Source)->Serialize(Writer);
    Test.TestFalse(TEXT("Optional save succeeds"), Writer.IsError());

    Component* Loaded = NewObject<Component>();
    FMemoryReader MemoryReader(Bytes);
    FObjectAndNameAsStringProxyArchive Reader(MemoryReader, false);
    static_cast<UObject*>(Loaded)->Serialize(Reader);
    static_cast<UObject*>(Loaded)->PostLoad();
    Test.TestFalse(TEXT("Optional reload succeeds"), Reader.IsError());
    Test.TestTrue(
        TEXT("Fresh object reload preserves optional selection"), Loaded->*Field == Source->*Field);
  }
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiOptionalColliderSelectionTest,
    "SuperDexPhysics.Enums.OptionalColliderSelections",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)

bool FMochiOptionalColliderSelectionTest::RunTest(const FString& Parameters) {
  CheckOptionalRoundTrip(
      *this,
      &UMochiRodMeshComponent::PointCloudColliderSegmentElementType,
      EMochiActorSegmentElementType::P1Q2);
  CheckOptionalRoundTrip(
      *this,
      &UMochiShellMeshComponent::PointCloudColliderTriangleElementType,
      EMochiActorBoundaryElementType::P1Q6);
  return true;
}

#endif
