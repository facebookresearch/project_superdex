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

#include "mochi_physics_test_fixture.h"
#include "reflection_fixture.h"

#include <mochi_core/utils/defer.h>
#include <mochi_physics/src/generated/mochi_physics_reflection_declarations.generated.h>
#include <mochi_physics/src/mochi_reflection_adapters.h>
#include <simple_reflection/simple_reflection_console.h>

#include <cstdint>
#include <string>

using namespace mochi;
using GeneratedReflection = test::MochiSceneTestBase;

TEST_F(GeneratedReflection, InheritedBindingsDispatchVirtualOverrides) {
  reflection_fixture::VirtualDerived data;
  auto const& info = SReflect::GetTypeInfo<reflection_fixture::VirtualDerived>();
  auto const* method = info.FindMethod("Run");
  ASSERT_NE(nullptr, method);
  int result = 0;
  ASSERT_TRUE(method->Call(&data, &result, {}));
  EXPECT_EQ(3, result);

  auto const* property = info.FindField("value");
  ASSERT_NE(nullptr, property);
  int value = 4;
  ASSERT_TRUE(property->SetValue(&data, &value));
  property->GetValue(&data, &result);
  EXPECT_EQ(18, result);
}

TEST_F(GeneratedReflection, FieldConflictKeepsAccessorMethods) {
  reflection_fixture::FieldAndAccessors data;
  auto const& info = SReflect::GetTypeInfo<reflection_fixture::FieldAndAccessors>();
  auto const* field = info.FindField("value");
  auto const* get = info.FindMethod("GetValue");
  auto const* set = info.FindMethod("SetValue");
  ASSERT_NE(nullptr, field);
  ASSERT_NE(nullptr, get);
  ASSERT_NE(nullptr, set);
  int value = 7;
  ASSERT_TRUE(field->SetValue(&data, &value));
  int result = 0;
  ASSERT_TRUE(get->Call(&data, &result, {}));
  EXPECT_EQ(8, result);
  value = 3;
  void* args[] = {&value};
  ASSERT_TRUE(set->Call(&data, nullptr, args));
  field->GetValue(&data, &result);
  EXPECT_EQ(6, result);
}

TEST_F(GeneratedReflection, DerivedFieldKeepsInheritedProperty) {
  reflection_fixture::DerivedField data;
  auto const& info = SReflect::GetTypeInfo<reflection_fixture::DerivedField>();
  auto const* field = info.FindField("localValue");
  auto const* property = info.FindField("value");
  ASSERT_NE(nullptr, field);
  ASSERT_NE(nullptr, property);
  EXPECT_EQ(nullptr, info.FindMethod("GetValue"));
  EXPECT_EQ(nullptr, info.FindMethod("SetValue"));
  real value = 7_r;
  ASSERT_TRUE(field->SetValue(&data, &value));
  real result = 0_r;
  property->GetValue(&data, &result);
  EXPECT_EQ(1_r, result);
  value = 9_r;
  ASSERT_TRUE(property->SetValue(&data, &value));
  property->GetValue(&data, &result);
  EXPECT_EQ(9_r, result);
  field->GetValue(&data, &result);
  EXPECT_EQ(7_r, result);
}

TEST_F(GeneratedReflection, LiveObjectReference) {
  auto const* field = SReflect::GetTypeInfo<Context>().FindField("debug");
  ASSERT_NE(nullptr, field);
  void const* visited = nullptr;
  field->VisitValue(_mochiContext, [&](void const* value) { visited = value; });
  EXPECT_EQ(&_mochiContext->GetDebugServer(), visited);
  EXPECT_EQ(&SReflect::GetTypeInfo<DebugServer>(), field->_innerTypeInfo);
}

TEST_F(GeneratedReflection, MutableSpanUsesOwnedStorage) {
  DynamicArray<real> values = {1_r, 2_r};
  reflection::ReflectionArgument<Span<real>> argument(values, *_scene);
  argument.Get()[0] = 7_r;
  DynamicArray<real> const original = {1_r, 2_r};
  EXPECT_EQ(original, values);
  Span<real const> returned = argument.Get();
  values = reflection::ToReflectionValue(argument.Get());
  DynamicArray<real> const expected = {7_r, 2_r};
  EXPECT_EQ(expected, values);
  EXPECT_EQ(expected, reflection::ToReflectionValue(returned));
}

TEST_F(GeneratedReflection, ReturnedObjectPointersBecomeHandles) {
  auto const* create = SReflect::GetTypeInfo<Scene>().FindMethod("CreateRigidActor");
  ASSERT_NE(nullptr, create);
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  ActorHandle handle;
  void* args[] = {&params};
  ASSERT_TRUE(create->Call(_scene, &handle, args));
  ASSERT_TRUE(handle.IsValid());
  EXPECT_NE(nullptr, _scene->GetActor(handle));
}

TEST_F(GeneratedReflection, HandleArgumentsResolveToObjects) {
  auto const* valid = SReflect::GetTypeInfo<Context>().FindMethod("IsValidScene");
  ASSERT_NE(nullptr, valid);
  Scene* scene = _mochiContext->CreateScene("reflected scene");
  ASSERT_NE(nullptr, scene);
  SceneHandle handle = scene->GetHandle();
  bool isValid = false;
  void* args[] = {&handle};
  ASSERT_TRUE(valid->Call(_mochiContext, &isValid, args));
  EXPECT_TRUE(isValid);
  // A stale handle resolves to nullptr, which the native method receives.
  _mochiContext->DestroyScene(scene);
  isValid = true;
  ASSERT_TRUE(valid->Call(_mochiContext, &isValid, args));
  EXPECT_FALSE(isValid);
}

TEST_F(GeneratedReflection, PublicStructPropertiesMethodsAndInheritance) {
  reflection_fixture::HiddenSetterData hidden;
  auto const* inherited =
      SReflect::GetTypeInfo<reflection_fixture::HiddenSetterData>().FindField("value");
  ASSERT_NE(nullptr, inherited);
  real inheritedValue = 2.5_r;
  ASSERT_TRUE(inherited->SetValue(&hidden, &inheritedValue));
  EXPECT_EQ(inheritedValue, hidden.GetValue());

  reflection_fixture::Data data;
  auto const& info = SReflect::GetTypeInfo<reflection_fixture::Data>();
  EXPECT_EQ(nullptr, info.FindMethod("SetValue"));
  EXPECT_EQ(nullptr, info.FindMethod("GetValue"));
  auto const* property = info.FindField("value");
  ASSERT_NE(nullptr, property);
  auto const* description = property->GetAttribute<attribute::Description>();
  ASSERT_NE(nullptr, description);
  EXPECT_NE(std::string::npos, description->_description.find("nonnegative"));

  std::string error;
  ASSERT_TRUE(SReflect::ConsoleExecute(&data, "value 7", nullptr, &error)) << error;
  EXPECT_EQ(7_r, data.GetValue());
  EXPECT_FALSE(SReflect::ConsoleExecute(&data, "value -1", nullptr, &error));
  EXPECT_EQ(7_r, data.GetValue());
  ASSERT_TRUE(SReflect::ConsoleExecute(&data, "reference 9", nullptr, &error)) << error;
  EXPECT_EQ(9_r, data.GetValue());
  ASSERT_TRUE(SReflect::ConsoleExecute(&data, "label \"remote name\"", nullptr, &error)) << error;
  EXPECT_EQ("remote name", data.GetLabel());
  ASSERT_TRUE(SReflect::ConsoleExecute(&data, "ready true", nullptr, &error)) << error;
  EXPECT_TRUE(data.IsReady());

  DynamicArray<real> values = {2_r, 3_r};
  DynamicArray<real> result;
  auto const* scale = info.FindMethod("Scale");
  ASSERT_NE(nullptr, scale);
  void* args[] = {&values};
  ASSERT_TRUE(scale->Call(&data, &result, args));
  DynamicArray<real> const expected = {4_r, 6_r};
  EXPECT_EQ(expected, values);
  EXPECT_EQ(expected, result);

  auto const* read = info.FindMethod("Read");
  ASSERT_NE(nullptr, read);
  int selected = 0;
  ASSERT_TRUE(read->Call(&data, &selected, args));
  EXPECT_EQ(1, selected);

  auto const* factory = info.FindMethod("Default");
  ASSERT_NE(nullptr, factory);
  reflection_fixture::Data defaults;
  ASSERT_TRUE(factory->Call(&data, &defaults, {}));
  EXPECT_EQ(1_r, defaults.GetValue());
}

TEST_F(GeneratedReflection, PublicStructSnapshotOwnsBorrowedData) {
  reflection_fixture::Data data;
  auto const* field = SReflect::GetTypeInfo<reflection_fixture::Data>().FindField("data");
  ASSERT_NE(nullptr, field);
  auto const* type = dynamic_cast<SReflect::StructTypeInfo const*>(field->_innerTypeInfo);
  ASSERT_NE(nullptr, type);
  auto const* values = type->FindField("values");
  ASSERT_NE(nullptr, values);
  void* snapshot = type->New();
  ASSERT_NE(nullptr, snapshot);
  MOCHI_DEFER(type->Delete(snapshot));
  field->GetValue(&data, snapshot);
  DynamicArray<real> const replacement = {99_r};
  data.SetSamples(MakeConstSpan(replacement));
  DynamicArray<real> actual;
  values->GetValue(snapshot, &actual);
  DynamicArray<real> const expected = {1_r, 2_r};
  EXPECT_EQ(expected, actual);
}

TEST_F(GeneratedReflection, ViewsConvertToTheirOwners) {
  reflection_fixture::Data data;
  auto const& info = SReflect::GetTypeInfo<reflection_fixture::Data>();
  auto const* field = info.FindField("sampleView");
  ASSERT_NE(nullptr, field);
  EXPECT_EQ(&SReflect::GetTypeInfo<reflection_fixture::Samples>(), field->_innerTypeInfo);
  reflection_fixture::Samples samples;
  field->GetValue(&data, &samples);
  DynamicArray<real> const replacement = {99_r};
  data.SetSamples(MakeConstSpan(replacement));
  DynamicArray<real> const expected = {1_r, 2_r};
  EXPECT_EQ(expected, samples.values);

  auto const* count = info.FindMethod("CountSamples");
  ASSERT_NE(nullptr, count);
  int result = 0;
  void* args[] = {&samples};
  ASSERT_TRUE(count->Call(&data, &result, args));
  EXPECT_EQ(2, result);
}

TEST_F(GeneratedReflection, GetterErrorsReachTheConsole) {
  reflection_fixture::Data data;
  std::string error;
  EXPECT_FALSE(SReflect::ConsoleExecute(&data, "checked", nullptr, &error));
  EXPECT_NE(std::string::npos, error.find("Checked value is unavailable")) << error;

  auto entries = SReflect::ConsoleList(
      &data, SReflect::GetTypeInfo<reflection_fixture::Data>(), "", {}, &error);
  EXPECT_TRUE(error.empty()) << error;
  bool found = false;
  for (auto const& entry : entries) {
    if (entry._name == "checked") {
      found = true;
      EXPECT_FALSE(entry._valueError.empty());
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(GeneratedReflection, VoidPointersAreReadOnlyIntegers) {
  reflection_fixture::Data data;
  int target = 0;
  data.pointer = &target;
  auto const* field = SReflect::GetTypeInfo<reflection_fixture::Data>().FindField("pointer");
  ASSERT_NE(nullptr, field);
  EXPECT_TRUE(field->IsReadOnly());
  uint64_t value = 0;
  field->GetValue(&data, &value);
  EXPECT_EQ(reinterpret_cast<uint64_t>(&target), value);
}
