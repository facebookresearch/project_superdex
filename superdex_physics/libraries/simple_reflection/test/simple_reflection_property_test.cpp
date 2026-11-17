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

#include <simple_reflection/simple_reflection.h>

#include <gtest/gtest.h>

SR_WARNING_PUSH()
SR_WARNING_IGNORE_MSVC(4459) // declaration of 'last' hides global declaration
SR_WARNING_IGNORE_MSVC(4706) // assignment within conditional expression
#include <picojson/picojson.h> // picojson (third-party)
SR_WARNING_POP()

#include <algorithm>
#include <array>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SReflectPropertyTest {

static constexpr double kPi = 3.14159265358979323846;

static int g_strictChangedCount = 0;
static int g_referenceChangedCount = 0;

namespace {

struct InnerPair {
  int32_t a = 0;
  int32_t b = 0;

  SR_BeginStruct(SReflectPropertyTest::InnerPair);
  SR_Field(a);
  SR_Field(b);
  SR_EndStruct();
};

// Reflected with SR_BeginStructEx below. It is trivially copyable, so only its property stops it
// from being memcpy-safe.
struct Base {
  int32_t xStorage = 0;
  int32_t idStorage = 7;

  [[nodiscard]] int32_t GetX() const {
    return xStorage;
  }
  void SetX(int32_t v) {
    xStorage = v;
  }
  [[nodiscard]] int32_t GetId() const noexcept {
    return idStorage;
  }
};

} // namespace
} // namespace SReflectPropertyTest

SR_BeginStructEx(SReflectPropertyTest::Base);
SR_Property("x", &SReflectPropertyTest::Base::GetX, &SReflectPropertyTest::Base::SetX);
SR_EndStructEx();

namespace SReflectPropertyTest {
namespace {

// Covers most kinds of property. The comment above each one says which case it covers.
struct Gadget : Base {
  int32_t count = 0;
  float levelStorage = 0.5f;
  int32_t levelSetCount = 0;
  std::string nameStorage;
  std::string nameRefStorage;
  mutable int32_t nameRefReadCount = 0;
  int32_t nameRefWriteCount = 0;
  double radians = 0.0;
  int32_t strictStorage = 0;
  InnerPair innerStorage{1, 9};
  int32_t innerSetCount = 0;
  std::string payloadStorage;
  std::vector<InnerPair> components{InnerPair{}};
  mutable InnerPair cacheStorage;
  bool enabled = false;

  [[nodiscard]] float GetLevel() const {
    return levelStorage;
  }
  void SetLevel(float v) noexcept {
    levelStorage = std::clamp(v, 0.0f, 1.0f);
    ++levelSetCount;
  }
  void SetLevel(double /*unused*/) {
    ADD_FAILURE() << "Selected the wrong SetLevel overload";
  }

  [[nodiscard]] std::string const& GetName() const {
    return nameStorage;
  }
  std::string& GetName() {
    return nameStorage;
  }
  void SetName(std::string const& v) {
    nameStorage = v;
  }

  std::string const& GetReadOnlyName() const {
    return nameStorage;
  }

  std::string const& GetNameRef() const noexcept {
    ++nameRefReadCount;
    return nameRefStorage;
  }
  std::string& GetNameRef() noexcept {
    ++nameRefWriteCount;
    return nameRefStorage;
  }

  InnerPair& GetCache() const {
    return cacheStorage;
  }

  static double GetDegrees(Gadget const& g) {
    return g.radians * 180.0 / kPi;
  }
  static void SetDegrees(Gadget& g, double degrees) {
    g.radians = degrees * kPi / 180.0;
  }

  void SetStrict(int32_t v) {
    if (v < 0) {
      throw std::invalid_argument("negative");
    }
    strictStorage = v;
  }
  void SetStrict(double /*unused*/) {
    ADD_FAILURE() << "Selected the wrong SetStrict overload";
  }

  SR_BeginStruct(SReflectPropertyTest::Gadget);
  // Inherits property "x"
  SR_BaseClass(SReflectPropertyTest::Base);
  SR_Field(count);
  // Member functions: by-value getter, and an overloaded noexcept setter that clamps
  SR_Property("level", &Gadget::GetLevel, &Gadget::SetLevel);
  SRA_FloatRange(0.0f, 1.0f);
  SRA_DoNotSerializeDefaults();
  // Read-only, with a noexcept getter from the base class, and an explicit SRA_ReadOnly
  SR_Property("id", &Gadget::GetId);
  SRA_ReadOnly();
  // Member functions: an overloaded getter returning const&, and a setter taking const&
  SR_Property("name", &Gadget::GetName, &Gadget::SetName);
  // Function pointers (static members behave like free functions) that convert units
  SR_Property("degrees", GetDegrees, SetDegrees);
  // Lambda getter with an overloaded member setter that throws to reject a value
  SR_Property("strict", [](Gadget const& g) { return g.strictStorage; }, &Gadget::SetStrict);
  SRA_OnChanged([] { ++g_strictChangedCount; });
  // Generic lambdas, with a getter returning const& to a struct
  SR_Property(
      "inner",
      [](auto const& g) -> auto const& { return g.innerStorage; },
      [](auto& g, InnerPair const& v) {
        g.innerStorage = v;
        ++g.innerSetCount;
      });
  // By-value lambdas on a string that holds JSON
  SR_Property(
      "payload",
      [](Gadget const& g) { return g.payloadStorage; },
      [](Gadget& g, std::string v) { g.payloadStorage = std::move(v); });
  SRA_JsonString();
  // Read-only lambda, parenthesized because its body contains a comma
  SR_Property("innerArray", ([](Gadget const& g) {
                return std::array<int32_t, 2>{g.innerStorage.a, g.innerStorage.b};
              }));
  // Writable without a setter: a generic lambda returning a reference it looks up on each call
  SR_Property("component", [](auto& g) -> auto& { return g.components.front(); });
  SRA_OnChanged([] { ++g_referenceChangedCount; });
  // Writable without a setter: an overloaded member function, writing through the non-const one
  SR_Property("nameRef", &Gadget::GetNameRef);
  // Writable without a setter: a const member function returning a non-const reference
  SR_Property("cache", &Gadget::GetCache);
  // Read-only: a const member function returning a const reference
  SR_Property("readOnlyName", &Gadget::GetReadOnlyName);
  SR_Field(enabled);
  SR_EndStruct();
};

// A type without a default constructor, so a binary load has nothing to read a value into
struct NoDefault {
  explicit NoDefault(int32_t v) : value(v) {}
  int32_t value;

  SR_BeginStruct(SReflectPropertyTest::NoDefault);
  SR_Field(value);
  SR_EndStruct();
};

struct ReadOnlyNoDefault {
  [[nodiscard]] NoDefault GetFixed() const {
    return NoDefault{2};
  }

  SR_BeginStruct(SReflectPropertyTest::ReadOnlyNoDefault);
  SR_Property("fixed", &ReadOnlyNoDefault::GetFixed);
  SR_EndStruct();
};

struct ThrowingDefault {
  [[noreturn]] ThrowingDefault() {
    throw std::runtime_error("default construction failed");
  }
  explicit ThrowingDefault(int32_t v) : value(v) {}
  int32_t value = 0;

  SR_BeginStruct(SReflectPropertyTest::ThrowingDefault);
  SR_Field(value);
  SR_EndStruct();
};

struct ReadOnlyThrowingDefault {
  [[nodiscard]] ThrowingDefault GetFixed() const {
    return ThrowingDefault{2};
  }

  SR_BeginStruct(SReflectPropertyTest::ReadOnlyThrowingDefault);
  SR_Property("fixed", &ReadOnlyThrowingDefault::GetFixed);
  SR_EndStruct();
};

struct ReadOnlyScalar {
  [[nodiscard]] int32_t GetFixed() const {
    return 2;
  }

  SR_BeginStruct(SReflectPropertyTest::ReadOnlyScalar);
  SR_Property("fixed", &ReadOnlyScalar::GetFixed);
  SR_EndStruct();
};

class ThrowingReader final : public SReflect::StreamReader {
 public:
  [[noreturn]] bool Read(void* /*dst*/, size_t /*numBytes*/) override {
    throw std::bad_alloc{};
  }
};

class CountingAllocator final : public SReflect::Allocator {
 public:
  int allocations = 0;
  int deallocations = 0;

 private:
  void* do_allocate(size_t bytes, size_t alignment) override {
    ++allocations;
    return SReflect::GetDefaultAllocator()->allocate(bytes, alignment);
  }
  void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
    ++deallocations;
    SReflect::GetDefaultAllocator()->deallocate(ptr, bytes, alignment);
  }
  [[nodiscard]] bool do_is_equal(SReflect::Allocator const& other) const noexcept override {
    return this == &other;
  }
};

struct ReferenceNoDefault {
  NoDefault& external;

  SR_BeginStruct(SReflectPropertyTest::ReferenceNoDefault);
  SR_Property("value", [](auto& obj) -> auto& { return obj.external; });
  SR_EndStruct()
};

struct ValueGetterPair {
  mutable int32_t storage = 7;

  int32_t& Get() const {
    return storage;
  }
  int32_t Get() noexcept {
    return storage;
  }

  SR_BeginStruct(SReflectPropertyTest::ValueGetterPair);
  SR_Property("value", &ValueGetterPair::Get);
  SR_EndStruct()
};

class PrivateAccessors {
  int32_t _a = 0;

  [[nodiscard]] int32_t GetA() const {
    return _a;
  }

  SR_BeginClass(SReflectPropertyTest::PrivateAccessors);
  // A private member function getter, and a lambda setter that writes a private member
  SR_Property("a", &PrivateAccessors::GetA, [](PrivateAccessors& p, int32_t v) { p._a = v; });
  SR_EndClass();
};

} // namespace

static SReflect::FieldTypeInfo const& Field(char const* name) {
  SReflect::FieldTypeInfo const* field = SReflect::GetTypeInfo<Gadget>().FindField(name);
  if (field == nullptr) {
    throw std::runtime_error(std::string{"missing field "} + name);
  }
  return *field;
}

static picojson::value ParseJson(std::string const& text) {
  picojson::value value;
  std::istringstream stream(text);
  std::string const err = picojson::parse(value, stream);
  if (!err.empty()) {
    throw std::runtime_error(err);
  }
  return value;
}

// Every member differs from a default-constructed Gadget, including the read-only id.
static Gadget MakeGadget() {
  Gadget g;
  g.xStorage = 4;
  g.idStorage = 1;
  g.count = 3;
  g.levelStorage = 0.25f;
  g.nameStorage = "bob";
  g.nameRefStorage = "carol";
  g.radians = 1.0;
  g.strictStorage = 2;
  g.innerStorage = InnerPair{5, 6};
  g.payloadStorage = R"({"k":1})";
  g.components.front() = InnerPair{7, 8};
  g.cacheStorage = InnerPair{3, 4};
  g.enabled = true;
  return g;
}

static void ExpectWritableMembersEqual(Gadget const& expected, Gadget const& actual) {
  EXPECT_EQ(expected.xStorage, actual.xStorage);
  EXPECT_EQ(expected.count, actual.count);
  EXPECT_EQ(expected.levelStorage, actual.levelStorage);
  EXPECT_EQ(expected.nameStorage, actual.nameStorage);
  EXPECT_EQ(expected.nameRefStorage, actual.nameRefStorage);
  EXPECT_DOUBLE_EQ(expected.radians, actual.radians);
  EXPECT_EQ(expected.strictStorage, actual.strictStorage);
  EXPECT_EQ(expected.innerStorage.a, actual.innerStorage.a);
  EXPECT_EQ(expected.innerStorage.b, actual.innerStorage.b);
  EXPECT_EQ(expected.payloadStorage, actual.payloadStorage);
  EXPECT_EQ(expected.components.front().a, actual.components.front().a);
  EXPECT_EQ(expected.components.front().b, actual.components.front().b);
  EXPECT_EQ(expected.cacheStorage.a, actual.cacheStorage.a);
  EXPECT_EQ(expected.cacheStorage.b, actual.cacheStorage.b);
  EXPECT_EQ(expected.enabled, actual.enabled);
}

TEST(SReflectProperty, Registration) {
  std::vector<std::string> names;
  for (auto const* field : SReflect::GetTypeInfo<Gadget>()._fields) {
    names.emplace_back(field->_name);
  }
  std::vector<std::string> const expected{
      "x",
      "count",
      "level",
      "id",
      "name",
      "degrees",
      "strict",
      "inner",
      "payload",
      "innerArray",
      "component",
      "nameRef",
      "cache",
      "readOnlyName",
      "enabled"};
  EXPECT_EQ(expected, names);

  Gadget g;
  EXPECT_FALSE(Field("count")._isProperty);
  EXPECT_NE(nullptr, Field("count").GetFieldPtr(&g));
  EXPECT_TRUE(Field("level")._isProperty);
  EXPECT_EQ(nullptr, Field("level").GetFieldPtr(&g));
  EXPECT_EQ(&SReflect::GetTypeInfo<InnerPair>(), Field("inner")._innerTypeInfo);

  EXPECT_FALSE(Field("level").IsReadOnly());
  EXPECT_TRUE(Field("innerArray").IsReadOnly());
  EXPECT_TRUE(Field("innerArray").HasAttribute<SReflect::Attribute_ReadOnly>());
  EXPECT_TRUE(Field("readOnlyName").IsReadOnly());
  EXPECT_TRUE(Field("readOnlyName").HasAttribute<SReflect::Attribute_ReadOnly>());
  EXPECT_TRUE(Field("id").IsReadOnly());
  auto const& idAttributes = Field("id")._attributes;
  auto const isReadOnlyAttribute = [](auto const& attribute) {
    return attribute.first == SReflect::Attribute_ReadOnly::GetTypeId();
  };
  // The explicit attribute is not duplicated
  EXPECT_EQ(1, std::count_if(idAttributes.begin(), idAttributes.end(), isReadOnlyAttribute));
}

TEST(SReflectProperty, Accessors) {
  Gadget g;
  float const tooBig = 2.0f;
  EXPECT_TRUE(Field("level").SetValue(&g, &tooBig));
  EXPECT_EQ(1.0f, g.levelStorage); // The setter clamps

  double const degrees = 90.0;
  EXPECT_TRUE(Field("degrees").SetValue(&g, &degrees));
  EXPECT_NEAR(kPi / 2.0, g.radians, 1e-12);
  double readBack = 0.0;
  Field("degrees").GetValue(&g, &readBack);
  EXPECT_NEAR(90.0, readBack, 1e-9);

  int32_t const x = 4;
  EXPECT_TRUE(Field("x").SetValue(&g, &x));
  EXPECT_EQ(4, g.xStorage);

  int32_t const strict = 3;
  EXPECT_TRUE(Field("strict").SetValue(&g, &strict));
  EXPECT_EQ(3, g.strictStorage);

  g.nameStorage = "bob";
  std::string name;
  Field("name").GetValue(&g, &name);
  EXPECT_EQ("bob", name);

  void const* visited = nullptr;
  Field("inner").VisitValue(&g, [&](void const* value) { visited = value; });
  EXPECT_EQ(&g.innerStorage, visited); // A getter returning const& is not copied
  Field("inner").VisitValue(&g, [&](void const* value) { return visited = value; }); // Ignored
  Field("readOnlyName").VisitValue(&std::as_const(g), [&](void const* value) { visited = value; });
  EXPECT_EQ(&g.nameStorage, visited);
}

TEST(SReflectProperty, ModifyValueWritesBackOnlyWhenTheCallbackReturnsTrue) {
  Gadget g;
  EXPECT_TRUE(Field("level").ModifyValue(&g, [](void* value) {
    *static_cast<float*>(value) = 0.75f;
    return false;
  }));
  EXPECT_EQ(0, g.levelSetCount);
  EXPECT_EQ(0.5f, g.levelStorage);

  EXPECT_TRUE(Field("level").ModifyValue(&g, [](void* value) {
    *static_cast<float*>(value) = 0.75f;
    return true;
  }));
  EXPECT_EQ(1, g.levelSetCount);
  EXPECT_EQ(0.75f, g.levelStorage);

  EXPECT_TRUE(Field("inner").ModifyValue(&g, [](void* value) {
    static_cast<InnerPair*>(value)->a = 5;
    return false;
  }));
  EXPECT_EQ(1, g.innerStorage.a);
  EXPECT_EQ(0, g.innerSetCount);

  // A field is modified in place, so there is nothing to write back
  EXPECT_TRUE(Field("count").ModifyValue(&g, [](void* value) {
    *static_cast<int32_t*>(value) = 5;
    return false;
  }));
  EXPECT_EQ(5, g.count);

  EXPECT_FALSE(Field("id").ModifyValue(&g, [](void* /*value*/) {
    ADD_FAILURE() << "Called the callback of a read-only property";
    return true;
  }));
}

TEST(SReflectProperty, ReferenceGettersWriteInPlace) {
  Gadget g;
  int const changedCountBefore = g_referenceChangedCount;
  auto modifyInPlace = [&g](char const* name, auto& expected, auto const& replacement) {
    using T = std::decay_t<decltype(expected)>;
    EXPECT_FALSE(Field(name).IsReadOnly()) << name;
    EXPECT_FALSE(Field(name).HasAttribute<SReflect::Attribute_ReadOnly>()) << name;
    EXPECT_EQ(nullptr, Field(name).GetFieldPtr(&g)) << name;
    EXPECT_TRUE(Field(name).ModifyValue(&g, [&](void* value) {
      EXPECT_EQ(&expected, value) << name;
      *static_cast<T*>(value) = replacement;
      return false; // Ignored, because the value is modified in place
    }));
  };
  modifyInPlace("component", g.components.front(), InnerPair{2, 3});
  modifyInPlace("nameRef", g.nameRefStorage, "bob");
  modifyInPlace("cache", g.cacheStorage, InnerPair{4, 5});
  EXPECT_EQ(2, g.components.front().a);
  EXPECT_EQ("bob", g.nameRefStorage);
  EXPECT_EQ(4, g.cacheStorage.a);
  EXPECT_EQ(changedCountBefore, g_referenceChangedCount);

  g.components.reserve(g.components.capacity() + 1);

  std::string const name = "alice";
  EXPECT_TRUE(Field("nameRef").SetValue(&g, &name));
  EXPECT_EQ("alice", g.nameRefStorage);
  EXPECT_TRUE(g.nameStorage.empty());

  InnerPair const pair{5, 6};
  EXPECT_TRUE(Field("component").SetValue(&g, &pair));
  EXPECT_EQ(6, g.components.front().b);
  EXPECT_EQ(changedCountBefore + 1, g_referenceChangedCount);

  EXPECT_TRUE(SReflect::FromJsonString(g, R"({"component": {"a": 1}})"));
  EXPECT_EQ(1, g.components.front().a);
  EXPECT_EQ(6, g.components.front().b);
  EXPECT_EQ(changedCountBefore + 1, g_referenceChangedCount);
}

TEST(SReflectProperty, ReferenceGetterSelectsConstReadsAndNonConstWrites) {
  Gadget g;
  g.nameRefStorage = "bob";
  std::string value;
  Field("nameRef").GetValue(&std::as_const(g), &value);
  EXPECT_EQ("bob", value);
  EXPECT_EQ(1, g.nameRefReadCount);
  EXPECT_EQ(0, g.nameRefWriteCount);

  value = "alice";
  EXPECT_TRUE(Field("nameRef").SetValue(&g, &value));
  EXPECT_EQ("alice", g.nameRefStorage);
  EXPECT_EQ(1, g.nameRefReadCount);
  EXPECT_EQ(1, g.nameRefWriteCount);
}

TEST(SReflectProperty, ByValueNonConstGetterKeepsPropertyReadOnly) {
  ValueGetterPair obj;
  auto const* field = SReflect::GetTypeInfo<ValueGetterPair>().FindField("value");
  ASSERT_NE(nullptr, field);
  EXPECT_TRUE(field->IsReadOnly());
  EXPECT_TRUE(field->HasAttribute<SReflect::Attribute_ReadOnly>());
  int32_t const value = 9;
  EXPECT_FALSE(field->SetValue(&obj, &value));
  EXPECT_FALSE(field->ModifyValue(&obj, [](void*) {
    ADD_FAILURE() << "Called the callback of a read-only property";
    return true;
  }));
  EXPECT_EQ(7, obj.storage);
}

TEST(SReflectProperty, FailedSetsLeaveObjectUnchanged) {
  Gadget g;
  int const changedCountBefore = g_strictChangedCount;

  int32_t const id = 99;
  EXPECT_FALSE(Field("id").SetValue(&g, &id)); // Read-only
  EXPECT_EQ(7, g.idStorage);

  int32_t const negative = -1;
  EXPECT_THROW(Field("strict").SetValue(&g, &negative), std::invalid_argument);
  EXPECT_EQ(0, g.strictStorage);
  EXPECT_EQ(changedCountBefore, g_strictChangedCount);

  int32_t const positive = 2;
  EXPECT_TRUE(Field("strict").SetValue(&g, &positive));
  EXPECT_EQ(changedCountBefore + 1, g_strictChangedCount);
}

TEST(SReflectProperty, JsonRoundTrip) {
  EXPECT_FALSE(ParseJson(SReflect::ToJsonString(Gadget{}, false)).contains("level"));

  Gadget const src = MakeGadget();
  std::string const json = SReflect::ToJsonString(src, false);
  picojson::value const parsed = ParseJson(json);
  EXPECT_EQ(1.0, parsed["id"].get<double>()); // Read-only properties are written
  EXPECT_TRUE(parsed["payload"].is<picojson::object>());

  Gadget dst;
  std::vector<SReflect::IssueReport> issues;
  SReflect::FromJsonString(dst, json, SReflect::DeserializeFlags::WarnIfExtraneousFields, &issues);
  EXPECT_TRUE(issues.empty()); // Read-only properties are skipped on load, without warnings
  ExpectWritableMembersEqual(src, dst);
  EXPECT_EQ(7, dst.idStorage);
}

TEST(SReflectProperty, MissingReadOnlyPropertiesAreNotReported) {
  ReadOnlyNoDefault obj;
  EXPECT_TRUE(SReflect::FromJsonString(obj, "{}", SReflect::DeserializeFlags::MaximumWarnings));
}

TEST(SReflectProperty, JsonLoadIsReadModifyWrite) {
  Gadget g;
  EXPECT_TRUE(SReflect::FromJsonString(g, R"({"inner": {"a": 5}})"));
  EXPECT_EQ(5, g.innerStorage.a);
  EXPECT_EQ(9, g.innerStorage.b);
  EXPECT_EQ(1, g.innerSetCount);
}

TEST(SReflectProperty, ThrowingSetterDuringLoadIsReportedAsAnIssue) {
  Gadget g;
  std::vector<SReflect::IssueReport> issues;
  SReflect::FromJsonString(
      g, R"({"strict": -1, "enabled": true})", SReflect::DeserializeFlags::Default, &issues);
  ASSERT_EQ(1u, issues.size());
  EXPECT_EQ("strict", issues[0]._path);
  EXPECT_EQ("negative", issues[0]._description);
  EXPECT_EQ(0, g.strictStorage);
  EXPECT_TRUE(g.enabled); // Deserialization continues after the issue
}

TEST(SReflectProperty, BinaryRoundTrip) {
  static_assert(std::is_trivially_copyable_v<Base>);
  EXPECT_FALSE(SReflect::GetTypeInfo<Base>().IsMemCopySafe());

  Gadget const src = MakeGadget();
  std::vector<uint8_t> const bytes = SReflect::ToBytes(src);
  Gadget dst;
  SReflect::SpanStreamReader reader{SReflect::Span<const uint8_t>{bytes}};
  EXPECT_TRUE(SReflect::FromBytes(reader, dst));
  EXPECT_FALSE(reader.HasUnreadBytes());
  ExpectWritableMembersEqual(src, dst);
  EXPECT_EQ(7, dst.idStorage);
}

TEST(SReflectProperty, ThrowingSetterFailsBinaryLoad) {
  Gadget src;
  src.strictStorage = -1; // The setter rejects this value
  std::vector<uint8_t> const bytes = SReflect::ToBytes(src);
  Gadget dst;
  SReflect::SpanStreamReader reader{SReflect::Span<const uint8_t>{bytes}};
  EXPECT_FALSE(SReflect::FromBytes(reader, dst));
}

TEST(SReflectProperty, FailedBinaryReadSkipsTheSetter) {
  Gadget const src = MakeGadget();
  std::vector<uint8_t> bytes = SReflect::ToBytes(src);
  // End the stream after x and count, where level starts
  bytes.resize(SReflect::ToBytes(src.xStorage).size() + SReflect::ToBytes(src.count).size());
  Gadget dst;
  SReflect::SpanStreamReader reader{SReflect::Span<const uint8_t>{bytes}};
  EXPECT_FALSE(SReflect::FromBytes(reader, dst));
  EXPECT_EQ(src.count, dst.count);
  EXPECT_EQ(0, dst.levelSetCount);
}

TEST(SReflectProperty, BinaryLoadLogsAReadOnlyPropertyItCannotSkip) {
  std::vector<uint8_t> const bytes = SReflect::ToBytes(ReadOnlyNoDefault{});
  ReadOnlyNoDefault dst;
  SReflect::SpanStreamReader reader{SReflect::Span<const uint8_t>{bytes}};
  testing::internal::CaptureStdout();
  EXPECT_FALSE(SReflect::FromBytes(reader, dst));
  EXPECT_NE(std::string::npos, testing::internal::GetCapturedStdout().find("'fixed'"));
}

TEST(SReflectProperty, BinaryLoadLogsReadOnlyConstructionExceptions) {
  std::vector<uint8_t> const bytes = SReflect::ToBytes(ReadOnlyThrowingDefault{});
  ReadOnlyThrowingDefault dst;
  SReflect::SpanStreamReader reader{SReflect::Span<const uint8_t>{bytes}};
  testing::internal::CaptureStdout();
  EXPECT_FALSE(SReflect::FromBytes(reader, dst));
  std::string const log = testing::internal::GetCapturedStdout();
  EXPECT_NE(std::string::npos, log.find("'fixed'"));
  EXPECT_NE(std::string::npos, log.find("default construction failed"));
}

TEST(SReflectProperty, BinaryLoadLogsReadOnlyReadExceptions) {
  ReadOnlyScalar dst;
  ThrowingReader reader;
  testing::internal::CaptureStdout();
  EXPECT_FALSE(SReflect::FromBytes(reader, dst));
  EXPECT_NE(std::string::npos, testing::internal::GetCapturedStdout().find("'fixed'"));
}

TEST(SReflectProperty, NewReleasesMemoryWhenConstructionThrows) {
  CountingAllocator allocator;
  EXPECT_THROW(SReflect::GetTypeInfo<ThrowingDefault>().New(&allocator), std::runtime_error);
  EXPECT_EQ(1, allocator.allocations);
  EXPECT_EQ(allocator.allocations, allocator.deallocations);
}

TEST(SReflectProperty, ReferenceGetterLoadsBinaryWithoutDefaultConstructingItsValue) {
  static_assert(!std::is_default_constructible_v<NoDefault>);
  NoDefault source{9};
  NoDefault destination{1};
  ReferenceNoDefault const src{source};
  ReferenceNoDefault dst{destination};
  std::vector<uint8_t> const bytes = SReflect::ToBytes(src);
  SReflect::SpanStreamReader reader{SReflect::Span<uint8_t const>{bytes}};
  EXPECT_TRUE(SReflect::FromBytes(reader, dst));
  EXPECT_FALSE(reader.HasUnreadBytes());
  EXPECT_EQ(9, destination.value);
}

TEST(SReflectProperty, StructCopy) {
  Gadget const src = MakeGadget();
  Gadget dst;
  SReflect::GetTypeInfo<Gadget>().Set(&src, &dst);
  ExpectWritableMembersEqual(src, dst);
  EXPECT_EQ(1, dst.levelSetCount); // Copied through the setter
  EXPECT_EQ(1, dst.innerSetCount);
  EXPECT_EQ(7, dst.idStorage); // Read-only properties are skipped
}

TEST(SReflectProperty, ValidationUsesGetter) {
  Gadget g;
  EXPECT_TRUE(SReflect::IsValid(g));
  g.levelStorage = 2.0f;
  EXPECT_FALSE(SReflect::IsValid(g));
}

TEST(SReflectProperty, TypeInfoJsonDescribesProperties) {
  auto const& info = SReflect::GetTypeInfo<Gadget>();
  picojson::value const json = ParseJson(info.TypeInfoToJson(false));
  picojson::array const& fields = json[info._nameWithNamespace]["fields"].get<picojson::array>();
  auto find = [&](std::string const& name) -> picojson::value const& {
    for (auto const& f : fields) {
      if (f["name"].get<std::string>() == name) {
        return f;
      }
    }
    throw std::runtime_error("missing " + name);
  };

  picojson::value const& count = find("count");
  EXPECT_FALSE(count["isProperty"].get<bool>());
  EXPECT_FALSE(count["isReadOnly"].get<bool>());
  EXPECT_TRUE(count.contains("offset"));

  picojson::value const& level = find("level");
  EXPECT_TRUE(level["isProperty"].get<bool>());
  EXPECT_FALSE(level["isReadOnly"].get<bool>());
  EXPECT_FALSE(level.contains("offset"));

  EXPECT_TRUE(find("id")["isReadOnly"].get<bool>());
}

TEST(SReflectProperty, PrivateAccessorsInClass) {
  PrivateAccessors p;
  EXPECT_TRUE(SReflect::FromJsonString(p, R"({"a": 1})"));
  EXPECT_EQ(R"({"a":1})", SReflect::ToJsonString(p, false));
}

TEST(SReflectProperty, CapturingAccessorsAreRejected) {
  using SReflect::detail::IsStatelessAccessor;
  int32_t local = 0;
  auto captureless = [](Gadget const& g) { return g.count; };
  auto generic = [](auto const& g) { return g.count; };
  auto byCopy = [local](Gadget const&) { return local; };
  auto byReference = [&local](Gadget const&) { return local; };
  auto initCapture = [scale = 2](Gadget const& g) { return g.count * scale; };
  static_assert(IsStatelessAccessor<decltype(captureless)>());
  static_assert(IsStatelessAccessor<decltype(generic)>());
  static_assert(IsStatelessAccessor<decltype(&Gadget::GetDegrees)>());
  static_assert(IsStatelessAccessor<decltype(&Gadget::GetLevel)>());
  static_assert(!IsStatelessAccessor<decltype(byCopy)>());
  static_assert(!IsStatelessAccessor<decltype(byReference)>());
  static_assert(!IsStatelessAccessor<decltype(initCapture)>());
  EXPECT_EQ(0, byReference(Gadget{}));
}

} // namespace SReflectPropertyTest
