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

#include "type_declarations.h"

#include <simple_reflection/simple_reflection.h>

#include <gtest/gtest.h>

SR_WARNING_PUSH()
SR_WARNING_IGNORE_MSVC(4459) // declaration of 'last' hides global declaration
#include <picojson/picojson.h> // picojson (third-party)
SR_WARNING_POP()

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <optional>
#include <sstream>
#include <type_traits>
#include <unordered_map>

struct SReflectTest_GlobalStruct {
  SR_BeginStruct(SReflectTest_GlobalStruct);
  SR_EndStruct();
};

class SReflectTest_GlobalClass {
  SR_BeginClass(SReflectTest_GlobalClass);
  SR_EndClass();
};

enum SReflectTest_GlobalEnum { kValue };

SR_BeginEnum(SReflectTest_GlobalEnum);
SR_EnumItem(kValue);
SR_EndEnum();

enum class SReflectTest_GlobalEnumClass { kValue };

SR_BeginEnum(SReflectTest_GlobalEnumClass);
SR_EnumItem(kValue);
SR_EndEnum();

namespace SReflectTest {

// Helper to remove whitespace characters from a string
static std::string RemoveWhitespace(std::string const& str) {
  std::string ret;
  ret.reserve(str.length() + 1);
  for (char c : str) {
    if (!std::isspace(static_cast<int>(c))) {
      ret.push_back(c);
    }
  }
  return ret;
}

// Helper to compare strings without white space
static void ExpectJson(std::string const& expectedJson, std::string const& actualJson) {
  EXPECT_STREQ(RemoveWhitespace(expectedJson).c_str(), RemoveWhitespace(actualJson).c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Example Usage Syntax
///////////////////////////////////////////////////////////////////////////////////////////////

enum class MyFruit : int32_t { Apple, Orange, Kumquat };

} // namespace SReflectTest

// NOTE: The current MacOS compiler only allows SR_BeginEnum in the global
//       namespace because it is out-dated. This should not be necessary because
//       current day C++ allows template specialization within a nested namespace.
SR_BeginEnum(SReflectTest::MyFruit);
SR_EnumItem(Apple);
SR_EnumItem(Orange);
SR_EnumItem(Kumquat);
SR_EndEnum();

namespace SReflectTest {

// Prove that we can access type information at static initialization time.
static const bool kTypeInfoSuccessfullyAccessedBeforeMain = []() {
  auto const& enumInfo = SReflect::GetTypeInfo<MyFruit>();
  char const* enumName = enumInfo._nameWithNamespace;
  char const* innerName = enumInfo._innerTypeInfo ? enumInfo._innerTypeInfo->_name : nullptr;
  return enumName && (std::string_view{enumName} == "SReflectTest::MyFruit") && innerName &&
      (std::string_view{innerName} == "int32");
}();

struct MyPoint {
  int x = 0;
  int y = 0;

  bool operator==(MyPoint const& rhs) const {
    return x == rhs.x && y == rhs.y;
  }

  SR_BeginStruct(SReflectTest::MyPoint);
  SR_Field(x);
  SR_Field(y);
  SR_EndStruct();
};

class MyClass : public SReflect::BaseObject {
 public:
  MyFruit myFruit = MyFruit::Kumquat;
  MyPoint myPoint;
  bool myBool = false;
  float myFloat = 9.11f;
  std::vector<std::string> myStrings{"one", "two"};
  void* noNeedToReflectEveryMemember = nullptr;

  SR_BeginClass(SReflectTest::MyClass);
  SRA_Description("MyClass is the best!");
  SR_Field(myFruit);
  SR_Field(myPoint);
  SR_Field(myBool);
  SR_Field(myFloat);
  SRA_FloatRange(0.0f, 1.0f);
  SR_Field(myStrings);
  SR_EndClass();
};

class MyDerivedClass : public MyClass {
 public:
  int oneMoreThing = 0;

  SR_BeginClass(SReflectTest::MyDerivedClass);
  SR_BaseClass(MyClass);
  SR_Field(oneMoreThing);
  SR_EndClass();
};

TEST(SReflect, GetTypeInfoBeforeMain) {
  EXPECT_TRUE(kTypeInfoSuccessfullyAccessedBeforeMain);
}

TEST(SReflect, Example1) {
  static_assert(sizeof(MyPoint) == 2 * sizeof(int), "No v-table. No padding");

  // Create a simple struct
  MyPoint pt;
  pt.x = 123;
  pt.y = 456;

  // Serialize to JSON
  std::string json = SReflect::ToJsonString(pt, false);
  EXPECT_STREQ("{\"x\":123,\"y\":456}", json.c_str());

  // Deserialize from JSON
  int numIssuesDetected = 0;
  SReflect::FromJsonString(
      pt, "{\"y\":987,\"x\":654}", SReflect::DeserializeFlags::MaximumWarnings, numIssuesDetected);
  EXPECT_EQ(0, numIssuesDetected);
  EXPECT_EQ(654, pt.x);
  EXPECT_EQ(987, pt.y);

  // Get the TypeInfo and do whatever you want with it
  [[maybe_unused]] SReflect::TypeInfo const& typeInfo = SReflect::GetTypeInfo<MyPoint>();
}

TEST(SReflect, Example2) {
  // This time, create an object of derived type, but point to it with a generic base class pointer.
  auto derivedObj = std::make_unique<MyDerivedClass>();
  SReflect::BaseObject* baseObj = derivedObj.get();

  // We can get the derived by by calling GetFinalTypeInfo
  SReflect::StructTypeInfo const& typeInfo = baseObj->GetFinalTypeInfo();
  EXPECT_STREQ("MyDerivedClass", typeInfo._name); // final derived class
  EXPECT_STREQ("MyClass", typeInfo._baseClasses[0]->_name); // its base class

  // If we serialize using the base pointer, we still get the fields from the derived classes
  std::string expectedJson =
      R"({
        "myBool": false,
        "myFloat": 9.1099996566772461,
        "myFruit": "Kumquat",
        "myPoint": {
          "x": 0,
          "y": 0
        },
        "myStrings": [
          "one",
          "two"
        ],
        "oneMoreThing": 0
      })";

  std::string actualJson = SReflect::ToJsonString(*baseObj);
  ExpectJson(expectedJson, actualJson);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Systematic Tests
///////////////////////////////////////////////////////////////////////////////////////////////

// Short hand
template <typename T>
static std::string ToJson(const T& obj) {
  return SReflect::ToJsonString(obj, false);
}

// Short hand
template <typename T>
static T FromJson(std::string const& jsonStr) {
  T obj{};
  int numIssuesOut = 0;
  SReflect::FromJsonString(obj, jsonStr, SReflect::DeserializeFlags::MaximumWarnings, numIssuesOut);
  EXPECT_EQ(0, numIssuesOut);
  return obj;
}

// Utility for comparing equality of various types.
template <typename A, typename B>
static bool IsEqual(A const& a, B const& b) {
  return a == b;
}
template <typename T, size_t N>
static bool IsEqual(const T (&a)[N], const T (&b)[N]) {
  for (size_t i = 0; i < N; ++i) {
    if (!(a[i] == b[i])) {
      return false;
    }
  }
  return true;
}

// Utility for copy assignement, which allows assignment of T[N]
template <typename T>
static void Assign(T& to, const T& from) {
  if constexpr (std::is_array_v<T>) {
    std::copy(std::begin(from), std::end(from), std::begin(to));
  } else {
    to = from;
  }
}

template <typename T>
struct SRTestValue {
  T value; // Any value of any type
  char const* valueAsJson = ""; // The corresponding JSON
};

template <typename T>
struct TemplateStructWithValue {
  T value;

  // NOTE: Declaring a template struct like this does not generate a unique type name
  //       and hash code for each type T. Doing so would require a modified version of
  //       SR_BeginStruct.
  SR_BeginStruct(SReflectTest::TemplateStructWithValue<T>);
  SR_Field(value);
  SR_EndStruct();
};

template <typename T, typename TestValueT = SRTestValue<T>>
static void TestConstructDestroy(std::vector<TestValueT> const& testValues) {
  auto const& ti = SReflect::GetTypeInfo<T>();

  // All types passed to this function support default and copy constructors
  EXPECT_TRUE(ti._constructInPlace);
  EXPECT_TRUE(ti._constructInPlaceByCopy);
  EXPECT_TRUE(ti._destructInPlace);

  // Construct in place. Expect default values.
  {
    alignas(T) std::byte buffer[sizeof(T)];
    ti._constructInPlace(buffer);
    auto expectedJson = SReflect::ToJsonString<T>(T{}, false);
    auto actualJson = SReflect::ToJsonString<T>(*reinterpret_cast<T*>(buffer), false);
    EXPECT_STREQ(expectedJson.c_str(), actualJson.c_str()); // Compare via serialization
    ti._destructInPlace(buffer);
  }

  // Use reflection to clone each of the test values. This time, write it in a way that does not
  // depend on compiler knowledge of type T.
  for (auto const& test : testValues) {
    constexpr size_t kAlignedEnough = 64; // More than enough for these tests
    constexpr size_t kBigEnough = 512; // More than enough for these tests
    EXPECT_GT(kAlignedEnough, ti._alignment);
    EXPECT_GT(kBigEnough, ti._sizeInBytes);
    alignas(kAlignedEnough) std::byte cloneBuffer[kBigEnough];
    ti._constructInPlaceByCopy(cloneBuffer, &test.value);
    picojson::value cloneJson;
    ti.Serialize(cloneBuffer, cloneJson);
    ExpectJson(test.valueAsJson, cloneJson.serialize(false)); // Compare via serialization
    ti._destructInPlace(cloneBuffer);
  }

  // Repeat with Clone and Delete, if supported
  for (auto const& test : testValues) {
    void* clone = ti.Clone(&test.value);
    picojson::value cloneJson;
    ti.Serialize(clone, cloneJson);
    ExpectJson(test.valueAsJson, cloneJson.serialize(false)); // Compare via serialization
    ti.Delete(clone);
  }
}

// A common utility to test any supported type with a variety of values
template <typename T, typename TestValueT = SRTestValue<T>>
static void TestAnyType(
    char const* shortName,
    char const* fullName,
    SReflect::CoreType coreType,
    std::vector<TestValueT> const& testValues) {
  EXPECT_EQ(coreType, SReflectTypeTraits<T>::coreType);

  // GetTypeInfo
  auto& info = SReflect::GetTypeInfo<T>();
  EXPECT_EQ(sizeof(T), info._sizeInBytes);
  EXPECT_EQ(alignof(T), info._alignment);
  EXPECT_EQ(coreType, info._coreType);
  EXPECT_STREQ(shortName, info._name);
  if (fullName) {
    EXPECT_STREQ(fullName, info._nameWithNamespace);
  } else {
    // The exact full name is unknown, but we still expect it to end with the short name.
    EXPECT_GE(strlen(info._nameWithNamespace), strlen(info._name));
    EXPECT_EQ(
        0,
        strcmp(
            info._nameWithNamespace + strlen(info._nameWithNamespace) - strlen(info._name),
            info._name));
  }
  EXPECT_NE(0, info._typeId.value);
  EXPECT_EQ(SReflect::ComputeTypeId(info._nameWithNamespace).value, info._typeId.value);

  // GetFinalTypeInfo
  T obj{};
  EXPECT_EQ(
      &info,
      &SReflect::GetFinalTypeInfo(obj)); // Same TypeInfo address since since T is a concrete type

  for (size_t i = 0; i < testValues.size(); ++i) {
    // Index of some other value. Expected to be non-equal.
    size_t j = (i + 1) % testValues.size();
    EXPECT_NE(i, j) << "Need more than one test value";

    // Create two variables with different values
    T val1;
    T val2;
    Assign(val1, testValues[i].value);
    Assign(val2, testValues[j].value);

    // Copy val1 --> val2
    info.Set(&val1, &val2);
    EXPECT_TRUE(IsEqual(val1, val2));

    // Serialize val1 to JSON
    std::string json = SReflect::ToJsonString(val1);
    ExpectJson(testValues[i].valueAsJson, json);

    // Deserialize the JSON into val3
    T val3;
    int numIssues = 0;
    SReflect::FromJsonString(val3, json, SReflect::DeserializeFlags::MaximumWarnings, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_TRUE(IsEqual(val1, val3));

    // Copy the value into a struct with a reflection field
    TemplateStructWithValue<T> s;
    Assign(s.value, val1);
  }

  TestConstructDestroy<T, TestValueT>(testValues);
}

TEST(SReflect, TypeId) {
  SReflect::TypeId defaultId;
  EXPECT_EQ(false, defaultId.IsValid());

  auto a = SReflect::TypeId{1};
  auto b = SReflect::TypeId{1};
  auto c = SReflect::TypeId{2};
  EXPECT_EQ(true, a.IsValid());
  EXPECT_EQ(true, b.IsValid());
  EXPECT_EQ(true, c.IsValid());

  // operator==
  EXPECT_EQ(true, a == a);
  EXPECT_EQ(true, a == b);
  EXPECT_EQ(false, a == c);

  // operator!=
  EXPECT_EQ(false, a != a);
  EXPECT_EQ(false, a != b);
  EXPECT_EQ(true, a != c);

  // std::hash
  std::unordered_map<SReflect::TypeId, bool> table;
  EXPECT_EQ(false, table[a]);
  EXPECT_EQ(false, table[b]);
  EXPECT_EQ(false, table[c]);
  table[a] = true;
  EXPECT_EQ(true, table[a]);
  EXPECT_EQ(true, table[b]);
  EXPECT_EQ(false, table[c]);
}

TEST(SReflect, DefaultAllocator) {
  auto* allocator = SReflect::GetDefaultAllocator();
  EXPECT_NE((SReflect::Allocator*)nullptr, allocator);
  EXPECT_TRUE(allocator->is_equal(*allocator));

  constexpr size_t kAlignments[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  for (size_t sz = 1; sz <= 1024; ++sz) {
    for (size_t align : kAlignments) {
      auto* ptr = static_cast<std::byte*>(allocator->allocate(sz, align));
      EXPECT_EQ(0, reinterpret_cast<std::uintptr_t>(ptr) % align);
      EXPECT_TRUE(ptr || !sz);
      if (ptr) {
        memset(ptr, 0, sz);
      }
      allocator->deallocate(ptr, sz, align);
    }
  }
}

TEST(SReflect, PrimitiveTypes) {
  static_assert(
      (int)SReflect::CoreType::X_Count == 22, "Please update this code if you add a CoreType");

  // clang-format off
  TestAnyType<bool>("bool", "bool",                 SReflect::CoreType::CT_bool,    {{true, "true"}, {false, "false"}});
  TestAnyType<uint8_t>("uint8","uint8",             SReflect::CoreType::CT_uint8,   {{0, "0"}, {UINT8_MAX, "255"}});
  TestAnyType<int8_t>("int8","int8",                SReflect::CoreType::CT_int8,    {{INT8_MIN, "-128"}, {INT8_MAX, "127"}});
  TestAnyType<uint16_t>("uint16","uint16",          SReflect::CoreType::CT_uint16,  {{0, "0"}, {UINT16_MAX, "65535"}});
  TestAnyType<int16_t>("int16","int16",             SReflect::CoreType::CT_int16,   {{INT16_MIN, "-32768"}, {INT16_MAX, "32767"}});
  TestAnyType<uint32_t>("uint32","uint32",          SReflect::CoreType::CT_uint32,  {{0, "0"}, {UINT32_MAX, "4294967295"}});
  TestAnyType<int32_t>("int32","int32",             SReflect::CoreType::CT_int32,   {{INT32_MIN, "-2147483648"}, {INT32_MAX, "2147483647"}});
  TestAnyType<uint64_t>("uint64","uint64",          SReflect::CoreType::CT_uint64,  {{0, "0"}, {INT64_MAX, "9223372036854775807"}}); // INT64_MAX is the largest integer value supported by picojson
  TestAnyType<int64_t>("int64","int64",             SReflect::CoreType::CT_int64,   {{INT64_MIN, "-9223372036854775808"}, {INT64_MAX, "9223372036854775807"}});
  TestAnyType<float>("float","float",               SReflect::CoreType::CT_float,   {{0.0f, "0"}, {-FLT_MAX, "-3.4028234663852886e+38"}, {FLT_MAX, "3.4028234663852886e+38"}});
  TestAnyType<double>("double","double",            SReflect::CoreType::CT_double,  {{0.0, "0"}, {-DBL_MAX, "-1.7976931348623157e+308"}, {DBL_MAX, "1.7976931348623157e+308"}});
  // clang-format on
}

TEST(SReflect, String) {
  TestAnyType<std::string>(
      "string", "std::string", SReflect::CoreType::CT_string, {{"", "\"\""}, {"woot", "\"woot\""}});

  // Editing & inspection via StringTypeInfo
  SReflect::StringTypeInfo const& ti = SReflect::GetTypeInfo<std::string>();
  void* str = ti.New();
  EXPECT_TRUE(ti._isNullTerminated);
  EXPECT_FALSE(ti._isReadOnly);
  EXPECT_EQ(std::string_view(""), ti.GetString(str));
  EXPECT_TRUE(ti.SetString(str, "hello"));
  EXPECT_EQ(std::string_view("hello"), ti.GetString(str));
  EXPECT_TRUE(ti.SetString(str, "new"));
  EXPECT_EQ(std::string_view("new"), ti.GetString(str));
  EXPECT_EQ('\0', ti.GetString(str).data()[3]) << "Expected null terminator";
  ti.Delete(str);
}

// A 1-byte enum class
enum class TestEnumU8 : uint8_t { min = 0, one = 1, two = 2, max = 255 };

enum class TestEnumI64 : int64_t { min = INT64_MIN, one = 1, two = 2, max = INT64_MAX };

namespace TestNamespace { // Show that we can use the SR_BeginEnum/SR_EndEnum macros in a namespace
enum class TestEnumU64 : uint64_t { min = 0, one = 1, two = 2, max = UINT64_MAX };
} // namespace TestNamespace

} // namespace SReflectTest

SR_BeginEnum(SReflectTest::TestEnumI64);
SR_EnumItem(min);
SR_EnumItem(one);
SR_EnumItem(two);
SR_EnumItem(max);
SR_EndEnum();

SR_BeginEnum(SReflectTest::TestEnumU8);
SR_EnumItem(min);
SR_EnumItem(one);
SR_EnumItem(two);
SR_EnumItem(max);
SR_EndEnum();

SR_BeginEnum(SReflectTest::TestNamespace::TestEnumU64);
SR_EnumItem(max); // out-of-order is OK
SR_EnumItem(two);
SR_EnumItem(min);
SR_EnumItem(one);
SR_EndEnum();

namespace SReflectTest {

template <typename EnumT>
static void
TestEnumType(const char* expectedShortName, const char* expectedFullName, size_t expectedSize) {
  EXPECT_EQ(expectedSize, sizeof(EnumT));

  const std::vector<SRTestValue<EnumT>> testValues = {
      {EnumT::min, "\"min\""},
      {EnumT::one, "\"one\""},
      {EnumT::two, "\"two\""},
      {EnumT::max, "\"max\""},
  };

  // Run generic tests
  TestAnyType<EnumT>(expectedShortName, expectedFullName, SReflect::CoreType::CT_enum, testValues);

  // Use the EnumTypeInfo to inspect it
  SReflect::EnumTypeInfo const& info = SReflect::GetTypeInfo<EnumT>();
  EXPECT_EQ(info._innerTypeInfo, &SReflect::GetTypeInfo<std::underlying_type_t<EnumT>>());
  EXPECT_EQ(4, info._items.size());

  // FindItemByName
  EXPECT_EQ(EnumT::min, static_cast<EnumT>(info.FindItemByName("min")->_value));
  EXPECT_EQ(EnumT::one, static_cast<EnumT>(info.FindItemByName("one")->_value));
  EXPECT_EQ(EnumT::two, static_cast<EnumT>(info.FindItemByName("two")->_value));
  EXPECT_EQ(EnumT::max, static_cast<EnumT>(info.FindItemByName("max")->_value));
  EXPECT_EQ(nullptr, info.FindItemByName("no_such_item"));

  // FindItemByValue
  EXPECT_STREQ("min", info.FindItemByValue(static_cast<uint64_t>(EnumT::min))->_name);
  EXPECT_STREQ("one", info.FindItemByValue(static_cast<uint64_t>(EnumT::one))->_name);
  EXPECT_STREQ("two", info.FindItemByValue(static_cast<uint64_t>(EnumT::two))->_name);
  EXPECT_STREQ("max", info.FindItemByValue(static_cast<uint64_t>(EnumT::max))->_name);
  EXPECT_EQ(nullptr, info.FindItemByValue(911)); // no such value

  // EnumToString (shorthand for FindItemByValue)
  EXPECT_STREQ("min", SReflect::EnumToString(EnumT::min));
  EXPECT_STREQ("one", SReflect::EnumToString(EnumT::one));
  EXPECT_STREQ("two", SReflect::EnumToString(EnumT::two));
  EXPECT_STREQ("max", SReflect::EnumToString(EnumT::max));
  EXPECT_STREQ("", SReflect::EnumToString(static_cast<EnumT>(911))); // no such value

  // GetValue (as uint64_t)
  EnumT value = EnumT::one;
  EXPECT_EQ(static_cast<uint64_t>(EnumT::one), info.GetValue(&value));
  value = EnumT::two;
  EXPECT_EQ(static_cast<uint64_t>(EnumT::two), info.GetValue(&value));

  // SetValue (from uint64_t)
  info.SetValue(&value, static_cast<uint64_t>(EnumT::max));
  EXPECT_EQ(EnumT::max, value);
  info.SetValue(&value, static_cast<uint64_t>(EnumT::min));
  EXPECT_EQ(EnumT::min, value);

  // GetTypeIndex
  EXPECT_EQ(std::type_index{typeid(EnumT)}, info.GetTypeIndex());
}

TEST(SReflect, Enum) {
  // Do similar tests with enums of different underlying types
  TestEnumType<TestEnumU8>("TestEnumU8", "SReflectTest::TestEnumU8", 1);
  TestEnumType<SReflectTest::TestEnumI64>("TestEnumI64", "SReflectTest::TestEnumI64", 8);
  TestEnumType<TestNamespace::TestEnumU64>(
      "TestEnumU64", "SReflectTest::TestNamespace::TestEnumU64", 8);
}

TEST(SReflect, std_vector) {
  // std::vector<int32_t>
  {
    using VecType = std::vector<int32_t>;
    std::vector<SRTestValue<VecType>> testValues = {
        {VecType{}, "[]"},
        {VecType{1}, "[1]"},
        {VecType{1, 2, 3}, "[1, 2, 3]"},
    };
    TestAnyType<VecType>(
        "vector<int32>", "std::vector<int32>", SReflect::CoreType::CT_array, testValues);
  }

  // std::vector<bool>
  {
    using VecType = std::vector<bool>;
    std::vector<SRTestValue<VecType>> testValues = {
        {VecType{}, "[]"},
        {VecType{true}, "[true]"},
        {VecType{true, false, true}, "[true, false, true]"},
    };
    TestAnyType<VecType>(
        "vector<bool>", "std::vector<bool>", SReflect::CoreType::CT_array, testValues);
  }

  // std::vector<std::string>
  {
    using VecType = std::vector<std::string>;
    std::vector<SRTestValue<VecType>> testValues = {
        {VecType{}, "[]"},
        {VecType{"one"}, "[\"one\"]"},
        {VecType{"one", "two", "three"}, "[\"one\", \"two\", \"three\"]"},
    };
    TestAnyType<VecType>(
        "vector<string>", "std::vector<std::string>", SReflect::CoreType::CT_array, testValues);
  }
}

TEST(SReflect, std_array) {
  // std::array<int32_t, 1>
  {
    using ArrType = std::array<int32_t, 1>;
    std::vector<SRTestValue<ArrType>> testValues = {
        {ArrType{-1}, "[-1]"},
        {ArrType{0}, "[0]"},
        {ArrType{1}, "[1]"},
    };
    TestAnyType<ArrType>(
        "array<int32,1>", "std::array<int32,1>", SReflect::CoreType::CT_array, testValues);
  }

  // std::array<std::string, 2>
  {
    using ArrType = std::array<std::string, 2>;
    std::vector<SRTestValue<ArrType>> testValues = {
        SRTestValue<ArrType>{{"", "word"}, "[\"\", \"word\"]"},
        SRTestValue<ArrType>{{"word", ""}, "[\"word\", \"\"]"},
        SRTestValue<ArrType>{{"one", "two"}, "[\"one\", \"two\"]"},
    };
    TestAnyType<ArrType>(
        "array<string,2>", "std::array<std::string,2>", SReflect::CoreType::CT_array, testValues);
  }
}

TEST(SReflect, c_array) {
  // int32_t[1]
  {
    struct ArrTestValue {
      int32_t value[1];
      char const* valueAsJson = "";
    };
    std::vector<ArrTestValue> testValues = {
        ArrTestValue{{-1}, "[-1]"},
        ArrTestValue{{0}, "[0]"},
        ArrTestValue{{1}, "[1]"},
    };
    TestAnyType<int32_t[1], ArrTestValue>(
        "int32[1]", "int32[1]", SReflect::CoreType::CT_array, testValues);
  }

  // std::string[2]
  {
    struct ArrTestValue {
      std::string value[2];
      char const* valueAsJson = "";
    };
    std::vector<ArrTestValue> testValues = {
        ArrTestValue{{"", "word"}, "[\"\", \"word\"]"},
        ArrTestValue{{"word", ""}, "[\"word\", \"\"]"},
        ArrTestValue{{"one", "two"}, "[\"one\", \"two\"]"},
    };
    TestAnyType<std::string[2], ArrTestValue>(
        "string[2]", "std::string[2]", SReflect::CoreType::CT_array, testValues);
  }
}

struct StructWithOptionals {
  std::optional<int> myInt;
  std::optional<std::vector<std::string>> myStringVector;
  std::optional<MyPoint> myPoint;

  bool operator==(const StructWithOptionals& rhs) const {
    return myInt == rhs.myInt && myStringVector == rhs.myStringVector && myPoint == rhs.myPoint;
  }

  SR_BeginStruct(SReflectTest::StructWithOptionals);
  SR_Field(myInt);
  SR_Field(myStringVector);
  SR_Field(myPoint);
  SR_EndStruct();
};

TEST(SReflect, std_optional) {
  // optional bool
  {
    struct OptionalTestValue {
      std::optional<bool> value;
      char const* valueAsJson = "";
    };
    std::vector<OptionalTestValue> testValues = {
        OptionalTestValue{{}, "null"}, // no value
        OptionalTestValue{{true}, "true"},
        OptionalTestValue{{false}, "false"},
    };
    TestAnyType<std::optional<bool>, OptionalTestValue>(
        "optional<bool>", "std::optional<bool>", SReflect::CoreType::CT_optional, testValues);
  }

  // optional vector
  {
    struct OptionalTestValue {
      std::optional<std::vector<std::string>> value;
      char const* valueAsJson = "";
    };
    std::vector<OptionalTestValue> testValues = {
        OptionalTestValue{{}, "null"}, // no value
        OptionalTestValue{{std::vector<std::string>{}}, "[]"}, // empty array
        OptionalTestValue{{std::vector<std::string>{"one"}}, "[\"one\"]"},
        OptionalTestValue{
            {std::vector<std::string>{"one", "two", "three"}}, "[\"one\",\"two\",\"three\"]"},
    };
    TestAnyType<std::optional<std::vector<std::string>>, OptionalTestValue>(
        "optional<vector<string>>",
        "std::optional<std::vector<std::string>>",
        SReflect::CoreType::CT_optional,
        testValues);
  }

  // optional field within a struct (writes nothing if no value)
  {
    struct OptionalTestValue {
      StructWithOptionals value;
      char const* valueAsJson = "";
    };
    std::vector<OptionalTestValue> testValues = {
        OptionalTestValue{StructWithOptionals{}, "{}"},
        OptionalTestValue{StructWithOptionals{42, std::nullopt, std::nullopt}, "{\"myInt\":42}"},
        OptionalTestValue{
            StructWithOptionals{std::nullopt, std::vector<std::string>{"one", "two"}, std::nullopt},
            "{\"myStringVector\":[\"one\",\"two\"]}"},
        OptionalTestValue{
            StructWithOptionals{std::nullopt, std::nullopt, MyPoint{9, 11}},
            "{\"myPoint\":{\"x\":9,\"y\":11}}"},
        OptionalTestValue{
            StructWithOptionals{42, std::nullopt, MyPoint{9, 11}},
            "{\"myInt\":42,\"myPoint\":{\"x\":9,\"y\":11}}"},
    };
    TestAnyType<StructWithOptionals, OptionalTestValue>(
        "StructWithOptionals",
        "SReflectTest::StructWithOptionals",
        SReflect::CoreType::CT_struct,
        testValues);
  }

  // vector of optional
  {
    struct OptionalTestValue {
      std::vector<std::optional<int32_t>> value;
      char const* valueAsJson = "";
    };
    std::vector<OptionalTestValue> testValues = {
        OptionalTestValue{{}, "[]"}, // empty array
        OptionalTestValue{{{}}, "[null]"}, // contains one element with no value
        OptionalTestValue{{{123}}, "[123]"}, // contains one element with a value
        OptionalTestValue{
            {{123, {}, 456, {}}}, "[123, null, 456, null]"}, // mix of value and no value
    };
    TestAnyType<std::vector<std::optional<int32_t>>, OptionalTestValue>(
        "vector<optional<int32>>",
        "std::vector<std::optional<int32>>",
        SReflect::CoreType::CT_array,
        testValues);
  }

  // SReflect Manipulation
  {
    std::optional<std::string> optionalStr;
    SReflect::OptionalTypeInfo const& optionalTypeInfo =
        SReflect::GetTypeInfo<std::optional<std::string>>();
    EXPECT_STREQ("optional<string>", optionalTypeInfo._name);
    EXPECT_EQ(sizeof(optionalStr), optionalTypeInfo._sizeInBytes);
    EXPECT_EQ(&SReflect::GetTypeInfo<std::string>(), optionalTypeInfo._innerTypeInfo);

    // No value by default
    EXPECT_EQ(nullptr, optionalTypeInfo.GetOptionalValue(&optionalStr));
    EXPECT_STREQ("null", SReflect::ToJsonString(optionalStr, false).c_str());

    // Give it a value by calling EnsureOptionalValue
    auto* pStrValue =
        reinterpret_cast<std::string*>(optionalTypeInfo.EnsureOptionalValue(&optionalStr));
    EXPECT_NE(static_cast<std::string*>(nullptr), pStrValue);
    EXPECT_STREQ("", pStrValue->c_str()); // EnsureOptionalValue gave it the default value

    // Get the value (same address)
    const auto& optionalStrConstRef = optionalStr;
    EXPECT_EQ((void*)pStrValue, optionalTypeInfo.GetOptionalValue(&optionalStr));
    EXPECT_EQ((void const*)pStrValue, optionalTypeInfo.GetOptionalValue(&optionalStrConstRef));

    // Set a different value
    std::string newStr = "woot";
    optionalTypeInfo.SetOptionalValue(&newStr, &optionalStr);
    EXPECT_STREQ("\"woot\"", SReflect::ToJsonString(optionalStr, false).c_str());
    EXPECT_EQ((void*)pStrValue, optionalTypeInfo.GetOptionalValue(&optionalStr)); // same address

    // Now that it has a value, EnsureOptionalValue is just like GetOptionalValue
    EXPECT_EQ((void*)pStrValue, optionalTypeInfo.EnsureOptionalValue(&optionalStr));

    // Clear the value
    optionalTypeInfo.SetOptionalValue(nullptr, &optionalStr);
    EXPECT_EQ(nullptr, optionalTypeInfo.GetOptionalValue(&optionalStr));

    // Use SetOptionalValue to give it a non-default value
    newStr = "Awesome Sauce";
    optionalTypeInfo.SetOptionalValue(&newStr, &optionalStr);
    EXPECT_STREQ("\"Awesome Sauce\"", SReflect::ToJsonString(optionalStr, false).c_str());
  }
}

TEST(SReflect, picojson_value) {
  {
    struct TestValue {
      picojson::value value;
      char const* valueAsJson = "";
    };
    std::vector<TestValue> testValues = {
        TestValue{picojson::value{}, "null"},
        TestValue{picojson::object{}, "{}"},
        TestValue{picojson::object{{"test", "1"}}, "{\"test\":\"1\"}"},
        TestValue{picojson::array{}, "[]"},
        TestValue{picojson::array{{"1"}}, "[\"1\"]"},
    };
    TestAnyType<picojson::value, TestValue>(
        "picojson_value", "picojson_value", SReflect::CoreType::CT_other, testValues);
  }
}

// A trival struct
struct MyTrivialInnerStruct {
  int32_t x = 0;

  bool operator==(MyTrivialInnerStruct const& rhs) const {
    return rhs.x == x;
  }

  SR_BeginStruct(SReflectTest::MyTrivialInnerStruct);
  SR_Field(x);
  SR_EndStruct();
};

// A struct with an arbitrary mix of types.
struct MyStructWithMixedFields {
  uint8_t myByte = 0;
  std::string myString = "woot";
  MyTrivialInnerStruct myStruct;
  std::vector<MyTrivialInnerStruct> myStructVec;

  bool operator==(MyStructWithMixedFields const& rhs) const {
    return (rhs.myByte == myByte) && (rhs.myString == myString) && (rhs.myStruct == myStruct) &&
        (myStructVec == myStructVec);
  }

  SR_BeginStruct(SReflectTest::MyStructWithMixedFields);
  SR_Field(myByte);
  SR_Field(myString);
  SR_Field(myStruct);
  SR_Field(myStructVec);
  SR_EndStruct();
};

TEST(SReflect, Struct) {
  std::vector<SRTestValue<MyStructWithMixedFields>> testValues = {
      SRTestValue<MyStructWithMixedFields>{
          {0, "", {0}, {}},
          "{\"myByte\":0, \"myString\":\"\", \"myStruct\":{\"x\":0}, \"myStructVec\":[]}"},
      SRTestValue<MyStructWithMixedFields>{
          {123, "zoink", {456}, {{1}, {2}}},
          "{\"myByte\":123, \"myString\":\"zoink\", \"myStruct\":{\"x\":456}, \"myStructVec\":[{\"x\":1}, {\"x\":2}]}"},
  };

  // Run generic tests using these sample values.
  TestAnyType<MyStructWithMixedFields>(
      "MyStructWithMixedFields",
      "SReflectTest::MyStructWithMixedFields",
      SReflect::CoreType::CT_struct,
      testValues);

  // Use StructTypeInfo to inspect it
  MyStructWithMixedFields obj;
  SReflect::StructTypeInfo const& info = SReflect::GetTypeInfo<MyStructWithMixedFields>();
  EXPECT_STREQ("MyStructWithMixedFields", info._name);
  EXPECT_STREQ("SReflectTest::MyStructWithMixedFields", info._nameWithNamespace);
  EXPECT_NE(0, info._typeId.value);
  EXPECT_EQ(MyStructWithMixedFields::GetTypeId(), info._typeId);
  EXPECT_EQ(info._typeId, SReflect::GetTypeId<MyStructWithMixedFields>());
  EXPECT_EQ(info._typeId, SReflect::GetFinalTypeId(obj));
  EXPECT_EQ(sizeof(MyStructWithMixedFields), info._sizeInBytes);
  EXPECT_EQ(std::type_index(typeid(MyStructWithMixedFields)), info._typeIndex);
  EXPECT_EQ(4, info._fields.size());
  EXPECT_STREQ("uint8", info.FindField("myByte")->_innerTypeInfo->_name);
  EXPECT_STREQ("string", info.FindField("myString")->_innerTypeInfo->_name);
  EXPECT_STREQ("std::string", info.FindField("myString")->_innerTypeInfo->_nameWithNamespace);
  EXPECT_STREQ("MyTrivialInnerStruct", info.FindField("myStruct")->_innerTypeInfo->_name);
  EXPECT_STREQ(
      "SReflectTest::MyTrivialInnerStruct",
      info.FindField("myStruct")->_innerTypeInfo->_nameWithNamespace);
  EXPECT_STREQ(
      "vector<MyTrivialInnerStruct>", info.FindField("myStructVec")->_innerTypeInfo->_name);
  EXPECT_STREQ(
      "std::vector<SReflectTest::MyTrivialInnerStruct>",
      info.FindField("myStructVec")->_innerTypeInfo->_nameWithNamespace);
  EXPECT_EQ(nullptr, info.FindField("no_such_field"));
}

struct StructWithNamedField {
  int _myValue = 0;
  bool operator==(StructWithNamedField const& rhs) const {
    return _myValue == rhs._myValue;
  }
  SR_BeginStruct(SReflectTest::StructWithNamedField);
  SR_Field_Name(_myValue, "value"); // Exposes name "value" not "_myValue"
  SR_EndStruct();
};

TEST(SReflect, FieldName) {
  StructWithNamedField obj;
  obj._myValue = 123; // Actual member name differs from reflected name
  ExpectJson(R"({"value":123})", SReflect::ToJsonString(obj));
  EXPECT_EQ(obj, SReflect::FromJsonString<StructWithNamedField>(R"({"value":123})"));
}

TEST(SReflect, GetOffsetFromDerivedToBase) {
  struct MyBase1 {
    int b1 = 0;
  };
  struct MyBase2 {
    int b2 = 0;
  };
  struct MyDerived : public MyBase1, public MyBase2 {};

  // Assumptions made about the above classes
  static_assert(sizeof(MyBase1) == sizeof(int));
  static_assert(sizeof(MyBase2) == sizeof(int));
  static_assert(sizeof(MyDerived) == sizeof(MyBase1) + sizeof(MyBase2));

  const ptrdiff_t offsetFromDerivedToBase1 =
      SReflect::GetOffsetFromDerivedToBase<MyDerived, MyBase1>();
  const ptrdiff_t offsetFromDerivedToBase2 =
      SReflect::GetOffsetFromDerivedToBase<MyDerived, MyBase2>();

  if (offsetFromDerivedToBase1 == 0) {
    // The memory layout puts MyBase1 first, and then MyBase2
    EXPECT_EQ(sizeof(MyBase1), offsetFromDerivedToBase2);
  } else {
    // The memory layout puts MyBase2 first, and then MyBase1
    EXPECT_EQ(sizeof(MyBase2), offsetFromDerivedToBase1);
  }

  // Default state
  MyDerived obj;
  EXPECT_EQ(0, obj.b1);
  EXPECT_EQ(0, obj.b2);

  // Use our knowledge of the offsets to modify the int members
  const int kValue1 = 111;
  const int kValue2 = 222;
  memcpy(reinterpret_cast<uint8_t*>(&obj) + offsetFromDerivedToBase1, &kValue1, sizeof(int));
  memcpy(reinterpret_cast<uint8_t*>(&obj) + offsetFromDerivedToBase2, &kValue2, sizeof(int));
  EXPECT_EQ(kValue1, obj.b1);
  EXPECT_EQ(kValue2, obj.b2);
}

namespace ClassTest {
struct MyClass : public SReflect::BaseObject {
  MyClass(int32_t val = 0) : x(val) {};

  int32_t x = 0;

  bool operator==(MyClass const& rhs) const {
    return rhs.x == x;
  }

  SR_BeginStruct(SReflectTest::ClassTest::MyClass);
  SR_Field(x);
  SR_EndStruct();
};
} // namespace ClassTest

TEST(SReflect, Class) {
  using MyClass = ClassTest::MyClass;
  MyClass obj;
  EXPECT_EQ(&SReflect::GetTypeInfo<MyClass>(), &MyClass::GetTypeInfo()); // Same address
  EXPECT_EQ(&SReflect::GetFinalTypeInfo(obj), &MyClass::GetTypeInfo()); // Same address
  EXPECT_EQ(SReflect::GetTypeId<MyClass>(), MyClass::GetTypeInfo()._typeId);
  EXPECT_EQ(SReflect::GetFinalTypeId(obj), MyClass::GetTypeInfo()._typeId);

  std::vector<SRTestValue<MyClass>> testValues = {
      SRTestValue<MyClass>{{123}, "{\"x\":123}"},
      SRTestValue<MyClass>{{456}, "{\"x\":456}"},
  };

  // NOTE: The full name of a function-local class is implementation defined
  TestAnyType<MyClass>("MyClass", nullptr, SReflect::CoreType::CT_struct, testValues);
}

namespace DerivedClassTest {
struct MyBase : public SReflect::BaseObject {
  MyBase(bool b = false, int32_t n = -1) : baseBool(b), baseInt(n) {}

  bool baseBool = false;
  int32_t baseInt = -1;

  SR_BeginStruct(SReflectTest::DerivedClassTest::MyBase);
  SR_Field(baseBool);
  SR_Field(baseInt);
  SR_EndStruct();
};

struct MyDerived : public MyBase {
  MyDerived(std::string s = "", bool b = false, int32_t n = -1) : MyBase(b, n), derivedStr(s) {}
  std::string derivedStr;

  bool operator==(const MyDerived& rhs) const {
    return (rhs.derivedStr == derivedStr) && (rhs.baseBool == baseBool) && (rhs.baseInt == baseInt);
  }

  SR_BeginStruct(SReflectTest::DerivedClassTest::MyDerived);
  SR_BaseClass(MyBase); // Inherits two fields
  SR_Field(derivedStr);
  SR_EndStruct();
};

struct MyOtherClass {
  SR_BeginStruct(SReflectTest::DerivedClassTest::MyOtherClass);
  SR_EndStruct();
};
} // namespace DerivedClassTest

TEST(SReflect, DerivedClass) {
  using MyBase = DerivedClassTest::MyBase;
  using MyDerived = DerivedClassTest::MyDerived;
  using MyOtherClass = DerivedClassTest::MyOtherClass;

  MyBase baseObj;
  MyDerived derivedObj;
  const MyBase& baseDerivedObj = derivedObj;
  EXPECT_EQ(&SReflect::GetTypeInfo<MyBase>(), &MyBase::GetTypeInfo()); // Same address...
  EXPECT_EQ(&SReflect::GetFinalTypeInfo(baseObj), &MyBase::GetTypeInfo());
  EXPECT_EQ(&SReflect::GetTypeInfo<MyDerived>(), &MyDerived::GetTypeInfo());
  EXPECT_EQ(&SReflect::GetFinalTypeInfo(derivedObj), &MyDerived::GetTypeInfo());
  EXPECT_EQ(
      &SReflect::GetFinalTypeInfo(baseDerivedObj),
      &MyDerived::GetTypeInfo()); // Gets the final type, even through reference to base

  EXPECT_EQ(SReflect::GetTypeId<MyBase>(), MyBase::GetTypeId()); // Same address...
  EXPECT_EQ(SReflect::GetFinalTypeId(baseObj), MyBase::GetTypeId());
  EXPECT_EQ(SReflect::GetTypeId<MyDerived>(), MyDerived::GetTypeId());
  EXPECT_EQ(SReflect::GetFinalTypeId(derivedObj), MyDerived::GetTypeId());
  EXPECT_EQ(
      SReflect::GetFinalTypeId(baseDerivedObj),
      MyDerived::GetTypeId()); // Gets the final type, even through reference to base

  // IsSameOrDerivedFrom
  EXPECT_TRUE(SReflect::GetTypeInfo<MyBase>().IsSameOrDerivedFrom<MyBase>());
  EXPECT_TRUE(SReflect::GetTypeInfo<MyDerived>().IsSameOrDerivedFrom<MyDerived>());
  EXPECT_TRUE(SReflect::GetTypeInfo<MyDerived>().IsSameOrDerivedFrom<MyBase>());
  EXPECT_FALSE(SReflect::GetTypeInfo<MyBase>().IsSameOrDerivedFrom<MyDerived>());
  EXPECT_FALSE(SReflect::GetTypeInfo<MyDerived>().IsSameOrDerivedFrom<MyOtherClass>());

  std::vector<SRTestValue<MyDerived>> testValues = {
      SRTestValue<MyDerived>{
          {"one", true, -3}, "{\"baseBool\":true, \"baseInt\":-3, \"derivedStr\":\"one\"}"},
      SRTestValue<MyDerived>{
          {"two", false, 9}, "{\"baseBool\":false, \"baseInt\":9, \"derivedStr\":\"two\"}"},
  };

  // NOTE: The full name of a function-local struct is implementation defined
  TestAnyType<MyDerived>("MyDerived", nullptr, SReflect::CoreType::CT_struct, testValues);
}

namespace StructAttributesTest {
struct MyEmptyStruct {
  SR_BeginStruct(SReflectTest::StructAttributesTest::MyEmptyStruct);
  SR_EndStruct();
};

struct MyStruct {
  SR_BeginStruct(SReflectTest::StructAttributesTest::MyStruct);
  SRA_DisplayName("My Struct");
  SRA_Description("Coolest struct evar!");
  SR_EndStruct();
};
} // namespace StructAttributesTest

TEST(SReflect, StructAttributes) {
  // A struct with no attributes
  {
    using MyEmptyStruct = StructAttributesTest::MyEmptyStruct;
    auto& info = SReflect::GetTypeInfo<MyEmptyStruct>();
    EXPECT_EQ(0, info._attributes.size());
    EXPECT_FALSE(info.HasAttribute<SReflect::Attribute_Description>());
    EXPECT_EQ(nullptr, info.GetAttribute<SReflect::Attribute_Description>());
  }

  // A struct with 2 attributes
  {
    using MyStruct = StructAttributesTest::MyStruct;
    auto& info = SReflect::GetTypeInfo<MyStruct>();
    EXPECT_EQ(2, info._attributes.size());
    EXPECT_TRUE(info.HasAttribute<SReflect::Attribute_DisplayName>());
    EXPECT_STREQ(
        "My Struct", info.GetAttribute<SReflect::Attribute_DisplayName>()->_displayName.c_str());
    EXPECT_TRUE(info.HasAttribute<SReflect::Attribute_Description>());
    EXPECT_STREQ(
        "Coolest struct evar!",
        info.GetAttribute<SReflect::Attribute_Description>()->_description.c_str());
  }
}

namespace FieldAttributesTest {
struct MyStruct {
  int x;
  int y;
  int z;

  SR_BeginStruct(SReflectTest::FieldAttributesTest::MyStruct);
  SR_Field(x);
  SR_Field(y);
  SRA_DisplayName("Fancy Field");
  SR_Field(z);
  SR_EndStruct();
};
} // namespace FieldAttributesTest

TEST(SReflect, FieldAttributes) {
  using MyStruct = FieldAttributesTest::MyStruct;

  auto& info = SReflect::GetTypeInfo<MyStruct>();
  EXPECT_EQ(3, info._fields.size());
  auto* fieldX = info.FindField("x");
  auto* fieldY = info.FindField("y");
  auto* fieldZ = info.FindField("z");
  EXPECT_EQ(nullptr, info.GetAttribute<SReflect::Attribute_DisplayName>());
  EXPECT_EQ(nullptr, fieldX->GetAttribute<SReflect::Attribute_DisplayName>());
  EXPECT_STREQ(
      "Fancy Field", fieldY->GetAttribute<SReflect::Attribute_DisplayName>()->_displayName.c_str());
  EXPECT_EQ(nullptr, fieldZ->GetAttribute<SReflect::Attribute_DisplayName>());
}

struct StructWithCustomTraits {
  int32_t x;
  int32_t y;

  bool operator==(const StructWithCustomTraits& rhs) const {
    return (x == rhs.x) && (y == rhs.y);
  }
};

} // namespace SReflectTest

// This specialization adds custom reflection support for StructWithCustomTraits
template <>
struct SReflectTypeTraits<SReflectTest::StructWithCustomTraits> {
  static constexpr SReflect::CoreType coreType =
      SReflect::CoreType::CT_array; // lie about the type because we can

  static SReflect::TypeInfo const& GetTypeInfo() { // Custom implementation treats it like int32[2]
    static_assert(
        sizeof(SReflectTest::StructWithCustomTraits) == sizeof(int32_t) * 2, "Unexpected padding");
    static SReflect::TypeInfo const* s_typeInfo =
        SReflect::MakeFixedArrayTypeInfo<SReflectTest::StructWithCustomTraits, int32_t, 2>(
            "KindOfLikeTwoInts");
    return *s_typeInfo;
  }
};
namespace SReflectTest {

TEST(SReflect, StructWithCustomTraits) {
  // MyNonintrusiveStruct has custom non-intrusive reflection support (see above) which
  // treats it as an array of int32[2] instead of a struct.
  std::vector<SRTestValue<StructWithCustomTraits>> testValues = {
      {{0, 0}, "[0,0]"},
      {{-123, 456}, "[-123,456]"},
  };
  TestAnyType<StructWithCustomTraits>(
      "KindOfLikeTwoInts", nullptr, SReflect::CoreType::CT_array, testValues);
}

template <class T>
struct TemplateWithCustomTraits {
  T x;
  T y;
  bool operator==(const TemplateWithCustomTraits& rhs) const {
    return (x == rhs.x) && (y == rhs.y);
  }
};

} // namespace SReflectTest

// This specialization adds custom reflection support for TemplateWithCustomTraits
template <typename T>
struct SReflectTypeTraits<SReflectTest::TemplateWithCustomTraits<T>> {
  static constexpr SReflect::CoreType coreType =
      SReflect::CoreType::CT_array; // lie about the type because we can
  static SReflect::TypeInfo const& GetTypeInfo() {
    static_assert(
        sizeof(SReflectTest::TemplateWithCustomTraits<T>) == sizeof(T) * 2, "Unexpected padding");
    static SReflect::TypeInfo const* s_typeInfo = []() {
      // Use custom formatting this time for the name with namespace.
      char const* nameWithNamespace = SReflect::detail::MakeTypeName(
          "SReflectTest::TemplateWithCustomTraits<",
          SReflect::GetTypeInfo<T>()._nameWithNamespace,
          ">");
      return SReflect::MakeFixedArrayTypeInfo<SReflectTest::TemplateWithCustomTraits<T>, T, 2>(
          nameWithNamespace, /*formatAsTemplate*/ false);
    }();
    return *s_typeInfo;
  }
};

namespace SReflectTest {

TEST(SReflect, TemplateWithCustomTraits) {
  std::vector<SRTestValue<TemplateWithCustomTraits<std::array<int, 2>>>> testValues = {
      {{std::array<int, 2>{1, 2}, std::array<int, 2>{3, 4}}, "[[1,2],[3,4]]"},
      {{std::array<int, 2>{5, 6}, std::array<int, 2>{7, 8}}, "[[5,6],[7,8]]"},
  };
  TestAnyType<TemplateWithCustomTraits<std::array<int, 2>>>(
      "TemplateWithCustomTraits<array<int32,2>>", // Both "SReflectTest::" and "std::" were pruned
      "SReflectTest::TemplateWithCustomTraits<std::array<int32,2>>",
      SReflect::CoreType::CT_array,
      testValues);
}

TEST(SReflect, StructWithExternalReflectionDeclaration) {
  // MyDeclaredStructEx uses SR_DeclareStructEx to declare (but not define) reflection markup
  // outside of the class. Reflection support is defined in a separate cpp file.
  std::vector<SRTestValue<MyDeclaredStructEx>> testValues = {
      {{0}, R"({"value":0})"}, {{123}, R"({"value":123})"}};
  TestAnyType<MyDeclaredStructEx>(
      "MyDeclaredStructEx",
      "SReflectTest::MyDeclaredStructEx",
      SReflect::CoreType::CT_struct,
      testValues);
}

TEST(SReflect, StructWithExternalReflectionDefinition) {
  // MyStructEx uses SR_BeginStructEx/SR_EndStructEx to add reflection markup outside of the class
  // declaration. Reflection support is both declared and defined in a header file.
  std::vector<SRTestValue<MyStructEx>> testValues = {
      {{0}, R"({"value":0})"}, {{123}, R"({"value":123})"}};
  TestAnyType<MyStructEx>(
      "MyStructEx", "SReflectTest::MyStructEx", SReflect::CoreType::CT_struct, testValues);
}

TEST(SReflect, TypeNamesInGlobalScope) {
  EXPECT_STREQ(
      "SReflectTest_GlobalStruct", SReflect::GetTypeInfo<SReflectTest_GlobalStruct>()._name);
  EXPECT_STREQ(
      "SReflectTest_GlobalStruct",
      SReflect::GetTypeInfo<SReflectTest_GlobalStruct>()._nameWithNamespace);
  EXPECT_STREQ("SReflectTest_GlobalClass", SReflect::GetTypeInfo<SReflectTest_GlobalClass>()._name);
  EXPECT_STREQ(
      "SReflectTest_GlobalClass",
      SReflect::GetTypeInfo<SReflectTest_GlobalClass>()._nameWithNamespace);
  EXPECT_STREQ("SReflectTest_GlobalEnum", SReflect::GetTypeInfo<SReflectTest_GlobalEnum>()._name);
  EXPECT_STREQ(
      "SReflectTest_GlobalEnum",
      SReflect::GetTypeInfo<SReflectTest_GlobalEnum>()._nameWithNamespace);
  EXPECT_STREQ("SReflectTest_GlobalEnum", SReflect::GetTypeInfo<SReflectTest_GlobalEnum>()._name);
  EXPECT_STREQ(
      "SReflectTest_GlobalEnum",
      SReflect::GetTypeInfo<SReflectTest_GlobalEnum>()._nameWithNamespace);
  EXPECT_STREQ(
      "SReflectTest_GlobalEnumClass", SReflect::GetTypeInfo<SReflectTest_GlobalEnumClass>()._name);
  EXPECT_STREQ(
      "SReflectTest_GlobalEnumClass",
      SReflect::GetTypeInfo<SReflectTest_GlobalEnumClass>()._nameWithNamespace);

  // Expect stable hash values
  EXPECT_EQ(
      0x75EF1C77AF699F8ALLU, SReflect::GetTypeInfo<SReflectTest_GlobalStruct>()._typeId.value);
  EXPECT_EQ(0x1BF5E2D87FF23E45LLU, SReflect::GetTypeInfo<SReflectTest_GlobalClass>()._typeId.value);
  EXPECT_EQ(0xD0CEE8B75503D17DLLU, SReflect::GetTypeInfo<SReflectTest_GlobalEnum>()._typeId.value);
  EXPECT_EQ(
      0xEAE9BD020684ADBELLU, SReflect::GetTypeInfo<SReflectTest_GlobalEnumClass>()._typeId.value);
}

namespace MyNamespace {

struct StructInNamespace {
  SR_BeginStruct(SReflectTest::MyNamespace::StructInNamespace);
  SR_EndStruct();
};

class ClassInNamespace {
 public:
  SR_BeginStruct(SReflectTest::MyNamespace::ClassInNamespace);
  SR_EndStruct();
};

enum EnumInNamespace { kValueInNamespace };

enum class EnumClassInNamespace { kValueInNamespace };

} // namespace MyNamespace

} // namespace SReflectTest
SR_BeginEnum(SReflectTest::MyNamespace::EnumInNamespace);
SR_EnumItem(kValueInNamespace);
SR_EndEnum();

SR_BeginEnum(SReflectTest::MyNamespace::EnumClassInNamespace);
SR_EnumItem(kValueInNamespace);
SR_EndEnum();
namespace SReflectTest {

TEST(SReflect, TypeNamesInNamespaceScope) {
  EXPECT_STREQ("StructInNamespace", SReflect::GetTypeInfo<MyNamespace::StructInNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::MyNamespace::StructInNamespace",
      SReflect::GetTypeInfo<MyNamespace::StructInNamespace>()._nameWithNamespace);
  EXPECT_STREQ("ClassInNamespace", SReflect::GetTypeInfo<MyNamespace::ClassInNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::MyNamespace::ClassInNamespace",
      SReflect::GetTypeInfo<MyNamespace::ClassInNamespace>()._nameWithNamespace);
  EXPECT_STREQ("EnumInNamespace", SReflect::GetTypeInfo<MyNamespace::EnumInNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::MyNamespace::EnumInNamespace",
      SReflect::GetTypeInfo<MyNamespace::EnumInNamespace>()._nameWithNamespace);
  EXPECT_STREQ("EnumInNamespace", SReflect::GetTypeInfo<MyNamespace::EnumInNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::MyNamespace::EnumInNamespace",
      SReflect::GetTypeInfo<MyNamespace::EnumInNamespace>()._nameWithNamespace);
  EXPECT_STREQ(
      "EnumClassInNamespace", SReflect::GetTypeInfo<MyNamespace::EnumClassInNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::MyNamespace::EnumClassInNamespace",
      SReflect::GetTypeInfo<MyNamespace::EnumClassInNamespace>()._nameWithNamespace);

  // Expect stable hash values
  EXPECT_EQ(
      0xCFDE1607DA0B6C3DLLU, SReflect::GetTypeInfo<MyNamespace::StructInNamespace>()._typeId.value);
  EXPECT_EQ(
      0xBDDE9D30183746A8LLU, SReflect::GetTypeInfo<MyNamespace::ClassInNamespace>()._typeId.value);
  EXPECT_EQ(
      0xADB93BEEF0586CBDLLU, SReflect::GetTypeInfo<MyNamespace::EnumInNamespace>()._typeId.value);
  EXPECT_EQ(
      0xEF5A21177D699DB0LLU,
      SReflect::GetTypeInfo<MyNamespace::EnumClassInNamespace>()._typeId.value);
}

namespace {

struct StructInAnonymousNamespace {
  SR_BeginStruct(SReflectTest::StructInAnonymousNamespace);
  SR_EndStruct();
};

class ClassInAnonymousNamespace {
 public:
  SR_BeginStruct(SReflectTest::ClassInAnonymousNamespace);
  SR_EndStruct();
};

enum EnumInAnonymousNamespace { kValueInAnonymousNamespace };

enum class EnumClassInAnonymousNamespace { kValueInAnonymousNamespace };

} // anonymous namespace

} // namespace SReflectTest
SR_BeginEnum(SReflectTest::EnumInAnonymousNamespace);
SR_EnumItem(kValueInAnonymousNamespace);
SR_EndEnum();

SR_BeginEnum(SReflectTest::EnumClassInAnonymousNamespace);
SR_EnumItem(kValueInAnonymousNamespace);
SR_EndEnum();
namespace SReflectTest {

TEST(SReflect, TypeNamesInAnonymousNamespace) {
  // Types declared in an anonymous namespace. These are normally compiler specific, but we
  // standardize them by replacing the anonymous namespace (and everything before it) with the
  // word "namespace".
  EXPECT_STREQ(
      "StructInAnonymousNamespace", SReflect::GetTypeInfo<StructInAnonymousNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::StructInAnonymousNamespace",
      SReflect::GetTypeInfo<StructInAnonymousNamespace>()._nameWithNamespace);
  EXPECT_STREQ(
      "ClassInAnonymousNamespace", SReflect::GetTypeInfo<ClassInAnonymousNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::ClassInAnonymousNamespace",
      SReflect::GetTypeInfo<ClassInAnonymousNamespace>()._nameWithNamespace);
  EXPECT_STREQ("EnumInAnonymousNamespace", SReflect::GetTypeInfo<EnumInAnonymousNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::EnumInAnonymousNamespace",
      SReflect::GetTypeInfo<EnumInAnonymousNamespace>()._nameWithNamespace);
  EXPECT_STREQ("EnumInAnonymousNamespace", SReflect::GetTypeInfo<EnumInAnonymousNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::EnumInAnonymousNamespace",
      SReflect::GetTypeInfo<EnumInAnonymousNamespace>()._nameWithNamespace);
  EXPECT_STREQ(
      "EnumClassInAnonymousNamespace",
      SReflect::GetTypeInfo<EnumClassInAnonymousNamespace>()._name);
  EXPECT_STREQ(
      "SReflectTest::EnumClassInAnonymousNamespace",
      SReflect::GetTypeInfo<EnumClassInAnonymousNamespace>()._nameWithNamespace);

  // Expect stable hash values
  EXPECT_EQ(
      0x5296E6DB491D575BLLU, SReflect::GetTypeInfo<StructInAnonymousNamespace>()._typeId.value);
  EXPECT_EQ(
      0x62D37995E11397D1LLU, SReflect::GetTypeInfo<ClassInAnonymousNamespace>()._typeId.value);
  EXPECT_EQ(0xBFC7C48CA5F78178LLU, SReflect::GetTypeInfo<EnumInAnonymousNamespace>()._typeId.value);
  EXPECT_EQ(
      0x3C02CCC8FA1520B2LLU, SReflect::GetTypeInfo<EnumClassInAnonymousNamespace>()._typeId.value);
}

struct MyStructToSave {
  int32_t value1 = 0;
  int32_t value2 = 0;

  SR_BeginStruct(SReflectTest::MyStructToSave);
  SR_Field(value1);
  SR_Field(value2);
  SR_EndStruct();
};

// This test is disabled by default because it performs file IO.
// Use argument "--gtest_also_run_disabled_tests" to run it locally.
TEST(SReflect, DISABLED_SaveToJsonFile) {
  const char* kTempFileName = "./__simple_reflection_test_temp_file__.json";

  // Save a single value to a new file
  int32_t value = 112358;
  bool result = SReflect::SaveToJsonFile(value, kTempFileName);
  EXPECT_TRUE(result);

  // Load a single value from a file
  int32_t value2 = 0;
  int numIssues = 911;
  result = SReflect::LoadFromJsonFile(
      value2, kTempFileName, SReflect::DeserializeFlags::MaximumWarnings, numIssues);
  EXPECT_TRUE(result);
  EXPECT_EQ(0, numIssues);
  EXPECT_EQ(112358, value2);

  // Save a struct, overwriting a file
  MyStructToSave obj;
  obj.value1 = 12345;
  obj.value2 = 98765;
  result = SReflect::SaveToJsonFile(obj, kTempFileName);
  EXPECT_TRUE(result);

  // Load a struct from a file
  MyStructToSave obj2;
  numIssues = 911;
  result = SReflect::LoadFromJsonFile(
      obj2, kTempFileName, SReflect::DeserializeFlags::MaximumWarnings, numIssues);
  EXPECT_TRUE(result);
  EXPECT_EQ(0, numIssues);
  EXPECT_EQ(12345, obj.value1);
  EXPECT_EQ(98765, obj.value2);

  // Cleanup
  std::remove(kTempFileName);
}

struct BaseWithoutTypeInfo {};

struct BaseWithTypeInfo : BaseWithoutTypeInfo {
  SR_BeginStruct(SReflectTest::BaseWithTypeInfo);
  SR_EndStruct();
};

struct InheritsTypeInfo : public BaseWithTypeInfo {
  // This class inherits reflection functions from its base,
  // but does not override.
};

struct OverridesTypeInfo : public BaseWithTypeInfo {
  // This class overrides the reflection functions of its base, thus
  // defining a new derived type.
  SR_BeginStruct(SReflectTest::OverridesTypeInfo);
  SR_BaseClass(BaseWithTypeInfo);
  SR_EndStruct()
};

TEST(SReflect, HasMemberFn_GetFinalTypeInfo) {
  // This test passes or fails at compile time.
  static_assert(false == SReflect::HasMemberFn_GetFinalTypeInfo<BaseWithoutTypeInfo>::value);
  static_assert(true == SReflect::HasMemberFn_GetFinalTypeInfo<BaseWithTypeInfo>::value);
  static_assert(false == SReflect::HasMemberFn_GetFinalTypeInfo<InheritsTypeInfo>::value);
  static_assert(true == SReflect::HasMemberFn_GetFinalTypeInfo<OverridesTypeInfo>::value);
}

struct StructWithAttribute_DoNotSerialize {
  int normal = 1;
  int noSerialize = 2;

  SR_BeginStruct(SReflectTest::StructWithAttribute_DoNotSerialize);
  SR_Field(normal);
  SR_Field(noSerialize) SRA_DoNotSerialize();
  SR_EndStruct();
};

TEST(SReflect, Attribute_DoNotSerialize) {
  StructWithAttribute_DoNotSerialize obj;

  // Do not serialize the 2nd field
  ExpectJson(R"({"normal": 1})", SReflect::ToJsonString(obj));

  // Do not load the 2nd field
  SReflect::FromJsonString(obj, R"({"normal": 123, "noSerialize": 456})");
  EXPECT_EQ(123, obj.normal); // Loaded
  EXPECT_EQ(2, obj.noSerialize); // Still the default value
}

struct StructWithAttribute_PreviouslyKnownAs {
  int fred = 1;
  int george = 2;

  SR_BeginStruct(SReflectTest::StructWithAttribute_PreviouslyKnownAs);
  SR_Field(fred) SRA_PreviouslyKnownAs("old_fred");
  SR_Field(george) SRA_PreviouslyKnownAs("old_george", "old_george2");
  SR_EndStruct();
};

TEST(SReflect, Attribute_PreviouslyKnownAs) {
  StructWithAttribute_PreviouslyKnownAs obj;

  // The new field names can be saved
  ExpectJson(R"({"fred": 1, "george": 2})", SReflect::ToJsonString(obj));

  // The new field names can be loaded
  SReflect::FromJsonString(obj, R"({"fred": 123, "george": 456})");
  EXPECT_EQ(123, obj.fred);
  EXPECT_EQ(456, obj.george);

  // The old names can also be loaded
  SReflect::FromJsonString(obj, R"({"old_fred": 111, "old_george": 222})");
  EXPECT_EQ(111, obj.fred);
  EXPECT_EQ(222, obj.george);

  // Some fields can have multiple old names
  SReflect::FromJsonString(obj, R"({"old_fred": 333, "old_george2": 444})");
  EXPECT_EQ(333, obj.fred);
  EXPECT_EQ(444, obj.george);
}

// Test enum with previous names (simulating renames: AncientName → OldName → NewName, V2 → Current)
enum class TestEnumPreviouslyKnownAs : uint8_t {
  NewName = 0,
  Current = 1,
};

} // namespace SReflectTest

SR_BeginEnum(SReflectTest::TestEnumPreviouslyKnownAs);
SR_EnumItem(NewName) SRA_PreviouslyKnownAs("OldName", "AncientName");
SR_EnumItem(Current) SRA_PreviouslyKnownAs("V2");
SR_EndEnum();

namespace SReflectTest {

TEST(SReflect, EnumPreviouslyKnownAs) {
  using E = TestEnumPreviouslyKnownAs;
  auto const& info = SReflect::GetTypeInfo<E>();

  // Current names work
  EXPECT_EQ(E::NewName, static_cast<E>(info.FindItemByName("NewName")->_value));
  EXPECT_EQ(E::Current, static_cast<E>(info.FindItemByName("Current")->_value));

  // Multiple previous names in one SRA_PreviouslyKnownAs call
  auto const* item = info.FindItemByName("OldName");
  ASSERT_NE(nullptr, item);
  EXPECT_EQ(static_cast<uint64_t>(E::NewName), item->_value);
  EXPECT_STREQ("NewName", item->_name);

  auto const* item2 = info.FindItemByName("AncientName");
  ASSERT_NE(nullptr, item2);
  EXPECT_EQ(static_cast<uint64_t>(E::NewName), item2->_value);

  // Previous name on a different item
  auto const* item3 = info.FindItemByName("V2");
  ASSERT_NE(nullptr, item3);
  EXPECT_EQ(static_cast<uint64_t>(E::Current), item3->_value);
  EXPECT_STREQ("Current", item3->_name);

  // Unknown name returns nullptr
  EXPECT_EQ(nullptr, info.FindItemByName("NoSuchName"));

  // Previous names are stored as attributes on items, not as separate entries
  EXPECT_EQ(2u, info._items.size());

  // FindItemByValue is unaffected
  EXPECT_STREQ("NewName", info.FindItemByValue(0)->_name);
  EXPECT_STREQ("Current", info.FindItemByValue(1)->_name);
  EXPECT_EQ(nullptr, info.FindItemByValue(99));

  // Serialization uses canonical name
  E value = E::NewName;
  ExpectJson("\"NewName\"", SReflect::ToJsonString(value));

  // Deserialization with previous names
  SReflect::FromJsonString(value, "\"OldName\"");
  EXPECT_EQ(E::NewName, value);
  SReflect::FromJsonString(value, "\"AncientName\"");
  EXPECT_EQ(E::NewName, value);
  SReflect::FromJsonString(value, "\"V2\"");
  EXPECT_EQ(E::Current, value);
}

// TODO: Add tests for other reflection attributes

struct MyNonTrivialInnerStruct {
  int third = 3;
  float fourth = 4.0;
  bool fifth = false;

  SR_BeginStruct(SReflectTest::MyNonTrivialInnerStruct);
  SR_Field(third);
  SR_Field(fourth);
  SR_Field(fifth);
  SR_EndStruct();
};

struct MyOuterStruct {
  int first = 1;
  float second = 2.0;
  MyNonTrivialInnerStruct inner;
  std::string sixth = "six";

  SR_BeginStruct(SReflectTest::MyOuterStruct);
  SR_Field(first);
  SR_Field(second);
  SR_Field(inner);
  SR_Field(sixth);
  SR_EndStruct();
};

TEST(SReflect, DeserializeMalformedJson) {
  // This test makes sure that malformed JSON strings correctly result in a runtime error.
  MyStructToSave obj;
  EXPECT_TRUE(SReflect::FromJsonString(obj, R"({"value1": 123, "value2": 456})")); // Valid
  EXPECT_EQ(123, obj.value1);
  EXPECT_EQ(456, obj.value2);
  obj = {};
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"({"value1": 123 "value2": 456})")); // Missing ,
  EXPECT_FALSE(
      SReflect::FromJsonString(
          obj, R"({"value1": 123, "value2": 123,})")); // Trailing comma (because JSON hates you)
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"({"value1": 123, value2: 456})")); // 2 missing "
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"({"value1": 123, "value2: 123})")); // One missing "
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"({"value1" 123, "value2": 456})")); // Missing :
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"("value1": 123, "value2": 456)")); // Missing {}
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"("value1": 123, "value2": 456})")); // Missing {
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"({"value1": 123, "value2": 456)")); // Missing }
  EXPECT_FALSE(SReflect::FromJsonString(obj, R"({"value1": 123, "value2":})")); // Missing value
}

TEST(SReflect, DeserializeMissingFields) {
  MyOuterStruct obj;
  obj.inner.fourth = 7.0;

  int numIssuesDetected = 0;
  SReflect::FromJsonString(
      obj,
      "{\"second\":8.0,\"inner\":{\"third\":9},\"sixth\":\"ten\"}",
      SReflect::DeserializeFlags::MaximumWarnings,
      numIssuesDetected);

  EXPECT_EQ(3, numIssuesDetected); // 3 fields missing from json string
  EXPECT_EQ(1, obj.first); // unchanged default
  EXPECT_EQ(8.0, obj.second); // changed
  EXPECT_EQ(9, obj.inner.third); // changed
  EXPECT_EQ(7.0, obj.inner.fourth); // unchanged non-default
  EXPECT_EQ(false, obj.inner.fifth); // unchanged default
  EXPECT_EQ("ten", obj.sixth); // changed
}

TEST(SReflect, DeserializeExtraFields) {
  MyOuterStruct obj;
  obj.inner.fourth = 7.0;

  int numIssuesDetected = 0;
  SReflect::FromJsonString(
      obj,
      "{\"second1\":8.0,\"inner\":{\"third2\":9},\"sixth\":\"ten\"}",
      SReflect::DeserializeFlags::MaximumWarnings,
      numIssuesDetected);

  EXPECT_EQ(7, numIssuesDetected); // 5 fields missing from json string, 2 extra fields
  EXPECT_EQ(1, obj.first); // unchanged default
  EXPECT_EQ(2.0, obj.second); // unchanged default
  EXPECT_EQ(3, obj.inner.third); // unchanged default
  EXPECT_EQ(7.0, obj.inner.fourth); // unchanged non-default
  EXPECT_EQ(false, obj.inner.fifth); // unchanged default
  EXPECT_EQ("ten", obj.sixth); // changed
}

struct StructIgnoringExtraneousFields {
  int a = 1;
  int b = 2;
  SR_BeginStruct(SReflectTest::StructIgnoringExtraneousFields);
  SRA_IgnoreExtraneousFields();
  SR_Field(a);
  SR_Field(b);
  SR_EndStruct();
};

struct StructNotIgnoringExtraneousFields {
  int a = 1;
  int b = 2;
  SR_BeginStruct(SReflectTest::StructNotIgnoringExtraneousFields);
  SR_Field(a);
  SR_Field(b);
  SR_EndStruct();
};

TEST(SReflect, Attribute_IgnoreExtraneousFields) {
  char const* json = R"({"a": 10, "b": 20, "extra": 99, "another": "x"})";

  // Without the attribute, the two unknown keys are counted as issues.
  StructNotIgnoringExtraneousFields control;
  int controlIssues = 0;
  SReflect::FromJsonString(
      control, json, SReflect::DeserializeFlags::WarnIfExtraneousFields, controlIssues);
  EXPECT_EQ(2, controlIssues);

  // With the attribute, the unknown keys are tolerated (fields still deserialize).
  StructIgnoringExtraneousFields obj;
  int numIssues = 0;
  SReflect::FromJsonString(
      obj, json, SReflect::DeserializeFlags::WarnIfExtraneousFields, numIssues);
  EXPECT_EQ(0, numIssues);
  EXPECT_EQ(10, obj.a);
  EXPECT_EQ(20, obj.b);
}

struct StructWithJsonStringField {
  std::string payload;
  int tag = 0;
  SR_BeginStruct(SReflectTest::StructWithJsonStringField);
  SR_Field(payload) SRA_JsonString();
  SR_Field(tag);
  SR_EndStruct();
};

struct StructWithJsonStringOnNonString {
  int badField = 0;
  SR_BeginStruct(SReflectTest::StructWithJsonStringOnNonString);
  SR_Field(badField) SRA_JsonString();
  SR_EndStruct();
};

TEST(SReflect, Attribute_JsonString) {
  // Deserialize: a nested JSON object at the tagged field is stored as its serialized text.
  StructWithJsonStringField obj;
  int numIssues = 0;
  SReflect::FromJsonString(
      obj,
      R"({"payload": {"x": 1, "y": 2}, "tag": 7})",
      SReflect::DeserializeFlags::MaximumWarnings,
      numIssues);
  EXPECT_EQ(0, numIssues);
  EXPECT_EQ(7, obj.tag);
  ExpectJson(R"({"x":1,"y":2})", obj.payload);

  // Serialize: the tagged string is emitted as the JSON value it represents (a nested object), not
  // a quoted string.
  ExpectJson(R"({"payload":{"x":1,"y":2},"tag":7})", SReflect::ToJsonString(obj, false));

  // A deserialize/serialize round trip is identity.
  std::string const json1 = SReflect::ToJsonString(obj, false);
  StructWithJsonStringField obj2;
  SReflect::FromJsonString(obj2, json1);
  EXPECT_EQ(json1, SReflect::ToJsonString(obj2, false));
}

TEST(SReflect, Attribute_JsonString_NonStringFieldReportsIssue) {
  // Applying JsonString to a field that cannot accept a string must fail to deserialize a JSON
  // object into it and report an issue -- the conversion routes through the normal string
  // assignment.
  StructWithJsonStringOnNonString obj;
  int numIssues = 0;
  SReflect::FromJsonString(
      obj, R"({"badField": {"x": 1}})", SReflect::DeserializeFlags::None, numIssues);
  EXPECT_NE(0, numIssues);
}

// Templated utility to verify input value can be roundtripped to json and back
template <typename T>
void SerdeJson_Roundtrip_Helper(T const& v1, T& v2) {
  // Serialize
  picojson::value json;
  SReflect::ToJsonValue(v1, json);

  // Deserialize
  int numIssues = 0;
  SReflect::FromJsonValue(v2, json, SReflect::DeserializeFlags::MaximumWarnings, numIssues);
  EXPECT_EQ(numIssues, 0);

  // Compare
  EXPECT_EQ(v1, v2);
}

template <typename T>
void SerdeJson_Roundtrip_Helper(T const& v1) {
  T v2;
  SerdeJson_Roundtrip_Helper(v1, v2);
}

template <typename T>
void SerdeBinary_Roundtrip_Helper(T const& v1, T& v2) {
  // Serialize
  SReflect::VecStreamWriter writer;
  EXPECT_TRUE(SReflect::ToBytes(v1, writer));
  EXPECT_TRUE(writer.GetNumBytesWritten() > 0);

  // Deserialize
  SReflect::SpanStreamReader reader(writer.GetBytes());
  EXPECT_TRUE(SReflect::FromBytes<T>(reader, v2));
  EXPECT_FALSE(reader.HasUnreadBytes());

  // Compare
  EXPECT_EQ(v1, v2);
}

template <typename T>
void SerdeBinary_Roundtrip_Helper(T const& v1) {
  T v2{};
  SerdeBinary_Roundtrip_Helper(v1, v2);
}

// Test if each basic primitive can be roundtripped to bytes and back
TEST(SReflect, SerdeBinary_Primitives) {
  SerdeBinary_Roundtrip_Helper<uint8_t>(0x2A);
  SerdeBinary_Roundtrip_Helper<uint16_t>(0xDEAD);
  SerdeBinary_Roundtrip_Helper<uint32_t>(0xDEADBEEF);
  SerdeBinary_Roundtrip_Helper<uint64_t>(0xDEADBEEFCAFEBABE);

  SerdeBinary_Roundtrip_Helper<int8_t>(0x2A);
  SerdeBinary_Roundtrip_Helper<int16_t>(0x1337);
  SerdeBinary_Roundtrip_Helper<int32_t>(0x1337BEEF);
  SerdeBinary_Roundtrip_Helper<int64_t>(0x1337BEEFCAFEBABE);

  SerdeBinary_Roundtrip_Helper<float>(42.0f);
  SerdeBinary_Roundtrip_Helper<double>(3.14159265359f);
}

// Utility struct for testing binary serializtion of POD types
struct SerdeBinarySimpleStruct {
  bool b;
  uint8_t u8;
  float f;

  SR_BeginStruct(SReflectTest::SerdeBinarySimpleStruct);
  SR_Field(b);
  SR_Field(u8);
  SR_Field(f);
  SR_EndStruct();
};
bool operator==(SerdeBinarySimpleStruct const& a, SerdeBinarySimpleStruct const& b) {
  return a.b == b.b && a.u8 == b.u8 && a.f == b.f;
}

// Test POD struct can be roundtripped
TEST(SReflect, SerdeBinary_SimpleStruct) {
  SerdeBinarySimpleStruct v1;
  v1.b = true;
  v1.u8 = 42;
  v1.f = 13.37f;

  SerdeBinary_Roundtrip_Helper(v1);
}

// Test std::optional for binary serialization
TEST(SReflect, SerdeBinary_Optional) {
  SerdeBinary_Roundtrip_Helper<std::optional<float>>(std::nullopt);
  SerdeBinary_Roundtrip_Helper<std::optional<int32_t>>(0x1337beef);
}

struct SerdeBinaryStringStruct {
  std::string s;

  SR_BeginStruct(SReflectTest::SerdeBinaryStringStruct);
  SR_Field(s);
  SR_EndStruct();
};
bool operator==(SerdeBinaryStringStruct const& a, SerdeBinaryStringStruct const& b) {
  return a.s == b.s;
}

TEST(SReflect, SerdeBinary_String) {
  SerdeBinaryStringStruct v1 = {"hello world"};
  SerdeBinary_Roundtrip_Helper(v1);
}

struct SerdeBinaryVecStruct {
  std::vector<uint16_t> values;

  SR_BeginStruct(SReflectTest::SerdeBinaryVecStruct);
  SR_Field(values);
  SR_EndStruct();
};

TEST(SReflect, SerdeBinary_Vec) {
  SerdeBinaryVecStruct v1 = {{1, 2, 3, 4, 1337}};

  SReflect::VecStreamWriter writer;
  EXPECT_TRUE(SReflect::ToBytes(v1, writer));

  SReflect::SpanStreamReader reader(writer.GetBytes());
  auto v2 = SReflect::FromBytes<SerdeBinaryVecStruct>(reader);
  EXPECT_FALSE(reader.HasUnreadBytes());

  EXPECT_EQ(v1.values.size(), v2.values.size());
  for (size_t i = 0; i < v1.values.size(); ++i) {
    EXPECT_EQ(v1.values[i], v2.values[i]);
  }
}

// TODO: Change to class
struct SerdeBinaryComplexStructA {
  int i;
  float f;
  std::string s;

  SR_BeginStruct(SReflectTest::SerdeBinaryComplexStructA);
  SR_Field(i);
  SR_Field(f);
  SR_Field(s);
  SR_EndStruct();
};
bool operator==(SerdeBinaryComplexStructA const& a, SerdeBinaryComplexStructA const& b) {
  return a.i == b.i && a.f == b.f && a.s == b.s;
}

struct SerdeBinaryComplexStructB : SerdeBinaryComplexStructA {
  bool b;

  SR_BeginStruct(SReflectTest::SerdeBinaryComplexStructB);
  SR_BaseClass(SReflectTest::SerdeBinaryComplexStructA);
  SR_Field(b);
  SR_EndStruct();
};

bool operator==(SerdeBinaryComplexStructB const& a, SerdeBinaryComplexStructB const& b) {
  return a.b == b.b &&
      (*static_cast<SerdeBinaryComplexStructA const*>(&a) ==
       *static_cast<SerdeBinaryComplexStructA const*>(&b));
}

class SerdeBinaryComplexStructC {
 public:
  std::vector<SerdeBinaryComplexStructB> bs;

  SR_BeginClass(SReflectTest::SerdeBinaryComplexStructC);
  SR_Field(bs);
  SR_EndClass();
};
bool operator==(SerdeBinaryComplexStructC const& a, SerdeBinaryComplexStructC const& b) {
  if (a.bs.size() != b.bs.size()) {
    return false;
  }

  for (size_t i = 0; i < a.bs.size(); ++i) {
    if ((a.bs[i] == b.bs[i]) == false) {
      return false;
    }
  }

  return true;
}

TEST(SReflect, SerdeBinary_ComplexStruct) {
  SerdeBinaryComplexStructC v1 = {{
      {{5, 13.37f, "hello"}, true},
      {{42, 31337.f, "goodbye"}, false},
  }};

  SerdeBinary_Roundtrip_Helper(v1);
}

struct SerdeBinaryEnumStruct {
  MyFruit fruit;
  SR_BeginStruct(SReflectTest::SerdeBinaryEnumStruct);
  SR_Field(fruit);
  SR_EndStruct();
};
bool operator==(SerdeBinaryEnumStruct const& a, SerdeBinaryEnumStruct const& b) {
  return a.fruit == b.fruit;
}

TEST(SReflect, SerdeBinary_Enum) {
  SerdeBinaryEnumStruct v1{MyFruit::Kumquat};

  SerdeBinary_Roundtrip_Helper(v1);
}

struct SerdeBinaryPicojsonStruct {
  picojson::value value;

  SR_BeginStruct(SReflectTest::SerdeBinaryPicojsonStruct);
  SR_Field(value);
  SR_EndStruct();
};
bool operator==(SerdeBinaryPicojsonStruct const& a, SerdeBinaryPicojsonStruct const& b) {
  return a.value == b.value;
}

TEST(SReflect, SerdeBinary_Picojson) {
  // number value
  {
    SerdeBinaryPicojsonStruct v1;
    v1.value = picojson::value(13.37f);

    SerdeBinary_Roundtrip_Helper(v1);
  }

  // string value
  {
    SerdeBinaryPicojsonStruct v1;
    v1.value = picojson::value("hello world");

    SerdeBinary_Roundtrip_Helper(v1);
  }
}

TEST(SReflect, SerdeBinary_VectorBool) {
  std::vector<bool> v1 = {true, false, true, false, true, false, true, false, true, false, true};

  SerdeBinary_Roundtrip_Helper(v1);
}

struct SerdeMemCopy_SafeRoot {
  bool b = true;
  int i = 42;
  float f = 13.37f;

  SR_BeginStruct(SReflectTest::SerdeMemCopy_SafeRoot);
  SR_Field(b);
  SR_Field(i);
  SR_Field(f);
  SR_EndStruct();
};
bool operator==(SerdeMemCopy_SafeRoot const& a, SerdeMemCopy_SafeRoot const& b) {
  return a.b == b.b && a.i == b.i && a.f == b.f;
}

struct SerdeMemCopy_SafeNested {
  SerdeMemCopy_SafeRoot a;
  SerdeMemCopy_SafeRoot b;

  SR_BeginStruct(SReflectTest::SerdeMemCopy_SafeNested);
  SR_Field(a);
  SR_Field(b);
  SR_EndStruct();
};
bool operator==(SerdeMemCopy_SafeNested const& a, SerdeMemCopy_SafeNested const& b) {
  return a.a == b.a && a.b == b.b;
}

struct SerdeMemCopy_UnsafeRoot {
  std::string s;

  SR_BeginStruct(SReflectTest::SerdeMemCopy_UnsafeRoot);
  SR_Field(s);
  SR_EndStruct();
};
bool operator==(SerdeMemCopy_UnsafeRoot const& a, SerdeMemCopy_UnsafeRoot const& b) {
  return a.s == b.s;
}

struct SerdeMemCopy_UnsafeNested {
  int x = 7;
  SerdeMemCopy_UnsafeRoot unsafe;

  SR_BeginStruct(SReflectTest::SerdeMemCopy_UnsafeNested);
  SR_Field(x);
  SR_Field(unsafe);
  SR_EndStruct();
};

class SerdeMemCopy_SafeClass {
 public:
  int i = 42;
  float f = 13.37f;

  SR_BeginClass(SReflectTest::SerdeMemCopy_SafeClass);
  SR_Field(i);
  SR_Field(f);
  SR_EndClass();
};

class SerdeMemCopy_UnsafeClass {
  int x = 7;
  SerdeMemCopy_UnsafeRoot unsafe;

  SR_BeginClass(SReflectTest::SerdeMemCopy_UnsafeClass);
  SR_Field(x);
  SR_Field(unsafe);
  SR_EndClass();
};

class SerdeMemCopy_SafeDerived : public SerdeMemCopy_SafeClass {
  bool b = true;

  SR_BeginClass(SReflectTest::SerdeMemCopy_SafeDerived);
  SR_BaseClass(SReflectTest::SerdeMemCopy_SafeClass);
  SR_Field(b);
  SR_EndClass();
};

class SerdeMemCopy_UnsafeVirtual {
 public:
  int x = 7;
  virtual ~SerdeMemCopy_UnsafeVirtual() = default;

  SR_BeginClass(SReflectTest::SerdeMemCopy_UnsafeVirtual);
  SR_Field(x);
  SR_EndClass();
};

TEST(SReflect, Serde_MemCopySafe) {
  // Safe
  EXPECT_TRUE(SReflect::GetFinalTypeInfo((int)0).IsMemCopySafe());
  EXPECT_TRUE(SReflect::GetFinalTypeInfo((float)0).IsMemCopySafe());
  EXPECT_TRUE(SReflect::GetFinalTypeInfo(SerdeMemCopy_SafeRoot{}).IsMemCopySafe());
  EXPECT_TRUE(SReflect::GetFinalTypeInfo(SerdeMemCopy_SafeNested{}).IsMemCopySafe());
  EXPECT_TRUE(SReflect::GetFinalTypeInfo(SerdeMemCopy_SafeClass{}).IsMemCopySafe());
  EXPECT_TRUE(SReflect::GetFinalTypeInfo(SerdeMemCopy_SafeDerived{}).IsMemCopySafe());

  // NOT Safe
  EXPECT_FALSE(SReflect::GetFinalTypeInfo(SerdeMemCopy_UnsafeRoot{}).IsMemCopySafe());
  EXPECT_FALSE(SReflect::GetFinalTypeInfo(SerdeMemCopy_UnsafeNested{}).IsMemCopySafe());
  EXPECT_FALSE(SReflect::GetFinalTypeInfo(SerdeMemCopy_UnsafeClass{}).IsMemCopySafe());
  EXPECT_FALSE(SReflect::GetFinalTypeInfo(SerdeMemCopy_UnsafeVirtual{}).IsMemCopySafe());
}

TEST(SReflect, Serde_BadDeserializeInput) {
  // throws because there are unconsumed bytes
  std::vector<uint8_t> bytes = {0, 1, 2, 3, 4};
  SReflect::SpanStreamReader reader(bytes);

  EXPECT_TRUE(SReflect::FromBytes<uint32_t>(reader));
  EXPECT_EQ(reader.GetNumBytesRemaining(), 1);
}

TEST(SReflect, Serde_IntoSpan) {
  // Create a buffer
  std::array<uint8_t, 2048> buffer;

  // Create multiple serializable types
  float v1 = 73.31f;
  SerdeBinaryComplexStructC v2 = {{
      {{5, 13.37f, "hello"}, true},
      {{42, 31337.f, "goodbye"}, false},
  }};
  SerdeBinaryEnumStruct v3{MyFruit::Kumquat};
  SerdeBinaryPicojsonStruct v4;
  v4.value = picojson::value(13.37f);
  SerdeBinaryPicojsonStruct v5;
  v5.value = picojson::value("hello world");
  SerdeMemCopy_SafeNested v6;
  SerdeMemCopy_UnsafeRoot v7;

  // Serialize them all!
  SReflect::SpanStreamWriter writer(buffer);
  EXPECT_TRUE(SReflect::ToBytes(v1, writer));
  EXPECT_TRUE(SReflect::ToBytes(v2, writer));
  EXPECT_TRUE(SReflect::ToBytes(v3, writer));
  EXPECT_TRUE(SReflect::ToBytes(v4, writer));
  EXPECT_TRUE(SReflect::ToBytes(v5, writer));
  EXPECT_TRUE(SReflect::ToBytes(v6, writer));
  EXPECT_TRUE(SReflect::ToBytes(v7, writer));

  // Deserialze them all
  SReflect::SpanStreamReader reader(writer.GetBytes());
  EXPECT_EQ(v1, SReflect::FromBytes<decltype(v1)>(reader));
  EXPECT_EQ(v2, SReflect::FromBytes<decltype(v2)>(reader));
  EXPECT_EQ(v3, SReflect::FromBytes<decltype(v3)>(reader));
  EXPECT_EQ(v4, SReflect::FromBytes<decltype(v4)>(reader));
  EXPECT_EQ(v5, SReflect::FromBytes<decltype(v5)>(reader));
  EXPECT_EQ(v6, SReflect::FromBytes<decltype(v6)>(reader));
  EXPECT_EQ(v7, SReflect::FromBytes<decltype(v7)>(reader));
}

TEST(SReflect, Serde_IntoSpan_TooSmall) {
  // Create a too small buffer
  std::array<uint8_t, 2> buffer;

  SReflect::SpanStreamWriter writer(buffer);
  EXPECT_FALSE(SReflect::ToBytes((int32_t)5, writer));
}

struct SerdeBinaryArraysStruct {
  std::array<float, 3> a;
  double b[2];

  SR_BeginClass(SReflectTest::SerdeBinaryArraysStruct);
  SR_Field(a);
  SR_Field(b);
  SR_EndClass();
};
bool operator==(SerdeBinaryArraysStruct const& a, SerdeBinaryArraysStruct const& b) {
  for (size_t i = 0; i < a.a.size(); ++i) {
    if (a.a[i] != b.a[i]) {
      return false;
    }
  }

  for (size_t i = 0; i < sizeof(a.b) / sizeof(*a.b); ++i) {
    if (a.b[i] != b.b[i]) {
      return false;
    }
  }
  return true;
}

TEST(SReflect, Serde_Arrays) {
  SerdeBinaryArraysStruct v1 = {{3.f, 1.f, 3.f}, {3.0, 7.0}};

  SerdeBinary_Roundtrip_Helper(v1);
}

#if SR_USE_NLOHMANN_JSON
TEST(SReflect, nlohmann) {
  SerdeBinaryComplexStructC v1 = {{
      {{5, 13.37f, "hello"}, true},
      {{42, 31337.f, "goodbye"}, false},
  }};

  nlohmann::json json;
  SReflect::ToJson(json, v1);

  SerdeBinaryComplexStructC v2;
  SReflect::FromJson(json, v2);

  ASSERT_EQ(v1, v2);
}
#endif

struct PairStruct {
  std::pair<int, float> p1;
  std::pair<int, std::string> p2;
  std::pair<std::string, int> p3;
  std::pair<std::string, std::string> p4;
  std::pair<std::string, std::vector<std::string>> p5;

  SR_BeginStruct(SReflectTest::PairStruct);
  SR_Field(p1);
  SR_Field(p2);
  SR_Field(p3);
  SR_Field(p4);
  SR_Field(p5);
  SR_EndStruct();
};
bool operator==(PairStruct const& a, PairStruct const& b) {
  return a.p1 == b.p1 && a.p2 == b.p2 && a.p3 == b.p3 && a.p4 == b.p4 && a.p5 == b.p5;
}

TEST(SReflect, Serde_Pair) {
  PairStruct ps;
  ps.p1 = {42, 13.37f};
  ps.p2 = {84, "hi there"};
  ps.p3 = {"goodbye", 1337};
  ps.p4 = {"hello", "world"};
  ps.p5 = {"numbers", {"foo", "bar", "baz"}};

  SerdeJson_Roundtrip_Helper(ps);
  SerdeBinary_Roundtrip_Helper(ps);
}

TEST(SReflect, StdUnorderedMap) {
  using MapType = std::unordered_map<std::string, MyStructEx>;
  MapType map;

  // Inspect MapTypeInfo
  SReflect::MapTypeInfo const& ti = SReflect::GetTypeInfo<MapType>();
  EXPECT_EQ(SReflect::CoreType::CT_map, ti._coreType);
  EXPECT_EQ(sizeof(MapType), ti._sizeInBytes);
  EXPECT_EQ(alignof(MapType), ti._alignment);
  EXPECT_STREQ("unordered_map<string,MyStructEx>", ti._name);
  EXPECT_STREQ("std::unordered_map<std::string,SReflectTest::MyStructEx>", ti._nameWithNamespace);
  EXPECT_EQ(&SReflect::GetTypeInfo<std::string>(), ti._keyTypeInfo);
  EXPECT_EQ(&SReflect::GetTypeInfo<MyStructEx>(), ti._valueTypeInfo);
  EXPECT_EQ(0, ti.GetNumKeys(&map));
  ti.Enumerate(&map, [](void const* /*key*/, void* /*value*/) {
    EXPECT_TRUE(false); // Should be empty. Nothing to enumerate.
    return true; // keep going
  });

  // Serialize (empty)
  ExpectJson(R"({})", SReflect::ToJsonString(map, false));
  SerdeJson_Roundtrip_Helper(map);
  SerdeBinary_Roundtrip_Helper(map);

  // Add key-value pairs
  std::string key0 = "answer";
  MyStructEx value0;
  value0.value = 42;
  EXPECT_TRUE(ti.Insert(&map, &key0, &value0));
  std::string key1 = "question";
  MyStructEx value1;
  value1.value = 911;
  EXPECT_TRUE(ti.Insert(&map, &key1, &value1));

  // Inspect native
  EXPECT_EQ(2, map.size());
  EXPECT_EQ(value0, map[key0]);
  EXPECT_EQ(value1, map[key1]);

  // Inspect using reflection
  EXPECT_EQ(2, ti.GetNumKeys(&map));
  int numEnumerated = 0;
  ti.Enumerate(
      static_cast<MapType const*>(&map),
      [&](void const* key, void const* value) { // const enumeration this time
        auto const& typedKey = *reinterpret_cast<std::string const*>(key);
        auto const& typedValue = *reinterpret_cast<MyStructEx const*>(value);
        if (typedKey == "answer") {
          EXPECT_EQ(42, typedValue.value);
        } else {
          EXPECT_STREQ("question", typedKey.c_str());
          EXPECT_EQ(911, typedValue.value);
        }
        ++numEnumerated;
        return true; // keep going
      });
  EXPECT_EQ(2, numEnumerated);
  ExpectJson(
      R"({"answer":{"value":42},"question":{"value":911}})", SReflect::ToJsonString(map, false));
  SerdeJson_Roundtrip_Helper(map);
  SerdeBinary_Roundtrip_Helper(map);

  // Modify a value
  MyStructEx newValue0;
  newValue0.value = -1;
  EXPECT_TRUE(ti.Insert(&map, &key0, &newValue0));
  EXPECT_EQ(2, map.size());
  EXPECT_EQ(newValue0, map[key0]);
  EXPECT_EQ(value1, map[key1]);
  EXPECT_EQ(2, ti.GetNumKeys(&map));
  ExpectJson(
      R"({"answer":{"value":-1},"question":{"value":911}})", SReflect::ToJsonString(map, false));

  // Clone
  void* map2 = ti.Clone(&map);

  // Remove a value
  EXPECT_TRUE(ti.Remove(&map, &key0));
  EXPECT_EQ(1, map.size());
  EXPECT_EQ(value1, map[key1]);
  EXPECT_EQ(1, ti.GetNumKeys(&map));
  ExpectJson(R"({"question":{"value":911}})", SReflect::ToJsonString(map, false));

  // Clear
  EXPECT_TRUE(ti.Clear(&map));
  EXPECT_EQ(0, map.size());
  EXPECT_EQ(0, ti.GetNumKeys(&map));
  ExpectJson(R"({})", SReflect::ToJsonString(map, false));

  // Clone unaffected
  EXPECT_EQ(2, ti.GetNumKeys(map2));
  picojson::value jsonValue = picojson::object();
  ti.Serialize(map2, jsonValue);
  ExpectJson(R"({"answer":{"value":-1},"question":{"value":911}})", jsonValue.serialize(false));

  // Cleanup
  ti.Delete(map2);
}

struct SimpleMapStruct {
  std::unordered_map<int, float> nums;
  std::unordered_map<std::string, std::string> words;

  SR_BeginStruct(SReflectTest::SimpleMapStruct);
  SR_Field(nums);
  SR_Field(words);
  SR_EndStruct();
};
bool operator==(SimpleMapStruct const& a, SimpleMapStruct const& b) {
  return a.nums == b.nums && a.words == b.words;
}
bool operator!=(SimpleMapStruct const& a, SimpleMapStruct const& b) {
  return !(a == b);
}

TEST(SReflect, Serde_Maps) {
  SimpleMapStruct v;
  v.nums[42] = 13.37f;
  v.nums[16] = 9000.0f;
  v.words["hello"] = "world";
  v.words["goodbye"] = "universe";

  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);
}

TEST(SReflect, Serde_MapJsonString) {
  // Initial version
  SimpleMapStruct v1;
  v1.nums[42] = 13.37f;
  v1.words["hello"] = "world";

  // Deserialize from hand written JSON
  // Should be equal
  std::string rawJson1 = R"({
  "nums" : { "42" : 13.37 },
  "words" : { "hello" : "world" }
  })";

  int numIssues = 0;
  SimpleMapStruct v2;
  SReflect::FromJsonString(v2, rawJson1, SReflect::DeserializeFlags::MaximumWarnings, numIssues);
  EXPECT_EQ(numIssues, 0);
  EXPECT_EQ(v1, v2);

  // Deserialize from incorrectly written JSON
  // Should not be equal
  std::string rawJson2 = R"({
  "nums": { "XXXXXXXXXXXXXX" : 42 },
  "words": { }
  })";
  SimpleMapStruct v3;
  SReflect::FromJsonString(v3, rawJson2, SReflect::DeserializeFlags::None, numIssues);
  EXPECT_NE(numIssues, 0);
  EXPECT_NE(v1, v3);
}

struct MapSortedKeys {
  std::unordered_map<int, int> nums;
  std::unordered_map<std::string, std::string> words;

  SR_BeginStruct(SReflectTest::MapSortedKeys);
  SR_Field(nums);
  SR_Field(words);
  SR_EndStruct();
};
bool operator==(MapSortedKeys const& a, MapSortedKeys const& b) {
  return a.nums == b.nums && a.words == b.words;
}

TEST(SReflect, Serde_MapSortedKeys) {
  constexpr size_t kCount = 5;

  // Initial version
  MapSortedKeys v;
  for (int i = 0; i < kCount; ++i) {
    v.nums[i] = i * 2;
  }
  v.words["a"] = "A";
  v.words["b"] = "B";
  v.words["c"] = "C";
  v.words["d"] = "D";
  v.words["e"] = "E";

  // Make sure it round trips
  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);

  std::string json = SReflect::ToJsonString(v, false);
  std::string expectedJson =
      R"({"nums":{"0":0,"1":2,"2":4,"3":6,"4":8},"words":{"a":"A","b":"B","c":"C","d":"D","e":"E"}})";
  EXPECT_EQ(json, expectedJson);
}

struct TypeWithNoDefaultConstructor {
  explicit TypeWithNoDefaultConstructor(int in) : value(in) {}
  int value;
  static int s_destroyCount;

  ~TypeWithNoDefaultConstructor() {
    ++s_destroyCount;
  }

  SR_BeginStruct(SReflectTest::TypeWithNoDefaultConstructor);
  SR_Field(value);
  SR_EndStruct();
};
int TypeWithNoDefaultConstructor::s_destroyCount = 0;

TEST(SReflect, NoDefaultConstructor) {
  // _constructInPlace should be null with no default constructor
  auto const& ti = SReflect::GetTypeInfo<TypeWithNoDefaultConstructor>();
  EXPECT_EQ(nullptr, ti._constructInPlace);
  EXPECT_NE((decltype(ti._constructInPlaceByCopy))nullptr, ti._constructInPlaceByCopy);
  EXPECT_NE((decltype(ti._destructInPlace))nullptr, ti._destructInPlace);

  // But other stuff should still work
  TypeWithNoDefaultConstructor obj(123);
  ExpectJson(R"({"value": 123})", SReflect::ToJsonString(obj));
  alignas(TypeWithNoDefaultConstructor) std::byte cloneBuffer[sizeof(TypeWithNoDefaultConstructor)];
  ti._constructInPlaceByCopy(cloneBuffer, &obj);
  picojson::value cloneJson;
  ti.Serialize(cloneBuffer, cloneJson);
  ExpectJson(R"({"value": 123})", cloneJson.serialize(false));

  // Prove that _destructInPlace calls the destructor.
  TypeWithNoDefaultConstructor::s_destroyCount = 0;
  ti._destructInPlace(cloneBuffer);
  EXPECT_EQ(1, TypeWithNoDefaultConstructor::s_destroyCount);
}

struct HashableFoo {
  HashableFoo() = default;
  HashableFoo(int _x) : x(_x) {}
  int x = 0;

  SR_BeginStruct(SReflectTest::HashableFoo);
  SR_Field(x);
  SR_EndStruct();
};
bool operator==(HashableFoo const& a, HashableFoo const& b) {
  return a.x == b.x;
}

} // namespace SReflectTest

namespace std {
template <>
struct hash<SReflectTest::HashableFoo> {
  size_t operator()(const SReflectTest::HashableFoo& f) const {
    return hash<int>()(f.x);
  }
};
} // namespace std

namespace SReflectTest {

struct MapSortedOk {
  std::unordered_map<HashableFoo, int> foos;

  SR_BeginStruct(SReflectTest::MapSortedOk);
  SR_Field(foos);
  SR_EndStruct();
};
bool operator==(MapSortedOk const& a, MapSortedOk const& b) {
  return a.foos == b.foos;
}

TEST(SReflect, Serde_MapCustomHash) {
  MapSortedOk v;
  v.foos[HashableFoo(7)] = 42;

  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);
}

struct DoNotSerializeDefaults_DoubleNested {
  int magic = 11235;

  bool operator==(DoNotSerializeDefaults_DoubleNested const& other) const {
    return magic == other.magic;
  }

  SR_BeginStruct(SReflectTest::DoNotSerializeDefaults_DoubleNested);
  SR_Field(magic);
  SR_EndStruct()
};

struct DoNotSerializeDefaults_Nested {
  std::string name = "nested";
  DoNotSerializeDefaults_DoubleNested deeper;
  std::optional<DoNotSerializeDefaults_DoubleNested> deeperOptionalStruct;
  std::optional<std::unordered_map<std::string, DoNotSerializeDefaults_DoubleNested>>
      deeperOptionalMap;

  bool operator==(DoNotSerializeDefaults_Nested const& other) const {
    return name == other.name && deeper == other.deeper &&
        deeperOptionalStruct == other.deeperOptionalStruct &&
        deeperOptionalMap == other.deeperOptionalMap;
  }

  SR_BeginStruct(SReflectTest::DoNotSerializeDefaults_Nested);
  SR_Field(name);
  SR_Field(deeper);
  SR_Field(deeperOptionalStruct);
  SR_Field(deeperOptionalMap);
  SR_EndStruct()
};

struct DoNotSerializeDefaults_Struct {
  std::string str = "hello";
  int value = 123;
  DoNotSerializeDefaults_Nested nested;
  DoNotSerializeDefaults_Nested nestedOurDefaults = {"Bob", {911}};
  std::array<std::string, 2> arrayOfStr = {"one", "two"};
  std::vector<std::string> vectorOfStr;
  std::array<DoNotSerializeDefaults_Nested, 2> arrayOfNested = {
      DoNotSerializeDefaults_Nested{"Fred", {42}},
      DoNotSerializeDefaults_Nested{"Fred's friend", {43}}};
  std::vector<DoNotSerializeDefaults_Nested> vectorOfNested;

  bool operator==(DoNotSerializeDefaults_Struct const& other) const {
    return str == other.str && value == other.value && nested == other.nested &&
        nestedOurDefaults == other.nestedOurDefaults && arrayOfStr == other.arrayOfStr &&
        vectorOfStr == other.vectorOfStr && arrayOfNested == other.arrayOfNested &&
        vectorOfNested == other.vectorOfNested;
  }

  SR_BeginStruct(SReflectTest::DoNotSerializeDefaults_Struct);
  SRA_DoNotSerializeDefaults(); // Class attribute
  SR_Field(str);
  SR_Field(value);
  SR_Field(nested);
  SR_Field(nestedOurDefaults);
  SR_Field(arrayOfStr);
  SR_Field(vectorOfStr);
  SR_Field(arrayOfNested);
  SR_Field(vectorOfNested);
  SR_EndStruct()
};

struct DoNotSerializeDefaults_PerField {
  DoNotSerializeDefaults_Nested nested1 = {"Billy", {11}};
  DoNotSerializeDefaults_Nested nested2 = {"Bilbo", {111}};

  bool operator==(DoNotSerializeDefaults_PerField const& other) const {
    return nested1 == other.nested1 && nested2 == other.nested2;
  }

  SR_BeginStruct(SReflectTest::DoNotSerializeDefaults_PerField);
  SR_Field(nested1);
  SR_Field(nested2) SRA_DoNotSerializeDefaults(); // Applies to nested2 (only)
  SR_EndStruct()
};

template <class T>
static void ExpectJsonRoundTrip(T const& obj, std::string const& expectedJson) {
  std::string actualJson = SReflect::ToJsonString(obj, false);
  EXPECT_STREQ(expectedJson.c_str(), actualJson.c_str());
  T reloadedObj{};
  EXPECT_TRUE(
      SReflect::FromJsonString(
          reloadedObj, actualJson, SReflect::DeserializeFlags::WarnIfExtraneousFields));
  EXPECT_EQ(obj, reloadedObj);
}

TEST(SReflect, SRA_DoNotSerializeDefaults_Struct) {
  DoNotSerializeDefaults_Struct obj;

  // Every field has its default value
  ExpectJsonRoundTrip(obj, "{}");

  // Non-default string field
  obj.str = "changed";
  ExpectJsonRoundTrip(obj, R"({"str":"changed"})");

  // Non-default int field
  obj = {};
  obj.value = 911;
  ExpectJsonRoundTrip(obj, R"({"value":911})");

  // Non-default string in nested struct
  obj = {};
  obj.nested.name = "best ever";
  ExpectJsonRoundTrip(obj, R"({"nested":{"name":"best ever"}})");

  // Non-default int in nested struct in nested struct
  obj = {};
  obj.nested.deeper.magic = 42;
  ExpectJsonRoundTrip(obj, R"({"nested":{"deeper":{"magic":42}}})");

  // std::optional field with a value (default is std::nullopt) in a nested struct.
  // That value is a default-constructed struct. It's nested fields should be pruned.
  obj = {};
  obj.nested.deeperOptionalStruct = DoNotSerializeDefaults_DoubleNested{};
  ExpectJsonRoundTrip(obj, R"({"nested":{"deeperOptionalStruct":{}}})");

  // Same as above, but the optional struct contains a non-default field.
  obj = {};
  obj.nested.deeperOptionalStruct = DoNotSerializeDefaults_DoubleNested{};
  obj.nested.deeperOptionalStruct->magic = 54321;
  ExpectJsonRoundTrip(obj, R"({"nested":{"deeperOptionalStruct":{"magic":54321}}})");

  // std::optional field with a value (default is std::nullopt) in a nested struct.
  // That value is a default-constructed map with a prunable value type.
  obj = {};
  obj.nested.deeperOptionalMap =
      std::unordered_map<std::string, DoNotSerializeDefaults_DoubleNested>{};
  ExpectJsonRoundTrip(obj, R"({"nested":{"deeperOptionalMap":{}}})");

  // Same as a bove but the optional map contains a default-constructed key-value pair.
  obj = {};
  obj.nested.deeperOptionalMap =
      std::unordered_map<std::string, DoNotSerializeDefaults_DoubleNested>{};
  (*obj.nested.deeperOptionalMap)[std::string{}] = DoNotSerializeDefaults_DoubleNested{};
  ExpectJsonRoundTrip(obj, R"({"nested":{"deeperOptionalMap":{"":{}}}})");

  // Same as above but the optional map contains a non-default key-value pair.
  obj = {};
  obj.nested.deeperOptionalMap =
      std::unordered_map<std::string, DoNotSerializeDefaults_DoubleNested>{};
  (*obj.nested.deeperOptionalMap)["myKey"] = DoNotSerializeDefaults_DoubleNested{98765};
  ExpectJsonRoundTrip(obj, R"({"nested":{"deeperOptionalMap":{"myKey":{"magic":98765}}}})");

  // Non-default string in nested struct (default comes from outer struct this time)
  obj = {};
  obj.nestedOurDefaults.name = "Sam";
  ExpectJsonRoundTrip(obj, R"({"nestedOurDefaults":{"name":"Sam"}})");

  // Non-default int in nested struct in nested struct (default comes form outer struct)
  obj = {};
  obj.nestedOurDefaults.deeper.magic = 42;
  ExpectJsonRoundTrip(obj, R"({"nestedOurDefaults":{"deeper":{"magic":42}}})");

  // Non-default std::array field
  obj = {};
  obj.arrayOfStr[1] = "new";
  ExpectJsonRoundTrip(obj, R"({"arrayOfStr":["one","new"]})");

  // Non-default std::vector field
  obj = {};
  obj.vectorOfStr.emplace_back("woot");
  ExpectJsonRoundTrip(obj, R"({"vectorOfStr":["woot"]})");

  // Non-default within a std::array of nested structs (defaults come from
  // DoNotSerializeDefaults_Struct)
  obj = {};
  obj.arrayOfNested[0].name = "New Guy";
  ExpectJsonRoundTrip(obj, R"({"arrayOfNested":[{"name":"New Guy"},{}]})");

  // Non-default within a std::array of nested structs (defaults come from
  // DoNotSerializeDefaults_Struct)
  obj = {};
  obj.arrayOfNested[1].deeper.magic = 1111;
  ExpectJsonRoundTrip(obj, R"({"arrayOfNested":[{},{"deeper":{"magic":1111}}]})");

  // Non-default within a std::vector of nested structs (defaults come from
  // DoNotSerializeDefaults_Nested)
  obj = {};
  obj.vectorOfNested.emplace_back(DoNotSerializeDefaults_Nested{"Nancy", {}}); // Non-default name
  obj.vectorOfNested.emplace_back(
      DoNotSerializeDefaults_Nested{"nested", {555}}); // Non-default deeper.magic
  obj.vectorOfNested.emplace_back(
      DoNotSerializeDefaults_Nested{"George", {666}}); // Non-default both
  ExpectJsonRoundTrip(
      obj,
      R"({"vectorOfNested":[{"name":"Nancy"},{"deeper":{"magic":555}},{"deeper":{"magic":666},"name":"George"}]})");

  // Many non-defaults
  obj = {};
  obj.str = "aaa";
  obj.nested.name = "bbb";
  obj.nested.deeper.magic = 111;
  obj.vectorOfStr.emplace_back("ccc");
  obj.arrayOfNested[0].name = "ddd";
  obj.arrayOfNested[1].name = "eee";
  ExpectJsonRoundTrip(
      obj,
      R"({"arrayOfNested":[{"name":"ddd"},{"name":"eee"}],"nested":{"deeper":{"magic":111},"name":"bbb"},"str":"aaa","vectorOfStr":["ccc"]})");
}

TEST(SReflect, SRA_DoNotSerializeDefaults_PerField) {
  // Show that SRA_DoNotSerializeDefaults can be applied to a single field
  DoNotSerializeDefaults_PerField obj;

  // "nested" should be saved even though it matches the default
  // "neste2" should not because of the attribute
  ExpectJsonRoundTrip(obj, R"({"nested1":{"deeper":{"magic":11},"name":"Billy"}})");

  obj.nested2.name = "new";
  ExpectJsonRoundTrip(
      obj, R"({"nested1":{"deeper":{"magic":11},"name":"Billy"},"nested2":{"name":"new"}})");
}

struct DoNotSerializeDefaults_NonRecursiveElem {
  std::string link;
  bool enable = false;

  bool operator==(DoNotSerializeDefaults_NonRecursiveElem const& other) const {
    return link == other.link && enable == other.enable;
  }

  SR_BeginStruct(SReflectTest::DoNotSerializeDefaults_NonRecursiveElem);
  SR_Field(link);
  SR_Field(enable);
  SR_EndStruct();
};

struct DoNotSerializeDefaults_NonRecursive {
  std::vector<DoNotSerializeDefaults_NonRecursiveElem> recursiveOverrides;
  std::vector<DoNotSerializeDefaults_NonRecursiveElem> nonRecursiveOverrides;

  bool operator==(DoNotSerializeDefaults_NonRecursive const& other) const {
    return recursiveOverrides == other.recursiveOverrides &&
        nonRecursiveOverrides == other.nonRecursiveOverrides;
  }

  SR_BeginStruct(SReflectTest::DoNotSerializeDefaults_NonRecursive);
  SR_Field(recursiveOverrides) SRA_DoNotSerializeDefaults(); // recursive (default)
  SR_Field(nonRecursiveOverrides) SRA_DoNotSerializeDefaults(false); // non-recursive
  SR_EndStruct();
};

TEST(SReflect, SRA_DoNotSerializeDefaults_NonRecursive) {
  // SRA_DoNotSerializeDefaults(false) omits the field when it equals its default (e.g. an empty
  // container), but serializes it in full when present -- WITHOUT pruning element sub-fields that
  // happen to equal their defaults (e.g. enable == false).
  DoNotSerializeDefaults_NonRecursive obj;

  // Both containers empty -> both omitted, regardless of the recursive flag.
  ExpectJsonRoundTrip(obj, "{}");

  // Recursive field: an element whose "enable" matches the default (false) has it pruned.
  obj = {};
  obj.recursiveOverrides.push_back({"a_b", false});
  ExpectJsonRoundTrip(obj, R"({"recursiveOverrides":[{"link":"a_b"}]})");

  // Non-recursive field: the same element keeps "enable":false because sub-fields are not pruned.
  obj = {};
  obj.nonRecursiveOverrides.push_back({"a_b", false});
  ExpectJsonRoundTrip(obj, R"({"nonRecursiveOverrides":[{"enable":false,"link":"a_b"}]})");
}

struct BaseWithFieldsYouMayNotWant {
  int a = 1;
  int b = 2;
  int c = 3;

  SR_BeginStruct(SReflectTest::BaseWithFieldsYouMayNotWant);
  SR_Field(a);
  SR_Field(b);
  SR_Field(c);
  SR_EndStruct();
};

struct StructWithFieldsRemoved : BaseWithFieldsYouMayNotWant {
  std::string newA = "hello";
  int d = 4;

  // For ExpectJsonRoundTrip
  bool operator==(const StructWithFieldsRemoved& other) const {
    // Ignore inherited a and c
    return newA == other.newA && b == other.b && d == other.d;
  }

  SR_BeginStruct(SReflectTest::StructWithFieldsRemoved);

  // Inherit fields, but hide "a" and "c"
  SR_BaseClass(SReflectTest::BaseWithFieldsYouMayNotWant);
  SR_RemoveField("a");
  SR_RemoveField("c");

  // This field takes the place of the inherited "a"
  SR_Field_Name(newA, "a");

  // This field is a new addition
  SR_Field(d);
  SR_EndStruct();
};

TEST(SReflect, RemoveField) {
  StructWithFieldsRemoved obj;
  auto& ti = SReflect::GetTypeInfo<StructWithFieldsRemoved>();

  // Check the fields
  EXPECT_EQ(3, ti._fields.size());
  EXPECT_STREQ("b", ti._fields[0]->_name); // Inherited
  EXPECT_STREQ("a", ti._fields[1]->_name); // Now exposing obj.newA
  EXPECT_EQ(SReflect::CoreType::CT_string, ti._fields[1]->_innerTypeInfo->_coreType);
  EXPECT_STREQ("d", ti._fields[2]->_name); // Normal field

  // Check serialization
  ExpectJsonRoundTrip(obj, R"({"a":"hello","b":2,"d":4})");
}

// Helper for testing AppendTemplateArgStr.
// NOTE: This can't be a variadic template the list includes non-types.
#define SRTEST_FORMAT_TARGS(...)                              \
  ([]() {                                                     \
    std::string str;                                          \
    SReflect::detail::AppendTemplateArgStr<__VA_ARGS__>(str); \
    return str;                                               \
  }())                                                        \
      .c_str()

TEST(SReflect, AppendTemplateArgStr) {
  constexpr int kExampleValue = 911;
  using StdStringCantHide = std::string;
  using MyFruitCantHide = MyFruit;

  // clang-format off

  // type
  EXPECT_STREQ("int32,", SRTEST_FORMAT_TARGS(int));
  EXPECT_STREQ("std::string,", SRTEST_FORMAT_TARGS(std::string));
  EXPECT_STREQ("std::string,", SRTEST_FORMAT_TARGS(StdStringCantHide));
  EXPECT_STREQ("SReflectTest::MyFruit,", SRTEST_FORMAT_TARGS(MyFruit));
  EXPECT_STREQ("SReflectTest::MyFruit,", SRTEST_FORMAT_TARGS(MyFruitCantHide));
  EXPECT_STREQ("std::vector<int32>,", SRTEST_FORMAT_TARGS(std::vector<int>));

  // value
  EXPECT_STREQ("123,", SRTEST_FORMAT_TARGS(123));
  EXPECT_STREQ("911,", SRTEST_FORMAT_TARGS(kExampleValue));
  EXPECT_STREQ("kValue,", SRTEST_FORMAT_TARGS(kValue)); // plain enum, global namespace
  EXPECT_STREQ("SReflectTest::MyNamespace::kValueInNamespace,", SRTEST_FORMAT_TARGS(MyNamespace::kValueInNamespace)); // plain enum, user namespace
  EXPECT_STREQ("SReflectTest_GlobalEnumClass::kValue,", SRTEST_FORMAT_TARGS(SReflectTest_GlobalEnumClass::kValue)); // enum class, global namespace
  EXPECT_STREQ("SReflectTest::MyFruit::Orange,", SRTEST_FORMAT_TARGS(MyFruit::Orange)); // enum class, user namespace

  // type, type
  EXPECT_STREQ("int32,float,", SRTEST_FORMAT_TARGS(int, float));
  EXPECT_STREQ("float,std::pair<std::string,uint16>,", SRTEST_FORMAT_TARGS(float, std::pair<std::string, uint16_t>));

  // type, value
  EXPECT_STREQ("std::string,42,", SRTEST_FORMAT_TARGS(std::string, 42));

  // value, type
  EXPECT_STREQ("42,std::string,", SRTEST_FORMAT_TARGS(42, std::string));

  // value, value
  EXPECT_STREQ("123,456,", SRTEST_FORMAT_TARGS(123, 456));

  // type, type, type, (type)
  EXPECT_STREQ("uint8,uint16,uint32,", SRTEST_FORMAT_TARGS(uint8_t, uint16_t, uint32_t));
  EXPECT_STREQ("uint8,uint16,uint32,uint64,", SRTEST_FORMAT_TARGS(uint8_t, uint16_t, uint32_t, uint64_t));

  // type, type, value, (type)
  EXPECT_STREQ("uint8,uint16,42,", SRTEST_FORMAT_TARGS(uint8_t, uint16_t, 42));
  EXPECT_STREQ("uint8,uint16,42,uint64,", SRTEST_FORMAT_TARGS(uint8_t, uint16_t, 42, uint64_t));

  // type, value, type, (type)
  EXPECT_STREQ("uint8,42,uint32,", SRTEST_FORMAT_TARGS(uint8_t, 42, uint32_t));
  EXPECT_STREQ("uint8,42,uint32,uint64,", SRTEST_FORMAT_TARGS(uint8_t, 42, uint32_t, uint64_t));

  // type, value, value, (type)
  EXPECT_STREQ("uint8,42,7,uint64,", SRTEST_FORMAT_TARGS(uint8_t, 42, 7, uint64_t));
  EXPECT_STREQ("uint8,42,7,", SRTEST_FORMAT_TARGS(uint8_t, 42, 7));

  // value, type, type, (type)
  EXPECT_STREQ("42,uint16,uint32,", SRTEST_FORMAT_TARGS(42, uint16_t, uint32_t));
  EXPECT_STREQ("42,uint16,uint32,uint64,", SRTEST_FORMAT_TARGS(42, uint16_t, uint32_t, uint64_t));

  // value, type, value, (type)
  EXPECT_STREQ("42,uint16,7,", SRTEST_FORMAT_TARGS(42, uint16_t, 7));
  EXPECT_STREQ("42,uint16,7,uint64,", SRTEST_FORMAT_TARGS(42, uint16_t, 7, uint64_t));

  // value, value, type, (type)
  EXPECT_STREQ("42,7,uint32,", SRTEST_FORMAT_TARGS(42, 7, uint32_t));
  EXPECT_STREQ("42,7,uint32,uint64,", SRTEST_FORMAT_TARGS(42, 7, uint32_t, uint64_t));

  // value, value, value, (type)
  EXPECT_STREQ("42,7,-1,", SRTEST_FORMAT_TARGS(42, 7, -1));
  EXPECT_STREQ("42,7,-1,uint64,", SRTEST_FORMAT_TARGS(42, 7, -1, uint64_t));

  // clang-format on
}

template <class T, int N>
struct MyArray {
  std::array<T, N> values;

  SR_BeginStructTemplate(SReflectTest::MyArray, T, N);
  SR_Field(values);
  SR_EndStruct();
};

template <class A, auto B, auto C>
class MyTemplateClass {
 public:
 private:
  SR_BeginClassTemplate(SReflectTest::MyTemplateClass, A, B, C);
  SR_EndClass();
};

template <class... Args>
struct MyVariadicTemplate {
  SR_BeginStructTemplate(SReflectTest::MyVariadicTemplate, Args...);
  SR_EndStruct();
};

TEST(SReflect, SR_BeginStructTemplate) {
  // NOTE: This function does not attempt to test all the various combinations of type and non-type
  // arguments. Those variations matter for detail::AppendTemplateArgStr, which is tested
  // separately.

  {
    auto const& ti = SReflect::GetTypeInfo<MyArray<std::string, 4>>();
    EXPECT_EQ(SReflect::CoreType::CT_struct, ti._coreType);
    EXPECT_STREQ("SReflectTest::MyArray<std::string,4>", ti._nameWithNamespace);
    EXPECT_STREQ("MyArray<string,4>", ti._name);
    EXPECT_EQ(1, ti._fields.size());
    EXPECT_STREQ("values", ti._fields[0]->_name);
  }

  {
    using Arg0 = bool;
    auto constexpr Arg1 = MyNamespace::kValueInNamespace;
    auto constexpr Arg2 = 42;
    using T = MyTemplateClass<Arg0, Arg1, Arg2>;
    auto const& ti = SReflect::GetTypeInfo<T>();
    EXPECT_EQ(SReflect::CoreType::CT_struct, ti._coreType);
    EXPECT_STREQ(
        "SReflectTest::MyTemplateClass<bool,SReflectTest::MyNamespace::kValueInNamespace,42>",
        ti._nameWithNamespace);
    EXPECT_STREQ("MyTemplateClass<bool,kValueInNamespace,42>", ti._name);
  }

  {
    using T = MyVariadicTemplate<bool>;
    auto const& ti = SReflect::GetTypeInfo<T>();
    EXPECT_EQ(SReflect::CoreType::CT_struct, ti._coreType);
    EXPECT_STREQ("SReflectTest::MyVariadicTemplate<bool>", ti._nameWithNamespace);
    EXPECT_STREQ("MyVariadicTemplate<bool>", ti._name);
  }

  {
    // Lots of type arguments
    using T = MyVariadicTemplate<
        std::string,
        float,
        std::pair<std::string, double>,
        int8_t,
        int16_t,
        int32_t,
        int64_t>;
    auto const& ti = SReflect::GetTypeInfo<T>();
    EXPECT_EQ(SReflect::CoreType::CT_struct, ti._coreType);
    EXPECT_STREQ(
        "SReflectTest::MyVariadicTemplate<std::string,float,std::pair<std::string,double>,int8,int16,int32,int64>",
        ti._nameWithNamespace);
    EXPECT_STREQ(
        "MyVariadicTemplate<string,float,pair<string,double>,int8,int16,int32,int64>", ti._name);
  }

  // Verify TypeId uniqueness
  EXPECT_NE(
      (SReflect::GetTypeId<MyArray<int32_t, 2>>()),
      (SReflect::GetTypeId<MyArray<uint32_t, 2>>())); // Different type
  EXPECT_NE(
      (SReflect::GetTypeId<MyArray<int32_t, 2>>()),
      (SReflect::GetTypeId<MyArray<int32_t, 3>>())); // Different non-type value
  EXPECT_NE(
      (SReflect::GetTypeId<MyVariadicTemplate<int32_t>>()),
      (SReflect::GetTypeId<MyVariadicTemplate<uint32_t>>())); // Different type
  EXPECT_NE(
      (SReflect::GetTypeId<MyVariadicTemplate<int32_t, int64_t, int16_t>>()),
      (SReflect::GetTypeId<MyVariadicTemplate<int32_t, int16_t, int64_t>>())); // Different order
  EXPECT_NE(
      (SReflect::GetTypeId<MyVariadicTemplate<int32_t>>()),
      (SReflect::GetTypeId<MyVariadicTemplate<int32_t, int32_t>>())); // Different number
}

TEST(SReflect, IsSupportedType) {
  struct UnknownStruct {};
  enum class UknownEnum {};
  static_assert(!SReflect::IsSupportedType<float*>());
  static_assert(!SReflect::IsSupportedType<float&>());
  static_assert(!SReflect::IsSupportedType<UnknownStruct>());
  static_assert(!SReflect::IsSupportedType<UknownEnum>());

  static_assert(SReflect::IsSupportedType<uint8_t>());
  static_assert(SReflect::IsSupportedType<int8_t>());
  static_assert(SReflect::IsSupportedType<uint16_t>());
  static_assert(SReflect::IsSupportedType<int16_t>());
  static_assert(SReflect::IsSupportedType<uint32_t>());
  static_assert(SReflect::IsSupportedType<int32_t>());
  static_assert(SReflect::IsSupportedType<uint64_t>());
  static_assert(SReflect::IsSupportedType<int64_t>());
  static_assert(SReflect::IsSupportedType<std::string>());
  static_assert(SReflect::IsSupportedType<std::array<std::string, 3>>());
  static_assert(SReflect::IsSupportedType<std::vector<std::string>>());
  static_assert(SReflect::IsSupportedType<std::optional<int>>());
  static_assert(SReflect::IsSupportedType<std::unordered_map<int, int>>());
  static_assert(SReflect::IsSupportedType<SReflectTest::MyStructWithMixedFields>());
  static_assert(SReflect::IsSupportedType<SReflectTest::MyFruit>());
}

TEST(SReflect, TryGetStructTypeInfo) {
  struct UnknownStruct {};
  enum class UknownEnum {};

  EXPECT_EQ((SReflect::TypeInfo const*)nullptr, SReflect::TryGetTypeInfo<float*>());
  EXPECT_EQ((SReflect::TypeInfo const*)nullptr, SReflect::TryGetTypeInfo<float&>());
  EXPECT_EQ((SReflect::TypeInfo const*)nullptr, SReflect::TryGetTypeInfo<UnknownStruct>());
  EXPECT_EQ((SReflect::TypeInfo const*)nullptr, SReflect::TryGetTypeInfo<UknownEnum>());

  // clang-format off
  EXPECT_EQ((&SReflect::GetTypeInfo<uint8_t>()), (SReflect::TryGetTypeInfo<uint8_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<int8_t>()), (SReflect::TryGetTypeInfo<int8_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<uint16_t>()), (SReflect::TryGetTypeInfo<uint16_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<int16_t>()), (SReflect::TryGetTypeInfo<int16_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<uint32_t>()), (SReflect::TryGetTypeInfo<uint32_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<int32_t>()), (SReflect::TryGetTypeInfo<int32_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<uint64_t>()), (SReflect::TryGetTypeInfo<uint64_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<int64_t>()), (SReflect::TryGetTypeInfo<int64_t>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<std::string>()), (SReflect::TryGetTypeInfo<std::string>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<std::array<std::string, 3>>()), (SReflect::TryGetTypeInfo<std::array<std::string, 3>>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<std::vector<std::string>>()), (SReflect::TryGetTypeInfo<std::vector<std::string>>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<std::optional<int>>()), (SReflect::TryGetTypeInfo<std::optional<int>>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<std::unordered_map<int, int>>()), (SReflect::TryGetTypeInfo<std::unordered_map<int, int>>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<SReflectTest::MyStructWithMixedFields>()), (SReflect::TryGetTypeInfo<SReflectTest::MyStructWithMixedFields>()));
  EXPECT_EQ((&SReflect::GetTypeInfo<SReflectTest::MyFruit>()), (SReflect::TryGetTypeInfo<SReflectTest::MyFruit>()));
  // clang-format on
}

// Used to test various types of matrices
template <
    typename T,
    bool kIsRowMajor = true,
    bool kIsRunRowsDynamic = true,
    bool kIsRunColumnsDynamic = true>
class TestMatrix {
 public:
  SReflect::MatrixTypeInfo::Layout layout;
  std::vector<T> data;

  bool operator==(TestMatrix const& rhs) const {
    return (layout._numRows == rhs.layout._numRows) &&
        (layout._numColumns == rhs.layout._numColumns) &&
        (layout._leadingDim == rhs.layout._leadingDim ||
         !(layout._leadingDim *
           rhs.layout._leadingDim)) && // Allow inequality if leading dim is auto-calculated
        (data == rhs.data);
  }
};

// MatrixTypeInfo derivation for TestMatrix
template <
    typename T,
    bool kIsRowMajor = true,
    bool kIsRunRowsDynamic = true,
    bool kIsRunColumnsDynamic = true>
class TestMatrixTypeInfo final : public SReflect::MatrixTypeInfo {
 public:
  using TestMatrixT = TestMatrix<T, kIsRowMajor, kIsRunRowsDynamic, kIsRunColumnsDynamic>;

  // Could be modified to trigger an error in TryResize
  mutable bool _canResizeRows = kIsRunRowsDynamic;
  mutable bool _canResizeColumns = kIsRunColumnsDynamic;

  TestMatrixTypeInfo() {
    // TypeInfo fields
    _coreType = SReflect::CoreType::CT_matrix;
    _alignment = alignof(TestMatrixT);
    _sizeInBytes = sizeof(TestMatrixT);
    _name = ""; // Not used by this test
    _nameWithNamespace = ""; // Not used by this test
    _typeId = SReflect::ComputeTypeId(_nameWithNamespace);

    // MatrixTypeInfo fields
    _innerTypeInfo = &SReflect::GetTypeInfo<T>();
    _isRowMajor = kIsRowMajor;
    _isNumRowsDynamic = kIsRunRowsDynamic;
    _isNumColumnsDynamic = kIsRunColumnsDynamic;
  }

  // SReflect::MatrixTypeInfo overrides
  Layout GetLayoutImpl(void const* obj) const override {
    return GetMat(obj)->layout;
  }
  void* GetData(void* obj) const override {
    return GetMat(obj)->data.data();
  }
  bool TryResize(void* obj, size_t numRows, size_t numColumns) const override {
    auto* mat = GetMat(obj);
    if ((numRows != mat->layout._numRows) && !_canResizeRows) {
      return false; // Can't resize rows
    }
    if ((numColumns != mat->layout._numColumns) && !_canResizeColumns) {
      return false; // Can't resize columns
    }
    // Actual matrix classes probably won't allow resizing when the leading dimension is different
    // from the number of rows or columns. That typically only happens when it is a non-owning view
    // matrix (e.g. a view of a subset of a larger matrix). For this test, we can handle it anyway.
    mat->layout._numRows = numRows;
    mat->layout._numColumns = numColumns;
    mat->layout._leadingDim = _isRowMajor ? numColumns : numRows;
    mat->data.clear(); // We don't claim to preserve values.
    mat->data.resize(numRows * numColumns);
    return true;
  }

 private:
  TestMatrixT* GetMat(void* obj) const {
    return static_cast<TestMatrixT*>(obj);
  }
  TestMatrixT const* GetMat(void const* obj) const {
    return static_cast<TestMatrixT const*>(obj);
  }
};

} // namespace SReflectTest

template <typename T, bool kIsRowMajor, bool kIsRunRowsDynamic, bool kIsRunColumnsDynamic>
struct SReflectTypeTraits<
    SReflectTest::TestMatrix<T, kIsRowMajor, kIsRunRowsDynamic, kIsRunColumnsDynamic>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_matrix;
  static auto const& GetTypeInfo() {
    static const SReflectTest::
        TestMatrixTypeInfo<T, kIsRowMajor, kIsRunRowsDynamic, kIsRunColumnsDynamic>
            s_typeInfo;
    return s_typeInfo;
  }
};

// Check IsSupportedType
static_assert(SReflect::IsSupportedType<SReflectTest::TestMatrix<int, true, true, true>>());
static_assert(
    SReflect::IsSupportedType<SReflectTest::TestMatrix<std::string, false, true, false>>());

namespace SReflectTest {

TEST(SReflect, Matrix_GetLayout) {
  TestMatrixTypeInfo<int> derivedInfo;
  SReflect::MatrixTypeInfo const& ti = derivedInfo; // Access via generic base
  TestMatrix<int> m;
  auto const& cm = m;

  // Empty layout
  m.layout._numRows = 0;
  m.layout._numColumns = 0;
  m.layout._leadingDim = 0;
  auto layout = ti.GetLayout(&cm);
  EXPECT_EQ(0, layout._numRows);
  EXPECT_EQ(0, layout._numColumns);
  EXPECT_EQ(0, layout._leadingDim);

  // Non-empty layout
  m.layout._numRows = 2;
  m.layout._numColumns = 3;
  m.layout._leadingDim = 3;
  layout = ti.GetLayout(&cm);
  EXPECT_EQ(2, layout._numRows);
  EXPECT_EQ(3, layout._numColumns);
  EXPECT_EQ(3, layout._leadingDim);
}

TEST(SReflect, Matrix_GetData) {
  TestMatrixTypeInfo<int> derivedInfo;
  SReflect::MatrixTypeInfo const& ti = derivedInfo; // Access via generic base
  TestMatrix<int> m;
  auto const& cm = m;

  // No data
  EXPECT_EQ((void*)nullptr, ti.GetData(&m));
  EXPECT_EQ((void*)nullptr, ti.GetData(&cm));

  // Valid data
  m.data.resize(8);
  EXPECT_EQ(m.data.data(), ti.GetData(&m));
  EXPECT_EQ(m.data.data(), ti.GetData(&cm));
}

TEST(SReflect, MatrixT_TryResize) {
  // Fixed rows, fixed columns
  {
    TestMatrix<int, true, false, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.data.resize(2 * 3);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(true, ti.TryResize(&m, 2, 3)); // same size
    EXPECT_EQ(false, ti.TryResize(&m, 3, 3)); // not allowed to change rows
    EXPECT_EQ(false, ti.TryResize(&m, 2, 4)); // not allowed to change columns
    EXPECT_EQ(2 * 3, m.data.size()); // no change
  }

  // Fixed rows, dynamic columns
  {
    TestMatrix<int, true, false, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.data.resize(2 * 3);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(true, ti.TryResize(&m, 2, 3)); // same size
    EXPECT_EQ(false, ti.TryResize(&m, 3, 3)); // not allowed to change rows
    EXPECT_EQ(true, ti.TryResize(&m, 2, 4)); // successfully change columns
    EXPECT_EQ(4 * 2, m.data.size());
  }

  // Dynamic rows, fixed columns
  {
    TestMatrix<int, true, true, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.data.resize(2 * 3);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(true, ti.TryResize(&m, 2, 3)); // same size
    EXPECT_EQ(false, ti.TryResize(&m, 2, 4)); // not allowed to change columns
    EXPECT_EQ(true, ti.TryResize(&m, 3, 3)); // successfully change rows
    EXPECT_EQ(3 * 3, m.data.size());
  }

  // Dynamic rows, dynamic columns
  {
    TestMatrix<int, true, true, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.data.resize(2 * 3);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(true, ti.TryResize(&m, 2, 3)); // same size
    EXPECT_EQ(true, ti.TryResize(&m, 2, 4)); // successfully change columns
    EXPECT_EQ(2 * 4, m.data.size());
    EXPECT_EQ(true, ti.TryResize(&m, 3, 4)); // successfully change rows
    EXPECT_EQ(3 * 4, m.data.size());
    EXPECT_EQ(true, ti.TryResize(&m, 1, 5)); // successfully change both
    EXPECT_EQ(1 * 5, m.data.size());
  }
}

TEST(SReflect, Matrix_GetElement) {
  // Row-major
  // [0, 1, 2]
  // [3, 4, 5]
  {
    TestMatrix<int, true> m;
    auto const& cm = m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.layout._leadingDim = 0; // Auto-computed
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(&m.data[0], ti.GetElement(&cm, 0, 0));
    EXPECT_EQ(&m.data[1], ti.GetElement(&cm, 0, 1));
    EXPECT_EQ(&m.data[2], ti.GetElement(&cm, 0, 2));
    EXPECT_EQ(&m.data[3], ti.GetElement(&cm, 1, 0));
    EXPECT_EQ(&m.data[4], ti.GetElement(&cm, 1, 1));
    EXPECT_EQ(&m.data[5], ti.GetElement(&cm, 1, 2));
  }

  // Column-major
  // [0, 2, 4]
  // [1, 3, 5]
  {
    TestMatrix<int, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.layout._leadingDim = 0; // Auto-computed
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(&m.data[0], ti.GetElement(&m, 0, 0)); // access via non-const m this time
    EXPECT_EQ(&m.data[2], ti.GetElement(&m, 0, 1));
    EXPECT_EQ(&m.data[4], ti.GetElement(&m, 0, 2));
    EXPECT_EQ(&m.data[1], ti.GetElement(&m, 1, 0));
    EXPECT_EQ(&m.data[3], ti.GetElement(&m, 1, 1));
    EXPECT_EQ(&m.data[5], ti.GetElement(&m, 1, 2));
  }

  // Row-major with padding
  // [0, 1], padding,
  // [3, 4], padding,
  {
    TestMatrix<int, true> m;
    auto const& cm = m;
    m.layout._numRows = 2;
    m.layout._numColumns = 2;
    m.layout._leadingDim = 3; // Larger than _numColumns
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(&m.data[0], ti.GetElement(&cm, 0, 0));
    EXPECT_EQ(&m.data[1], ti.GetElement(&cm, 0, 1));
    EXPECT_EQ(&m.data[3], ti.GetElement(&cm, 1, 0));
    EXPECT_EQ(&m.data[4], ti.GetElement(&cm, 1, 1));
  }

  // Column-major with padding
  // [0, 3]
  // [1, 4]
  // [padding]
  {
    TestMatrix<int, false> m;
    auto const& cm = m;
    m.layout._numRows = 2;
    m.layout._numColumns = 2;
    m.layout._leadingDim = 3; // Larger than _numRows
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_EQ(&m.data[0], ti.GetElement(&cm, 0, 0));
    EXPECT_EQ(&m.data[3], ti.GetElement(&cm, 0, 1));
    EXPECT_EQ(&m.data[1], ti.GetElement(&cm, 1, 0));
    EXPECT_EQ(&m.data[4], ti.GetElement(&cm, 1, 1));
  }
}

TEST(SReflect, Matrix_SerializeJSON) {
  picojson::value json;

  // Row Major
  // [0, 1]
  // [2, 3]
  // [4, 5]
  {
    TestMatrix<int, true> m;
    auto const& cm = m;
    m.layout._numRows = 3;
    m.layout._numColumns = 2;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    ti.Serialize(&cm, json);
    ExpectJson("[[0,1],[2,3],[4,5]]", json.serialize(false));
  }

  // Column Major
  // [0, 2, 4]
  // [1, 3, 5]
  {
    TestMatrix<int, false> m;
    auto const& cm = m;
    m.layout._numRows = 2;
    m.layout._numColumns = 3;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5}; // Same data as previous case
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    ti.Serialize(&cm, json);
    ExpectJson("[[0,2,4],[1,3,5]]", json.serialize(false));
  }

  // Row Vector (dynamic columns)
  // [0, 1, 2]
  {
    TestMatrix<int, true, false, true> m;
    auto const& cm = m;
    m.layout._numRows = 1;
    m.layout._numColumns = 3;
    m.data = std::vector<int>{0, 1, 2};
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    ti.Serialize(&cm, json);
    ExpectJson("[0,1,2]", json.serialize(false)); // As 1D array
  }

  // Column Vector (dynamic rows)
  // [0]
  // [1]
  // [2]
  {
    TestMatrix<int, false, true, false> m;
    auto const& cm = m;
    m.layout._numRows = 3;
    m.layout._numColumns = 1;
    m.data = std::vector<int>{0, 1, 2}; // Same data as previous case
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    ti.Serialize(&cm, json);
    ExpectJson("[0,1,2]", json.serialize(false)); // As 1D array
  }

  // Row Major with padding
  // [0, 1], padding
  // [3, 4], padding
  {
    TestMatrix<int, true> m;
    auto const& cm = m;
    m.layout._numRows = 2;
    m.layout._numColumns = 2;
    m.layout._leadingDim = 3;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5}; // Same data again
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    ti.Serialize(&cm, json);
    ExpectJson("[[0,1],[3,4]]", json.serialize(false));
  }

  // Column Major with padding
  // [0, 3]
  // [1, 4]
  // [padding]
  {
    TestMatrix<int, false> m;
    auto const& cm = m;
    m.layout._numRows = 2;
    m.layout._numColumns = 2;
    m.layout._leadingDim = 3;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5}; // Same data again
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    ti.Serialize(&cm, json);
    ExpectJson("[[0,3],[1,4]]", json.serialize(false));
  }
}

TEST(SReflect, Matrix_DeserializeJSON) {
  // Column-major, fixed rows, fixed columns
  {
    TestMatrix<int, false, false, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data.resize(8);
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows);
    EXPECT_EQ(4, m.layout._numColumns); // was resized
    EXPECT_EQ(m.data, (std::vector<int>{0, 4, 1, 5, 2, 6, 3, 7}));
  }

  // Column-major, fixed rows, dynamic columns
  {
    TestMatrix<int, false, false, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 0; // To be resized
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows);
    EXPECT_EQ(4, m.layout._numColumns); // was resized
    EXPECT_EQ(m.data, (std::vector<int>{0, 4, 1, 5, 2, 6, 3, 7}));
  }

  // Column-major, dynamic rows, fixed columns
  {
    TestMatrix<int, false, true, false> m;
    m.layout._numRows = 0; // To be resized
    m.layout._numColumns = 4;
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows); // was resized
    EXPECT_EQ(4, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{0, 4, 1, 5, 2, 6, 3, 7}));
  }

  // Column-major, dynamic rows, dynamic columns
  {
    TestMatrix<int, false, true, true> m;
    m.layout._numRows = 0; // To be resized
    m.layout._numColumns = 0; // To be resized
    m.data.clear(); // To be resized
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows); // was resized
    EXPECT_EQ(4, m.layout._numColumns); // was resized
    EXPECT_EQ(m.data, (std::vector<int>{0, 4, 1, 5, 2, 6, 3, 7}));
  }

  // Row-major, fixed rows, fixed columns
  {
    TestMatrix<int, true, false, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data.resize(8);
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  }

  // Row-major, fixed rows, dynamic columns
  {
    TestMatrix<int, true, false, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 0; // To be resized
    m.data.clear(); // To be resized
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows);
    EXPECT_EQ(4, m.layout._numColumns); // was resized
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  }

  // Row-major, dynamic rows, fixed columns
  {
    TestMatrix<int, true, true, false> m;
    m.layout._numRows = 0; // To be resized
    m.layout._numColumns = 4;
    m.data.clear(); // To be resized
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows); // was resized
    EXPECT_EQ(4, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  }

  // Row-major, dynamic rows, dynamic columns
  {
    TestMatrix<int, true, true, true> m;
    m.layout._numRows = 0; // To be resized
    m.layout._numColumns = 0; // To be resized
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(2, m.layout._numRows); // was resized
    EXPECT_EQ(4, m.layout._numColumns); // was resized
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  }

  // Failure to resize rows
  {
    TestMatrix<int, true, false, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[9, 9, 9, 9]]", {}, numIssues);
    EXPECT_NE(0, numIssues); // Can't resize down to 1 row
    EXPECT_EQ(2, m.layout._numRows); // no change
    EXPECT_EQ(4, m.layout._numColumns); // no change
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7})); // no change
  }

  // Failure to resize columns
  {
    TestMatrix<int, true, true, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[9, 9, 9, 9, 9], [9, 9, 9, 9, 9]]", {}, numIssues);
    EXPECT_NE(0, numIssues); // Can't resize up to 5 columns
    EXPECT_EQ(2, m.layout._numRows); // no change
    EXPECT_EQ(4, m.layout._numColumns); // no change
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7})); // no change
  }

  // Failure due to inconsistent row width
  {
    TestMatrix<int, true, true, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[9, 9, 9, 9], [9, 9, 9]]", {}, numIssues);
    EXPECT_NE(0, numIssues); // 2nd row has a diffierent number of values
    EXPECT_EQ(2, m.layout._numRows); // no change
    EXPECT_EQ(4, m.layout._numColumns); // no change
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7})); // no change
  }

  // Failure due to incorrect JSON formatting
  {
    TestMatrix<int, true, true, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    int numIssues = 0;
    SReflect::FromJsonString(m, "123", {}, numIssues);
    EXPECT_NE(0, numIssues); // no outer array
    numIssues = 0;
    SReflect::FromJsonString(m, "[123]", {}, numIssues);
    EXPECT_NE(0, numIssues); // no inner array
    numIssues = 0;
    SReflect::FromJsonString(m, "[[\"oops\"]]", {}, numIssues);
    EXPECT_NE(0, numIssues); // wrong value type
    numIssues = 0;
    // This time, we allow that the data may have been modified via partial failure because
    // validating every value in the array up front would be a burden (and something that other type
    // traits don't do).
    SReflect::FromJsonString(m, "[[123]]", {}, numIssues);
    EXPECT_EQ(0, numIssues); // Succeed this time
    EXPECT_EQ(1, m.layout._numRows);
    EXPECT_EQ(1, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{123}));
    numIssues = 0;
  }

  // Row-major, fixed rows, fixed columns, leading dim
  {
    TestMatrix<int, true, false, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.layout._leadingDim = 5;
    m.data.resize(10, 42);
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(
        m.data, (std::vector<int>{0, 1, 2, 3, 42, 4, 5, 6, 7, 42})); // Some 42 padding remains
  }

  // Column-major, fixed rows, fixed columns, leading dim
  {
    TestMatrix<int, false, false, false> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.layout._leadingDim = 4;
    m.data.resize(16, 42);
    int numIssues = 0;
    SReflect::FromJsonString(m, "[[0, 1, 2, 3], [4, 5, 6, 7]]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(
        m.data,
        (std::vector<int>{
            0, 4, 42, 42, 1, 5, 42, 42, 2, 6, 42, 42, 3, 7, 42, 42})); // Some 42 padding remains
  }

  // "[]" loads as (0 x 0)
  {
    TestMatrix<int, true> m;
    m.layout._numRows = 2;
    m.layout._numColumns = 4;
    m.data = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    int numIssues = 0;
    SReflect::FromJsonString(m, "[]", {}, numIssues);
    EXPECT_EQ(0, numIssues); // Succeed this time
    EXPECT_EQ(0, m.layout._numRows);
    EXPECT_EQ(0, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{}));
  }

  // Fixed row vector (dynamic columns) can load from 1D array
  {
    TestMatrix<int, true, false, true> m;
    m.layout._numRows = 1;
    m.layout._numColumns = 3;
    m.data = std::vector<int>{0, 1, 2};
    int numIssues = 0;
    SReflect::FromJsonString(m, "[0, 1, 2]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(1, m.layout._numRows);
    EXPECT_EQ(3, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2}));
    SReflect::FromJsonString(m, "[[3, 4, 5]]", {}, numIssues); // 2D JSON syntax also works here
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(1, m.layout._numRows);
    EXPECT_EQ(3, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{3, 4, 5}));
  }

  // Fixed column vector (dynamic rows) can load from 1D array
  {
    TestMatrix<int, true, true, false> m;
    m.layout._numRows = 3;
    m.layout._numColumns = 1;
    m.data = std::vector<int>{0, 1, 2};
    int numIssues = 0;
    SReflect::FromJsonString(m, "[0, 1, 2]", {}, numIssues);
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(3, m.layout._numRows);
    EXPECT_EQ(1, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{0, 1, 2}));
    SReflect::FromJsonString(m, "[[3], [4], [5]]", {}, numIssues); // 2D JSON syntax also works here
    EXPECT_EQ(0, numIssues);
    EXPECT_EQ(3, m.layout._numRows);
    EXPECT_EQ(1, m.layout._numColumns);
    EXPECT_EQ(m.data, (std::vector<int>{3, 4, 5}));
  }
}

template <bool kIsRowMajor>
static void TestMatrixRoundTrip() {
  // This test uses JSON to configure the test cases, assuming that JSON deserialization
  // is solid. See tests above.
  const std::string kJsonStrings[] = {
      "[[123]]", "[[1, 2, 3]]", "[[1],[2],[3]]", "[[1,2,3],[4,5,6]]", "[[1,2],[3,4],[5,6]]"};

  for (auto const& json : kJsonStrings) {
    // Parse JSON string (dynamic rows, dynamic columns)
    TestMatrix<int, kIsRowMajor, true, true> src;
    int numIssues = 0;
    SReflect::FromJsonString(src, json, {}, numIssues);
    EXPECT_EQ(0, numIssues);

    // Fixed rows, fixed columns
    {
      TestMatrix<int, kIsRowMajor, false, false> m1;
      m1.layout._numRows = src.layout._numRows;
      m1.layout._numColumns = src.layout._numColumns;
      m1.data = src.data;
      auto m2 = m1;
      m2.data.clear();
      m2.data.resize(m1.data.size(), 0); // Fill zeros
      SerdeBinary_Roundtrip_Helper(m1, m2);

      // Give JSON serialization more test coverage while we're at it.
      m2 = m1;
      m2.data.clear();
      m2.data.resize(m1.data.size(), 0);
      SerdeJson_Roundtrip_Helper(m1, m2);

      // Give TypeInfo:Set more test coverage too
      m2 = m1;
      m2.data.clear();
      m2.data.resize(m1.data.size(), 0);
      SReflect::GetTypeInfo<decltype(m1)>().Set(&m1, &m2);
      EXPECT_EQ(m1, m2);
    }

    // Fixed rows, dynamic columns
    {
      TestMatrix<int, kIsRowMajor, false, true> m1;
      m1.layout._numRows = src.layout._numRows;
      m1.layout._numColumns = src.layout._numColumns;
      m1.data = src.data;
      auto m2 = m1;
      m2.layout._numColumns = 0; // To be resized
      m2.data.clear(); // To be resized
      SerdeBinary_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.layout._numColumns = 0; // To be resized
      m2.data.clear(); // To be resized
      SerdeJson_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.layout._numColumns = 0; // To be resized
      m2.data.clear(); // To be resized
      SReflect::GetTypeInfo<decltype(m1)>().Set(&m1, &m2);
      EXPECT_EQ(m1, m2);
    }

    // Dynamic rows, fixed columns
    {
      TestMatrix<int, kIsRowMajor, true, false> m1;
      m1.layout._numRows = src.layout._numRows;
      m1.layout._numColumns = src.layout._numColumns;
      m1.data = src.data;
      auto m2 = m1;
      m2.layout._numRows = 0; // To be resized
      m2.data.clear(); // To be resized
      SerdeBinary_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.layout._numRows = 0; // To be resized
      m2.data.clear(); // To be resized
      SerdeJson_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.layout._numRows = 0; // To be resized
      m2.data.clear(); // To be resized
      SReflect::GetTypeInfo<decltype(m1)>().Set(&m1, &m2);
      EXPECT_EQ(m1, m2);
    }

    // Dynamic rows, dynamic columns
    {
      auto m1 = src;
      auto m2 = m1;
      m2.layout._numRows = 0; // To be resized
      m2.layout._numColumns = 0; // To be resized
      m2.data.clear(); // To be resized
      SerdeBinary_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.layout._numRows = 0; // To be resized
      m2.layout._numColumns = 0; // To be resized
      m2.data.clear(); // To be resized
      SerdeJson_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.layout._numRows = 0; // To be resized
      m2.layout._numColumns = 0; // To be resized
      m2.data.clear(); // To be resized
      SReflect::GetTypeInfo<decltype(m1)>().Set(&m1, &m2);
      EXPECT_EQ(m1, m2);
    }

    // Fixed rows, fixed columns, leading dimension
    {
      TestMatrix<int, kIsRowMajor, false, false> m1;
      m1.layout._numRows = src.layout._numRows;
      m1.layout._numColumns = src.layout._numColumns;
      m1.data = src.data;
      if constexpr (kIsRowMajor) {
        m1.layout._leadingDim = m1.layout._numColumns + 1;
        for (int r = 0; r < m1.layout._numRows; ++r) {
          // Insert a 42 after each row, as padding.
          m1.data.insert(m1.data.begin() + m1.layout._numColumns + r * m1.layout._leadingDim, 42);
        }
      } else {
        m1.layout._leadingDim = m1.layout._numRows + 1;
        for (int c = 0; c < m1.layout._numColumns; ++c) {
          // Insert a 42 after each column, as padding.
          m1.data.insert(m1.data.begin() + m1.layout._numRows + c * m1.layout._leadingDim, 42);
        }
      }
      auto m2 = m1;
      m2.data.clear();
      m2.data.resize(m1.data.size(), 42); // 42 will remain in the padding.
      SerdeBinary_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.data.clear();
      m2.data.resize(m1.data.size(), 42);
      SerdeJson_Roundtrip_Helper(m1, m2);
      m2 = m1;
      m2.data.clear();
      m2.data.resize(m1.data.size(), 42);
      SReflect::GetTypeInfo<decltype(m1)>().Set(&m1, &m2);
      EXPECT_EQ(m1, m2);
    }
  }

  // Fail. Zero bytes.
  {
    TestMatrix<int, kIsRowMajor> m;
    SReflect::VecStreamWriter writer;
    SReflect::SpanStreamReader reader(writer.GetBytes());
    EXPECT_FALSE(SReflect::FromBytes(reader, m));
  }

  // Fail. Invalid header.
  {
    TestMatrix<int, kIsRowMajor> m;
    SReflect::VecStreamWriter writer;
    EXPECT_TRUE(writer.Write("no", 2));
    SReflect::SpanStreamReader reader(writer.GetBytes());
    EXPECT_FALSE(SReflect::FromBytes(reader, m));
  }

  // Fail. Insufficient bytes.
  {
    TestMatrix<int, kIsRowMajor> m;
    m.layout._numRows = 3;
    m.layout._numColumns = 3;
    m.data.resize(9);
    SReflect::VecStreamWriter writer;
    EXPECT_TRUE(SReflect::ToBytes(m, writer));
    auto buffer = writer.GetBytes();
    SReflect::SpanStreamReader reader1(buffer);
    EXPECT_TRUE(SReflect::FromBytes(reader1, m)); // Prove this is a valid buffer
    SReflect::SpanStreamReader reader2(
        SReflect::Span<uint8_t const>{buffer.data(), buffer.size() - 1});
    EXPECT_FALSE(SReflect::FromBytes(reader2, m)); // Fail because one byte is missing
  }

  // Fail. TryResizeRows returns false even though the type claimed to be resizable.
  {
    TestMatrix<int, kIsRowMajor, true, true> m1;
    m1.layout._numRows = 2;
    m1.layout._numColumns = 3;
    m1.data = std::vector<int>{1, 2, 3, 4, 5, 6};
    SReflect::VecStreamWriter writer;
    auto const& ti = SReflect::GetTypeInfo<decltype(m1)>();
    EXPECT_TRUE(ti.SerializeToBytes(&m1, writer));
    // Successfully deserialize to prove we can
    {
      decltype(m1) m2;
      SReflect::SpanStreamReader reader(writer.GetBytes());
      EXPECT_TRUE(SReflect::FromBytes(reader, m2));
      EXPECT_EQ(m1, m2);
    }
    // Now use the same stream but TryResize will return false
    {
      decltype(m1) m2;
      ti._canResizeRows = false;
      SReflect::SpanStreamReader reader(writer.GetBytes());
      EXPECT_FALSE(SReflect::FromBytes(reader, m2)); // TryResize failure
      ti._canResizeRows = true;
    }
  }
}

TEST(SReflect, Matrix_RoundTrip) {
  TestMatrixRoundTrip<true>(); // row-major
  TestMatrixRoundTrip<false>(); // column-major
}

struct StructWithFloatMatrix {
  TestMatrix<float> mat;

  SR_BeginStruct(SReflectTest::StructWithFloatMatrix);
  SR_Field(mat) SRA_FloatRange(0.0f, 1.0f);
  SR_EndStruct();
};

TEST(SReflect, Matrix_IsValid) {
  // IsValid is a rarely used feature, but MatrixTypeInfo supports it. One way to fail validation is
  // with an attribute that applies recursively to the inner type.

  // Empty matrix
  StructWithFloatMatrix obj;
  EXPECT_TRUE(SReflect::IsValid(obj));

  // Matrix with zeros
  obj.mat.layout._numRows = 3;
  obj.mat.layout._numColumns = 3;
  obj.mat.data.resize(9);
  EXPECT_TRUE(SReflect::IsValid(obj));

  // Matrix with one value outside the allowed range
  obj.mat.data[5] = 1.1f;
  EXPECT_FALSE(SReflect::IsValid(obj));
}

using TestVariant = std::variant<int, float, std::string, std::vector<int>>;

TEST(SReflect, Variant_TypeInfo) {
  auto const& ti = SReflect::GetTypeInfo<TestVariant>();
  EXPECT_EQ(ti._coreType, SReflect::CoreType::CT_variant);
  EXPECT_EQ(ti._alignment, alignof(TestVariant));
  EXPECT_EQ(ti._sizeInBytes, sizeof(TestVariant));
  EXPECT_STREQ(ti._name, "variant<int32,float,string,vector<int32>>");
  EXPECT_STREQ(ti._nameWithNamespace, "std::variant<int32,float,std::string,std::vector<int32>>");
  EXPECT_EQ(ti._typeId, SReflect::ComputeTypeId(ti._nameWithNamespace));
  EXPECT_EQ(false, ti.IsMemCopySafe());

  // Inner types
  ASSERT_EQ(4, ti._innerTypes.size());
  EXPECT_EQ(&SReflect::GetTypeInfo<int>(), ti._innerTypes[0]);
  EXPECT_EQ(&SReflect::GetTypeInfo<float>(), ti._innerTypes[1]);
  EXPECT_EQ(&SReflect::GetTypeInfo<std::string>(), ti._innerTypes[2]);
  EXPECT_EQ(&SReflect::GetTypeInfo<std::vector<int>>(), ti._innerTypes[3]);

  // Some variant types are memcpy safe
  EXPECT_EQ(true, (SReflect::GetTypeInfo<std::variant<int, float>>().IsMemCopySafe()));
  EXPECT_EQ(true, (SReflect::GetTypeInfo<std::variant<MyFruit, MyPoint>>().IsMemCopySafe()));
  EXPECT_EQ(false, (SReflect::GetTypeInfo<std::variant<int, std::string>>().IsMemCopySafe()));
}

TEST(SReflect, Variant_InnerTypeIndex) {
  auto const& ti = SReflect::GetTypeInfo<TestVariant>();
  TestVariant v;
  auto const& cv = v;

  // GetInnerTypeIndex
  v = 123;
  EXPECT_EQ(0, ti.GetInnerTypeIndex(&cv));
  v = 1.23f;
  EXPECT_EQ(1, ti.GetInnerTypeIndex(&cv));
  v = std::string{};
  EXPECT_EQ(2, ti.GetInnerTypeIndex(&cv));
  v = std::vector<int>{};
  EXPECT_EQ(3, ti.GetInnerTypeIndex(&cv));

  // SetInnerTypeIndex (emplaces a default value of the new type)
  constexpr char const* kExpectedDefaultJson[4] = {
      "{\"int32\":0}", // int
      "{\"float\":0}", // float
      "{\"string\":\"\"}", // std::string
      "{\"vector<int32>\":[]}"};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(true, ti.TrySetInnerTypeIndex(&v, i));
    EXPECT_EQ(i, ti.GetInnerTypeIndex(&cv));
    EXPECT_STREQ(kExpectedDefaultJson[i], SReflect::ToJsonString(cv, /*pretty*/ false).c_str());
  }

  // If a variant has a non-default-constructible inner type, then SetInnerTypeIndex will fail.
  std::variant<int, TypeWithNoDefaultConstructor, float> v2(123);
  auto const& ti2 = SReflect::GetTypeInfo<decltype(v2)>();
  EXPECT_EQ(0, ti2.GetInnerTypeIndex(&v2));
  v2 = TypeWithNoDefaultConstructor{42};
  EXPECT_EQ(1, ti2.GetInnerTypeIndex(&v2));
  v2 = 1.23f;
  EXPECT_EQ(2, ti2.GetInnerTypeIndex(&v2));
  EXPECT_EQ(true, ti2.TrySetInnerTypeIndex(&v2, 0)); // success
  EXPECT_EQ(0, v2.index());
  v2 = 123;
  EXPECT_EQ(false, ti2.TrySetInnerTypeIndex(&v2, 1)); // fail
  ASSERT_EQ(0, v2.index()); // no change
  EXPECT_EQ(123, std::get<0>(v2)); // no change
}

struct StructWithVariant {
  TestVariant var;
  SR_BeginStruct(SReflectTest::StructWithVariant);
  // This is a contrived example to test TypeInfo::IsValid.
  // SRA_FloatRange only applies when the inner type is float or double.
  SR_Field(var) SRA_FloatRange(0.0f, 1.0f);
  SR_EndStruct();
};

TEST(SReflect, Variant_IsValid) {
  // IsValid is a rarely used feature, but VariantTypeInfo supports it. One way to fail validation
  // is with an attribute that applies recursively to the inner type.

  // Default
  StructWithVariant obj;
  EXPECT_TRUE(SReflect::IsValid(obj));

  // Valid float value
  obj.var = 0.5f;
  EXPECT_TRUE(SReflect::IsValid(obj));

  // Out-of-range float value
  obj.var = 2.0f;
  EXPECT_FALSE(SReflect::IsValid(obj));
}

TEST(SReflect, Variant_Serialization) {
  auto const& ti = SReflect::GetTypeInfo<TestVariant>();
  TestVariant v, v2;

  // int
  v = 123;
  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);
  ti.Set(&v, &v2); // Copy via TypeInfo::Set
  EXPECT_EQ(v, v2);

  // float
  v = 1.23f;
  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);
  ti.Set(&v, &v2); // Copy via TypeInfo::Set
  EXPECT_EQ(v, v2);

  // std::string
  v = std::string{"Hello"};
  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);
  ti.Set(&v, &v2); // Copy via TypeInfo::Set
  EXPECT_EQ(v, v2);

  // std::vector<int>
  v = std::vector<int>{1, 2, 3, 4, 5};
  SerdeJson_Roundtrip_Helper(v);
  SerdeBinary_Roundtrip_Helper(v);
  ti.Set(&v, &v2); // Copy via TypeInfo::Set
  EXPECT_EQ(v, v2);
}

TEST(SReflect, CalcHash64) {
  // SReflect::CalcHash64 simply exposes CityHash64 so people can use it without another dependency.
  // This is just a sanity check that it is hooked up and working as expected.
  constexpr std::string_view kText =
      "Once upon a time, there was a hashing function that did more or less what it was supposed to do.";
  std::vector<uint64_t> hashes(kText.size());
  for (size_t i = 0; i < kText.size(); ++i) {
    hashes[i] = SReflect::CalcHash64(kText.data(), i + 1);
  }
  std::sort(hashes.begin(), hashes.end());
  EXPECT_EQ(hashes.end(), std::unique(hashes.begin(), hashes.end())); // All unique
}

static picojson::value ParseJsonForTypeInfoTest(std::string const& json) {
  picojson::value result;
  std::istringstream stream(json);
  std::string error = picojson::parse(result, stream);
  EXPECT_STREQ("", error.c_str());
  return result;
}

static picojson::object const& JsonObject(picojson::value const& value) {
  static const picojson::object kEmptyObject;
  EXPECT_TRUE(value.is<picojson::object>());
  return value.is<picojson::object>() ? value.get<picojson::object>() : kEmptyObject;
}

static picojson::array const& JsonArray(picojson::value const& value) {
  static const picojson::array kEmptyArray;
  EXPECT_TRUE(value.is<picojson::array>());
  return value.is<picojson::array>() ? value.get<picojson::array>() : kEmptyArray;
}

static picojson::value const& JsonAt(picojson::object const& object, char const* key) {
  static const picojson::value kNullValue;
  auto it = object.find(key);
  EXPECT_NE(object.end(), it) << key;
  return it != object.end() ? it->second : kNullValue;
}

static picojson::object const& JsonObjectAt(picojson::object const& object, char const* key) {
  return JsonObject(JsonAt(object, key));
}

static picojson::array const& JsonArrayAt(picojson::object const& object, char const* key) {
  return JsonArray(JsonAt(object, key));
}

static std::string JsonStringAt(picojson::object const& object, char const* key) {
  picojson::value const& value = JsonAt(object, key);
  EXPECT_TRUE(value.is<std::string>()) << key;
  return value.is<std::string>() ? value.get<std::string>() : std::string{};
}

static double JsonNumberAt(picojson::object const& object, char const* key) {
  picojson::value const& value = JsonAt(object, key);
  EXPECT_TRUE(value.is<double>()) << key;
  return value.is<double>() ? value.get<double>() : 0.0;
}

static bool JsonBoolAt(picojson::object const& object, char const* key) {
  picojson::value const& value = JsonAt(object, key);
  EXPECT_TRUE(value.is<bool>()) << key;
  return value.is<bool>() ? value.get<bool>() : false;
}

static picojson::object const* FindNamedJsonObject(
    picojson::array const& array,
    std::string_view name) {
  for (picojson::value const& value : array) {
    if (!value.is<picojson::object>()) {
      continue;
    }
    picojson::object const& object = value.get<picojson::object>();
    auto nameIt = object.find("name");
    if (nameIt != object.end() && nameIt->second.is<std::string>() &&
        nameIt->second.get<std::string>() == name) {
      return &object;
    }
  }
  return nullptr;
}

// The flat-dictionary key for a type: its fully-qualified name, as used by SerializeTypeInfo.
static std::string TypeKey(SReflect::TypeInfo const& ti) {
  return ti._nameWithNamespace ? ti._nameWithNamespace : "";
}

// Get the flat-dictionary entry for `ti`'s own type (the root entry).
static picojson::object const& RootEntry(
    picojson::value const& dict,
    SReflect::TypeInfo const& ti) {
  return JsonObjectAt(JsonObject(dict), TypeKey(ti).c_str());
}

// Resolve a type reference (a fully-qualified-name json string) to its entry in the flat
// dictionary.
static picojson::object const& ResolveRef(
    picojson::object const& dict,
    picojson::value const& ref) {
  EXPECT_TRUE(ref.is<std::string>());
  return JsonObjectAt(dict, ref.is<std::string>() ? ref.get<std::string>().c_str() : "");
}

// Given the flat dictionary and an "attributes" object (keyed by attribute-type name), find the
// serialized value for the attribute whose struct type has the given short name. Returns nullptr
// if absent.
static picojson::value const* FindAttributeValueByTypeName(
    picojson::object const& dict,
    picojson::object const& attributes,
    std::string_view attributeTypeName) {
  for (auto const& [typeKey, value] : attributes) {
    auto typeIt = dict.find(typeKey);
    if (typeIt == dict.end() || !typeIt->second.is<picojson::object>()) {
      continue;
    }
    picojson::object const& typeEntry = typeIt->second.get<picojson::object>();
    auto nameIt = typeEntry.find("name");
    if (nameIt != typeEntry.end() && nameIt->second.is<std::string>() &&
        nameIt->second.get<std::string>() == attributeTypeName) {
      return &value;
    }
  }
  return nullptr;
}

struct TypeInfoJsonStruct {
  float normalized = 0.0f;
  std::string label;
  std::optional<MyPoint> maybePoint;
  std::unordered_map<std::string, MyPoint> pointsByName;
  TestMatrix<float, true, true, false> matrix;
  TestVariant variant;
  int32_t _renamed = 0;

  SR_BeginStruct(SReflectTest::TypeInfoJsonStruct);
  SRA_DisplayName("Type Info JSON Struct");
  SR_Field(normalized) SRA_FloatRange(0.0f, 1.0f) SRA_Units("meters");
  SR_Field(label) SRA_PreviouslyKnownAs("oldLabel");
  SR_Field(maybePoint) SRA_DoNotSerializeDefaults(false);
  SR_Field(pointsByName);
  SR_Field(matrix);
  SR_Field(variant);
  SR_Field_Name(_renamed, "renamed");
  SR_EndStruct();
};

struct TypeInfoJsonStringStruct {
  std::string text;

  SR_BeginStruct(SReflectTest::TypeInfoJsonStringStruct);
  SR_Field(text);
  SR_EndStruct();
};

struct TypeInfoJsonDedupStruct {
  MyPoint a;
  MyPoint b;

  SR_BeginStruct(SReflectTest::TypeInfoJsonDedupStruct);
  SR_Field(a);
  SR_Field(b);
  SR_EndStruct();
};

struct TypeInfoJsonUnionA {
  MyPoint point;

  SR_BeginStruct(SReflectTest::TypeInfoJsonUnionA);
  SR_Field(point);
  SR_EndStruct();
};

struct TypeInfoJsonUnionB {
  MyPoint point;
  int32_t count = 0;

  SR_BeginStruct(SReflectTest::TypeInfoJsonUnionB);
  SR_Field(point);
  SR_Field(count);
  SR_EndStruct();
};

TEST(SReflect, TypeInfoToJson_Primitive) {
  auto const& ti = SReflect::GetTypeInfo<int32_t>();
  picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
  picojson::object const& object = RootEntry(json, ti);

  EXPECT_EQ("CT_int32", JsonStringAt(object, "coreType"));
  EXPECT_EQ(static_cast<double>(alignof(int32_t)), JsonNumberAt(object, "alignment"));
  EXPECT_EQ(static_cast<double>(sizeof(int32_t)), JsonNumberAt(object, "sizeInBytes"));
  EXPECT_EQ(object.end(), object.find("attributes")); // Omitted when empty.
  EXPECT_EQ("int32", JsonStringAt(object, "name"));
  EXPECT_EQ("int32", JsonStringAt(object, "nameWithNamespace"));

  std::string typeId = JsonStringAt(object, "typeId");
  EXPECT_EQ(16, typeId.size());
  EXPECT_NE("0000000000000000", typeId);

  picojson::value prettyJson = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(true));
  EXPECT_TRUE(prettyJson.is<picojson::object>());
}

TEST(SReflect, TypeInfoToJson_StructFieldsAndAttributes) {
  auto const& ti = SReflect::GetTypeInfo<TypeInfoJsonStruct>();
  picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
  picojson::object const& dict = JsonObject(json);
  picojson::object const& object = RootEntry(json, ti);

  EXPECT_EQ("CT_struct", JsonStringAt(object, "coreType"));
  EXPECT_EQ("TypeInfoJsonStruct", JsonStringAt(object, "name"));
  EXPECT_EQ("SReflectTest::TypeInfoJsonStruct", JsonStringAt(object, "nameWithNamespace"));
  EXPECT_FALSE(JsonBoolAt(object, "isMemCopySafe"));
  EXPECT_EQ(object.end(), object.find("baseClasses")); // Omitted when empty.

  picojson::object const& structAttributes = JsonObjectAt(object, "attributes");
  ASSERT_EQ(1u, structAttributes.size());
  picojson::value const* displayNameValue =
      FindAttributeValueByTypeName(dict, structAttributes, "Attribute_DisplayName");
  ASSERT_NE(nullptr, displayNameValue);
  EXPECT_EQ("Type Info JSON Struct", JsonStringAt(JsonObject(*displayNameValue), "displayName"));

  picojson::array const& fields = JsonArrayAt(object, "fields");
  ASSERT_EQ(7, fields.size());

  picojson::object const* normalizedField = FindNamedJsonObject(fields, "normalized");
  ASSERT_NE(nullptr, normalizedField);
  EXPECT_EQ(normalizedField->end(), normalizedField->find("coreType"));
  EXPECT_EQ(
      "CT_float",
      JsonStringAt(ResolveRef(dict, JsonAt(*normalizedField, "innerType")), "coreType"));
  EXPECT_EQ(0.0, JsonNumberAt(*normalizedField, "offset"));

  picojson::object const& normalizedAttributes = JsonObjectAt(*normalizedField, "attributes");
  ASSERT_EQ(2u, normalizedAttributes.size());
  picojson::value const* floatRangeValueRef =
      FindAttributeValueByTypeName(dict, normalizedAttributes, "Attribute_FloatRange");
  ASSERT_NE(nullptr, floatRangeValueRef);
  picojson::object const& floatRangeValue = JsonObject(*floatRangeValueRef);
  EXPECT_EQ(floatRangeValue.end(), floatRangeValue.find("_min"));
  EXPECT_EQ(0.0, JsonNumberAt(floatRangeValue, "min"));
  EXPECT_EQ(1.0, JsonNumberAt(floatRangeValue, "max"));

  picojson::value const* unitsValueRef =
      FindAttributeValueByTypeName(dict, normalizedAttributes, "Attribute_Units");
  ASSERT_NE(nullptr, unitsValueRef);
  EXPECT_EQ("meters", JsonStringAt(JsonObject(*unitsValueRef), "units"));

  picojson::object const* labelField = FindNamedJsonObject(fields, "label");
  ASSERT_NE(nullptr, labelField);
  picojson::value const* previousNamesValueRef = FindAttributeValueByTypeName(
      dict, JsonObjectAt(*labelField, "attributes"), "Attribute_PreviouslyKnownAs");
  ASSERT_NE(nullptr, previousNamesValueRef);
  picojson::array const& previousNames =
      JsonArrayAt(JsonObject(*previousNamesValueRef), "previousNames");
  ASSERT_EQ(1, previousNames.size());
  EXPECT_EQ("oldLabel", previousNames[0].get<std::string>());

  picojson::object const* maybePointField = FindNamedJsonObject(fields, "maybePoint");
  ASSERT_NE(nullptr, maybePointField);
  picojson::value const* defaultsValueRef = FindAttributeValueByTypeName(
      dict, JsonObjectAt(*maybePointField, "attributes"), "Attribute_DoNotSerializeDefaults");
  ASSERT_NE(nullptr, defaultsValueRef);
  EXPECT_FALSE(JsonBoolAt(JsonObject(*defaultsValueRef), "recursive"));

  picojson::object const* renamedField = FindNamedJsonObject(fields, "renamed");
  ASSERT_NE(nullptr, renamedField);
  EXPECT_EQ(nullptr, FindNamedJsonObject(fields, "_renamed"));
}

TEST(SReflect, TypeInfoToJson_DerivedStructBaseClasses) {
  auto const& ti = SReflect::GetTypeInfo<MyDerivedClass>();
  picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
  picojson::object const& dict = JsonObject(json);
  picojson::object const& object = RootEntry(json, ti);

  picojson::array const& baseClasses = JsonArrayAt(object, "baseClasses");
  ASSERT_EQ(1, baseClasses.size());
  EXPECT_EQ("MyClass", JsonStringAt(ResolveRef(dict, baseClasses[0]), "name"));

  picojson::array const& fields = JsonArrayAt(object, "fields");
  EXPECT_NE(nullptr, FindNamedJsonObject(fields, "myFloat"));
  EXPECT_NE(nullptr, FindNamedJsonObject(fields, "oneMoreThing"));
}

TEST(SReflect, TypeInfoToJson_EnumTypeInfo) {
  auto const& ti = SReflect::GetTypeInfo<MyFruit>();
  picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
  picojson::object const& dict = JsonObject(json);
  picojson::object const& object = RootEntry(json, ti);

  EXPECT_EQ("CT_enum", JsonStringAt(object, "coreType"));
  EXPECT_EQ("MyFruit", JsonStringAt(object, "name"));
  EXPECT_EQ("CT_int32", JsonStringAt(ResolveRef(dict, JsonAt(object, "innerType")), "coreType"));

  picojson::array const& items = JsonArrayAt(object, "items");
  ASSERT_EQ(3, items.size());
  picojson::object const* apple = FindNamedJsonObject(items, "Apple");
  ASSERT_NE(nullptr, apple);
  char expectedValue[17];
  snprintf(
      expectedValue, sizeof(expectedValue), "%016" PRIx64, static_cast<uint64_t>(MyFruit::Apple));
  EXPECT_EQ(expectedValue, JsonStringAt(*apple, "value"));
  EXPECT_EQ(apple->end(), apple->find("attributes")); // Omitted when empty.
}

TEST(SReflect, TypeInfoToJson_StringTypeInfo) {
  auto const& ti = SReflect::GetTypeInfo<TypeInfoJsonStringStruct>();
  picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
  picojson::object const& dict = JsonObject(json);
  picojson::object const& object = RootEntry(json, ti);

  picojson::array const& fields = JsonArrayAt(object, "fields");
  picojson::object const* textField = FindNamedJsonObject(fields, "text");
  ASSERT_NE(nullptr, textField);

  picojson::object const& stringTypeInfo = ResolveRef(dict, JsonAt(*textField, "innerType"));
  EXPECT_EQ("CT_string", JsonStringAt(stringTypeInfo, "coreType"));
  EXPECT_TRUE(JsonBoolAt(stringTypeInfo, "isNullTerminated"));
  EXPECT_FALSE(JsonBoolAt(stringTypeInfo, "isReadOnly"));
}

TEST(SReflect, TypeInfoToJson_CollectionAndOtherTypeInfo) {
  {
    auto const& ti = SReflect::GetTypeInfo<std::vector<int32_t>>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& dict = JsonObject(json);
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_array", JsonStringAt(object, "coreType"));
    EXPECT_TRUE(JsonBoolAt(object, "canResize"));
    EXPECT_EQ(object.end(), object.find("numElements")); // Resizable: size is per-instance.
    EXPECT_EQ("CT_int32", JsonStringAt(ResolveRef(dict, JsonAt(object, "innerType")), "coreType"));
  }
  {
    auto const& ti = SReflect::GetTypeInfo<std::array<int32_t, 3>>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_array", JsonStringAt(object, "coreType"));
    EXPECT_FALSE(JsonBoolAt(object, "canResize"));
    EXPECT_EQ(3.0, JsonNumberAt(object, "numElements")); // Fixed size is preserved.
  }
  {
    auto const& ti = SReflect::GetTypeInfo<TestMatrix<float, true, true, false>>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& dict = JsonObject(json);
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_matrix", JsonStringAt(object, "coreType"));
    EXPECT_TRUE(JsonBoolAt(object, "isRowMajor"));
    EXPECT_TRUE(JsonBoolAt(object, "isNumRowsDynamic"));
    EXPECT_FALSE(JsonBoolAt(object, "isNumColumnsDynamic"));
    EXPECT_EQ("CT_float", JsonStringAt(ResolveRef(dict, JsonAt(object, "innerType")), "coreType"));
  }
  {
    auto const& ti = SReflect::GetTypeInfo<std::unordered_map<std::string, MyPoint>>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& dict = JsonObject(json);
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_map", JsonStringAt(object, "coreType"));
    EXPECT_EQ("CT_string", JsonStringAt(ResolveRef(dict, JsonAt(object, "keyType")), "coreType"));
    EXPECT_EQ("MyPoint", JsonStringAt(ResolveRef(dict, JsonAt(object, "valueType")), "name"));
  }
  {
    auto const& ti = SReflect::GetTypeInfo<std::optional<MyPoint>>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& dict = JsonObject(json);
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_optional", JsonStringAt(object, "coreType"));
    EXPECT_EQ("MyPoint", JsonStringAt(ResolveRef(dict, JsonAt(object, "innerType")), "name"));
  }
  {
    auto const& ti = SReflect::GetTypeInfo<TestVariant>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_variant", JsonStringAt(object, "coreType"));
    EXPECT_FALSE(JsonBoolAt(object, "isMemCopySafe"));
    EXPECT_EQ(4, JsonArrayAt(object, "innerTypes").size());
  }
  {
    using Pair = std::pair<std::string, MyPoint>;
    auto const& ti = SReflect::GetTypeInfo<Pair>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& dict = JsonObject(json);
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_other", JsonStringAt(object, "coreType"));
    EXPECT_EQ("CT_string", JsonStringAt(ResolveRef(dict, JsonAt(object, "typeT")), "coreType"));
    EXPECT_EQ("MyPoint", JsonStringAt(ResolveRef(dict, JsonAt(object, "typeU")), "name"));
    EXPECT_TRUE(JsonAt(object, "offsetT").is<double>());
    EXPECT_TRUE(JsonAt(object, "offsetU").is<double>());
  }
  {
    auto const& ti = SReflect::GetTypeInfo<picojson::value>();
    picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
    picojson::object const& object = RootEntry(json, ti);
    EXPECT_EQ("CT_other", JsonStringAt(object, "coreType"));
    EXPECT_EQ("picojson_value", JsonStringAt(object, "name"));
    EXPECT_EQ(object.end(), object.find("typeT"));
  }
}

TEST(SReflect, TypeInfoToJson_MatrixDimensions) {
  // Fixed rows and columns: both dimensions are preserved.
  {
    TestMatrixTypeInfo<float, /*rowMajor*/ true, /*rowsDynamic*/ false, /*colsDynamic*/ false> info;
    info._fixedNumRows = 2;
    info._fixedNumColumns = 3;
    picojson::value json;
    info.SerializeTypeInfo(json);
    picojson::object const& object = RootEntry(json, info);
    EXPECT_FALSE(JsonBoolAt(object, "isNumRowsDynamic"));
    EXPECT_FALSE(JsonBoolAt(object, "isNumColumnsDynamic"));
    EXPECT_EQ(2.0, JsonNumberAt(object, "numRows"));
    EXPECT_EQ(3.0, JsonNumberAt(object, "numColumns"));
  }
  // Fixed rows, dynamic columns: only the fixed dimension is preserved.
  {
    TestMatrixTypeInfo<float, /*rowMajor*/ true, /*rowsDynamic*/ false, /*colsDynamic*/ true> info;
    info._fixedNumRows = 4;
    info._fixedNumColumns = 0;
    picojson::value json;
    info.SerializeTypeInfo(json);
    picojson::object const& object = RootEntry(json, info);
    EXPECT_FALSE(JsonBoolAt(object, "isNumRowsDynamic"));
    EXPECT_TRUE(JsonBoolAt(object, "isNumColumnsDynamic"));
    EXPECT_EQ(4.0, JsonNumberAt(object, "numRows"));
    EXPECT_EQ(object.end(), object.find("numColumns")); // Dynamic: size is per-instance.
  }
  // Fully dynamic: neither dimension is present.
  {
    TestMatrixTypeInfo<float, /*rowMajor*/ true, /*rowsDynamic*/ true, /*colsDynamic*/ true> info;
    picojson::value json;
    info.SerializeTypeInfo(json);
    picojson::object const& object = RootEntry(json, info);
    EXPECT_TRUE(JsonBoolAt(object, "isNumRowsDynamic"));
    EXPECT_TRUE(JsonBoolAt(object, "isNumColumnsDynamic"));
    EXPECT_EQ(object.end(), object.find("numRows"));
    EXPECT_EQ(object.end(), object.find("numColumns"));
  }
}

TEST(SReflect, TypeInfoToJson_DedupSharedFieldType) {
  auto const& ti = SReflect::GetTypeInfo<TypeInfoJsonDedupStruct>();
  picojson::value json = ParseJsonForTypeInfoTest(ti.TypeInfoToJson(false));
  picojson::object const& dict = JsonObject(json);
  picojson::object const& object = RootEntry(json, ti);

  picojson::array const& fields = JsonArrayAt(object, "fields");
  picojson::object const* fieldA = FindNamedJsonObject(fields, "a");
  picojson::object const* fieldB = FindNamedJsonObject(fields, "b");
  ASSERT_NE(nullptr, fieldA);
  ASSERT_NE(nullptr, fieldB);

  std::string const refA = JsonAt(*fieldA, "innerType").get<std::string>();
  std::string const refB = JsonAt(*fieldB, "innerType").get<std::string>();
  EXPECT_EQ(refA, refB); // Two fields of the same type share one reference.
  EXPECT_EQ(1u, dict.count(refA)); // ...and a single dictionary entry.
  EXPECT_EQ("MyPoint", JsonStringAt(JsonObjectAt(dict, refA.c_str()), "name"));
}

TEST(SReflect, TypeInfoToJson_AdditiveUnion) {
  auto const& tiA = SReflect::GetTypeInfo<TypeInfoJsonUnionA>();
  auto const& tiB = SReflect::GetTypeInfo<TypeInfoJsonUnionB>();

  picojson::value dst;
  tiA.SerializeTypeInfo(dst);
  tiB.SerializeTypeInfo(dst); // Additive: accumulates into the same dictionary.
  picojson::object const& dict = JsonObject(dst);

  // Both unrelated root types are present.
  EXPECT_EQ(1u, dict.count(TypeKey(tiA)));
  EXPECT_EQ(1u, dict.count(TypeKey(tiB)));

  // Their shared dependency is present exactly once.
  auto const& pointTi = SReflect::GetTypeInfo<MyPoint>();
  EXPECT_EQ(1u, dict.count(TypeKey(pointTi)));
}

TEST(SReflect, TypeInfoListToJson) {
  std::vector<SReflect::TypeInfo const*> types;
  std::string json;

  // Shorthand
  auto const& classInfo = SReflect::GetTypeInfo<MyDerivedClass>();
  auto const& enumInfo = SReflect::GetTypeInfo<MyFruit>();

  // No types
  EXPECT_STREQ("{}", SReflect::TypeInfoListToJson(nullptr, 0, /*pretty*/ false).c_str());
  EXPECT_STREQ("{}\n", SReflect::TypeInfoListToJson(nullptr, 0, /*pretty*/ true).c_str());

  // One type: MyClass
  types.push_back(&classInfo);
  std::string classJons = SReflect::TypeInfoListToJson(types.data(), types.size());
  EXPECT_STREQ(classInfo.TypeInfoToJson().c_str(), classJons.c_str());

  // Two types: MyClass + MyFruit (already referenced by a field of MyClass)
  types.push_back(&enumInfo);
  std::string classAndEnumJson = SReflect::TypeInfoListToJson(types.data(), types.size());
  EXPECT_STREQ(
      classJons.c_str(), classAndEnumJson.c_str()); // Same because MyFruit was already included.

  // Three types: MyClass + MyFruit + SReflectTest_GlobalStruct (not previously included)
  types.push_back(&SReflect::GetTypeInfo<SReflectTest_GlobalStruct>());
  std::string threeTypesJson = SReflect::TypeInfoListToJson(types.data(), types.size());
  EXPECT_STRNE(classJons.c_str(), threeTypesJson.c_str()); // Not equal now

  // Prase threeTypesJson
  picojson::object dictionary = ParseJsonForTypeInfoTest(threeTypesJson).get<picojson::object>();

  // Expect the 3 types we added explicitly
  EXPECT_NE(dictionary.end(), dictionary.find("SReflectTest::MyDerivedClass"));
  EXPECT_NE(dictionary.end(), dictionary.find("SReflectTest::MyFruit"));
  EXPECT_NE(dictionary.end(), dictionary.find("SReflectTest_GlobalStruct"));

  // Expect some of the nested types as well
  EXPECT_NE(dictionary.end(), dictionary.find("SReflectTest::MyClass")); // Base of MyDerivedClass
  EXPECT_NE(dictionary.end(), dictionary.find("SReflectTest::MyPoint")); // Field of MyClass
  EXPECT_NE(dictionary.end(), dictionary.find("std::vector<std::string>")); // Field of MyClass
}

} // namespace SReflectTest
