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

#pragma once

// If you don't want to use SimpleReflection then define SIMPLE_REFLECTION_ENABLE to 0. That
// will strip out all SimpleReflection code, including all of the markup macros. This is useful for
// inclusion into Unreal Engine code, which interacts with our libraries via pure C APIs.
#ifndef SIMPLE_REFLECTION_ENABLE
#ifdef __UNREAL__ // (if being built by the Unreal Engine build system)
#define SIMPLE_REFLECTION_ENABLE 0
#else
// SimpleReflection only works if RTTI is enabled for this compilation unit.
#if defined(__GXX_RTTI) || defined(__cpp_rtti) || (defined(_MSC_VER) && defined(_CPPRTTI))
#define SIMPLE_REFLECTION_ENABLE 1
#else
#define SIMPLE_REFLECTION_ENABLE 0
#endif
#endif
#endif

#if SIMPLE_REFLECTION_ENABLE

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Optional third-party dependencies
#ifndef SR_USE_NLOHMANN_JSON
#define SR_USE_NLOHMANN_JSON 1
#endif

// Detect availability of <memory_resource>
#ifndef SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE
#ifdef __has_include
#if __has_include(<memory_resource>)
#define SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE 1
#else
#define SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE 0
#endif
#endif // defined(__has_include)
#endif // !defined(SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE)
#ifndef SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE
#define SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE (__cplusplus >= 201703L) // C++17 or newer
#endif

#if SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE
#include <memory_resource>
#endif

// Forward
namespace picojson {
class value;
}

#if SR_USE_NLOHMANN_JSON
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4459) // declaration of 'last' hides global declaration
#endif
#include <nlohmann/json.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif // SR_USE_NLOHMANN_JSON

/**
Relatively simple reflection system. Provides a powerful and flexible set of features
for any supported types. To add support for a new type, simply add the SR_* macros to the
class/struct declaration (see examples). You can also add attributes for customized behavior.

Then you can:
  Serialize/deserialize the struct to/from json
  Construct a debug panel to edit the struct's fields
  Do customized copy of a struct instance into another instance
  Support generic 'SetProperty(x)' interface functions
  Create dynamic types that have fields that are not known at compile time
  Validate the fields of a class/struct instance
  Detect when field values are changed via e.g. DebugConsole
  Add metadata to fields for presentation in a user-facing UI
  Add custom attributes to fields to do anything you want
  ...and more!

Does not currently support (but can be added as needed):
  Some STL containers (list, map, set, ...)
  Binary serialization
  Pointers (hopefully can avoid this)
  C++ types with no default/zero-argument constructor

Example 1 - Simple struct with some attributes:

  struct Thing {
    float f1 = 0;
    std::vector<int> f2;
    std::string f3;
    bool f4[23];

    SR_BeginStruct(Thing); SRA_Description("Super useful struct");
    SR_Field(f1); SRA_FloatRange(0.0f, 1.0f);
    SR_Field(f2);
    SR_Field(f3);
    SR_Field(f4);
    SR_EndStruct();
  };

  void UseThing() {
    Thing thing;
    thing.f1 = 15.0f;

    // Validation will fail in this case, because f1 is out-of-range.
    bool isValid = SReflect::IsValid(thing);

    // Serialization
    SReflect::SaveToJsonFile(thing, "somewhere/thing.json");

    // Deserialization
    int numIssuesDetected;
    SReflect::LoadFromJsonFile(thing, "somewhere/thing.json", numIssuesDetected);

    // Get the TypeInfo and do whatever you need...
    const SReflect::StructTypeInfo& typeInfo = thing.GetTypeInfo();
  }

Example 2 - Class using BaseObject

  class Thing : public SReflect::BaseObject {
    std::string name;

    SR_BeginClass(Thing);
    SR_Field(name);
    SR_EndClass();
  };

  void UseAnyObject(SReflect::BaseObject const& obj) {
    // BaseObject has a virtual function that gives us the derived TypeInfo.
    // Thus, we can now do everything without templates.

    // Serialization
    SReflect::SaveToJsonFile(obj, "somewhere/any_object.json");

    // Deserialization
    int numIssuesDetected;
    SReflect::LoadFromJsonFile(thing, "somewhere/any_object.json", numIssuesDetected);

    // Get the derived TypeInfo and do whatever you need...
    const SReflect::StructTypeInfo& typeInfo = thing.GetFinalTypeInfo();
  }

  void UseThing() {
    Thing thing;
    thing.name = "Bob";
    UseAnyObject(thing);
  }
*/

///////////////////////////////////////////////////////////////////////////
// Forwards
///////////////////////////////////////////////////////////////////////////
namespace SReflect {
struct Attribute;
class TypeInfo;
class StructTypeInfo;
} // namespace SReflect

namespace picojson {
class value;
}

///////////////////////////////////////////////////////////////////////////
// Macros (implemented in simple_reflection_inl.h)
///////////////////////////////////////////////////////////////////////////

// clang-format off

/**
To add reflection support to a class or struct intrusively, add an SR_BeginStruct/SR_EndStruct
block inside the class/struct declaration. This declares the static member function GetTypeInfo
and the non-static member function GetFinalTypeInfo. It does NOT change the size nor layout of
the type. Your class/struct can optionally derive from BaseObject, in which case GetFinalTypeInfo
will be a virtual override.

Example:
  struct MyStruct {
    int value = 0;

    SR_BeginStruct(MyStruct);
    SR_Field(value);
    SR_EndStruct();
  };

*/
#define SR_BeginStruct(structType)                IMPL_SR_BeginStruct(structType, #structType)
#define SR_EndStruct()                            IMPL_SR_EndStruct()

/**
To add reflection support to a template class or struct intrusively, use SR_BeginStructTemplate
instead of SR_BeginStruct. This allows Simple Reflection to format the correct type name when instantiated,
and it ensures that every unique instantiation of the type will have a unique TypeId.
SR_BeginClassTemplate is the same, except that it declares "public:" access. These macros typically
go at the END of the class declaration.

Example:
  template<class T, int N>
  struct MyTemplate {
    std::array<T, N> values{};

    SR_BeginStructTemplate(MyTemplate, T, N);
    SR_Field(values);
    SR_EndStruct();
  };

Limitations:
  - All argument types must be supported by Simple Reflection.
  - You can use an enum constant as a non-type argument, but it must be exposed via SR_Item.
  - Unfortunately, non-type arguments are only supported in the first 3 argument positions.
    If you need a non-type argument after that, then your options are:
    1) Use SR_BeginStruct(YourTemplate<Args>). This will work, but all instantiations of the template will
       have the same name and TypeId.
    2) Extend Simple Reflection support (see SReflect::detail::AppendTemplateArgStr).
    3) Implement your own specialization of SReflectTypeTraits and format the name there.
*/
#define SR_BeginStructTemplate(structType, ...)   IMPL_SR_BeginStructTemplate(structType, __VA_ARGS__)
#define SR_BeginClassTemplate(classType, ...)     public: IMPL_SR_BeginStructTemplate(classType, __VA_ARGS__)

/**
To add reflection support to a class or struct non-intrusively, add an SR_BeginStructEx/SR_EndStructEx block
outside of your class/struct declaration ("ex" is for external). This defines a specialization of SReflectTypeTraits.

Example:
  struct MyStruct {
    int value = 0;
  };

  SR_BeginStructEx(MyStruct);
  SR_Field(value);
  SR_EndStructEx();

*/
#define SR_BeginStructEx(structType)    IMPL_SR_BeginStructEx(structType)
#define SR_EndStructEx()                IMPL_SR_EndStructEx()

/**
To reduce code bloat, you can split up the declaration and definition of a reflection class/struct.

Example:
  // In header file
  struct MyStruct {
    int value = 0;
  };
  SR_DeclareStructEx(MyStruct);

  // In cpp file:
  SR_BeginStructDefinitionEx(MyStruct)
  SR_Field(value);
  SR_EndStructDefinitionEx();
*/

#define SR_DeclareStructEx(structType)            IMPL_SR_DeclareStructEx(structType)
#define SR_BeginStructDefinitionEx(structType)    IMPL_SR_BeginStructDefinitionEx(structType)
#define SR_EndStructDefinitionEx()                IMPL_SR_EndStructDefinitionEx()

/**
Adds a field (member variable) to the class or struct being defined.
Goes between the SR_BeginStruct and SR_EndStruct macros. See examples (above).
*/
#define SR_Field(member)                          IMPL_SR_Field_Name(member, #member)

/**
Adds a field (member variable) to the class or struct, but give it a name that is different
from the actual class member. Example: SR_Field_Name(_myValue, "value")
*/
#define SR_Field_Name(member, name)               IMPL_SR_Field_Name(member, name)

/**
Remove a field that was previously added. This can be used to hide a base class field that in
case you want to expose the information in a different way.
*/
#define SR_RemoveField(name)                      IMPL_SR_RemoveField(name)

/**
Same as SR_BeginStruct except that it declares reflection support in public scope.
Anything after SR_EndClass() will be in private scope. Thus, a SR_BeginClass/SR_EndClass
pair can always go at the end of a class declaration.
*/
#define SR_BeginClass(classType)        public: IMPL_SR_BeginStruct(classType, #classType)
#define SR_EndClass()                   IMPL_SR_EndStruct(); private: static_assert(true)

/**
When reflection support is added non-intrusively, there is no difference between classes
and structs. These macros exist just for symmetry.
*/
#define SR_BeginClassEx(classType)      IMPL_SR_StructEx(classType)
#define SR_EndClassEx()                 IMPL_SR_EndStructEx()

/**
Adds a base class to the class/struct that is currently being defined, including all of its fields.
Goes between the SR_BeginStruct/SR_EndStruct macros. Does not allow virtual inheritence, diamond
inheritience, nor multiple inheritence of base classes with non-static data members (but inheritence
of multiple interface classes is fine).
*/
#define SR_BaseClass(baseClass)       IMPL_SR_BaseClass(baseClass)

/**
Adds an attribute to provide additional information, or customize behavior.
If the attribute comes after SR_BeginStruct/SR_BeginClass then it describes the class/struct itself.
If the attribute comes after an SR_Field, then it describes the field.
See the individual attribute classes (below) for more details.
*/
#define SRA_DoNotSerialize()          { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_DoNotSerialize); }
#define SRA_DoNotSerializeDefaults(...)  { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_DoNotSerializeDefaults(__VA_ARGS__)); } // Currently only supported for JSON serialization
#define SRA_ReadOnly()                { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_ReadOnly); }
#define SRA_FloatRange(min1, max1)    { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_FloatRange{min1, max1}); }
#define SRA_IntRange(min1, max1)      { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_IntRange{min1, max1}); }
#define SRA_UIntRange(min1, max1)     { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_UIntRange{min1, max1}); }
#define SRA_OnChanged(cb)             { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_OnChanged{cb}); }
#define SRA_Units(str)                { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_Units{str}); }
#define SRA_DisplayName(str)          { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_DisplayName{str}); }
#define SRA_Description(str)          { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_Description{str}); }
#define SRA_Category(str)             { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_Category{str}); }
#define SRA_PreviouslyKnownAs(...)    { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_PreviouslyKnownAs{__VA_ARGS__}); }
#define SRA_HideFromEditor()          { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_HideFromEditor{}); }
#define SRA_OnDeserialize(cb)         { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_OnDeserialize{cb}); }
#define SRA_IgnoreExtraneousFields()  { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_IgnoreExtraneousFields{}); }
#define SRA_JsonString()              { SReflect::detail::AddAttribute(*myInfo, new SReflect::Attribute_JsonString{}); }

/**
Provide the reflection information for an enum class. Must go in the global namespace.
Example usage:
    enum class Fruit { Apple, Orange };

    SR_BeginEnum(Fruit)
    SR_EnumItem(Apple)
    SR_EnumItem(Orange)
    SR_EndEnum()

If an enum value is renamed, use SRA_PreviouslyKnownAs to keep the old name recognized during
deserialization. Serialization always uses the current name.

    enum class Fruit { GrannySmith, Orange };

    SR_BeginEnum(Fruit)
    SR_EnumItem(GrannySmith) SRA_PreviouslyKnownAs("Apple")
    SR_EnumItem(Orange)
    SR_EndEnum()
*/
#define SR_BeginEnum(enumType)      IMPL_SR_BeginEnum(enumType)
#define SR_EnumItem(enumValue)      IMPL_SR_EnumItem(enumValue)
#define SR_EndEnum()                IMPL_SR_EndEnum()

// clang-format on

/**
Simple Reflection
*/
namespace SReflect {

/**
Interface used to customize memory allocation. Compatible with std::pmr::memory_resource (if
supported by your compiler). Note that all methods that take an Allocator* argument will use a
default allocator if you pass nullptr.
*/
#if SIMPLE_REFLECTION_HAS_MEMORY_RESOURCE
using Allocator = std::pmr::memory_resource;
#else
class Allocator {
 public:
  virtual ~Allocator() = default;

  // Allocate memory with the specified alignment (must be a power of 2)
  void* allocate(size_t bytes, size_t alignment) {
    return do_allocate(bytes, alignment);
  }

  // Deallocate memory that was previously allocated with the same allocator.
  void deallocate(void* ptr, size_t bytes, size_t alignment) {
    return do_deallocate(ptr, bytes, alignment);
  }

  // Return true if the other allocator is equivalent to this one, meaning that memory allocated by
  // one can be deallocated by the other.
  bool is_equal(Allocator const& other) const noexcept {
    return do_is_equal(other);
  }

 protected:
  virtual void* do_allocate(size_t bytes, size_t alignment) = 0;
  virtual void do_deallocate(void* ptr, size_t bytes, size_t alignment) = 0;
  virtual bool do_is_equal(Allocator const& other) const noexcept = 0;
};
#endif

/**
Get the default Allocator implementation.
*/
Allocator* GetDefaultAllocator();

/**
Interface for opaquely writing bytes into a stream
*/
class StreamWriter {
 public:
  [[nodiscard]] virtual bool Write(void const* src, size_t numBytes) = 0;
  virtual ~StreamWriter() = default;
};

/**
Interface for opaquely reading bytes out of a stream
*/
class StreamReader {
 public:
  [[nodiscard]] virtual bool Read(void* dst, size_t numBytes) = 0;
  virtual ~StreamReader() = default;
};

/**
Contiguous span (array) of elements. T may be const or non-const.
*/
template <typename T>
class Span {
 public:
  // clang-format off
  Span() = default;
  template<typename RHS> Span(RHS& rhs) : _ptr(std::data(rhs)), _len(std::size(rhs)) {}
  Span(T* ptr, size_t len) : _ptr(ptr), _len(len) {}
  [[nodiscard]] T* begin() const { return _ptr; }
  [[nodiscard]] T* end() const { return _ptr + _len; }
  [[nodiscard]] T* data() const { return _ptr; }
  [[nodiscard]] size_t size() const { return _len; }
  // clang-format on
 private:
  T* _ptr = nullptr;
  size_t _len = 0;
};

/**
StreamWriter implementation backed by std::vector<uint8_t>.
Vector grows as needed.
*/
class VecStreamWriter : public StreamWriter {
 public:
  // Constructors
  VecStreamWriter();
  explicit VecStreamWriter(size_t capacity);

  // StreamWriter interface
  [[nodiscard]] bool Write(void const* src, size_t numBytes) override;

  // Accessors
  std::vector<uint8_t> TakeBytes() {
    return std::move(dst);
  }
  [[nodiscard]] Span<const uint8_t> GetBytes() const;
  [[nodiscard]] size_t GetNumBytesWritten() const;

 private:
  std::vector<uint8_t> dst;
};

/**
StreamWriter that writes into a fixed-size span.
Writes fail once span is exhausted.
*/
class SpanStreamWriter : public StreamWriter {
 public:
  // Constructors
  explicit SpanStreamWriter(Span<uint8_t> buffer);

  // StreamWriter interface
  [[nodiscard]] bool Write(void const* src, size_t numBytes) override;

  // Accessors
  [[nodiscard]] Span<const uint8_t> GetBytes() const;
  [[nodiscard]] size_t GetNumBytesWritten() const;

 private:
  Span<uint8_t> dst;
  size_t iDst = 0;
};

/**
StreamReader that reads from fixed-size span
Reads fail once span is exhausted.
*/
class SpanStreamReader : public StreamReader {
 public:
  // Constructors
  SpanStreamReader(Span<const uint8_t> buffer);

  // StreamReader interface
  [[nodiscard]] bool Read(void* dst, size_t numBytes) override;

  // Utilities
  [[nodiscard]] bool HasUnreadBytes() const;
  [[nodiscard]] size_t GetNumBytesRead() const;
  [[nodiscard]] size_t GetNumBytesRemaining() const;

 private:
  Span<const uint8_t> src;
  size_t iSrc = 0;
};

///////////////////////////////////////////////////////////////////////////
// TypeId
///////////////////////////////////////////////////////////////////////////

/**
The SReflect::TypeId is an identifier based on the hash of TypeInfo::_nameWithNamespace.
It can be used for quick comparison, even across multiple DLLs. It is illegal to declare
two different types with the same TypeId (either by anonymous namespace or hash collision).
*/
struct TypeId {
  uint64_t value = 0;

  TypeId() = default;
  explicit TypeId(uint64_t val) : value(val) {}
  [[nodiscard]] bool IsValid() const {
    return value != 0;
  }
  bool operator==(TypeId const& rhs) const;
  bool operator!=(TypeId const& rhs) const;
  bool operator<(TypeId const& rhs) const;
};

///////////////////////////////////////////////////////////////////////////
// Core Types
///////////////////////////////////////////////////////////////////////////

// Note: changes to this enum must be reflected in simple_reflection.cpp.
enum class CoreType {
  X_Invalid,

  // std::string
  CT_string,

  // Built-in arithmetic types:
  CT_bool,
  CT_uint8,
  CT_int8,
  CT_uint16,
  CT_int16,
  CT_uint32,
  CT_int32,
  CT_uint64,
  CT_int64,
  CT_float,
  CT_double,

  // Enum class or enum struct (see EnumTypeInfo)
  CT_enum,

  // Array of elements (see ArrayTypeInfo)
  // Examples: std::vector<T>, std::array<T,N>, T[N]
  CT_array,

  // 2D array of elements (see MatrixTypeInfo)
  CT_matrix,

  // Class or struct (see StructTypeInfo)
  CT_struct,

  // Member of a class or struct (see FieldTypeInfo)
  CT_field,

  // Associative container (see MapTypeInfo)
  // Example: std::unordered_map
  CT_map,

  // Optional type (see OptionalTypeInfo)
  // Example: std::optional
  CT_optional,

  // Variant type where the inner type is selected at runtime (see VariantTypeInfo).
  // Example: std::variant
  CT_variant,

  // DEPRECATED
  CT_other,

  X_Count,
};

///////////////////////////////////////////////////////////////////////////
// BaseObject
///////////////////////////////////////////////////////////////////////////

/**
A common base for polymorphic classes wishing to support reflection via
SR_BeginStruct/SR_EndStruct or SR_BeginClass/SR_EndClass macros. The only real cost is that it
requires a v-table pointer (and thus should not be used on POD structures like float3)
*/
class BaseObject {
 public:
  virtual ~BaseObject() = default;
  [[nodiscard]] virtual StructTypeInfo const& GetFinalTypeInfo()
      const = 0; //<! Get the derived type info
  [[nodiscard]] virtual TypeId GetFinalTypeId()
      const = 0; //<! Get the hash code from the derived type info
};

///////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////

/**
Return true if T is a type that is supported by Simple Reflection. You can extend support to your
own types using macros like SR_BeginEnum/SR_EndEnum, SR_BeginStruct/SR_EndStruct, and others. For
more advanced cases, you can declare a specialization of SReflectTypeTraits directly.
*/
template <typename T>
constexpr bool IsSupportedType();

/**
Get the type information for any supported type.
The return type depends on the template parameter. For example:
  SReflect::TypeInfo const& // Core type
  SReflect::StructTypeInfo const& // Class or struct
  SReflect::EnumTypeInfo const& // Enum
  SReflect::ArrayTypeInfo const& // Array type like std::vector, std::array, T[N]
  SReflect::MapTypeInfo const& // Associative container like std::unordered_map
*/
// TODO: this function and friends should be refactored to SR_GetTypeInfo
template <typename T>
const auto& GetTypeInfo();

/**
Return a pointer to the type info structure, or nullptr.
Equivalent to: IsSupportedType<T>() ? &GetTypeInfo<T>() : nullptr
*/
template <typename T>
const auto* TryGetTypeInfo();

/**
Get the type information from an object of any supported type. If the object is a
class deriving from BaseObject, then it will call the virtual GetFinalTypeInfo() to
return the information from the concrete derived class. Otherwise, it is equivalent
to SReflect::GetTypeInfo<T>().
*/
template <typename T>
const auto& GetFinalTypeInfo(T const& obj);

/**
Equivalent to SReflect::GetTypeInfo<T>()._typeId
*/
template <typename T>
TypeId GetTypeId();

/**
Equivalent to SReflect::GetFinalTypeInfo<T>(obj)._typeId;
*/
template <typename T>
TypeId GetFinalTypeId(T const& obj);

/**
Use the TypeInfo to validate an object of any supported type. Return true on success.
*/
template <typename T>
bool IsValid(T const& obj);

/**
Return the name of an enum item, given the numeric value.
Return an empty string if not found.
*/
template <typename EnumT>
[[nodiscard]] char const* EnumToString(EnumT value);

/**
Serialize any supported type to a JSON string
*/
template <typename T>
std::string ToJsonString(T const& obj, bool pretty = true);

/**
Serialize any supported type to picojson
*/
template <typename T>
void ToJsonValue(T const& obj, picojson::value& outJson);

/**
Serialize a JSON dictionary containing information about the specified types and all of the
nested types (attributes, fields, etc...). Types are identified by their fully-qualified names
(`_nameWithNamespace`). See TypeInfo::TypeInfoToJson
*/
std::string
TypeInfoListToJson(SReflect::TypeInfo const* const* typeList, size_t numTypes, bool pretty = true);

#if SR_USE_NLOHMANN_JSON
template <typename T>
void ToJson(nlohmann::json& outJson, T const& value) {
  outJson = nlohmann::json::parse(SReflect::ToJsonString(value, false));
}
#endif // SR_USE_NLOHMANN_JSON

/**
Flags to control deserialization
*/
enum class DeserializeFlags : uint32_t {
  // clang-format off
  None                   = 0,
  WarnIfMissingFields    = 1U << 0,  // Warn if a field is missing from the serialized data; this is common if a new field is added to typeinfo, or serialized data is 'sparse', usually not a problem
  WarnIfExtraneousFields = 1U << 1,  // Warn if the serialized data has extra fields that do not exist in typeinfo; this is common if a typinfo field has been renamed, or hand-edited serialized data has a typo in a field name (and needs to be fixed)
  Default                = WarnIfExtraneousFields,
  MaximumWarnings        = WarnIfMissingFields | WarnIfExtraneousFields,
  // clang-format on
};

/**
Deserialize any supported type from a JSON string
*/
template <typename T>
void FromJsonString(
    T& objOut,
    std::string const& json,
    DeserializeFlags deserializeFlags,
    int& outNumIssues);

template <typename T>
bool FromJsonString(
    T& objOut,
    std::string const& json,
    DeserializeFlags deserializeFlags = DeserializeFlags::Default) {
  int numIssues = 0;
  FromJsonString(objOut, json, deserializeFlags, numIssues);
  return numIssues == 0;
}

// This overload throws an exception if deserialization fails
template <typename T>
T FromJsonString(
    std::string const& json,
    DeserializeFlags deserializeFlags = DeserializeFlags::Default) {
  T value{};
  bool success = FromJsonString<T>(value, json, deserializeFlags);
  if (!success) {
    throw std::runtime_error{"Failed to deserialize JSON string"};
  }
  return value;
}

#if SR_USE_NLOHMANN_JSON
template <typename T>
void FromJson(nlohmann::json const& j, T& outValue) {
  FromJsonString(outValue, j.dump());
}
#endif // SR_USE_NLOHMANN_JSON

/**
Deserialize any supported type from picojson
*/
template <typename T>
void FromJsonValue(
    T& objOut,
    picojson::value const& json,
    DeserializeFlags deserializeFlags,
    int& outNumIssues);

/**
Serialize any supported type to a JSON file. Return false if the IO failed.
*/
template <typename T>
bool SaveToJsonFile(T const& obj, char const* filePath, bool warnOnFailure = true);

/**
Deserialize any supported type from a JSON file.  Return false if the IO failed.
*/
template <typename T>
bool LoadFromJsonFile(
    T& objOut,
    char const* filePath,
    DeserializeFlags deserializeFlags,
    int& outNumIssues);

/**
    Serialize any supported type to binary blob.
    Binary format is not versioned and incompatible across any data structure change.
*/
template <typename T>
std::vector<uint8_t> ToBytes(T const& src) noexcept;

/**
Serialize any supported type to binary blob.
Binary format is not versioned and incompatible across any data structure change.
*/
template <typename T>
bool ToBytes(T const& src, StreamWriter& dst) noexcept;

/**
Deserializes any supported type from binary blob.
*/
template <typename T>
bool FromBytes(StreamReader& src, T& dst);

/**
Deserializes any supported type from binary blob.
throws std::exception on any error
*/
template <typename T>
T FromBytes(StreamReader& src);

/**
Compute a TypeId from the a full type name. For any type T, the following is always true:
  ComputeTypeId(GetTypeInfo<T>()._nameWithNamespace) == GetTypeId<T>()
*/
TypeId ComputeTypeId(char const* nameWithNamespace);

/**
Return the name of a CoreType as a null-terminated string.
*/
char const* CoreTypeToString(CoreType coreType);

/**
Compute a 64-bit hash for an array of bytes using the same hash algorithm as ComputeTypeId.
*/
uint64_t CalcHash64(void const* src, size_t numBytes);

///////////////////////////////////////////////////////////////////////////
// TypeInfo
///////////////////////////////////////////////////////////////////////////

/**
Provides runtime information about any supported type. Accessed via SReflect::GetTypeInfo<T>().
Some types use classes derived from TypeInfo to add more information (see StructTypeInfo,
EnumTypeInfo, etc...)
*/
class TypeInfo {
 public:
  using AttributeList = std::vector<std::pair<SReflect::TypeId, Attribute const*>>;

  CoreType _coreType = CoreType::X_Invalid;
  size_t _alignment = 0; // alignof(T) for type T
  size_t _sizeInBytes = 0; // sizeof(T) for type T
  AttributeList _attributes;

  /**
  Uses placement new to default construct an object at the specified address. Arithmetic types and
  arrays of arithmetic types will be zero-initialized. The address must have sufficient alignment
  (see _alignment) and sufficient size (see _sizeInBytes).
  WARNING: Will be nullptr for types that do not support default construction.
  */
  void (*_constructInPlace)(void* dst) = nullptr;

  /**
  Uses placement new to copy construct an object at the specified address. The address must have
  sufficient alignment (see _alignment) and sufficient size (see _sizeInBytes).
  WARNING: Will be nullptr for types that do not support default construction.
  */
  void (*_constructInPlaceByCopy)(void* dst, void const* src) = nullptr;

  /**
  Calls an object's destructor, but does NOT release the memory. This function should be used to
  clean up any object created by _defaultConstructInPlace or _copyConstructInPlace. Guranteed to
  be non-null if any of the construction methods are non-null.
  */
  void (*_destructInPlace)(void* ptr) = nullptr;

  /**
  Short name of type or field. No "::" separators.
  Examples: "float", "myField", "MyClass"
  */
  char const* _name = nullptr;

  /**
  Full name of the type including namespace path with "::" separators. The One Definition Rule
  dictates that this name will be unique within the program unless it is declared within an
  anonymous namespace or within the scope of a function. For primitive types and fields, the _name
  and _nameWithNamespace will be the same.
  Examples: "float", "myField", "MyGlobalClass", "MyNamespace::MyClass",
  "std::vector<std::string>"
  */
  char const* _nameWithNamespace = nullptr;

  /**
  A 64-bit hash of the _nameWithNamespace. Useful for fast comparison and for use in associative
  containers. Computed by frl::HashString(_nameWithNamespace)
  */
  TypeId _typeId;

  TypeInfo() = default;
  virtual ~TypeInfo() = default;

  /**
  Allocate and default construct a new object, similar to calling `new T{}`.
  You may specify a custom allocator. Returns nullptr if not supported.
  Call TypeInfo::Delete for cleanup.
  */
  void* New(Allocator* allocator = nullptr) const;

  /**
  Allocate and copy construct a new object, similar to calling `new T{src}`.
  You may optionally specify a custom allocator. Returns nullptr if not supported.
  Call TypeInfo::Delete for cleanup.
  */
  void* Clone(void const* src, Allocator* allocator = nullptr) const;

  /**
  Destruct and deallocates an object similar to calling `delete T`.
  If you specified a custom memory_resource when allocating the object, then you must provide the
  SAME one here.
  */
  void Delete(void* ptr, Allocator* allocator = nullptr) const;

  /**
  Copy data from src into dst
  */
  void Set(void const* src, void* dst) const;

  /**
  Serialize src into a picojson::value
  */
  void Serialize(void const* src, picojson::value& dst) const;

  /**
  Deserialize data from a picojson::value into dst
  */
  void Deserialize(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& outIssuesDetected) const;

  /**
  Serialize src in binary format into dst
  */
  bool SerializeToBytes(void const* src, StreamWriter& dst) const;

  /**
  Deserialize from src reader into dst
  */
  bool DeserializeFromBytes(StreamReader& src, void* dst) const;

  /**
  Serialize this TypeInfo metadata to a JSON string.

  Serializes the information for this type and all nested types (fields, attributes, base classes,
  etc...) as a single flat dictionary, keyed by the fully-qualified type names
  (`_nameWithNamespace`).
  */
  [[nodiscard]] std::string TypeInfoToJson(bool pretty = true) const;

  /**
  Serialize this TypeInfo metadata into dst as a flat dictionary keyed by fully-qualified name.

  On entry, if 'dst' is not a picojson::object it is set to one. This function is additive: it
  never clears 'dst'. If this type's key is already present it returns immediately, which breaks
  cycles and removes redundancy. Calling it repeatedly with different root types accumulates the
  union of all reachable types into one flat dictionary. Each entry describes a single type; nested
  types are referenced by their fully-qualified name, never embedded directly.
  */
  void SerializeTypeInfo(picojson::value& dst) const;

  /**
  Check if src is valid, based on attributes
  */
  bool IsValid(void const* src) const;

  /**
  Return a pointer to an attribute with the given type. Return nullptr if not found.
  */
  template <typename T>
  T const* GetAttribute() const {
    return static_cast<T const*>(GetAttribute(T::GetTypeId()));
  }
  [[nodiscard]] Attribute const* GetAttribute(TypeId attrType) const; // Non-template version

  /**
  Check if this TypeInfo has an attribute on it with the given type
  */
  template <typename T>
  [[nodiscard]] bool HasAttribute() const {
    return HasAttribute(T::GetTypeId());
  }
  [[nodiscard]] bool HasAttribute(TypeId attrType) const;

  /**
  Check if objects of this type can be safely memcopied
  */
  [[nodiscard]] virtual bool IsMemCopySafe() const;

  // Details to be implemented by subclasses:
  virtual uint64_t ToUInt64(void const* src) const;
  virtual void FromUInt64(uint64_t val, void* dst) const;
  virtual void SetInner(void const* src, void* dst) const = 0;

  // These functions should only be called from TypeInfo implementations
  virtual bool IsValidInner(void const* src, AttributeList const& attribs) const = 0;
  virtual void SerializeInner(void const* src, picojson::value& dst) const = 0;
  virtual void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const = 0;
  virtual bool SerializeToBytesInner(void const* src, StreamWriter& dst) const = 0;
  virtual bool DeserializeFromBytesInner(StreamReader& src, void* dst) const = 0;

 protected:
  virtual void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const;
};

///////////////////////////////////////////////////////////////////////////
// Fields
///////////////////////////////////////////////////////////////////////////

/**
Describes a field within a class or struct.
*/
class FieldTypeInfo : public TypeInfo {
 public:
  TypeInfo const* _innerTypeInfo = nullptr;
  size_t _offset = 0;

  uint8_t* GetFieldPtr(void* structBasePtr) const;
  uint8_t const* GetFieldPtr(void const* structBasePtr) const;
};

///////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////

/**
Describes a class or struct.
Created by the SR_BeginStruct or SR_BeginClass macro.
Accessed via SReflect::GetTypeInfo<T>() or T::GetTypeInfo().
*/
class StructTypeInfo : public TypeInfo {
 public:
  explicit StructTypeInfo(std::type_info const& rttiInfo);

  std::vector<FieldTypeInfo*> _fields; // Includes fields inherited via SR_BaseClass
  std::vector<StructTypeInfo const*> _baseClasses; // Includes all bases recursively
  std::type_index _typeIndex;
  bool _isMemCopySafe = false;

  /**
  Find a field by name (case-sensitive) or return nullptr if not found.
  */
  [[nodiscard]] FieldTypeInfo const* FindField(std::string_view name) const;

  /**
  Find a base class's info by SReflect::TypeId or return nullptr if not found.
  */
  [[nodiscard]] StructTypeInfo const* FindBaseClass(TypeId id) const;

  /**
  Return true if the TypeId belongs to the class/struct described by
  this StructTypeInfo or one of its base classes.
  */
  [[nodiscard]] bool IsSameOrDerivedFrom(TypeId typeId) const;

  /**
  Legacy version of IsSameOrDerivedFrom, which uses the std::type_index
  */
  [[nodiscard]] bool IsSameOrDerivedFrom(std::type_index typeIndex) const;

  /**
  Return true if type T is the class/struct being described by this StructTypeInfo
  or one of its base classes.
  */
  template <typename T>
  [[nodiscard]] bool IsSameOrDerivedFrom() const;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Enums
///////////////////////////////////////////////////////////////////////////

/**
Describes a single constant within an enum. Added via SR_EnumItem macro.
Enum items can have attributes (e.g. SRA_PreviouslyKnownAs) placed after the SR_EnumItem macro.
*/
struct EnumItem {
  char const* _name;
  uint64_t _value;
  TypeInfo::AttributeList _attributes;

  EnumItem(char const* name, uint64_t value) : _name(name), _value(value) {}

  template <typename T>
  T const* GetAttribute() const {
    return static_cast<T const*>(GetAttribute(T::GetTypeId()));
  }

  [[nodiscard]] Attribute const* GetAttribute(TypeId attrType) const;
};

/**
Describes an enum type. Created via SR_BeginEnum/SR_EndEnum macros.
Accessed via SReflect::GetTypeInfo<YourEnum>().
*/
class EnumTypeInfo : public TypeInfo {
 public:
  std::vector<EnumItem> _items;
  TypeInfo const* _innerTypeInfo = nullptr;

  [[nodiscard]] EnumItem const* FindItemByValue(uint64_t value) const;
  [[nodiscard]] EnumItem const* FindItemByName(std::string_view name) const;

  // Get the current value of an enum variable as a uint64_t
  virtual uint64_t GetValue(void const* obj) const = 0;

  // Set the value of an enum variable by uint64_t
  virtual void SetValue(void* obj, uint64_t value) const = 0;

  // Return the C++ type_index. This value should not be serialized to disk, but
  // can be used at runtime to uniquely identify the type.
  [[nodiscard]] virtual std::type_index GetTypeIndex() const = 0;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Strings
///////////////////////////////////////////////////////////////////////////

/**
Describes a string of characters like std::string.
*/
class StringTypeInfo : public TypeInfo {
 public:
  // If true, the string will be followed by a zero byte, which is legal to read.
  bool _isNullTerminated = false;

  // If the string is read-only, then calls to SetString will fail.
  bool _isReadOnly = false;

  // Get a view of the string (read only). It is legal to read one additional
  // byte beyond the length of the string if (and only if) _isNullTerminated,
  // in which case that byte will have value zero.
  virtual std::string_view GetString(void const* obj) const = 0;

  // Replace the contents of the string. Return false if read-only.
  virtual bool SetString(void* obj, std::string_view str) const = 0;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Vectors & Arrays
///////////////////////////////////////////////////////////////////////////

/**
Extends TypeInfo with additional functionality for types with CoreType::CT_array.
The defining characterisitc of an array is that it has some numer of elements that can be
accessed by index. The number of elements may be fixed, or may be resizable. Example types
include: std::vector<T>, std::array<T, N>, T[N], and other array-like types that the user
may define.
*/
class ArrayTypeInfo : public TypeInfo {
 public:
  TypeInfo const* _innerTypeInfo = nullptr;

  [[nodiscard]] virtual bool CanResize() const = 0; //<! Is SetNumElements supported?
  virtual size_t GetNumElements(void const* obj) const = 0; //<! Get the number of elements.
  virtual bool SetNumElements(void* obj, size_t numElements)
      const = 0; //<! Attempt to change the number of elements. Return true if successful.
  virtual void* GetElement(void* obj, size_t i) const = 0; //<! Get the address of an element.
  virtual void const* GetElement(void const* obj, size_t i)
      const = 0; //<! Get the address of an element.

 protected:
  bool IsValidInner(void const* src, AttributeList const& attribs) const override;
  void SetInner(void const* src, void* dst) const override;
  void SerializeInner(void const* src, picojson::value& dst) const override;
  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override;
  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override;
  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Matrices
///////////////////////////////////////////////////////////////////////////

/**
Extends TypeInfo with additional functionality for types with CoreType::CT_matrix.
Matrix types have the following characteristics:
  - A 2D array of elements of the inner type.
  - The inner type currently requires IsMemCopySafe() == true, but that could be changed.
  - Dimensions may be fixed or dynamic.
  - Memory storage is either row-major or column-major.
  - If row-major, then values in a row are stored contiguously.
  - If column-major, then values in a column are stored contiguously.
  - May have a "leading dimension" that is greater than the number of columns (for row-major) or
    greater than the number of rows (for column-major). This indicates that there is extra space
    ("stride") between the storage of these rows or columns.
JSON Serialization:
  - Writes nested arrays in row-major order. Example: "[[1,2,3], [4,5,6]]" (2 rows, 3 cols)
  - Exception: A matrix with 1 fixed row or 1 fixed column writes a 1D array. Example: "[1,2,3]"
Binary Serialization:
  - Only writes the dimensions if they are dynamic.
  - Then writes the values in the native memory order.
*/
class MatrixTypeInfo : public TypeInfo {
 public:
  TypeInfo const* _innerTypeInfo = nullptr;
  bool _isRowMajor = false; //<! True if row-major, false if column-major
  bool _isNumRowsDynamic = false; //<! True if numRows is a runtime property. False if fixed.
  bool _isNumColumnsDynamic = false; //<! True if numColumns is a runtime property. False if fixed.
  size_t _fixedNumRows = 0; //<! Fixed number of rows or zero if _isNumRowsDynamic
  size_t _fixedNumColumns = 0; //<! Fixed number of columns or zero if _isNumColumnsDynamic

  struct Layout {
    size_t _numRows = 0;
    size_t _numColumns = 0;
    size_t _leadingDim = 0;
  };

  // Get information about dimensions and memory layout.
  [[nodiscard]] Layout GetLayout(void const* obj) const;

  // Get the address of the element with coordinates (0, 0).
  // Could return nullptr if the matrix is zero size on either dimension.
  [[nodiscard]] virtual void* GetData(void* obj) const = 0;
  [[nodiscard]] void const* GetData(void const* obj) const;

  // Try to set the number of rows and columns in the matrix. Returns true if the resize was
  // successful, or if no resize was necessary. Existing values are not necessarily preserved.
  [[nodiscard]] virtual bool TryResize(void* obj, size_t numRows, size_t numColumns) const = 0;

  // Get the address of a single element within the matrix.
  // Triggers an SR_ASSERT if coordinates are out-of-bounds.
  [[nodiscard]] void* GetElement(void* obj, size_t row, size_t column) const;
  [[nodiscard]] void const* GetElement(void const* obj, size_t row, size_t column) const;

  // TypeInfo overrides:
  bool IsValidInner(void const* obj, AttributeList const& attribs) const final;
  void SetInner(void const* src, void* dst) const final;
  void SerializeInner(void const* src, picojson::value& dst) const final;
  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const final;
  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const final;
  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const final;

 protected:
  virtual Layout GetLayoutImpl(void const* obj) const = 0;
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Maps
///////////////////////////////////////////////////////////////////////////

class MapTypeInfo : public SReflect::TypeInfo {
 public:
  TypeInfo const* _keyTypeInfo = nullptr;
  TypeInfo const* _valueTypeInfo = nullptr;

  // Get the number of key-value pairs in the map
  virtual size_t GetNumKeys(void const* map) const = 0;

  // Enumerate the key-value pairs. If the callback returns false, then stop enumerating.
  using OnEach = std::function<bool(void const* key, void* value)>;
  virtual void Enumerate(void* map, OnEach const& callback) const = 0;

  // Enumerate const key-value pairs. If the callback returns false, then stop enumerating.
  using OnEachConst = std::function<bool(void const* key, void const* value)>;
  virtual void Enumerate(void const* map, OnEachConst const& callback) const;

  // Try to remove all key-value pairs from the map. Return false if read-only.
  virtual bool Clear(void* map) const = 0;

  // Try to insert a new key-value pair, or replace an exising one. Return false if read-only.
  virtual bool Insert(void* map, void const* key, void const* value) const = 0;

  // Try to remove a key-value pair. Return false if read-only or if key does not exist.
  virtual bool Remove(void* map, void const* key) const = 0;

  // TypeInfo overrides
  bool IsValidInner(void const* src, AttributeList const& attribs) const override;
  void SerializeInner(void const* src, picojson::value& dstJson) const override;
  void DeserializeInner(
      picojson::value const& srcJson,
      void* dst,
      SReflect::DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override;
  bool SerializeToBytesInner(void const* src, SReflect::StreamWriter& dst) const override;
  bool DeserializeFromBytesInner(SReflect::StreamReader& src, void* dst) const override;
  [[nodiscard]] bool IsMemCopySafe() const override;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Optional
///////////////////////////////////////////////////////////////////////////

/**
Describes a type like std::optional<InnerType> which may or may not have a value.
These types behave just like the inner type except they have one additional state,
which is the state of having no value at all. Optionals without a value serialize
to JSON as null.
*/
class OptionalTypeInfo : public TypeInfo {
 public:
  TypeInfo const* _innerTypeInfo = nullptr;

  /**
  Ensure that the optional<T> has a value. Default construct a new value if necessary.
  @param srcOptional The address of an object of type optional<T>
  @return The address of the optional<T>::value(). Will not be nullptr;
  */
  virtual void* EnsureOptionalValue(void* srcOptional) const = 0;

  /**
  Get the inner value from an optional<T> or return nullptr if it has none.
  @param srcOptional The address of an object of type optional<T>
  @return The address of the optional<T>::value() or nullptr if it has no value.
  */
  virtual void* GetOptionalValue(void* srcOptional) const = 0;
  virtual void const* GetOptionalValue(void const* srcOptional) const = 0;

  /**
  Set the inner value of an optional<T> using the address of a value of type T.
  If the value is nullptr, then clear the optional<T> so that it has no value.
  @param srcValue The address of a value of type T or nullptr (meaning "no value")
  @param dstOptional The address of an object of type optional<T>
  */
  virtual void SetOptionalValue(void const* srcValue, void* dstOptional) const = 0;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

///////////////////////////////////////////////////////////////////////////
// Variants
///////////////////////////////////////////////////////////////////////////

/**
Extends TypeInfo with additional functionality for types with CoreType::CT_variant.
Example type: std::variant
*/
class VariantTypeInfo : public TypeInfo {
 public:
  // All possible inner types are known up front
  std::vector<TypeInfo const*> _innerTypes;

  // Variants can be copied via memcpy if all of the inner types can be.
  bool _isMemCopySafe = false;

  // Return the index of the object's current type (up to _innerTypes.size() - 1).
  [[nodiscard]] virtual size_t GetInnerTypeIndex(void const* obj) const = 0;

  // Get the address of the current inner object.
  [[nodiscard]] virtual void* GetInnerObject(void* obj) const = 0;
  [[nodiscard]] void const* GetInnerObject(void const* obj) const;

  // Try to change the inner type index and default construct a new inner object.
  // Returns true if the change was successful, or if the type was already correct.
  [[nodiscard]] virtual bool TrySetInnerTypeIndex(void* obj, size_t typeIndex) const = 0;

  // TypeInfo overrides
  [[nodiscard]] bool IsMemCopySafe() const final;
  void SetInner(void const* src, void* dst) const final;
  [[nodiscard]] bool IsValidInner(void const* src, AttributeList const& attributes) const final;
  void SerializeInner(void const* src, picojson::value& dstJson) const final;
  void DeserializeInner(
      picojson::value const& srcJson,
      void* dst,
      SReflect::DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const final;
  [[nodiscard]] bool SerializeToBytesInner(void const* src, SReflect::StreamWriter& dst)
      const final;
  [[nodiscard]] bool DeserializeFromBytesInner(SReflect::StreamReader& src, void* dst) const final;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

} // namespace SReflect

// Include inline implementation before defining attribute classes, which depend on it.
#include "simple_reflection_inl.h"

namespace SReflect {

///////////////////////////////////////////////////////////////////////////
// Attributes
///////////////////////////////////////////////////////////////////////////

/**
Common base for all simple reflection attributes
*/
struct Attribute : public BaseObject {
  virtual ~Attribute() = default;
  SR_BeginStruct(SReflect::Attribute);
  SR_EndStruct();
};

/**
Disable serializing/deserializing of a particular field.
Does not disable getting nor setting the field.
*/
struct Attribute_DoNotSerialize : public Attribute {
  SR_BeginStruct(SReflect::Attribute_DoNotSerialize);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Used as a class or field attribute. Prevents serialization of fields whose values match their
default-constructed values.

When _recursive is true (the default, and the only behavior for the class-level attribute), pruning
also descends into nested fields/elements, omitting any nested values that match their defaults.

When _recursive is false (only meaningful as a field attribute), the field is omitted only if its
entire value matches the default; otherwise the field is serialized in full, including nested
sub-fields that equal their defaults. Useful for an optional container that should be omitted when
empty but written in full (all sub-fields shown) when present.
*/
struct Attribute_DoNotSerializeDefaults : public Attribute {
  bool _recursive{true};
  explicit Attribute_DoNotSerializeDefaults(bool recursive = true) : _recursive(recursive) {}
  SR_BeginStruct(SReflect::Attribute_DoNotSerializeDefaults);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_recursive, "recursive");
  SR_EndStruct();
};

/**
Applied to a struct. Suppresses the "extraneous field" issue that
DeserializeFlags::WarnIfExtraneousFields would otherwise raise for JSON keys that match no field of
this struct. Use for structs that intentionally carry data beyond their reflected schema (e.g. a
record whose extra keys are consumed elsewhere), so the surrounding document can still be
deserialized strictly while this one struct tolerates unknown keys. JSON only; the binary path is
positional and has no notion of extraneous fields.
*/
struct Attribute_IgnoreExtraneousFields : public Attribute {
  SR_BeginStruct(SReflect::Attribute_IgnoreExtraneousFields);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Applied to a string field that stores a serialized JSON value rather than a plain string. On JSON
deserialize, a JSON object/array at this field is serialized to text and stored in the string (as
though the file had held a string containing that text); on JSON serialize, the string is parsed and
emitted as the JSON value it represents. A deserialize/serialize round trip is therefore identity
for JSON object/array values. The conversion routes through the field's normal string assignment, so
applying this to a field that cannot accept a string fails to deserialize and reports an issue,
exactly as a type mismatch would. Plain-string and scalar JSON values pass through unchanged. The
binary path is unaffected (the raw string round-trips as-is).
*/
struct Attribute_JsonString : public Attribute {
  SR_BeginStruct(SReflect::Attribute_JsonString);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Disallow editing this type using editor/debug panel.
Does not disable deserialization nor setting of the field through code.
*/
struct Attribute_ReadOnly : public Attribute {
  SR_BeginStruct(SReflect::Attribute_ReadOnly);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Valid range for this type, for floats and doubles
*/
struct Attribute_FloatRange : public Attribute {
  double _min{};
  double _max{};
  explicit Attribute_FloatRange(float min1, float max1)
      : _min(static_cast<double>(min1)), _max(static_cast<double>(max1)) {}
  explicit Attribute_FloatRange(double min1, double max1) : _min(min1), _max(max1) {}
  SR_BeginStruct(SReflect::Attribute_FloatRange);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_min, "min");
  SR_Field_Name(_max, "max");
  SR_EndStruct();
};

/**
Valid range for this type, for signed integer types
*/
struct Attribute_IntRange : public Attribute {
  int64_t _min{};
  int64_t _max{};
  explicit Attribute_IntRange(int64_t min1, int64_t max1) : _min(min1), _max(max1) {}
  SR_BeginStruct(SReflect::Attribute_IntRange);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_min, "min");
  SR_Field_Name(_max, "max");
  SR_EndStruct();
};

/**
Valid range for this type, for unsigned integer types
*/
struct Attribute_UIntRange : public Attribute {
  uint64_t _min{};
  uint64_t _max{};
  explicit Attribute_UIntRange(uint64_t min1, uint64_t max1) : _min(min1), _max(max1) {}
  SR_BeginStruct(SReflect::Attribute_UIntRange);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_min, "min");
  SR_Field_Name(_max, "max");
  SR_EndStruct();
};

/**
Callback when type is changed via Set function
*/
struct Attribute_OnChanged : public Attribute {
  std::function<void()> _onChanged;
  explicit Attribute_OnChanged(std::function<void()> cb) : _onChanged(std::move(cb)) {}
  SR_BeginStruct(SReflect::Attribute_OnChanged);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Specify units that can be used by editor UI
*/
struct Attribute_Units : public Attribute {
  std::string _units;
  explicit Attribute_Units(std::string_view units) : _units(units) {}
  SR_BeginStruct(SReflect::Attribute_Units);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_units, "units");
  SR_EndStruct();
};

/**
Marks a 3- or 4-element float field as a linear RGB / RGBA color, so editor UI can present a color
picker instead of a row of numeric drags. Shape alone is ambiguous: a 4-element float field is just
as likely to be a quaternion, a translation or a pair of extents.
*/
struct Attribute_Color : public Attribute {
  SR_BeginStruct(SReflect::Attribute_Color);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Specify a different name to use in editor UI rather than the standard type name
*/
struct Attribute_DisplayName : public Attribute {
  std::string _displayName;
  explicit Attribute_DisplayName(std::string_view displayName) : _displayName(displayName) {}
  SR_BeginStruct(SReflect::Attribute_DisplayName);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_displayName, "displayName");
  SR_EndStruct();
};

/**
Description that can be used by editor UI
*/
struct Attribute_Description : public Attribute {
  std::string _description;
  explicit Attribute_Description(std::string_view description) : _description(description) {}
  SR_BeginStruct(SReflect::Attribute_Description);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_description, "description");
  SR_EndStruct();
};

/**
Category to group properties by in editor UI
*/
struct Attribute_Category : public Attribute {
  std::string _category;
  explicit Attribute_Category(std::string_view category) : _category(category) {}
  SR_BeginStruct(SReflect::Attribute_Category);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_category, "category");
  SR_EndStruct();
};

/**
If a field is renamed, you can use this to specify the old name,
then during json deserialization, fields with the old name will be mapped into the field
*/
struct Attribute_PreviouslyKnownAs : public Attribute {
  std::vector<std::string> _previousNames;
  explicit Attribute_PreviouslyKnownAs(std::initializer_list<std::string_view> const& previousNames)
      : _previousNames(previousNames.begin(), previousNames.end()) {}
  explicit Attribute_PreviouslyKnownAs(std::string_view previousName)
      : _previousNames(1, std::string{previousName}) {}
  SR_BeginStruct(SReflect::Attribute_PreviouslyKnownAs);
  SR_BaseClass(SReflect::Attribute);
  SR_Field_Name(_previousNames, "previousNames");
  SR_EndStruct();
};

/**
Do not show this field in an editor
*/
struct Attribute_HideFromEditor : public Attribute {
  SR_BeginStruct(SReflect::Attribute_HideFromEditor);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

/**
Callback after data is loaded from json but before data is moved from json into struct instance
This can be used to pre-process the json data, e.g. changing field values or computing values for
new fields Can be useful to support loading older versions of a json file
*/
struct Attribute_OnDeserialize : public Attribute {
  std::function<void(picojson::value&)> _onDeserialize;
  explicit Attribute_OnDeserialize(std::function<void(picojson::value&)> cb) : _onDeserialize(cb) {}
  SR_BeginStruct(SReflect::Attribute_OnDeserialize);
  SR_BaseClass(SReflect::Attribute);
  SR_EndStruct();
};

} // namespace SReflect

#else // if !SIMPLE_REFLECTION_ENABLE

#define SR_BeginStruct(structType)
#define SR_BeginStructTemplate(structType, ...)
#define SR_EndStruct()
#define SR_Field(member)
#define SR_Field_Name(member, name)
#define SR_BeginClass(classType)
#define SR_BeginClassTemplate(classType, ...)
#define SR_EndClass()
#define SR_BaseClass(baseClass)
#define SRA_DoNotSerialize()
#define SRA_DoNotSerializeDefaults()
#define SRA_ReadOnly()
#define SRA_FloatRange(min1, max1)
#define SRA_IntRange(min1, max1)
#define SRA_UIntRange(min1, max1)
#define SRA_OnChanged(cb)
#define SRA_Units(str)
#define SRA_DisplayName(str)
#define SRA_Description(str)
#define SRA_Category(str)
#define SRA_PreviouslyKnownAs(...)
#define SRA_HideFromEditor()
#define SRA_OnDeserialize(cb)
#define SRA_IgnoreExtraneousFields()
#define SRA_JsonString()
#define SR_BeginEnum(enumType)
#define SR_EnumItem(enumValue)
#define SR_EndEnum()

#endif
