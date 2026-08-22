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

#include <mochi_core/mochi_config.h>

/**************************************************************************************************
 *  Mochi Reflection
 *
 *    Mochi uses a first-party C++ reflection library called Simple Reflection. It is rebranded here
 *    using "MOCHI_" prefixed macros, which usually compile out when mochi headers are included
 *    externally.
 *
 *    Reflection metadata can be used for things like:
 *      - Serialization
 *      - Automatic UI generation
 *      - Portable type identification
 *      - And more...
 *
 *    Enum Example:
 *
 *      enum class MyEnum {
 *        Option1,
 *        Option2,
 *        Count
 *      };
 *
 *      // Then, in the globl namespace:
 *      MOCHI_ENUM_BEGIN(mochi::MyEnum)
 *      MOCHI_ENUM_ITEM(Option1)
 *      MOCHI_ENUM_ITEM(Option2)
 *      MOCHI_ENUM_COUNT(Count)
 *      MOCHI_ENUM_END()
 *
 *    Struct Example (declared inside the class):
 *
 *      struct MyStruct {
 *        int someValue = 0;
 *        DynamicArray<Real3> coords;
 *
 *        MOCHI_STRUCT_BEGIN(mochi::MyStruct)
 *        MOCHI_FIELD(someValue);
 *        MOCHI_FIELD(coords);
 *        MOCHI_STRUCT_END();
 *      };
 *
 *    Struct Example (declared outside the class):
 *
 *      struct MyStruct {
 *        int someValue = 0;
 *        DynamicArray<Real3> coords;
 *      };
 *
 *      // Then, in the global namespace:
 *      MOCHI_STRUCT_BEGIN_EX(mochi::MyStruct)
 *      MOCHI_FIELD(someValue);
 *      MOCHI_FIELD(coords);
 *      MOCHI_STRUCT_END_EX();
 *
 *    Template Example (internal):
 *
 *      template <class T, int N>
 *      struct MyTemplate {
 *        std::array<T, N> values;
 *
 *        MOCHI_TEMPLATE_BEGIN(mochi::Template, T, N)
 *        MOCHI_FIELD(values);
 *        MOCHI_TEMPLATE_END();
 *      };
 *
 *    Attributes Example (they go AFTER that which they describe):
 *
 *      MOCHI_STRUCT_BEGIN_EX(mochi::MyStruct)
 *      MOCHI_ATTRIBUTE(Description("This string describes the whole class"));
 *      MOCHI_FIELD(someValue) MOCHI_ATTRIBUTE(ReadOnly);
 *      MOCHI_FIELD(coords) MOCHI_ATTRIBUTE(Units("m"));
 *      MOCHI_STRUCT_END_EX();
 *
 */

/**
 * When MOCHI_USE_REFLECTION is defined to 0, all Mochi reflection macros will compile out. This is
 * the default so Mochi headers can be included externally without external users taking a
 * dependency on Simple Reflection. Mochi's build system defins MOCHI_USE_REFLECTION=1 when
 * compiling Mochi source code, so these features are always available internally.
 */
#ifndef MOCHI_USE_REFLECTION
#define MOCHI_USE_REFLECTION 0
#endif

#if MOCHI_USE_REFLECTION

#ifdef SIMPLE_REFLECTION_ENABLE
#if !SIMPLE_REFLECTION_ENABLE
#error MOCHI_USE_REFLECTION cannot be defined to 1 (true) if SIMPLE_REFLECTION_ENABLE has already been defined to 0 (false).
#endif
#else
#define SIMPLE_REFLECTION_ENABLE 1
#endif

// Mochi reflection uses only Simple Reflection's picojson/string APIs; the nlohmann::json bridge
// (SReflect::ToJson/FromJson) is never called. Disabling it keeps the heavy <nlohmann/json.hpp>
// out of every TU that includes this header. Matches the CMake build (SR_USE_NLOHMANN_JSON=0).
#ifndef SR_USE_NLOHMANN_JSON
#define SR_USE_NLOHMANN_JSON 0
#endif

#ifdef assert_invariant
#pragma push_macro("assert_invariant")
#undef assert_invariant
#define MOCHI_RESTORE_ASSERT_INVARIANT
#endif

#include <simple_reflection/simple_reflection.h>

#ifdef MOCHI_RESTORE_ASSERT_INVARIANT
#pragma pop_macro("assert_invariant")
#undef MOCHI_RESTORE_ASSERT_INVARIANT
#endif

#define MOCHI_ENUM_BEGIN(name) \
  IMPL_SR_BeginEnum(name);     \
  [[maybe_unused]] static constexpr int kEnumFirstItemLine = __LINE__ + 1;
#define MOCHI_ENUM_ITEM(name) IMPL_SR_EnumItem(name)
#define MOCHI_ENUM_COUNT(name)                                           \
  static_assert(                                                         \
      static_cast<int>(MyEnum::name) == (__LINE__ - kEnumFirstItemLine), \
      "Unexpected number of lines in reflection enum definition. Please make sure that every enum value is listed here (one per line).");
#define MOCHI_ENUM_END() IMPL_SR_EndEnum()

#define MOCHI_STRUCT_BEGIN(name) IMPL_SR_BeginStruct(name, #name)
#define MOCHI_STRUCT_END() IMPL_SR_EndStruct()
#define MOCHI_STRUCT(name) MOCHI_STRUCT_BEGIN(name) MOCHI_STRUCT_END()
#define MOCHI_STRUCT_WITH_BASE(name, base) \
  MOCHI_STRUCT_BEGIN(name) MOCHI_BASE_CLASS(base) MOCHI_STRUCT_END()

#define MOCHI_STRUCT_BEGIN_EX(name) IMPL_SR_BeginStructEx(name)
#define MOCHI_STRUCT_END_EX() IMPL_SR_EndStructEx()
#define MOCHI_STRUCT_EX(name) MOCHI_STRUCT_BEGIN_EX(name) MOCHI_STRUCT_END_EX()
#define MOCHI_STRUCT_WITH_BASE_EX(name, base) \
  MOCHI_STRUCT_BEGIN_EX(name) MOCHI_BASE_CLASS(base) MOCHI_STRUCT_END_EX()

#define MOCHI_TEMPLATE_BEGIN(name, ...) IMPL_SR_BeginStructTemplate(name, __VA_ARGS__)
#define MOCHI_TEMPLATE_END() IMPL_SR_EndStruct()
#define MOCHI_TEMPLATE(name, ...) MOCHI_TEMPLATE_BEGIN(name, __VA_ARGS__) MOCHI_TEMPLATE_END()

#define MOCHI_FIELD(name) IMPL_SR_Field_Name(name, #name)
#define MOCHI_FIELD_NAME(realName, customName) IMPL_SR_Field_Name(realName, customName)
#define MOCHI_REMOVE_FIELD(name) IMPL_SR_RemoveField(name)
#define MOCHI_REPLACE_FIELD_NAME(realName, fieldNameToReplace) \
  MOCHI_REMOVE_FIELD(fieldNameToReplace) MOCHI_FIELD_NAME(realName, fieldNameToReplace)

#define MOCHI_BASE_CLASS(name) IMPL_SR_BaseClass(name)

// MOCHI_ATTRIBUTE goes after the thing it describes.
// See examples in the comment block above.
#define MOCHI_ATTRIBUTE(...)                                                      \
  {                                                                               \
    SReflect::detail::AddAttribute(*myInfo, new ::mochi::attribute::__VA_ARGS__); \
  }

// MOCHI_ATTRIBUTE_IF adds an attribute conditionally (must be constexpr). Used in templates where
// the use of the attribute depends on the template parameters.
#define MOCHI_ATTRIBUTE_IF(condition, ...) \
  if constexpr (condition) {               \
    MOCHI_ATTRIBUTE(__VA_ARGS__);          \
  }

namespace mochi {

// Base class for all attributes
using Attribute = SReflect::Attribute;

} // namespace mochi

namespace mochi::attribute {

// [Struct Attribute] Assign a category string to the class/struct, for organization in UI.
// Example: MOCHI_ATTRIBUTE(Category("text"))
using Category = SReflect::Attribute_Category;

// [Field Attribute] Marks a 3- or 4-element float field as a linear RGB / RGBA color, so tools can
// show a color picker rather than numeric drags.
// Example: MOCHI_ATTRIBUTE(Color)
using Color = SReflect::Attribute_Color;

// [General Attribute] Add a descriptive string to the struct/field/enum/item.
// Example: MOCHI_ATTRIBUTE(Description("text"))
using Description = SReflect::Attribute_Description;

// [General Attribute] Changes how the struct/field/enum/item will appear in UI.
// Example: MOCHI_ATTRIBUTE(DisplayName("text"))
using DisplayName = SReflect::Attribute_DisplayName;

// [General Attribute] Do not display the struct/field/enum/item in UI.
// Example: MOCHI_ATTRIBUTE(HideFromEditor)
using HideFromEditor = SReflect::Attribute_HideFromEditor;

// [Field Attribute] Indicates that a field should never be serialized/deserialized.
// Example: MOCHI_ATTRIBUTE(NoSerialize)
using NoSerialize = SReflect::Attribute_DoNotSerialize;

// [Struct/Field Attribute] Indicates that only fields with non-default values should be serialized.
// As a field attribute, pass recursive=false to omit the field only when it equals its default,
// while serializing it in full (all sub-fields shown) when present:
//   MOCHI_ATTRIBUTE(NoSerializeDefaults)          // omit defaults, recursing into sub-fields
//   MOCHI_ATTRIBUTE(NoSerializeDefaults(false))   // omit only if the whole field is default
using NoSerializeDefaults = SReflect::Attribute_DoNotSerializeDefaults;

// [General Attribute] Indicates that something was renamed. Used for backward compatibility.
// Example: MOCHI_ATTRIBUTE(PreviouslyKnownAs)
using PreviouslyKnownAs = SReflect::Attribute_PreviouslyKnownAs;

// [Field Attribute] Indicates the legal range of values for a field of signed integer type (any
// size) Example: MOCHI_ATTRIBUTE(IntRange(-1, 1))
using IntRange = SReflect::Attribute_IntRange;

// [Field Attribute] Indicates the legal range of values for a field of type float or double.
// Example: MOCHI_ATTRIBUTE(FloatRange(0_r, 1_r))
using FloatRange = SReflect::Attribute_FloatRange;

// [Field Attribute] Indicates the legal range of values for a field of unsigned integer type (any
// size) Example: MOCHI_ATTRIBUTE(UIntRange(0, 1))
using UIntRange = SReflect::Attribute_UIntRange;

// [Field Attribute] Indicates that the field should appear read-only in tools.
// Example: MOCHI_ATTRIBUTE(ReadOnly)
// NOTE: This does not prevent serialization/deserialization.
using ReadOnly = SReflect::Attribute_ReadOnly;

// [Field Attribute] Indicates the SI unit of measure (short form like "m", "s", "m/s", etc...)
// Example: MOCHI_ATTRIBUTE(Units("kg"))
using Units = SReflect::Attribute_Units;

// [Struct Attribute] Suppresses the extraneous-field issue that
// DeserializeFlags::WarnIfExtraneousFields raises for JSON keys matching no field of this struct.
// Use for structs that intentionally carry data beyond their reflected schema, so the surrounding
// document still deserializes strictly. Example: MOCHI_ATTRIBUTE(IgnoreExtraneousFields)
using IgnoreExtraneousFields = SReflect::Attribute_IgnoreExtraneousFields;

// [Field Attribute] Marks a string field that stores a serialized JSON value: a JSON object/array
// at this field is stored as its serialized text on load and re-emitted as that JSON value on save
// (a deserialize/serialize round trip is identity). Applying it to a non-string field fails to
// deserialize and reports an issue. Example: MOCHI_ATTRIBUTE(JsonString)
using JsonString = SReflect::Attribute_JsonString;

} // namespace mochi::attribute

#else // if !MOCHI_USE_REFLECTION

// It can be useful to add a semicolon after an attribute because that causes clang-format
// to keep it on the same line as the thing it modifies. However, the compiler will see this
// as a stray semicolon within when !MOCHI_USE_REFLECTION. Therefore we suppress the warning
// within reflection definition blocks.
#define IMPL_MOCHI_PUSH_IGNORE_EXTRA_SEMI() \
  MOCHI_WARNING_PUSH()                      \
  MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wextra-semi")

#define MOCHI_ENUM_BEGIN(name) IMPL_MOCHI_PUSH_IGNORE_EXTRA_SEMI()
#define MOCHI_ENUM_ITEM(name)
#define MOCHI_ENUM_COUNT(name)
#define MOCHI_ENUM_END() MOCHI_WARNING_POP()

#define MOCHI_STRUCT_BEGIN(name) IMPL_MOCHI_PUSH_IGNORE_EXTRA_SEMI()
#define MOCHI_STRUCT_END() MOCHI_WARNING_POP()
#define MOCHI_STRUCT(name)
#define MOCHI_STRUCT_WITH_BASE(name, base)

#define MOCHI_STRUCT_BEGIN_EX(name) IMPL_MOCHI_PUSH_IGNORE_EXTRA_SEMI()
#define MOCHI_STRUCT_END_EX() MOCHI_WARNING_POP()
#define MOCHI_STRUCT_EX(name)
#define MOCHI_STRUCT_WITH_BASE_EX(name, base)

#define MOCHI_TEMPLATE_BEGIN(name, ...)
#define MOCHI_TEMPLATE_END()
#define MOCHI_TEMPLATE(name, ...)

#define MOCHI_BASE_CLASS(name)
#define MOCHI_FIELD(name)
#define MOCHI_FIELD_NAME(realName, customName)
#define MOCHI_REMOVE_FIELD(name)
#define MOCHI_REPLACE_FIELD_NAME(realName, fieldNameToReplace)

#define MOCHI_ATTRIBUTE(...)
#define MOCHI_ATTRIBUTE_IF(condition, ...)

#endif // !MOCHI_USE_REFLECTION
