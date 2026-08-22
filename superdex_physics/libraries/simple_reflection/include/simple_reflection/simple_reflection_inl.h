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

#include "simple_reflection.h" // Reverse include for intellisense

#if SIMPLE_REFLECTION_ENABLE

///////////////////////////////////////////////////////////////////////////
// Compiler Warning Suppression
///////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER) && !defined(__GNUC__) && !defined(__clang__)
#define SR_WARNING_PUSH() __pragma(warning(push))
#define SR_WARNING_POP() __pragma(warning(pop))
#define SR_WARNING_IGNORE_MSVC(X) __pragma(warning(disable : X))
#define SR_WARNING_IGNORE_GCC(X)
#define SR_WARNING_IGNORE_CLANG(X)
#elif defined(__GNUC__) and !defined(__clang__)
#define SR_WARNING_PUSH() _Pragma("GCC diagnostic push")
#define SR_WARNING_POP() _Pragma("GCC diagnostic pop")
#define SR_WARNING_IGNORE_MSVC(X)
#define SR_WARNING_IGNORE_GCC(X) _Pragma(#X)
#define SR_WARNING_IGNORE_CLANG(X)
#elif defined(__clang__)
#define SR_WARNING_PUSH() _Pragma("clang diagnostic push")
#define SR_WARNING_POP() _Pragma("clang diagnostic pop")
#define SR_WARNING_IGNORE_MSVC(X)
#define SR_WARNING_IGNORE_GCC(X) _Pragma(#X)
#define SR_WARNING_IGNORE_CLANG(X) _Pragma(#X)
#else
#define SR_WARNING_PUSH()
#define SR_WARNING_POP()
#define SR_WARNING_IGNORE_MSVC(X)
#define SR_WARNING_IGNORE_GCC(X)
#endif

/////////////////////////////////////////////////////////////////////////////
// TypeId
/////////////////////////////////////////////////////////////////////////////

inline bool SReflect::TypeId::operator==(SReflect::TypeId const& rhs) const {
  return value == rhs.value;
}
inline bool SReflect::TypeId::operator!=(SReflect::TypeId const& rhs) const {
  return value != rhs.value;
}
inline bool SReflect::TypeId::operator<(SReflect::TypeId const& rhs) const {
  return value < rhs.value;
}

namespace std {
template <>
struct hash<SReflect::TypeId> {
 public:
  size_t operator()(SReflect::TypeId const& hash) const {
    return static_cast<size_t>(hash.value);
  }
};
} // namespace std

/////////////////////////////////////////////////////////////////////////////
// HasMemberFn_GerFinalTypeInfo<T>
/////////////////////////////////////////////////////////////////////////////

namespace SReflect {

/**
Template type trait for decomposing the types that make up a pointer-to-member-function type.
*/
template <class MemFn>
struct DecomposeMemFn {};
template <class C, class R>
struct DecomposeMemFn<R (C::*)() const> {
  using ReturnType = R;
  using ClassType = C;
};

/**
Template type trait deriving from std::true iff type T is a class or struct with the member function
T::GetFinalTypeInfo. This is typically added by including macros like SR_BeginClass or
SR_BeginStruct inside the class declaration.
*/
template <typename T, typename DefaultVoid = void>
struct HasMemberFn_GetFinalTypeInfo : public std::false_type {};
template <typename T>
struct HasMemberFn_GetFinalTypeInfo<
    T,
    std::enable_if_t<
        std::is_same_v<T, typename DecomposeMemFn<decltype(&T::GetFinalTypeInfo)>::ClassType>>>
    : public std::true_type {};

} // namespace SReflect

/////////////////////////////////////////////////////////////////////////////
// SReflectTypeTraits
/////////////////////////////////////////////////////////////////////////////

/**
Simple Reflection can support any type as long as a specialization of SReflectTypeTraits can be
found. This generic implementation is selected for all unsupported types. It is also selected for
for classes and structs that use macros like SR_BeginClass or SR_BeginStruct internally.
*/
template <typename T>
struct SReflectTypeTraits {
  // If T is a class or struct with macros like SR_BeginClass or SR_BeginStruct inside the class
  // declaration, then we can support it here. We detect such a type by looking for the member
  // function T::GetFinalTypeInfo. If T is any other type, then it is one that Simple Reflection
  // doesn't understand, so isSupportedType will be false.
  static constexpr bool isSupportedType = SReflect::HasMemberFn_GetFinalTypeInfo<T>::value;

  // If this is not a supported type, then set CoreType::X_Invalid. This provides a way to check for
  // support at compile time (see IsSupportedType).
  static constexpr SReflect::CoreType coreType =
      isSupportedType ? SReflect::CoreType::CT_struct : SReflect::CoreType::X_Invalid;

  static const auto& GetTypeInfo() {
    // If someone calls SReflectTypeTraits<T>::GetTypeInfo() on an unsupported type, then they will
    // get compiler errors. We do our best to ensure that the first compiler error will give them
    // helpful information.
    if constexpr (!isSupportedType) {
      static_assert(
          !std::is_enum_v<T>,
          "To use this enum with reflection, please add an SR_BeginEnum/SR_EndEnum block.");
      static_assert(
          !std::is_class_v<T>,
          "To use this class or struct with reflection, please and an SR_BeginStruct/SR_EndStruct block inside "
          "the class declaration, or an SR_BeginStructEx/SR_EndStructEx block outside the class declaration. "
          "For more advanced cases, it is also possible to declare your own specialization of SReflectTypeTraits<T>. "
          "See <simple_reflection/simple_reflection.h> for more information.");
      static_assert(
          !std::is_pointer_v<T>,
          "Raw pointer types are not currently supported by Simple Reflection.");
      static_assert(
          !std::is_reference_v<T>,
          "Reference types are not currently supported by Simple Reflection.");
      static_assert(
          isSupportedType,
          "Unable to find SReflectTypeTraits for this type. Please make sure that you have included the appripriate "
          "header and that it is a fully qualified type, not a forward declaration only.");
    }

    return T::GetTypeInfo();
  }
};

namespace SReflect {

/////////////////////////////////////////////////////////////////////////////
// StructTypeInfo
/////////////////////////////////////////////////////////////////////////////

template <typename T>
inline bool StructTypeInfo::IsSameOrDerivedFrom() const {
  static_assert(std::is_class_v<T>, "IsSameOrDerivedFrom<T>() requires a class or struct T.");
  return IsSameOrDerivedFrom(GetTypeId<T>());
}

/////////////////////////////////////////////////////////////////////////////
// details
/////////////////////////////////////////////////////////////////////////////

namespace detail {

/**
Add stuff to the TypeInfo that is currently being built. Used by the SR_* family of macros.
*/
void AddAttribute(TypeInfo& info, Attribute const* a);
void AddAttribute(StructTypeInfo& info, Attribute const* a);
void AddBaseClass(StructTypeInfo& info, StructTypeInfo const& base, ptrdiff_t baseOffset);
void AddField(StructTypeInfo& info, char const* name, size_t offset, TypeInfo const& innerType);
void RemoveField(StructTypeInfo& info, char const* name);
void AddEnumItem(EnumTypeInfo& info, char const* name, uint64_t value);
void AddAttribute(EnumTypeInfo& info, Attribute const* a);

std::string SaveToJsonString(void const* obj, TypeInfo const& info, bool pretty);
void LoadFromJsonString(
    void* outObj,
    TypeInfo const& info,
    std::string const& jsonStr,
    DeserializeFlags deserializeFlags,
    int& outNumIssues);
bool SaveToJsonFile(
    void const* obj,
    TypeInfo const& info,
    char const* filePath,
    bool warnOnFailure);
bool LoadFromJsonFile(
    void* outObj,
    TypeInfo const& info,
    char const* filePath,
    DeserializeFlags deserializeFlags,
    int& outNumIssues);

bool ToBytes(void const* src, TypeInfo const& srcInfo, StreamWriter& dst) noexcept;
bool FromBytes(StreamReader& src, void* dst, TypeInfo const& dstInfo) noexcept;

TypeInfo* MakePicojsonValueTypeInfo();

/**
Call TypeInfo::_constructInPlace for each element of a contiguous array
*/
void ArrayConstructInPlace(TypeInfo const& elementType, void* dst, size_t count);

/**
Call TypeInfo::_constructInPlaceByCopy for each element of a contiguous array
*/
void ArrayConstructInPlaceByCopy(
    TypeInfo const& elementType,
    void* dst,
    size_t count,
    void const* src);

/**
Call TypeInfo::_destructInPlace for each element of a contiguous array
*/
void ArrayDestructInPlace(TypeInfo const& elementType, void* ptr, size_t count);

/**
Make the ArrayTypeInfo for a fixed-size array. The caller is responsible for initializing the
creation/destruction functions. See MakeFixedArrayTypeInfo template.
*/
ArrayTypeInfo* MakeFixedArrayTypeInfoImpl(
    char const* nameWithNamespace,
    bool formatNameAsTemplate,
    TypeInfo const& innerTypeInfo,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    size_t fixedArraySize,
    bool isMemCopySafe);

/**
Initialize the ArrayTypeInfo for dynamically resizable array type. The caller is responsible for
creating the ArrayTypeInfo (or derived) and for initialization the creation/destruction functions.
If (formatNameAsTemplate == true), then the name will be formatted like "name<T>" were "T" is the
inner type name.
*/
void InitDynamicArrayTypeInfoImpl(
    ArrayTypeInfo* ti,
    char const* nameWithNamespace,
    bool formatNameAsTemplate,
    TypeInfo const& innerTypeInfo,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment);

/**
Allocate a new EnumTypeInfo to which enum items can be added. Used by SR_BeginEnum.
*/
EnumTypeInfo* MakeEnum(
    char const* nameWithNamespace,
    std::type_info const& rttiTypeInfo,
    TypeInfo const& innerTypeInfo);

/**
Allocate a formatted string by joining the parameters. Used for template type names
*/
char const* MakeTypeName(
    char const* a,
    char const* b = "",
    char const* c = "",
    char const* d = "",
    const char* e = "",
    char const* f = "");

/**
Make the StructTypeInfo for a class or struct.
*/
SReflect::StructTypeInfo* MakeStructImpl(
    char const* nameWithNamespace,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    bool isMemCopySafe);

SReflect::StructTypeInfo* CloneStructImpl(const SReflect::StructTypeInfo* structInfo);

template <class T>
bool isAligned(void* dst) {
  // Performing modulo on the pointer is apparently not a standard compliant way of checking
  // alignment, so call std::align and check if the pointer was modified assume the memory is sized
  // exactly for the object
  size_t size = sizeof(T);
  auto alignedPtr = dst;
  // std::align will leave the pointer alone if there's not room, but it will return a nullptr if
  // that happens
  return std::align(alignof(T), sizeof(T), alignedPtr, size) && alignedPtr == dst;
}

void InitMapTypeInfo(
    MapTypeInfo* ti,
    char const* nameWithNamespace,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    TypeInfo const& keyTypeInfo,
    TypeInfo const& valueTypeInfo);

template <class... Args>
std::enable_if_t<sizeof...(Args) == 0, void> AppendTemplateArgStr(std::string& /*unused*/) {}

template <class T0, class... Args>
void AppendTemplateArgStr(std::string& out) {
  out += SReflect::GetTypeInfo<T0>()._nameWithNamespace;
  out += ",";
  AppendTemplateArgStr<Args...>(out);
}

void AppendTemplateArgStr_EnumValue(
    EnumTypeInfo const& enumInfo,
    uint64_t value,
    bool isEnumClass,
    std::string& out);

template <auto V0, class... Args>
void AppendTemplateArgStr(std::string& out) {
  using T0 = decltype(V0);
  if constexpr (std::is_enum_v<T0>) {
    // Implemented in the cpp to reduce code bloat
    static constexpr bool kIsEnumClass = !std::is_convertible_v<T0, std::underlying_type_t<T0>>;
    AppendTemplateArgStr_EnumValue(
        SReflect::GetTypeInfo<T0>(), static_cast<uint64_t>(V0), kIsEnumClass, out);
  } else {
    static_assert(
        std::is_arithmetic_v<T0>,
        "SReflect template name formatting currently only supports enums and arithmetic types.");
    out += std::to_string(V0);
  }
  out += ",";
  AppendTemplateArgStr<Args...>(out);
}

// Unfortunately C++ does not support variadic templates with mixed type and non-type args.
// The following overloads support all combinations with non-type arguments in any of the
// first the argument positions.
template <class T0, auto V1, class... Args>
void AppendTemplateArgStr(std::string& out) {
  AppendTemplateArgStr<T0>(out);
  AppendTemplateArgStr<V1>(out);
  AppendTemplateArgStr<Args...>(out);
}
template <auto V0, auto V1, class... Args>
void AppendTemplateArgStr(std::string& out) {
  AppendTemplateArgStr<V0>(out);
  AppendTemplateArgStr<V1>(out);
  AppendTemplateArgStr<Args...>(out);
}
template <class T0, class T1, auto V2, class... Args>
void AppendTemplateArgStr(std::string& out) {
  AppendTemplateArgStr<T0>(out);
  AppendTemplateArgStr<T1>(out);
  AppendTemplateArgStr<V2>(out);
  AppendTemplateArgStr<Args...>(out);
}
template <class T0, auto V1, auto V2, class... Args>
void AppendTemplateArgStr(std::string& out) {
  AppendTemplateArgStr<T0>(out);
  AppendTemplateArgStr<V1>(out);
  AppendTemplateArgStr<V2>(out);
  AppendTemplateArgStr<Args...>(out);
}
template <auto V0, class T1, auto V2, class... Args>
void AppendTemplateArgStr(std::string& out) {
  AppendTemplateArgStr<V0>(out);
  AppendTemplateArgStr<T1>(out);
  AppendTemplateArgStr<V2>(out);
  AppendTemplateArgStr<Args...>(out);
}
template <auto T0, auto V1, auto V2, class... Args>
void AppendTemplateArgStr(std::string& out) {
  AppendTemplateArgStr<T0>(out);
  AppendTemplateArgStr<V1>(out);
  AppendTemplateArgStr<V2>(out);
  AppendTemplateArgStr<Args...>(out);
}

// Take the struct name (with namespace) and the output of AppendTemplateArgStr (see above).
// Allocates a new name string like "MyTemplate<int,N>". Used by SR_BeginStructTemplate.
char const* MakeTemplateTypeName(char const* structNameWithNamespace, char const* formattedArgList);

// Lookup enum item name or return nullptr
char const* EnumToStringImpl(EnumTypeInfo const& ti, uint64_t value);

// Initialize the basic fields of VariantTypeInfo in the cpp to reduce bloat.
void InitVariantTypeInfo(
    VariantTypeInfo& ti,
    char const* classNameWithNamespace,
    size_t sizeInBytes,
    size_t alignment,
    bool isTriviallyCopyable);

} // namespace detail

/////////////////////////////////////////////////////////////////////////////
// Public Utility Functions
/////////////////////////////////////////////////////////////////////////////

/**
Allocate a new TypeInfo equivalent to GetPrimitiveInfo(type) but with a different name.
*/
TypeInfo* MakePrimitiveInfo(CoreType coreType, char const* nameWithNamespace);

/**
Compute the hash code using SReflect's chosen algorithm
*/
TypeId ComputeTypeId(char const* nameWithNamespace);

/**
Assert if the type name has already been defined
*/
void VerifyTypeIdIsUnique(TypeInfo const& typeInfo, std::type_info const& stdInfo);

/**
Initialize the consturct/destruct members of a TypeInfo struct.
*/
template <typename T>
void InitTypeInfoFunctionPointers(TypeInfo* ti) {
  if constexpr (std::is_destructible_v<T>) {
    if constexpr (std::is_default_constructible_v<T>) {
      ti->_constructInPlace = [](void* dst) { new (dst) T{}; };
    }
    if constexpr (std::is_copy_constructible_v<T>) {
      ti->_constructInPlaceByCopy = [](void* dst, void const* src) {
        new (dst) T{*static_cast<const T*>(src)};
      };
    }
    if constexpr (std::is_default_constructible_v<T> || std::is_copy_constructible_v<T>) {
      ti->_destructInPlace = [](void* p) { static_cast<T*>(p)->~T(); };
    }
  }
}

/**
Make the StructTypeInfo for a class or struct of type T.
*/
template <typename T>
StructTypeInfo* MakeStructTypeInfo(char const* nameWithNamespace) {
  StructTypeInfo* ti = detail::MakeStructImpl(
      nameWithNamespace, typeid(T), sizeof(T), alignof(T), std::is_trivially_copyable_v<T>);
  InitTypeInfoFunctionPointers<T>(ti);
  return ti;
}

/**
Make the ArrayTypeInfo for an array type with fixed size.
The name will be formatted as follows (for inner type T, size N):
  - If (nameWithNamespace == nullptr), then name is "T[N]"
  - Else if (formatNameAsTemplate == true), then name is "nameWithNamespace<T,N>"
  - Else name is just "nameWithNamespace".
*/
template <class OuterType, class InnerType, size_t N>
ArrayTypeInfo* MakeFixedArrayTypeInfo(
    char const* nameWithNamespace,
    bool formatNameAsTemplate = false) {
  auto* ti = detail::MakeFixedArrayTypeInfoImpl(
      nameWithNamespace,
      formatNameAsTemplate,
      SReflect::GetTypeInfo<InnerType>(),
      typeid(OuterType),
      sizeof(OuterType),
      alignof(OuterType),
      N,
      std::is_trivially_copyable_v<OuterType>);
  InitTypeInfoFunctionPointers<OuterType>(ti);
  return ti;
}

/**
Make the DerivedArrayTypeInfo for a dynamically resizable array. If (formatNameAsTemplate == true),
then the name will be formatted as "name<T>" where T is the inner type name.
*/
template <class DerivedArrayTypeInfo, class OuterType, class InnerType>
DerivedArrayTypeInfo* MakeDynamicArrayTypeInfo(
    char const* nameWithNamespace,
    bool formatNameAsTemplate = false) {
  auto* ti = new DerivedArrayTypeInfo;
  detail::InitDynamicArrayTypeInfoImpl(
      ti,
      nameWithNamespace,
      formatNameAsTemplate,
      SReflect::GetTypeInfo<InnerType>(),
      typeid(OuterType),
      sizeof(OuterType),
      alignof(OuterType));
  InitTypeInfoFunctionPointers<OuterType>(ti);
  return ti;
}

/**
Finalize construction of struct
*/
void FinalizeStruct(StructTypeInfo* info);

/**
Get the type info for one of the primitive types (numbers and string).
*/
TypeInfo const& GetPrimitiveInfo(CoreType type);

template <typename T>
constexpr bool IsSupportedType() {
  return SReflectTypeTraits<T>::coreType != CoreType::X_Invalid;
}

template <typename T>
const auto& GetTypeInfo() {
  return SReflectTypeTraits<T>::GetTypeInfo();
}

template <typename T>
const auto* TryGetTypeInfo() {
  if constexpr (IsSupportedType<T>()) {
    return &SReflectTypeTraits<T>::GetTypeInfo();
  } else {
    return static_cast<SReflect::TypeInfo const*>(nullptr);
  }
}

template <typename T>
inline const auto& GetFinalTypeInfo(T const& obj) {
  if constexpr (std::is_convertible_v<T*, BaseObject*>) {
    return obj.GetFinalTypeInfo(); // virtual call
  } else {
    return SReflectTypeTraits<T>::GetTypeInfo();
  }
}

template <typename T>
inline TypeId GetTypeId() {
  return SReflectTypeTraits<T>::GetTypeInfo()._typeId;
}

template <typename T>
TypeId GetFinalTypeId(T const& obj) {
  if constexpr (std::is_convertible_v<T*, BaseObject*>) {
    return obj.GetFinalTypeId(); // virtual call
  } else {
    return GetTypeId<T>();
  }
}

template <typename T>
inline bool IsValid(T const& obj) {
  return GetFinalTypeInfo(obj).IsValid(&obj);
}

template <typename EnumT>
inline char const* EnumToString(EnumT value) {
  static_assert(std::is_enum_v<EnumT>, "Requires an enum type");
  return detail::EnumToStringImpl(GetTypeInfo<EnumT>(), static_cast<uint64_t>(value));
}

template <typename T>
inline std::string ToJsonString(T const& obj, bool pretty) {
  return detail::SaveToJsonString(&obj, GetFinalTypeInfo(obj), pretty);
}

template <typename T>
inline void FromJsonString(
    T& objOut,
    std::string const& json,
    DeserializeFlags deserializeFlags,
    int& numIssuesOut) {
  return detail::LoadFromJsonString(
      &objOut, GetFinalTypeInfo(objOut), json, deserializeFlags, numIssuesOut);
}

template <typename T>
inline void ToJsonValue(T const& obj, picojson::value& jsonOut) {
  GetFinalTypeInfo(obj).Serialize(&obj, jsonOut);
}

template <typename T>
inline void FromJsonValue(
    T& objOut,
    picojson::value const& json,
    DeserializeFlags deserializeFlags,
    int& numIssuesOut) {
  GetFinalTypeInfo(objOut).Deserialize(json, &objOut, deserializeFlags, numIssuesOut);
}

template <typename T>
inline bool SaveToJsonFile(T const& obj, char const* filePath, bool warnOnFailure) {
  return detail::SaveToJsonFile(&obj, GetFinalTypeInfo(obj), filePath, warnOnFailure);
}

template <typename T>
inline bool LoadFromJsonFile(
    T& objOut,
    char const* filePath,
    DeserializeFlags deserializeFlags,
    int& numIssuesOut) {
  return detail::LoadFromJsonFile(
      &objOut, GetFinalTypeInfo(objOut), filePath, deserializeFlags, numIssuesOut);
}

template <typename T>
std::vector<uint8_t> ToBytes(T const& src) noexcept {
  std::vector<uint8_t> result;

  VecStreamWriter writer(4096);
  if (SReflect::ToBytes(src, writer)) {
    result = writer.TakeBytes();
  }

  return result;
}

template <typename T>
inline bool ToBytes(T const& src, StreamWriter& dst) noexcept {
  return detail::ToBytes(&src, GetFinalTypeInfo(src), dst);
}

template <typename T>
inline bool FromBytes(StreamReader& src, T& dst) {
  return detail::FromBytes(src, &dst, GetFinalTypeInfo(dst));
}

template <typename T>
T FromBytes(StreamReader& src) {
  T result;
  if (!FromBytes(src, result)) {
    throw std::runtime_error("SReflect::FromBytes failed to parse stream");
  }
  return result;
}

/**
Return the byte offset from the start of an object of derived type to the start of
its base class. May return a non-zero value in case of multiple inheritence of
base non-zero-sized base classes (classes with non-static data members).
*/
template <class FromDerived, class ToBase>
inline ptrdiff_t GetOffsetFromDerivedToBase() {
  static_assert(std::is_convertible_v<FromDerived*, ToBase*>, "Expected a base class");
  return static_cast<ptrdiff_t>(
      reinterpret_cast<char*>(
          static_cast<ToBase*>(reinterpret_cast<FromDerived*>(sizeof(FromDerived)))) -
      reinterpret_cast<char*>(sizeof(FromDerived)));
}

/////////////////////////////////////////////////////////////////////////////
// Support for std::vector, or any other dynamic array type that has
// size(), resize(), and operator[].
/////////////////////////////////////////////////////////////////////////////

template <typename VectorT>
class VectorTypeInfo final : public ArrayTypeInfo {
 public:
  [[nodiscard]] bool CanResize() const override {
    return true;
  }

  size_t GetNumElements(void const* obj) const override {
    auto& vec = *(reinterpret_cast<VectorT const*>(obj));
    return vec.size();
  }

  bool SetNumElements(void* obj, size_t numElements) const override {
    auto& vec = *(reinterpret_cast<VectorT*>(obj));
    vec.resize(numElements);
    return true;
  }

  void* GetElement(void* obj, size_t i) const override {
    auto& vec = *(reinterpret_cast<VectorT*>(obj));
    return &vec[i];
  }

  void const* GetElement(void const* obj, size_t i) const override {
    auto& vec = *(reinterpret_cast<VectorT const*>(obj));
    return &vec[i];
  }

  bool IsMemCopySafe() const override {
    return false;
  }
};

/**
vector<bool> needs a special case since std::vector<bool> is stored as a bitset rather than
normal array so the memory layout of the buffer isn't the same as real bools, so we can't just
pass the address of the target memory.
*/
template <>
class VectorTypeInfo<std::vector<bool>> final : public ArrayTypeInfo {
 public:
  [[nodiscard]] bool CanResize() const override {
    return true;
  }
  size_t GetNumElements(void const* obj) const override;
  bool SetNumElements(void* obj, size_t numElements) const override;
  bool IsMemCopySafe() const override;

 private:
  bool IsValidInner(void const* obj, AttributeList const& attribs) const override;
  void SetInner(void const* src, void* dst) const override;
  void SerializeInner(void const* obj, picojson::value& dst) const override;
  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override;
  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override;
  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override;
  void* GetElement(void* /*obj*/, size_t /*i*/) const override; // Not implemented
  void const* GetElement(void const* /*obj*/, size_t /*i*/) const override; // Not implemented
};

/////////////////////////////////////////////////////////////////////////////
// Support for std::pair
/////////////////////////////////////////////////////////////////////////////
class PairTypeInfo : public SReflect::TypeInfo {
 public:
  SReflect::TypeInfo const* _infoT = nullptr;
  SReflect::TypeInfo const* _infoU = nullptr;
  uint64_t _offsetT = 0;
  uint64_t _offsetU = 0;

 protected:
  void SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const override;
};

namespace detail {
PairTypeInfo* MakePairTypeInfo(
    std::type_info const& stdTypeInfo,
    TypeInfo const* infoT,
    TypeInfo const* infoU,
    size_t sizeInBytes,
    size_t alignment,
    size_t offsetT,
    size_t offsetU);
} // namespace detail
} // namespace SReflect

template <typename T, typename U>
struct SReflectTypeTraits<std::pair<T, U>> {
  using PairType = std::pair<T, U>;
  // TODO: Replace PairTypeInfo with something more generic like TupleTypeInfo,
  // which could support an arbitrary number of inner types. Then add CT_tuple and
  // use it here.
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_other;

  static SReflect::PairTypeInfo const& GetTypeInfo() {
    static auto* s_typeInfo = []() {
      SR_WARNING_PUSH();
      SR_WARNING_IGNORE_GCC(GCC diagnostic ignored "-Winvalid-offsetof");
      auto* ti = SReflect::detail::MakePairTypeInfo(
          typeid(PairType),
          &SReflect::GetTypeInfo<T>(),
          &SReflect::GetTypeInfo<U>(),
          sizeof(PairType),
          alignof(PairType),
          offsetof(PairType, first),
          offsetof(PairType, second));
      SR_WARNING_POP();
      SReflect::InitTypeInfoFunctionPointers<PairType>(ti);
      return ti;
    }();
    return *s_typeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// Support for std::unordered_map
/////////////////////////////////////////////////////////////////////////////

// Implementation of std::unordered_map
//
// Note: JSON object keys must be strings. This is annoying.
// To use T as a key this implementation serializes T to JSON then serializes that as a JSON string
// Consider: std::unordered_map<int, float> foo. It would serialize as:
//
//     foo : {
//         "42" : 13.37
//         "256" : 9000.0
//     }
//
// Notice how the integers are encased in quotes.
// This implementation allows arbitrarily complex objects to be used as keys and serialized.
// It is not recommended to use complex types as keys. But you can.
namespace SReflect {
template <typename KeyT, typename ValueT, typename HashT, typename KeyEqualT, typename AllocT>
class UnorderedMapTypeInfo final : public SReflect::MapTypeInfo {
 public:
  // typedefs
  using MapType = std::unordered_map<KeyT, ValueT, HashT, KeyEqualT, AllocT>;

  // accessors
  MapType const& GetMap(void const* ptr) const {
    return *reinterpret_cast<MapType const*>(ptr);
  }
  MapType& GetMap(void* ptr) const {
    return *reinterpret_cast<MapType*>(ptr);
  }

  size_t GetNumKeys(void const* map) const override {
    return GetMap(map).size();
  }

  void Enumerate(void* map, OnEach const& callback) const override {
    for (auto& [key, value] : GetMap(map)) {
      callback(&key, &value);
    }
  }

  bool Clear(void* map) const override {
    GetMap(map).clear();
    return true;
  }

  bool Insert(void* map, void const* key, void const* value) const override {
    GetMap(map)[*reinterpret_cast<KeyT const*>(key)] = *reinterpret_cast<ValueT const*>(value);
    return true;
  }

  bool Remove(void* map, void const* key) const override {
    return (GetMap(map).erase(*reinterpret_cast<KeyT const*>(key)) != 0);
  }

  void SetInner(void const* src, void* dst) const override {
    GetMap(dst) = GetMap(src);
  }
};
} // namespace SReflect

template <typename KeyT, typename ValueT, typename HashT, typename KeyEqualT, typename AllocT>
struct SReflectTypeTraits<std::unordered_map<KeyT, ValueT, HashT, KeyEqualT, AllocT>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_map;
  static SReflect::MapTypeInfo const& GetTypeInfo() {
    static auto* sTypeInfo = []() {
      using MapType = std::unordered_map<KeyT, ValueT, HashT, KeyEqualT, AllocT>;
      auto* ti = new SReflect::UnorderedMapTypeInfo<KeyT, ValueT, HashT, KeyEqualT, AllocT>;
      SReflect::detail::InitMapTypeInfo(
          ti,
          "std::unordered_map",
          typeid(MapType),
          sizeof(MapType),
          alignof(MapType),
          SReflect::GetTypeInfo<KeyT>(),
          SReflect::GetTypeInfo<ValueT>());
      SReflect::InitTypeInfoFunctionPointers<MapType>(ti);
      return ti;
    }();
    return *sTypeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// Support for std::optional
/////////////////////////////////////////////////////////////////////////////

namespace SReflect::detail {
OptionalTypeInfo* MakeOptionalTypeInfo(
    std::type_info const& stdTypeInfo,
    TypeInfo const& innerTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    void* (*ensureValueFn)(void* optional),
    void* (*getValueFn)(void* optional),
    void (*setValueFn)(void const* srcValue, void* dstOptional));
} // namespace SReflect::detail

template <typename T>
struct SReflectTypeTraits<std::optional<T>> {
  using OptionalType = std::optional<T>;
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_optional;
  static SReflect::OptionalTypeInfo const& GetTypeInfo() {
    static auto* s_typeInfo = []() {
      auto* ti = SReflect::detail::MakeOptionalTypeInfo(
          typeid(OptionalType),
          SReflect::GetTypeInfo<T>(),
          sizeof(OptionalType),
          alignof(OptionalType),
          [](void* dstOptional) -> void* {
            auto* dstOpt = reinterpret_cast<OptionalType*>(dstOptional);
            if (!*dstOpt) {
              dstOpt->emplace(T{}); // Construct value
            }
            return &dstOpt->value();
          }, // Get inner value or nullptr
          [](void* srcOptional) -> void* {
            auto* srcOpt = reinterpret_cast<OptionalType*>(srcOptional);
            return (*srcOpt) ? &srcOpt->value() : nullptr;
          }, // Get inner value or nullptr
          [](void const* srcValue, void* dstOptional) { // Set inner value or reset
            auto* dstOpt = reinterpret_cast<OptionalType*>(dstOptional);
            auto* srcVal = reinterpret_cast<T const*>(srcValue);
            if (srcVal) {
              if (!*dstOpt) {
                dstOpt->emplace(T{}); // Construct value
              }
              SReflect::GetTypeInfo<T>().Set(srcVal, &**dstOpt); // Copy value
            } else {
              dstOpt->reset(); // No more value
            }
          });
      SReflect::InitTypeInfoFunctionPointers<OptionalType>(ti);
      return ti;
    }();
    return *s_typeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// Support for std::variant
/////////////////////////////////////////////////////////////////////////////

namespace SReflect {
template <typename... Types>
class StdVariantTypeInfo final : public SReflect::VariantTypeInfo {
 public:
  static_assert(
      sizeof...(Types) <= 255,
      "This variant has too many inner types. Binary serialization currently assumes the number can fit in a single byte.");
  using VariantT = std::variant<Types...>;
  size_t GetInnerTypeIndex(void const* obj) const final {
    return reinterpret_cast<VariantT const*>(obj)->index();
  }
  void* GetInnerObject(void* obj) const final {
    auto& var = *reinterpret_cast<VariantT*>(obj);
    return std::visit([](auto& inner) -> void* { return &inner; }, var);
  }
  bool TrySetInnerTypeIndex(void* obj, size_t typeIndex) const final {
    auto& var = *reinterpret_cast<VariantT*>(obj);
    if (typeIndex == var.index()) {
      return true;
    }
    if (typeIndex < sizeof...(Types)) {
      // Emplace a default-constructed object into the variant (if possible)
      return kEmplaceDefault[typeIndex](var);
    }
    return false;
  }

 private:
  // Static function table such that kEmplaceFnTable[i](var) emplaces a default constructed object
  // of type index i into the variant.
  using EmplaceFn = bool (*)(VariantT&);
  static constexpr std::array<EmplaceFn, sizeof...(Types)> kEmplaceDefault{+[](VariantT& v) {
    if constexpr (std::is_default_constructible_v<Types>) {
      v.template emplace<Types>();
      return true;
    } else {
      return false;
    }
  }...};
};
} // namespace SReflect

template <typename... Types>
struct SReflectTypeTraits<std::variant<Types...>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_variant;
  static SReflect::VariantTypeInfo const& GetTypeInfo() {
    static auto* typeInfo = []() {
      using VariantT = std::variant<Types...>;
      auto* ti = new SReflect::StdVariantTypeInfo<Types...>;
      ti->_innerTypes =
          std::initializer_list<SReflect::TypeInfo const*>{&SReflect::GetTypeInfo<Types>()...};
      SReflect::detail::InitVariantTypeInfo(
          *ti,
          "std::variant",
          sizeof(VariantT),
          alignof(VariantT),
          std::is_trivially_copyable_v<VariantT>);
      SReflect::InitTypeInfoFunctionPointers<VariantT>(ti);
      return ti;
    }();
    return *typeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// Core Types
/////////////////////////////////////////////////////////////////////////////

/**
Declares all the primitive types that are directly supported by the system. All types not
declared are supported as either array of data or a struct.
*/
#define SR_DECLARE_CORE_TYPE(T, typeID)                    \
  template <>                                              \
  struct SReflectTypeTraits<T> {                           \
    static constexpr SReflect::CoreType coreType = typeID; \
    static const auto& GetTypeInfo() {                     \
      return SReflect::GetPrimitiveInfo(typeID);           \
    }                                                      \
  }

// clang-format off
SR_DECLARE_CORE_TYPE(bool,        SReflect::CoreType::CT_bool);
SR_DECLARE_CORE_TYPE(uint8_t,     SReflect::CoreType::CT_uint8);
SR_DECLARE_CORE_TYPE(int8_t,      SReflect::CoreType::CT_int8);
SR_DECLARE_CORE_TYPE(char,        SReflect::CoreType::CT_int8);
SR_DECLARE_CORE_TYPE(uint16_t,    SReflect::CoreType::CT_uint16);
SR_DECLARE_CORE_TYPE(int16_t,     SReflect::CoreType::CT_int16);
SR_DECLARE_CORE_TYPE(uint32_t,    SReflect::CoreType::CT_uint32);
SR_DECLARE_CORE_TYPE(int32_t,     SReflect::CoreType::CT_int32);
SR_DECLARE_CORE_TYPE(uint64_t,    SReflect::CoreType::CT_uint64);
SR_DECLARE_CORE_TYPE(int64_t,     SReflect::CoreType::CT_int64);
SR_DECLARE_CORE_TYPE(float,       SReflect::CoreType::CT_float);
SR_DECLARE_CORE_TYPE(double,      SReflect::CoreType::CT_double);
// clang-format on

#ifdef __APPLE__
// The Mac compiler defines size_t as unsigned long which is not the same type as uint64_t.
// Simple Reflection will support it but treat it the same as uint64_t.
SR_DECLARE_CORE_TYPE(size_t, SReflect::CoreType::CT_uint64);
#endif //__APPLE__

#undef SR_DECLARE_CORE_TYPE

/////////////////////////////////////////////////////////////////////////////
// std::string
/////////////////////////////////////////////////////////////////////////////
template <>
struct SReflectTypeTraits<std::string> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_string;
  static SReflect::StringTypeInfo const& GetTypeInfo();
};

/////////////////////////////////////////////////////////////////////////////
// T[N] c-style array
/////////////////////////////////////////////////////////////////////////////
template <typename T, size_t N>
struct SReflectTypeTraits<T[N]> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    static auto* s_typeInfo = []() {
      using OuterType = T[N];
      auto const& inner = SReflect::GetTypeInfo<T>();
      auto* ti = SReflect::detail::MakeFixedArrayTypeInfoImpl(
          nullptr,
          false,
          inner,
          typeid(OuterType),
          sizeof(OuterType),
          alignof(OuterType),
          N,
          std::is_trivially_copyable_v<T>);
      ti->_constructInPlace = inner._constructInPlace ? +[](void* dst) {
        SReflect::detail::ArrayConstructInPlace(SReflect::GetTypeInfo<T>(), dst, N);
      } : nullptr;
      ti->_constructInPlaceByCopy = inner._constructInPlaceByCopy ? +[](void* dst, void const* src) {
        SReflect::detail::ArrayConstructInPlaceByCopy(SReflect::GetTypeInfo<T>(), dst, N, src);
      } : nullptr;
      ti->_destructInPlace = inner._destructInPlace ? +[](void* ptr) {
        SReflect::detail::ArrayDestructInPlace(SReflect::GetTypeInfo<T>(), ptr, N);
      } : nullptr;
      return ti;
    }();
    return *s_typeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// std::array<T,N>
/////////////////////////////////////////////////////////////////////////////
template <typename T, size_t N>
struct SReflectTypeTraits<std::array<T, N>> {
  using ArrayType = std::array<T, N>;
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    static auto* s_typeInfo =
        SReflect::MakeFixedArrayTypeInfo<std::array<T, N>, T, N>("std::array", true);
    return *s_typeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// std::vector<T>
/////////////////////////////////////////////////////////////////////////////
template <typename T, typename AllocT>
struct SReflectTypeTraits<std::vector<T, AllocT>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    using VectorType = std::vector<T, AllocT>;
    static auto* s_typeInfo =
        SReflect::MakeDynamicArrayTypeInfo<SReflect::VectorTypeInfo<VectorType>, VectorType, T>(
            "std::vector", true);
    return *s_typeInfo;
  }
};

/////////////////////////////////////////////////////////////////////////////
// picojson::value
/////////////////////////////////////////////////////////////////////////////
template <>
struct SReflectTypeTraits<picojson::value> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_other;
  static SReflect::TypeInfo const& GetTypeInfo() {
    static auto* s_typeInfo(SReflect::detail::MakePicojsonValueTypeInfo());
    return *s_typeInfo;
  }
};

///////////////////////////////////////////////////////////////////////////
// Macro Implementations
///////////////////////////////////////////////////////////////////////////

// clang-format off

// If someone forgets to use the full namespace path with SR_BeginStruct or SR_BeginEnum,
// then they will get a compiler error including a special message formatted with underscored.
//
// Example:
//    If you write SR_BeginClass(MyClass), where MyClass is in a nested namespace
//    then you might see compiler error output like this (from MSVC):
//
//    error C2039: 'MyClass': is not a member of '`global namespace''
//    error C2061 : syntax error : identifier 'MyClass'
//    error C2065 : '______PLEASE_PROVIDE_THE_FULL_NAMESPACE_OF_THIS_TYPE_WHEN_USING_SIMPLE_REFLECTION______' : undeclared identifier
//
// If you double click on any of those lines in the Visual Studio output window, it will
// take you to the SR_BeginStruct or SR_BeginEnum macro that needs to be changed. It would be
// nice if we could format the message with a static_assert, but that doesn't work. In the
// error case, the compiler would generate a warning about the malformed type name and stop
// evaluating the rest of the static_assert.
//
#define IMPL_SR_CHECK_NAMESPACE \
  ______PLEASE_PROVIDE_THE_FULL_NAMESPACE_OF_THIS_TYPE_WHEN_USING_SIMPLE_REFLECTION______

// See SR_BeginStruct
#define IMPL_SR_BeginStruct(structType, structName)                                         \
  SR_WARNING_PUSH()                                                                         \
  /* The compiler generates a warning when GetFinalTypeInfo and GetFileTypeId are virtual*/ \
  /* overrides that lack the 'override' keyword. This case happens when deriving from    */ \
  /* SReflect::BaseObject. We do not use the 'virtual' nor 'override' keywords here      */ \
  /* because SR_BeginStruct is also used in classes/structs that do not derive from      */ \
  /* SReflect::Base object and in fact do not have a v-table at all. The current syntax  */ \
  /* accommodates both cases.                                                            */ \
  SR_WARNING_IGNORE_CLANG(clang diagnostic ignored "-Winconsistent-missing-override")       \
  SReflect::StructTypeInfo const& GetFinalTypeInfo() const /*maybe an override*/ {          \
    static_assert(std::is_same_v<const structType, std::remove_pointer_t<decltype(this)>>,  \
       "Incorrect type parameter passed to SR_BeginStruct macro");                          \
    using IMPL_SR_CHECK_NAMESPACE = ::structType;                                           \
    if constexpr (sizeof(IMPL_SR_CHECK_NAMESPACE) != 0) {                                   \
      return GetTypeInfo();                                                                 \
    }                                                                                       \
  }                                                                                         \
  SReflect::TypeId GetFinalTypeId() const /*maybe an override*/ {                           \
    return structType::GetTypeId();                                                         \
  }                                                                                         \
  SR_WARNING_POP()                                                                          \
  static SReflect::TypeId GetTypeId() {                                                     \
    static const SReflect::TypeId s_typeTypeId = GetTypeInfo()._typeId;                     \
    return s_typeTypeId;                                                                    \
  }                                                                                         \
  static SReflect::StructTypeInfo const& GetTypeInfo() {                                    \
    static SReflect::StructTypeInfo* s_typeInfo = []() {                                    \
        using MyStruct = structType;                                                        \
        SReflect::StructTypeInfo* myInfo =                                                  \
          SReflect::MakeStructTypeInfo<MyStruct>(structName);

// Helper macro to create the template type as a single argument
#define IMPL_SR_TEMPLATE_TYPE(structType, ...) structType<__VA_ARGS__>

#define IMPL_SR_BeginStructTemplate(structType, ...)                                        \
  IMPL_SR_BeginStruct(IMPL_SR_TEMPLATE_TYPE(structType, __VA_ARGS__), ([]() {               \
      std::string str;                                                                      \
      SReflect::detail::AppendTemplateArgStr<__VA_ARGS__>(str);                             \
      return SReflect::detail::MakeTemplateTypeName(#structType, str.c_str());              \
    }()));

// See SR_EndStruct
#define IMPL_SR_EndStruct()                                                                 \
        SReflect::FinalizeStruct(myInfo);                                                   \
        return myInfo;                                                                      \
      }();                                                                                  \
    return *s_typeInfo;                                                                     \
  }

#define IMPL_SR_DeclareStructEx(structType)                                                 \
  template<>                                                                                \
  struct SReflectTypeTraits<structType> {                                                   \
    static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_struct;           \
    static SReflect::StructTypeInfo const& GetTypeInfo();                                   \
  };

#define IMPL_SR_BeginStructDefinitionEx(structType)                                         \
  static_assert(                                                                            \
      !std::is_base_of_v<SReflect::BaseObject, structType>,                                 \
      "Classes deriving from SReflect::BaseObject should use SR_BeginClass/SR_EndClass"     \
      "within the class declaration. It will define the virtual method GetFinalTypeInfo."); \
  SReflect::StructTypeInfo const& SReflectTypeTraits<structType>::GetTypeInfo() {           \
    static SReflect::StructTypeInfo* s_typeInfo = []() {                                    \
      using MyStruct = structType;                                                          \
      SReflect::StructTypeInfo* myInfo = SReflect::MakeStructTypeInfo<MyStruct>(#structType);

#define IMPL_SR_EndStructDefinitionEx()                                                     \
        SReflect::FinalizeStruct(myInfo);                                                   \
        return myInfo;                                                                      \
      }();                                                                                  \
    return *s_typeInfo;                                                                     \
  }

#define IMPL_SR_BeginStructEx(structType)                                                   \
  template<>                                                                                \
  struct SReflectTypeTraits<structType> {                                                   \
    static_assert(                                                                          \
      !std::is_base_of_v<SReflect::BaseObject, structType>,                                 \
      "Classes deriving from SReflect::BaseObject should use SR_BeginClass/SR_EndClass"     \
      "within the class declaration. It will define the virtual method GetFinalTypeInfo."); \
    static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_struct;           \
    static SReflect::StructTypeInfo const& GetTypeInfo() {                                  \
      static SReflect::StructTypeInfo* s_typeInfo = []() {                                  \
        using MyStruct = structType;                                                        \
          SReflect::StructTypeInfo* myInfo = SReflect::MakeStructTypeInfo<MyStruct>(#structType);

#define IMPL_SR_EndStructEx()                                                               \
          SReflect::FinalizeStruct(myInfo);                                                 \
          return myInfo;                                                                    \
        }();                                                                                \
      return *s_typeInfo;                                                                   \
    }                                                                                       \
  };

// See SR_NamedField
#define IMPL_SR_Field_Name(memberName, fieldName)                                           \
  {                                                                                         \
    SR_WARNING_PUSH();                                                                      \
    /* The MacOS compiler complains about offsetof used with non-standard-layout types */   \
    /* (e.g. classes with inheritance). Ignore the warning. The tests show that it works.*/ \
    SR_WARNING_IGNORE_GCC(GCC diagnostic ignored "-Winvalid-offsetof");                     \
    SReflect::detail::AddField(                                                             \
        *myInfo,                                                                            \
        fieldName,                                                                          \
        offsetof(MyStruct, memberName),                                                     \
        SReflect::GetTypeInfo<decltype(MyStruct::memberName)>());                           \
    SR_WARNING_POP();                                                                       \
  }

  // See RemoveField
#define IMPL_SR_RemoveField(fieldName)                                                        \
    SReflect::detail::RemoveField(*myInfo, fieldName);

// See SR_BaseClass
#define IMPL_SR_BaseClass(BaseClass)                                                        \
  {                                                                                         \
    SReflect::detail::AddBaseClass(                                                         \
        *myInfo,                                                                            \
        SReflect::GetTypeInfo<BaseClass>(),                                                 \
        SReflect::GetOffsetFromDerivedToBase<MyStruct, BaseClass>());                       \
  }

// See SR_BeginEnum
#define IMPL_SR_BeginEnum(enumName)                                                         \
  static_assert(std::is_enum_v<enumName>,                                                   \
     "SR_BeginEnum can only be used with enums an enum classes.");                          \
  template <>                                                                               \
  struct SReflectTypeTraits<enumName> {                                                     \
    static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_enum;             \
    static SReflect::EnumTypeInfo const& GetTypeInfo() {                                    \
      using IMPL_SR_CHECK_NAMESPACE = ::enumName;                                           \
      if constexpr (sizeof(IMPL_SR_CHECK_NAMESPACE) != 0) {                                 \
        /* NOTE: We should be able to use the lambda syntax here, just like in */           \
        /*       SR_BeginStruct, but MSVC fails to compile it when the specialization */    \
        /*       of SReflectTypeInfo<T> is declared in a nested namespace. Must be a */     \
        /*       compiler bug. Calling a separately named static function is a */           \
        /*       work-around.*/                                                             \
        static SReflect::EnumTypeInfo const* s_typeInfo = IMPL_MakeTypeInfo();              \
        return *s_typeInfo;                                                                 \
      }                                                                                     \
    }                                                                                       \
    static SReflect::EnumTypeInfo const* IMPL_MakeTypeInfo() {                              \
      (void)(coreType); /* Suppress unused variable warnings. Sometimes it is used. */      \
      using MyEnum = enumName;                                                              \
      auto* myInfo = SReflect::detail::MakeEnum(                                            \
          #enumName,                                                                        \
          typeid(enumName),                                                                 \
          SReflect::GetTypeInfo<std::underlying_type<enumName>::type>());                   \
      SReflect::InitTypeInfoFunctionPointers<enumName>(myInfo);

// See SR_EnumItem
#define IMPL_SR_EnumItem(itemName)                                                          \
  { SReflect::detail::AddEnumItem(*myInfo, #itemName, (uint64_t)MyEnum::itemName); }

// See SR_EndEnum
#define IMPL_SR_EndEnum()                                                                   \
      return myInfo;                                                                        \
    }                                                                                       \
  };

// clang-format on

#endif // SIMPLE_REFLECTION_ENABLE
