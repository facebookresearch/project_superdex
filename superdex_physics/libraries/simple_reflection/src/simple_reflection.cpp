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

#if SIMPLE_REFLECTION_ENABLE

#include <cityhash/city.h> // CityHash by Google (third-party)

SR_WARNING_PUSH()
SR_WARNING_IGNORE_MSVC(4459) // declaration of 'last' hides global declaration
#include <picojson/picojson.h> // picojson (third-party)
SR_WARNING_POP()

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <unordered_map>

// Reflection of this enum type could be moved to the header if it is needed elsewhere.
// For now, it is restricted to the cpp file to reduce bloat.
SR_BeginEnum(SReflect::CoreType);
SR_EnumItem(X_Invalid);
SR_EnumItem(CT_string);
SR_EnumItem(CT_bool);
SR_EnumItem(CT_uint8);
SR_EnumItem(CT_int8);
SR_EnumItem(CT_uint16);
SR_EnumItem(CT_int16);
SR_EnumItem(CT_uint32);
SR_EnumItem(CT_int32);
SR_EnumItem(CT_uint64);
SR_EnumItem(CT_int64);
SR_EnumItem(CT_float);
SR_EnumItem(CT_double);
SR_EnumItem(CT_enum);
SR_EnumItem(CT_array);
SR_EnumItem(CT_matrix);
SR_EnumItem(CT_struct);
SR_EnumItem(CT_field);
SR_EnumItem(CT_map);
SR_EnumItem(CT_optional);
SR_EnumItem(CT_variant);
SR_EnumItem(CT_other);
SR_EnumItem(X_Count);
SR_EndEnum();

namespace SReflect {

// Maximum allowed length for a type name
static constexpr size_t kMaxTypeNameLen = 256;

/////////////////////////////////////////////////////////////////////////////
// Debugging
/////////////////////////////////////////////////////////////////////////////

// Helper function to format a string. Not static because SR_LOG might compile out.
namespace {
std::string Format(const char* format, ...) {
  va_list args1, args2;
  va_start(args1, format);
  va_copy(args2, args1);
  int len = vsnprintf(nullptr, 0, format, args1);
  std::string str;
  str.resize((size_t)len); // +1 makes it cheap to append "\n" for logging
  if (len) {
    vsnprintf(&str[0], len + 1, format, args2); // This use of &str[0] guaranteed by C++11
  }
  va_end(args2);
  va_end(args1);
  return str;
}
} // namespace

// Asserts are enabled by default
#ifndef SR_ASSERT_ENABLE
#define SR_ASSERT_ENABLE 1
#endif
#if SR_ASSERT_ENABLE
#define SR_ASSERT(condition, message)  \
  if (condition) {                     \
  } else {                             \
    throw std::runtime_error(message); \
  }
#else
#define SR_ASSERT(condition, message)
#endif

// Logging is enabled by default
#ifndef SR_LOG_ENABLE
#define SR_LOG_ENABLE 1
#endif
#if SR_LOG_ENABLE
// Log using std::cout because it is easier to intercept/redirect if people want to do that.
// Every line ends with std::endl, which causes a flush which is considered to be appropriate
// because this macro is generally used for errors/warnings which should appear immediately.
#define SR_LOG(...) std::cout << "[Simple Reflection] " << Format(__VA_ARGS__) << std::endl;
#else
#define SR_LOG(...)
#endif

/////////////////////////////////////////////////////////////////////////////
// Utilities
/////////////////////////////////////////////////////////////////////////////

static char const* MakeTypeNameWithoutAnyNamespace(char const* nameWithNamespace) {
  // Remove all namespaces from the name and return a newly allocated string.
  // Example: "std::vector<std::pair<std::string,X::Thing>>" --> "vector<pair<string,Thing>>"
  std::array<char, kMaxTypeNameLen> buf;
  auto dstBegin = buf.begin();
  auto dst = buf.begin();
  auto const srcLen = strlen(nameWithNamespace);
  char const* src = nameWithNamespace + srcLen - 1;
  char const* srcREnd = nameWithNamespace;
  while (src >= srcREnd) {
    // Look for "::"
    if (src >= srcREnd + 2 && *src == ':' && *(src - 1) == ':') {
      // Skip next identifier in reverse order
      src -= 2;
      while ((src >= srcREnd) &&
             ((*src >= 'A' && *src <= 'Z') || (*src >= 'a' && *src <= 'z') ||
              (*src >= '0' && *src <= '9') || *src == '_')) {
        src--;
      }
    } else {
      *dst = *src;
      ++dst;
      --src;
    }
  }
  // Allocate a copy of the buffer and reverse direction
  auto const dstLen = dst - dstBegin;
  char* out = new char[dstLen + 1];
  memcpy(out, buf.data(), dstLen);
  std::reverse(out, out + dstLen);
  out[dstLen] = '\0';
  return out;
}

static char const* GetTypeNameWithoutNamespace(const char* nameWithNamespace) {
  // Return just the type name without any namespaces.
  // Example: "MyNamespace::Thing" --> "Thing"
  char const* templateArgs = strchr(nameWithNamespace, '<');
  if (templateArgs) {
    // Remove namespaces from template argument types as well.
    return MakeTypeNameWithoutAnyNamespace(nameWithNamespace);
  } else {
    // Return the substring that starts after the last colon
    char const* lastColon = strrchr(nameWithNamespace, ':');
    return lastColon ? lastColon + 1 : nameWithNamespace;
  }
}

TypeId ComputeTypeId(const char* nameWithNamespace) {
  // This is the one location that selects the hashing algorithm to use
  return TypeId{CityHash64(nameWithNamespace, strlen(nameWithNamespace))};
}

[[nodiscard]] static bool StrEqualCaseInsensitive(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

template <class F>
class Defer final {
 public:
  Defer() = delete;
  Defer(const Defer&) = delete;
  Defer(Defer&&) = delete;
  Defer& operator=(const Defer&) = delete;
  Defer& operator=(Defer&&) = delete;

  explicit Defer(F&& f) : _fun(std::move(f)) {};

  ~Defer() {
    _fun();
  }

 private:
  F _fun;
};

#define SR_PP_CAT_IMPL(a, b) a##b
#define SR_PP_CAT(a, b) SR_PP_CAT_IMPL(a, b)
#define SR_DEFER(code) Defer SR_PP_CAT(_defer, __LINE__)([&]() { code; })

/////////////////////////////////////////////////////////////////////////////
// TypeId Enforcement
/////////////////////////////////////////////////////////////////////////////

/**
Called when a new type is registered. Enforces the policy that no two types
can have the same TypeId. In theory this could happen due to a hash collision or
(more likely) anonymous namespaces.
*/
void VerifyTypeIdIsUnique(TypeInfo const& typeInfo, std::type_info const& stdInfo) {
  struct RegisteredInfo {
    char const* name;
  };

  static std::mutex s_idTableMutex;
  std::lock_guard lock(s_idTableMutex);

  static std::unordered_map<TypeId, RegisteredInfo> s_idTable;

  auto it = s_idTable.find(typeInfo._typeId);
  if (it == s_idTable.end()) {
    s_idTable.emplace(std::make_pair(typeInfo._typeId, RegisteredInfo{stdInfo.name()}));
  } else {
    RegisteredInfo const& existingType = it->second;
    if (strcmp(existingType.name, stdInfo.name()) != 0) {
      SR_LOG(
          "Attempting to declare type \"%s\" with TypeId %" PRIu64
          ", but type \"%s\" is already declared with the same TypeId. If either of these "
          "names are in an anonymous namespace, then please give the namespace a name in "
          "order to disambiguate.",
          typeInfo._nameWithNamespace,
          typeInfo._typeId.value,
          existingType.name);
    }
  }
}

void detail::ArrayConstructInPlace(TypeInfo const& elementType, void* dst, size_t count) {
  assert(elementType._constructInPlace != nullptr);
  auto* itr = static_cast<std::byte*>(dst);
  auto const* end = itr + count * elementType._sizeInBytes;
  for (; itr != end; itr += elementType._sizeInBytes) {
    elementType._constructInPlace(itr);
  }
}

void detail::ArrayConstructInPlaceByCopy(
    TypeInfo const& elementType,
    void* dst,
    size_t count,
    void const* src) {
  assert(elementType._constructInPlaceByCopy != nullptr);
  auto const* srcItr = static_cast<std::byte const*>(src);
  auto* dstItr = static_cast<std::byte*>(dst);
  auto* dstEnd = dstItr + count * elementType._sizeInBytes;
  for (; dstItr != dstEnd; srcItr += elementType._sizeInBytes, dstItr += elementType._sizeInBytes) {
    elementType._constructInPlaceByCopy(dstItr, srcItr);
  }
}

void detail::ArrayDestructInPlace(TypeInfo const& elementType, void* ptr, size_t count) {
  assert(elementType._destructInPlace != nullptr);
  auto* itr = static_cast<std::byte*>(ptr);
  auto const* end = itr + count * elementType._sizeInBytes;
  for (; itr != end; itr += elementType._sizeInBytes) {
    elementType._destructInPlace(itr);
  }
}

namespace detail {
// Helper to get a typed value from picojson.
// Adapted from RTech's <Common/PicojsonUtils.h>
template <typename T>
static void GetJsonValue(picojson::value const& pjval, T& val) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<std::decay_t<T>, bool>) {
    // Get any signed or unsigned integer as int64_t
    val = static_cast<T>(pjval.get<int64_t>());
  } else if constexpr (std::is_floating_point_v<T>) {
    // Get any floating point value as double
    val = static_cast<T>(pjval.get<double>());
  } else {
    val = pjval.get<T>();
  }
}
} // namespace detail

namespace serde::bytes {
// TODO:
// * varint encode/decode for sizes (see SerializeStreams.h for example)

/////////////////////////////////////////////////////////////////////////////
// Serialize/Deserialize declarations
/////////////////////////////////////////////////////////////////////////////

// Serializers
template <typename T>
static bool Serialize(T const& src, StreamWriter& dst);
template <typename T>
static bool Serialize(Span<const T> src, StreamWriter& dst);
static bool Serialize(std::string const& src, StreamWriter& dst);

// Deserializers
template <typename T>
static bool Deserialize(StreamReader& src, T& dst);
template <typename Container>
static bool DeserializeContainer(StreamReader& src, Container& dst);
bool Deserialize(StreamReader& src, std::string& dst);

/////////////////////////////////////////////////////////////////////////////
// Serialize/Deserialize implementations
/////////////////////////////////////////////////////////////////////////////
template <typename T>
inline static bool Serialize(T const& src, StreamWriter& dst) {
  static_assert(std::is_arithmetic_v<T>, "Can not serialize non-arithmetic types via memcpy");
  return dst.Write(&src, sizeof(T));
}

template <typename T>
inline static bool Serialize(Span<const T> src, StreamWriter& dst) {
  static_assert(std::is_arithmetic_v<T>, "Can not serialize non-arithmetic types via memcpy");

  // Serialize size
  size_t numElements = src.size();
  if (!Serialize(numElements, dst)) {
    return false;
  }

  // Serialize full span of elements in single memcpy (because they are trivially copyable
  size_t numBytesToCopy = sizeof(T) * numElements;
  return dst.Write(src.data(), numBytesToCopy);
}

template <typename T, size_t N>
static bool Serialize(std::array<T, N> const& src, StreamWriter& dst) {
  static_assert(std::is_arithmetic_v<T>, "Can not serialize non-arithmetic types via memcpy");
  return dst.Write(src.data(), sizeof(T) * src.size());
}

template <typename T, size_t N>
static bool Serialize(T const src[N], StreamWriter& dst) {
  static_assert(std::is_arithmetic_v<T>, "Can not serialize non-arithmetic types via memcpy");
  return dst.Write(src, sizeof(T) * N);
}

static bool Serialize(std::string const& src, StreamWriter& dst) {
  return Serialize(Span{src.data(), src.size()}, dst);
}

template <typename T>
inline static bool Deserialize(StreamReader& src, T& dst) {
  return src.Read(&dst, sizeof(T));
}

template <typename T>
inline static bool Deserialize(StreamReader& src, Span<T> dst) {
  static_assert(std::is_arithmetic_v<T>, "Can not serialize non-arithmetic types via memcpy");

  return src.Read(dst.data(), sizeof(T) * dst.size());
}

// TODO: can maybe rename just Deserialize
template <typename Container>
inline static bool DeserializeContainer(StreamReader& src, Container& dst) {
  // Deserialize size
  size_t numElements = 0;
  if (!Deserialize(src, numElements)) {
    return false;
  }

  // Set container size
  dst.resize(numElements);

  // Deserialize elements
  for (size_t i = 0; i < numElements; ++i) {
    if (!Deserialize(src, dst[i])) {
      return false;
    }
  }

  return true;
}

bool Deserialize(StreamReader& src, std::string& dst) {
  return DeserializeContainer(src, dst);
}

} // namespace serde::bytes

/////////////////////////////////////////////////////////////////////////////
// TypeInfo
/////////////////////////////////////////////////////////////////////////////

static Attribute const* FindAttribute(TypeId attrType, TypeInfo::AttributeList const& list) {
  for (auto [type, ptr] : list) {
    if (type == attrType) {
      return ptr;
    }
  }
  return nullptr; // Not found
}

template <typename PrimType, typename AttribRangeType>
static bool IsRangeValid(void const* src, SReflect::TypeInfo::AttributeList const& attribs) {
  auto* range =
      static_cast<AttribRangeType const*>(FindAttribute(AttribRangeType::GetTypeId(), attribs));
  if (range) {
    PrimType const& v = *reinterpret_cast<PrimType const*>(src);
    if ((v < static_cast<PrimType>(range->_min)) || (v > static_cast<PrimType>(range->_max))) {
      return false;
    }
  }
  return true;
}

Attribute const* TypeInfo::GetAttribute(TypeId attrType) const {
  return FindAttribute(attrType, _attributes);
}

bool TypeInfo::IsMemCopySafe() const {
  // Can only true if overridden by derived class.
  return false;
}

bool TypeInfo::HasAttribute(TypeId attrType) const {
  return FindAttribute(attrType, _attributes) != nullptr;
}

namespace {
class DefaultAllocator final : public Allocator {
 public:
  static constexpr std::string_view kFooter = "EndAlloc";

  void* do_allocate(size_t bytes, size_t alignment) override {
    SR_ASSERT(
        (alignment > 0) && !(alignment & (alignment - 1)), "Alignment must be a power of two");
    if (bytes == 0) {
      throw std::bad_alloc{};
    }
    if (alignment <= sizeof(double)) {
      return std::malloc(bytes);
    } else {
      // Implement aligned allocation using std::malloc since some compiler fail to support
      // std::pmr::new_delete_resource, or std::aligned_alloc.
      auto* base = static_cast<std::byte*>(
          std::malloc(bytes + alignment - 1 + sizeof(uintptr_t) + kFooter.size()));
      auto aligned = (reinterpret_cast<uintptr_t>(base) + alignment - 1) & ~(alignment - 1);
      auto offset = aligned - reinterpret_cast<uintptr_t>(base);
      auto* alignedPtr = reinterpret_cast<std::byte*>(aligned);
      memcpy(alignedPtr + bytes, kFooter.data(), kFooter.size());
      memcpy(alignedPtr + bytes + kFooter.size(), &offset, sizeof(offset));
      return alignedPtr;
    }
  }

  void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
    if (ptr) {
      if (alignment <= sizeof(double)) {
        std::free(ptr);
      } else {
        auto aligned = static_cast<std::byte*>(ptr);
        SR_ASSERT(
            0 == memcmp(aligned + bytes, kFooter.data(), kFooter.size()),
            "Address is invalid or allocation footer was overwritten");
        std::uintptr_t offset;
        memcpy(&offset, aligned + bytes + kFooter.size(), sizeof(offset));
        auto* base = aligned - offset;
        std::free(base);
      }
    }
  }

  [[nodiscard]] bool do_is_equal(Allocator const& other) const noexcept override {
    return dynamic_cast<DefaultAllocator const*>(&other) != nullptr;
  }

  static DefaultAllocator* Get() {
    static DefaultAllocator instance;
    return &instance;
  }
};
} // namespace

Allocator* GetDefaultAllocator() {
  return DefaultAllocator::Get();
}

static Allocator* EnsureAllocator(Allocator* allocator) {
  return allocator ? allocator : DefaultAllocator::Get();
}

static std::string Uint64ToHexString(uint64_t value) {
  char buffer[17];
  snprintf(buffer, sizeof(buffer), "%016" PRIx64, value);
  return buffer;
}

static std::string TypeIdToString(TypeId typeId) {
  return Uint64ToHexString(typeId.value);
}

static std::string SafeString(char const* s) {
  return std::string{s ? s : ""};
}

// Inserts key=value into `obj` unless `value` is an empty array or empty object. Used for
// structural collection keys so that empty collections are omitted from the output entirely.
static void SetField(picojson::object& obj, char const* key, picojson::value value) {
  if (value.is<picojson::array>() && value.get<picojson::array>().empty()) {
    return;
  }
  if (value.is<picojson::object>() && value.get<picojson::object>().empty()) {
    return;
  }
  obj[key] = std::move(value);
}

// Ensures `info` is present in the flat dictionary `dict` and returns a reference to it as its
// fully-qualified name. Returns json null if `info` is nullptr.
static picojson::value AddTypeRef(TypeInfo const* info, picojson::value& dict) {
  if (info == nullptr) {
    return picojson::value{};
  }
  info->SerializeTypeInfo(dict);
  return picojson::value{SafeString(info->_nameWithNamespace)};
}

// Ensures each type in `infos` is present in `dict` and returns an array of their fully-qualified
// names.
template <typename InfoPtr>
static picojson::array AddTypeRefArray(std::vector<InfoPtr> const& infos, picojson::value& dict) {
  picojson::array result;
  result.reserve(infos.size());
  for (auto const* info : infos) {
    if (info == nullptr) {
      continue;
    }
    info->SerializeTypeInfo(dict);
    result.push_back(picojson::value{SafeString(info->_nameWithNamespace)});
  }
  return result;
}

// Serializes an attribute list into a dictionary keyed by each attribute's struct-type
// fully-qualified name. Each attribute's struct type is also added to the flat dictionary `dict`.
// The value is kept even when it is an empty object (marker attributes carry no fields).
static picojson::object WriteAttributes(
    TypeInfo::AttributeList const& attributes,
    picojson::value& dict) {
  picojson::object result;
  for (auto const& [unusedTypeId, attribute] : attributes) {
    (void)unusedTypeId;
    if (attribute == nullptr) {
      continue;
    }

    StructTypeInfo const& attributeTypeInfo = attribute->GetFinalTypeInfo();
    attributeTypeInfo.SerializeTypeInfo(dict);

    picojson::value valueJson = picojson::object{};
    attributeTypeInfo.Serialize(attribute, valueJson);
    result[SafeString(attributeTypeInfo._nameWithNamespace)] = std::move(valueJson);
  }
  return result;
}

static void
SerializeCommonTypeInfo(TypeInfo const& info, picojson::value& entry, picojson::value& dict) {
  picojson::object& obj = entry.get<picojson::object>();
  GetTypeInfo<CoreType>().Serialize(&info._coreType, obj["coreType"]);
  obj["alignment"] = static_cast<double>(info._alignment);
  obj["sizeInBytes"] = static_cast<double>(info._sizeInBytes);
  obj["name"] = SafeString(info._name);
  obj["nameWithNamespace"] = SafeString(info._nameWithNamespace);
  obj["typeId"] = TypeIdToString(info._typeId);
  SetField(obj, "attributes", picojson::value{WriteAttributes(info._attributes, dict)});
}

static picojson::value WriteEnumItem(EnumItem const& item, picojson::value& dict) {
  picojson::object result;
  result["name"] = SafeString(item._name);
  result["value"] = Uint64ToHexString(item._value);
  SetField(result, "attributes", picojson::value{WriteAttributes(item._attributes, dict)});
  return picojson::value{std::move(result)};
}

// Builds a field descriptor into `entry` (an object). A field is a descriptor, not a dictionary
// entry, so it does not write common info and is not added to the flat dictionary itself; its
// inner type is referenced by fully-qualified name.
static void
WriteFieldDescriptor(FieldTypeInfo const& field, picojson::value& entry, picojson::value& dict) {
  picojson::object& obj = entry.get<picojson::object>();
  obj["name"] = SafeString(field._name);
  obj["offset"] = static_cast<double>(field._offset);
  obj["innerType"] = AddTypeRef(field._innerTypeInfo, dict);
  SetField(obj, "attributes", picojson::value{WriteAttributes(field._attributes, dict)});
}

void* TypeInfo::New(Allocator* allocator) const {
  assert(_sizeInBytes != 0);
  assert(_alignment != 0);
  if (_constructInPlace) {
    auto* ptr = EnsureAllocator(allocator)->allocate(_sizeInBytes, _alignment);
    if (ptr) {
      _constructInPlace(ptr);
      return ptr;
    }
  }
  return nullptr;
}

void* TypeInfo::Clone(void const* src, Allocator* allocator) const {
  assert(_sizeInBytes != 0);
  assert(_alignment != 0);
  if (_constructInPlaceByCopy) {
    auto* ptr = EnsureAllocator(allocator)->allocate(_sizeInBytes, _alignment);
    if (ptr) {
      _constructInPlaceByCopy(ptr, src);
      return ptr;
    }
  }
  return nullptr;
}

void TypeInfo::Delete(void* ptr, Allocator* allocator) const {
  assert(_sizeInBytes != 0);
  assert(_alignment != 0);
  if (ptr) {
    assert(_destructInPlace); // Must exist if object is valid
    _destructInPlace(ptr);
    EnsureAllocator(allocator)->deallocate(ptr, _sizeInBytes, _alignment);
  }
}

void TypeInfo::Set(void const* src, void* dst) const {
  SetInner(src, dst);

  // Trigger onchanged callback, if we have one.
  auto const* onChanged = GetAttribute<Attribute_OnChanged>();
  if (onChanged != nullptr) {
    onChanged->_onChanged();
  }
}

void TypeInfo::Serialize(void const* src, picojson::value& dst) const {
  if (HasAttribute<Attribute_DoNotSerialize>()) {
    return;
  }
  SerializeInner(src, dst);
}

void TypeInfo::Deserialize(
    picojson::value const& src,
    void* dst,
    DeserializeFlags deserializeFlags,
    int& outIssuesDetected) const {
  outIssuesDetected = 0;
  if (HasAttribute<Attribute_DoNotSerialize>()) {
    return;
  }

  try {
    DeserializeInner(src, dst, deserializeFlags, outIssuesDetected);
  } catch (std::runtime_error const& e) {
    SR_LOG(
        "Failure while deserializing type '%s': %s",
        _nameWithNamespace ? _nameWithNamespace : "(unknown)",
        e.what());
    ++outIssuesDetected;
  }
}

bool TypeInfo::SerializeToBytes(void const* src, StreamWriter& dst) const {
  if (HasAttribute<Attribute_DoNotSerialize>()) {
    return true;
  }
  return SerializeToBytesInner(src, dst);
}

bool TypeInfo::DeserializeFromBytes(StreamReader& src, void* dst) const {
  if (HasAttribute<Attribute_DoNotSerialize>()) {
    return true;
  }
  return DeserializeFromBytesInner(src, dst);
}

std::string TypeInfo::TypeInfoToJson(bool pretty) const {
  picojson::value jsonValue;
  SerializeTypeInfo(jsonValue);
  return jsonValue.serialize(pretty);
}

void TypeInfo::SerializeTypeInfo(picojson::value& dst) const {
  if (!dst.is<picojson::object>()) {
    dst = picojson::value{picojson::object{}};
  }
  auto& dstObj = dst.get<picojson::object>();
  std::string const key = SafeString(_nameWithNamespace);
  if (dstObj.count(key) != 0) {
    return; // dedup + cycle break
  }
  // Reserve the slot before recursing so nested types that point back at this type terminate.
  dstObj[key] = picojson::value{picojson::object{}};
  picojson::value entry{picojson::object{}};
  SerializeTypeInfoImpl(entry, dst); // May recurse into dst; key already reserved.
  dstObj[key] = std::move(entry); // Re-index by key (map-safe).
}

// Public utility function:
std::string
TypeInfoListToJson(SReflect::TypeInfo const* const* typeList, size_t numTypes, bool pretty) {
  SR_ASSERT((typeList != nullptr) || (numTypes == 0), "Null input array");
  // TypeInfo::SerializeTypeInfo does not clear the input, so we can call it repeatedly to
  // accumulate a dictionary of all recursive types (recursive).
  picojson::value dictionary = picojson::object{};
  for (size_t i = 0; i < numTypes; ++i) {
    SR_ASSERT(typeList[i] != nullptr, "Null input");
    typeList[i]->SerializeTypeInfo(dictionary);
  }
  return dictionary.serialize(pretty);
}

void TypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  SerializeCommonTypeInfo(*this, entry, dict);
}

bool TypeInfo::IsValid(void const* src) const {
  return IsValidInner(src, _attributes);
}

uint64_t TypeInfo::ToUInt64(void const* /*src*/) const {
  throw std::runtime_error("TypeInfo::ToUInt64 not supported for this type");
}

void TypeInfo::FromUInt64(uint64_t /*val*/, void* /*dst*/) const {
  throw std::runtime_error("TypeInfo::FromUInt64 not supported for this type");
}

/////////////////////////////////////////////////////////////////////////////
// PrimitiveTypeInfo
/////////////////////////////////////////////////////////////////////////////

template <typename T>
class PrimitiveTypeInfo : public TypeInfo {
 public:
  using TypeInfo::TypeInfo;

 protected:
  bool IsValidInner(void const* src, AttributeList const& attribs) const override {
    if constexpr (std::is_floating_point_v<T>) {
      return IsRangeValid<T, Attribute_FloatRange>(src, attribs);
    } else if constexpr (std::is_signed_v<T>) {
      return IsRangeValid<T, Attribute_IntRange>(src, attribs);
    } else if constexpr (std::is_unsigned_v<T>) {
      return IsRangeValid<T, Attribute_UIntRange>(src, attribs);
    } else {
      static_assert(std::is_same_v<T, std::string> || std::is_same_v<T, bool>, "Unexpected type!");
      return true;
    }
  }

  void SetInner(void const* src, void* dst) const override {
    *reinterpret_cast<T*>(dst) = *reinterpret_cast<T const*>(src);
  }

  void SerializeInner(void const* src, picojson::value& dst) const override {
    dst = *static_cast<const T*>(src);
  }

  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    return serde::bytes::Serialize(*static_cast<T const*>(src), dst);
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    return serde::bytes::Deserialize(src, *static_cast<T*>(dst));
  }

  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags /*deserializeFlags*/,
      int& /*inOutIssuesDetected*/) const override {
    detail::GetJsonValue(src, *static_cast<T*>(dst));
  }

  uint64_t ToUInt64([[maybe_unused]] void const* src) const override {
    if constexpr (std::is_arithmetic_v<T>) {
      return static_cast<uint64_t>(*reinterpret_cast<const T*>(src));
    } else {
      throw std::runtime_error("TypeInfo::ToUInt64 not supported for this type");
    }
  }

  void FromUInt64([[maybe_unused]] uint64_t val, [[maybe_unused]] void* dst) const override {
    if constexpr (std::is_arithmetic_v<T>) {
      *reinterpret_cast<T*>(dst) = static_cast<T>(val);
    } else {
      throw std::runtime_error("TypeInfo::FromUInt64 not supported for this type");
    }
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return std::is_arithmetic_v<T> && !HasAttribute<Attribute_DoNotSerialize>();
  }
};

template <typename T>
static TypeInfo* MakePrimitiveInfoTyped(char const* name, const char* nameWithNamespace) {
  auto* ti = new PrimitiveTypeInfo<T>;
  ti->_sizeInBytes = sizeof(T);
  ti->_alignment = alignof(T);
  ti->_name = name;
  ti->_nameWithNamespace = nameWithNamespace ? nameWithNamespace : name;
  ti->_typeId = ComputeTypeId(ti->_nameWithNamespace);
  ti->_coreType = SReflectTypeTraits<T>::coreType;
  InitTypeInfoFunctionPointers<T>(ti);
  VerifyTypeIdIsUnique(*ti, typeid(T));
  return ti;
}

TypeInfo* MakePrimitiveInfo(CoreType coreType, char const* nameWithNamespace) {
  char const* name = GetTypeNameWithoutNamespace(nameWithNamespace);
  switch (coreType) {
      // clang-format off
    case CoreType::CT_bool:   return MakePrimitiveInfoTyped<bool>(name, nameWithNamespace);
    case CoreType::CT_uint8:  return MakePrimitiveInfoTyped<uint8_t>(name, nameWithNamespace);
    case CoreType::CT_int8:   return MakePrimitiveInfoTyped<int8_t>(name, nameWithNamespace);
    case CoreType::CT_uint16: return MakePrimitiveInfoTyped<uint16_t>(name, nameWithNamespace);
    case CoreType::CT_int16:  return MakePrimitiveInfoTyped<int16_t>(name, nameWithNamespace);
    case CoreType::CT_uint32: return MakePrimitiveInfoTyped<uint32_t>(name, nameWithNamespace);
    case CoreType::CT_int32:  return MakePrimitiveInfoTyped<int32_t>(name, nameWithNamespace);
    case CoreType::CT_uint64: return MakePrimitiveInfoTyped<uint64_t>(name, nameWithNamespace);
    case CoreType::CT_int64:  return MakePrimitiveInfoTyped<int64_t>(name, nameWithNamespace);
    case CoreType::CT_float:  return MakePrimitiveInfoTyped<float>(name, nameWithNamespace);
    case CoreType::CT_double: return MakePrimitiveInfoTyped<double>(name, nameWithNamespace);
    // clang-format on
    default:
      SR_ASSERT(
          coreType < CoreType::CT_string || coreType > CoreType::CT_double,
          "Expected a primitive CoreType for MakePrimitiveInfo");
      return nullptr;
  }
}

struct CoreTypeInfo {
  char const* name;
  TypeInfo const* primitiveTypeInfo;
};

// clang-format off
SR_WARNING_PUSH()
SR_WARNING_IGNORE_CLANG(clang diagnostic ignored "-Wglobal-constructors")
static const CoreTypeInfo kCoreTypeInfo[] = {
    {"Invalid",  nullptr},
    {"string",   nullptr}, // Not a primitive type
    {"bool",     MakePrimitiveInfo(CoreType::CT_bool, "bool")},
    {"uint8",    MakePrimitiveInfo(CoreType::CT_uint8, "uint8")},
    {"int8",     MakePrimitiveInfo(CoreType::CT_int8, "int8")},
    {"uint16",   MakePrimitiveInfo(CoreType::CT_uint16, "uint16")},
    {"int16",    MakePrimitiveInfo(CoreType::CT_int16, "int16")},
    {"uint32",   MakePrimitiveInfo(CoreType::CT_uint32, "uint32")},
    {"int32",    MakePrimitiveInfo(CoreType::CT_int32, "int32")},
    {"uint64",   MakePrimitiveInfo(CoreType::CT_uint64, "uint64")},
    {"int64",    MakePrimitiveInfo(CoreType::CT_int64, "int64")},
    {"float",    MakePrimitiveInfo(CoreType::CT_float, "float")},
    {"double",   MakePrimitiveInfo(CoreType::CT_double, "double")},
    {"enum",     nullptr}, // Not a primitive type
    {"array",    nullptr}, // Not a primitive type
    {"matrix",   nullptr}, // Not a primitive type
    {"struct",   nullptr}, // Not a primitive type
    {"field",    nullptr}, // Not a primitive type
    {"map",      nullptr}, // Not a primitive type
    {"optional", nullptr}, // Not a primitive type
    {"variant",  nullptr}, // Not a primitive type
    {"other",    nullptr}, // Not a primitive type
};
SR_WARNING_POP()
// clang-format on

static_assert(
    std::size(kCoreTypeInfo) == (size_t)CoreType::X_Count,
    "Please update this list if the SReflect::CoreType enum changes.");

char const* CoreTypeToString(CoreType coreType) {
  SR_ASSERT((size_t)coreType < std::size(kCoreTypeInfo), "Invalid CoreType for CoreTypeToString");
  return kCoreTypeInfo[(size_t)coreType].name;
}

uint64_t CalcHash64(void const* src, size_t numBytes) {
  SR_ASSERT(numBytes > 0, "Requires at least one byte");
  return CityHash64(static_cast<char const*>(src), numBytes);
}

/////////////////////////////////////////////////////////////////////////////
// EnumTypeInfo
/////////////////////////////////////////////////////////////////////////////

Attribute const* EnumItem::GetAttribute(TypeId attrType) const {
  return FindAttribute(attrType, _attributes);
}

EnumItem const* EnumTypeInfo::FindItemByValue(uint64_t value) const {
  auto it = std::find_if(
      _items.begin(), _items.end(), [value](EnumItem const& item) { return item._value == value; });
  if (it == _items.end()) {
    // Value not found in enum items
    return nullptr;
  }
  return &*it;
}

EnumItem const* EnumTypeInfo::FindItemByName(std::string_view name) const {
  auto it = std::find_if(
      _items.begin(), _items.end(), [name](EnumItem const& item) { return item._name == name; });
  if (it == _items.end()) {
    // Case-insensitive lookup is allowed for historical reasons
    it = std::find_if(_items.begin(), _items.end(), [name](EnumItem const& item) {
      return StrEqualCaseInsensitive(item._name, name);
    });
  }
  if (it == _items.end()) {
    // Check previous names via SRA_PreviouslyKnownAs attributes (exact match only)
    for (auto const& item : _items) {
      if (auto const* pna = item.GetAttribute<Attribute_PreviouslyKnownAs>()) {
        for (std::string const& prevName : pna->_previousNames) {
          if (prevName == name) {
            return &item;
          }
        }
      }
    }
    // Value not found in enum items or previous names
    return nullptr;
  }
  return &*it;
}

void EnumTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  picojson::array items;
  items.reserve(_items.size());
  for (EnumItem const& item : _items) {
    items.push_back(WriteEnumItem(item, dict));
  }
  SetField(obj, "items", picojson::value{std::move(items)});
  obj["innerType"] = AddTypeRef(_innerTypeInfo, dict);
}

/////////////////////////////////////////////////////////////////////////////
// EnumTypeInfoImpl
/////////////////////////////////////////////////////////////////////////////

class EnumTypeInfoImpl final : public EnumTypeInfo {
 public:
  explicit EnumTypeInfoImpl(std::type_index typeIndex) : _typeIndex(typeIndex) {}

 private:
  bool IsValidInner(void const* src, AttributeList const& attribs) const override {
#if 0
    // Check if value matches one of the enum values
    // TODO: add attribute to enable this, because sometimes it's ok to not match a single enum value
    uint64_t srcAsUInt64 = _innerTypeInfo->ToUInt64(src);
    if (FindItemByValue(srcAsUInt64) == nullptr) {
      return false;
    }
#endif
    return _innerTypeInfo->IsValidInner(src, attribs);
  }

  void SetInner(void const* src, void* dst) const override {
    _innerTypeInfo->SetInner(src, dst);
  }

  void SerializeInner(void const* src, picojson::value& dst) const override {
    // If possible, we want to emit a string instead of a number, which increases
    // robustness to changes in enum items in save files
    // See if this value is one of the enum values
    uint64_t srcAsUInt64 = _innerTypeInfo->ToUInt64(src);
    EnumItem const* item = FindItemByValue(srcAsUInt64);
    if (item != nullptr) {
      dst = item->_name;
    } else {
      // No item found - just serialize the number
      _innerTypeInfo->SerializeInner(src, dst);
    }
  }

  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override {
    // Get string, then check if this is an enum item name, or a numeric value
    std::string dstAsString;
    detail::GetJsonValue(src, dstAsString);
    if (!dstAsString.empty()) {
      if (isdigit(dstAsString[0])) {
        // Value is a number; read it directly as a number into dst
        _innerTypeInfo->DeserializeInner(src, dst, deserializeFlags, inOutIssuesDetected);
      } else {
        // Value is a string; find the enum item that matches
        EnumItem const* item = FindItemByName(dstAsString);
        if (item != nullptr) {
          // Found an enum item - set dst from its value
          _innerTypeInfo->FromUInt64(item->_value, dst);
        } else {
          // Json has a string that doesn't match any existing enum item; leave at default
          SR_LOG("Enum value '%s' not found, using default", dstAsString.c_str());
          ++inOutIssuesDetected;
        }
      }
    }
  }

  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    // Always serialize as number. Ignore string lookup.
    return _innerTypeInfo->SerializeToBytesInner(src, dst);
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    return _innerTypeInfo->DeserializeFromBytesInner(src, dst);
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return _innerTypeInfo->IsMemCopySafe();
  }

  uint64_t GetValue(void const* obj) const override {
    return _innerTypeInfo->ToUInt64(obj);
  }

  void SetValue(void* obj, uint64_t value) const override {
    return _innerTypeInfo->FromUInt64(value, obj);
  }

  [[nodiscard]] std::type_index GetTypeIndex() const override {
    return _typeIndex;
  }

  std::type_index _typeIndex;
};

/////////////////////////////////////////////////////////////////////////////
// PruneDefaultFieldValues (utility)
/////////////////////////////////////////////////////////////////////////////

static void PruneDefaultFieldValues(
    TypeInfo const* info,
    picojson::value const& defaultValue,
    picojson::value& actualValue) {
  auto const* structInfo = dynamic_cast<StructTypeInfo const*>(info);
  if (structInfo && defaultValue.is<picojson::object>() && actualValue.is<picojson::object>()) {
    // jsonValue should be a dictionary where the key names match the field names
    auto const& defaultObj = defaultValue.get<picojson::object>();
    auto& actualObj = actualValue.get<picojson::object>();
    for (auto const* field : structInfo->_fields) {
      std::string fieldName = field->_name;
      auto actualIt = actualObj.find(fieldName);
      auto defaultIt = defaultObj.find(fieldName);
      if (actualIt != actualObj.end() && defaultIt != defaultObj.end()) {
        // Compare the actual value to the default value by serializing both
        auto& actualField = actualIt->second;
        auto const& defaultField = defaultIt->second;
        if (actualField.serialize(false) == defaultField.serialize(false)) {
          // Default value detected. Prune it.
          actualObj.erase(actualIt);
          continue;
        }

        // Some fields may contains nested structs with their own default values.
        // Step into the nested field if we can.
        PruneDefaultFieldValues(field->_innerTypeInfo, defaultField, actualField);
      } else if (
          actualIt != actualObj.end() && defaultIt == defaultObj.end() &&
          field->_innerTypeInfo->_coreType == CoreType::CT_optional) {
        // This field has a type like std::optional<T>. It has a value but the default is to have no
        // value (std::nullopt). In this case, we attempt to prune defaults according to the
        // defaults of the inner type T.
        PruneDefaultFieldValues(field->_innerTypeInfo, picojson::value{}, actualIt->second);
      }
    }
    return;
  }

  auto const* arrayInfo = dynamic_cast<ArrayTypeInfo const*>(info);
  if (arrayInfo && defaultValue.is<picojson::array>() && actualValue.is<picojson::array>()) {
    picojson::array const& defaultArray = defaultValue.get<picojson::array>();
    picojson::array& actualArray = actualValue.get<picojson::array>();
    if (arrayInfo->CanResize()) {
      // If the array is resizable, then we treat it as a collection of the inner type. In this
      // case, we prune fields that are equal to the default value, as defined by the inner type.
      void* tempInstance = arrayInfo->_innerTypeInfo->New();
      SR_DEFER(arrayInfo->_innerTypeInfo->Delete(tempInstance));
      picojson::value defaultInnerValue = picojson::object();
      arrayInfo->_innerTypeInfo->Serialize(tempInstance, defaultInnerValue);
      for (auto& actualElement : actualArray) {
        PruneDefaultFieldValues(arrayInfo->_innerTypeInfo, defaultInnerValue, actualElement);
      }
    } else if (defaultArray.size() == actualArray.size()) {
      // If the array is NOT resizable, then we treat it as N nested objects. In this case, we prune
      // fields that are equal to the default value, as defined by the outer type.
      for (size_t i = 0; i < actualArray.size(); ++i) {
        PruneDefaultFieldValues(arrayInfo->_innerTypeInfo, defaultArray[i], actualArray[i]);
      }
    }
    return;
  }

  auto const* optionalInfo = dynamic_cast<OptionalTypeInfo const*>(info);
  if (optionalInfo && !actualValue.is<picojson::null>()) {
    // This optional must have a value because it serialized something.
    // Attempt to prune the inner type using a default-constructed object of that type.
    auto const* innerType = optionalInfo->_innerTypeInfo;
    void* tempInnerObj = innerType->New();
    SR_DEFER(innerType->Delete(tempInnerObj));
    picojson::value defaultInnerValue = picojson::object();
    innerType->Serialize(tempInnerObj, defaultInnerValue);
    PruneDefaultFieldValues(innerType, defaultInnerValue, actualValue);
  }

  auto const* mapInfo = dynamic_cast<MapTypeInfo const*>(info);
  if (mapInfo && actualValue.is<picojson::object>()) {
    // The value type may be something that we can prune. DO NOT prune the key type.
    picojson::object& actualObj = actualValue.get<picojson::object>();
    auto const* innerValueType = mapInfo->_valueTypeInfo;
    void* tempInnerValue = innerValueType->New();
    SR_DEFER(innerValueType->Delete(tempInnerValue));
    picojson::value defaultInnerValue = picojson::object();
    innerValueType->Serialize(tempInnerValue, defaultInnerValue);
    for (auto&& [key, actualInnerValue] : actualObj) {
      PruneDefaultFieldValues(innerValueType, defaultInnerValue, actualInnerValue);
    }
  }

  // If the type did not fall into one of the known categories (above), then leave it as-is.
}

/////////////////////////////////////////////////////////////////////////////
// FieldTypeInfo
/////////////////////////////////////////////////////////////////////////////

uint8_t* FieldTypeInfo::GetFieldPtr(void* structBasePtr) const {
  return reinterpret_cast<uint8_t*>(structBasePtr) + _offset;
}

uint8_t const* FieldTypeInfo::GetFieldPtr(void const* structBasePtr) const {
  return reinterpret_cast<uint8_t const*>(structBasePtr) + _offset;
}

/////////////////////////////////////////////////////////////////////////////
// FieldTypeInfoImpl
/////////////////////////////////////////////////////////////////////////////

class FieldTypeInfoImpl final : public FieldTypeInfo {
 protected:
  bool IsValidInner(void const* src, AttributeList const& /*attribs*/) const override {
    return _innerTypeInfo->IsValidInner(src, _attributes);
  }

  void SetInner(void const* src, void* dst) const override {
    _innerTypeInfo->Set(src, dst);
  }

  void SerializeInner(void const* src, picojson::value& dst) const override {
    // cannot serialize a field so we always call serialize on the inner type
    // src is a field pointer
    // dst is a picojson object
    if (HasAttribute<Attribute_JsonString>() && _innerTypeInfo->_coreType == CoreType::CT_string) {
      // This string field carries a serialized JSON value; emit it as that value (e.g. a nested
      // object) rather than as a quoted string, so a deserialize/serialize round trip is identity.
      // See Attribute_JsonString.
      auto const* strInfo = static_cast<StringTypeInfo const*>(_innerTypeInfo);
      std::string const text{strInfo->GetString(src)};
      picojson::value parsed;
      std::istringstream textStream(text);
      std::string const parseErr = picojson::parse(parsed, textStream);
      if (parseErr.empty() && (parsed.is<picojson::object>() || parsed.is<picojson::array>())) {
        dst[_name] = parsed;
        return;
      }
      // Not a JSON object/array (e.g. an empty or non-JSON string): fall through to normal
      // emission.
    }
    picojson::value fieldValue = picojson::object();
    _innerTypeInfo->SerializeInner(src, fieldValue);
    if (fieldValue.is<picojson::null>() && (_innerTypeInfo->_coreType == CoreType::CT_optional)) {
      // This is an optional field for which no value was specified. In this case we do not write
      // anything to JSON (neither the field name, nor the null value).
    } else {
      // Add a key-value pair to the JSON dictionary.
      dst[_name] = fieldValue;
    }
  }

  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override {
    // src is a picojson value
    // dst is a field pointer
    try {
      if (HasAttribute<Attribute_JsonString>() &&
          (src.is<picojson::object>() || src.is<picojson::array>())) {
        // This string field carries a serialized JSON value: store the JSON object/array as its
        // serialized text, routed through the field's normal string assignment so a non-string
        // target fails and is reported. See Attribute_JsonString.
        picojson::value const asString{src.serialize(false)};
        _innerTypeInfo->DeserializeInner(asString, dst, deserializeFlags, inOutIssuesDetected);
      } else {
        _innerTypeInfo->DeserializeInner(src, dst, deserializeFlags, inOutIssuesDetected);
      }
    } catch (std::runtime_error const& e) {
      SR_LOG(
          "Failure while getting json field '%s': '%s'. Is the field type correct? Is there an extra comma in the json?",
          _name,
          e.what());
      ++inOutIssuesDetected;
    }
  }

  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    return _innerTypeInfo->SerializeToBytesInner(src, dst);
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    return _innerTypeInfo->DeserializeFromBytesInner(src, dst);
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return _innerTypeInfo->IsMemCopySafe();
  }

  friend class StructTypeInfo;
};

/////////////////////////////////////////////////////////////////////////////
// StructTypeInfo
/////////////////////////////////////////////////////////////////////////////

StructTypeInfo::StructTypeInfo(std::type_info const& typeInfo) : _typeIndex(typeInfo) {}

FieldTypeInfo const* StructTypeInfo::FindField(std::string_view name) const {
  for (auto* f : _fields) {
    if (f->_name == name) {
      return f;
    }
  }
  return nullptr; // Not found
}

StructTypeInfo const* StructTypeInfo::FindBaseClass(SReflect::TypeId id) const {
  for (const auto* b : _baseClasses) {
    if (b->_typeId == id) {
      return b;
    }
  }
  return nullptr;
}

bool StructTypeInfo::IsSameOrDerivedFrom(TypeId typeId) const {
  return (_typeId == typeId) || (FindBaseClass(typeId) != nullptr);
}

bool StructTypeInfo::IsSameOrDerivedFrom(std::type_index typeIndex) const {
  if (typeIndex == _typeIndex) {
    return true;
  }
  for (StructTypeInfo const* base : _baseClasses) {
    if (base->_typeIndex == typeIndex) {
      return true;
    }
  }
  return false;
}

void StructTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  SetField(obj, "baseClasses", picojson::value{AddTypeRefArray(_baseClasses, dict)});
  picojson::array fields;
  fields.reserve(_fields.size());
  for (FieldTypeInfo const* field : _fields) {
    if (field == nullptr) {
      continue;
    }
    picojson::value fieldEntry{picojson::object{}};
    WriteFieldDescriptor(*field, fieldEntry, dict);
    fields.push_back(std::move(fieldEntry));
  }
  SetField(obj, "fields", picojson::value{std::move(fields)});
  obj["isMemCopySafe"] = IsMemCopySafe();
}

/////////////////////////////////////////////////////////////////////////////
// StructTypeInfoImpl
/////////////////////////////////////////////////////////////////////////////

class StructTypeInfoImpl final : public StructTypeInfo {
 public:
  using StructTypeInfo::StructTypeInfo;

 protected:
  bool IsValidInner(void const* src, AttributeList const& attribs) const override {
    for (FieldTypeInfo const* field : _fields) {
      uint8_t const* srcFieldPtr = field->GetFieldPtr(src);
      if (!field->IsValidInner(srcFieldPtr, attribs)) {
        return false;
      }
    }
    return true;
  }
  void SetInner(void const* src, void* dst) const override {
    for (FieldTypeInfo const* field : _fields) {
      uint8_t const* srcFieldPtr = field->GetFieldPtr(src);
      uint8_t* dstFieldPtr = field->GetFieldPtr(dst);
      field->Set(srcFieldPtr, dstFieldPtr);
    }
  }

  void SerializeInner(void const* src, picojson::value& dst) const override {
    dst = picojson::object();
    bool fieldHasAttrDoNotSerializeDefaults = false;
    for (FieldTypeInfo const* field : _fields) {
      uint8_t const* srcFieldPtr = field->GetFieldPtr(src);
      field->Serialize(srcFieldPtr, dst);
      fieldHasAttrDoNotSerializeDefaults = fieldHasAttrDoNotSerializeDefaults ||
          field->HasAttribute<Attribute_DoNotSerializeDefaults>();
    }

    // Optionally omit fields equal to their default values (recursively)
    bool const hasAttrDoNotSerializeDefaults = HasAttribute<Attribute_DoNotSerializeDefaults>();
    if (hasAttrDoNotSerializeDefaults || fieldHasAttrDoNotSerializeDefaults) {
      // Construct an object with default values
      void* tempInstance = this->New();
      SR_DEFER(this->Delete(tempInstance));
      auto& actualObj = dst.get<picojson::object>();
      for (FieldTypeInfo const* field : _fields) {
        auto const* fieldAttr = field->GetAttribute<Attribute_DoNotSerializeDefaults>();
        if (hasAttrDoNotSerializeDefaults || fieldAttr != nullptr) {
          auto actualFieldIt = actualObj.find(field->_name);
          if (actualFieldIt != actualObj.end()) {
            // Compare this field to its default value via serialization.
            picojson::value& actualFieldValue = actualFieldIt->second;
            picojson::value defaultFieldValue = picojson::object();
            field->_innerTypeInfo->Serialize(field->GetFieldPtr(tempInstance), defaultFieldValue);
            if (actualFieldValue.serialize(false) == defaultFieldValue.serialize(false)) {
              // The field exactly matches the default value. Prune it.
              actualObj.erase(actualFieldIt);
            } else {
              // The field differs from its default. A class-level attribute, or a field-level
              // attribute with _recursive == true, prunes nested sub-fields equal to their
              // defaults. A field-level attribute with _recursive == false leaves the field as-is.
              bool const recursive =
                  hasAttrDoNotSerializeDefaults || fieldAttr == nullptr || fieldAttr->_recursive;
              if (recursive) {
                PruneDefaultFieldValues(field->_innerTypeInfo, defaultFieldValue, actualFieldValue);
              }
            }
          }
        }
      }
    }
  }

  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override {
    auto flags = static_cast<uint32_t>(deserializeFlags);
    // Loop over fields, find them, and if exist, deserialize from them
    picojson::object const& srcObj = src.get<picojson::object>();
    int numSrcFieldsMatched = 0;
    for (FieldTypeInfo const* field : _fields) {
      if (field->HasAttribute<Attribute_DoNotSerialize>()) {
        // Do not attempt to deserialize this field
        continue;
      }
      auto srcFieldIt = srcObj.find(field->_name);
      if (srcFieldIt == srcObj.end()) {
        // Did not find field by its current name. Check if any of its previous names exist in the
        // json data
        auto const* pna = field->GetAttribute<Attribute_PreviouslyKnownAs>();
        if (pna != nullptr) {
          for (std::string const& previousName : pna->_previousNames) {
            srcFieldIt = srcObj.find(previousName);
            if (srcFieldIt != srcObj.end()) {
              break;
            }
          }
        }
      }
      if (srcFieldIt != srcObj.end()) {
        ++numSrcFieldsMatched;
        uint8_t* dstFieldPtr = field->GetFieldPtr(dst);
        int fieldIssues{0};
        field->Deserialize(srcFieldIt->second, dstFieldPtr, deserializeFlags, fieldIssues);
        inOutIssuesDetected += fieldIssues;
      } else {
        if (field->_innerTypeInfo->_coreType == CoreType::CT_optional) {
          // If the inner type is something like std::optional, then the missing field is not
          // considered to be a problem.
        } else if (flags & static_cast<uint32_t>(DeserializeFlags::WarnIfMissingFields)) {
          SR_LOG(
              "Missing json field '%s' in %s object",
              field->_name,
              _nameWithNamespace ? _nameWithNamespace : "(unknown)");
          ++inOutIssuesDetected;
        }
      }
    }
    if ((flags & static_cast<uint32_t>(DeserializeFlags::WarnIfExtraneousFields)) &&
        !HasAttribute<Attribute_IgnoreExtraneousFields>()) {
      if ((int)srcObj.size() > numSrcFieldsMatched) {
        // There are unmatched src fields, and user wants warnings for them.
        // Iterate over src fields so we can print out exactly which ones are extra.
        for (auto const& srcField : srcObj) {
          FieldTypeInfo const* fieldInfo = FindField(srcField.first);
          if (fieldInfo != nullptr) {
            if (fieldInfo->HasAttribute<Attribute_DoNotSerialize>()) {
              SR_LOG(
                  "JSON data for object [%s] contains field [%s], which is marked as 'do not serialize'. Please correct the JSON file or resave it. When resaved, any unsupported fields will be lost.",
                  _nameWithNamespace ? _nameWithNamespace : "(unknown)",
                  srcField.first.c_str());
              ++inOutIssuesDetected;
            }
          } else {
            SR_LOG(
                "JSON data for object [%s] contains unknown field name [%s]. It may be a typo or it may be a field that was removed from the code. Please correct the JSON file or resave it. When resaved, any unsupported fields will be lost.",
                _nameWithNamespace ? _nameWithNamespace : "(unknown)",
                srcField.first.c_str());
            ++inOutIssuesDetected;
          }
        }
      }
    }
  }

  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    if (IsMemCopySafe()) {
      return dst.Write(src, _sizeInBytes);
    } else {
      for (FieldTypeInfo const* field : _fields) {
        uint8_t const* srcFieldPtr = field->GetFieldPtr(src);
        if (!field->SerializeToBytes(srcFieldPtr, dst)) {
          return false;
        }
      }
      return true;
    }
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    if (IsMemCopySafe()) {
      return src.Read(dst, _sizeInBytes);
    } else {
      for (FieldTypeInfo const* field : _fields) {
        uint8_t* dstFieldPtr = field->GetFieldPtr(dst);
        if (!field->DeserializeFromBytes(src, dstFieldPtr)) {
          return false;
        }
      }
      return true;
    }
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return _isMemCopySafe;
  }
};

/////////////////////////////////////////////////////////////////////////////
// ArrayTypeInfo
/////////////////////////////////////////////////////////////////////////////

bool ArrayTypeInfo::IsValidInner(void const* src, AttributeList const& attribs) const {
  const size_t numElements = GetNumElements(src);
  for (size_t i = 0; i < numElements; ++i) {
    if (!_innerTypeInfo->IsValidInner(GetElement(src, i), attribs)) {
      return false;
    }
  }
  return true;
}

void ArrayTypeInfo::SetInner(void const* src, void* dst) const {
  const size_t numElements = GetNumElements(src);
  [[maybe_unused]] const bool success = SetNumElements(dst, numElements);
  SR_ASSERT(
      success,
      "SetNumElements should always succeed in this context. "
      "They are either both dynamic vectors or both equal sized arrays.");
  for (size_t i = 0; i < numElements; ++i) {
    _innerTypeInfo->Set(GetElement(src, i), GetElement(dst, i));
  }
}

void ArrayTypeInfo::SerializeInner(void const* src, picojson::value& dst) const {
  dst = picojson::array();
  picojson::array& dstArray = dst.get<picojson::array>();
  const size_t numElements = GetNumElements(src);
  dstArray.resize(numElements);
  // Add elements
  for (size_t i = 0; i < numElements; ++i) {
    _innerTypeInfo->SerializeInner(GetElement(src, i), dstArray[i]);
  }
}

void ArrayTypeInfo::DeserializeInner(
    picojson::value const& src,
    void* dst,
    DeserializeFlags deserializeFlags,
    int& inOutIssuesDetected) const {
  picojson::array const& srcArray = src.get<picojson::array>();
  const size_t numElements = srcArray.size();
  const bool success = SetNumElements(dst, numElements);
  if (success) {
    for (size_t i = 0; i < numElements; ++i) {
      _innerTypeInfo->DeserializeInner(
          srcArray[i], GetElement(dst, i), deserializeFlags, inOutIssuesDetected);
    }
  } else {
    SR_LOG("Array is the wrong size (got:%zu, expected:%zu)", srcArray.size(), GetNumElements(dst));
    ++inOutIssuesDetected;
  }
}

bool ArrayTypeInfo::SerializeToBytesInner(void const* src, StreamWriter& dst) const {
  // Serialize count
  const size_t numElements = GetNumElements(src);
  if (!serde::bytes::Serialize(numElements, dst)) {
    return false;
  }

  // Serialize each element
  for (size_t i = 0; i < numElements; ++i) {
    const void* element = GetElement(src, i);
    if (!_innerTypeInfo->SerializeToBytesInner(element, dst)) {
      return false;
    }
  }

  return true;
}

bool ArrayTypeInfo::DeserializeFromBytesInner(StreamReader& src, void* dst) const {
  // Deserialize count
  size_t numElements = 0;
  if (!serde::bytes::Deserialize(src, numElements)) {
    return false;
  }

  // Deserialize each elements
  SetNumElements(dst, numElements);
  for (size_t i = 0; i < numElements; ++i) {
    if (!_innerTypeInfo->DeserializeFromBytesInner(src, GetElement(dst, i))) {
      return false;
    }
  }

  return true;
}

void ArrayTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  bool const canResize = CanResize();
  obj["innerType"] = AddTypeRef(_innerTypeInfo, dict);
  obj["canResize"] = canResize;
  if (!canResize) {
    // A non-resizable array has a fixed element count; GetNumElements ignores its argument.
    obj["numElements"] = static_cast<double>(GetNumElements(nullptr));
  }
}

/////////////////////////////////////////////////////////////////////////////
// FixedArrayTypeInfo (supports non-resizable arrays like T[N] and std::array<T,N>)
/////////////////////////////////////////////////////////////////////////////

class FixedArrayTypeInfo final : public ArrayTypeInfo {
 public:
  size_t _numElements; // number of elements
  bool _isMemCopySafe = false;

  [[nodiscard]] bool CanResize() const override {
    return false;
  }

  size_t GetNumElements(void const* /*obj*/) const override {
    return _numElements;
  }

  bool SetNumElements(void* /*obj*/, size_t numElements) const override {
    return (numElements == _numElements); // Cannot change
  }

  void* GetElement(void* obj, size_t i) const override {
    return reinterpret_cast<uint8_t*>(obj) + (i * _innerTypeInfo->_sizeInBytes);
  }

  void const* GetElement(void const* obj, size_t i) const override {
    return reinterpret_cast<uint8_t const*>(obj) + (i * _innerTypeInfo->_sizeInBytes);
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return _isMemCopySafe;
  }
};

/////////////////////////////////////////////////////////////////////////////
// VectorTypeInfo<bool>
/////////////////////////////////////////////////////////////////////////////
size_t VectorTypeInfo<std::vector<bool>>::GetNumElements(void const* obj) const {
  const auto& vec = *(reinterpret_cast<std::vector<bool> const*>(obj));
  return vec.size();
}

bool VectorTypeInfo<std::vector<bool>>::SetNumElements(void* obj, size_t numElements) const {
  auto& vec = *(reinterpret_cast<std::vector<bool>*>(obj));
  vec.resize(numElements);
  return true;
}

bool VectorTypeInfo<std::vector<bool>>::IsMemCopySafe() const {
  return false;
}

void* VectorTypeInfo<std::vector<bool>>::GetElement(void* /*obj*/, size_t /*i*/) const {
  throw std::runtime_error("GetElement not supported for type std::vector<bool>");
}

void const* VectorTypeInfo<std::vector<bool>>::GetElement(void const* /*obj*/, size_t /*i*/) const {
  throw std::runtime_error("GetElement not supported for type std::vector<bool>");
}

bool VectorTypeInfo<std::vector<bool>>::IsValidInner(void const* obj, AttributeList const& attribs)
    const {
  const auto& vec = *(reinterpret_cast<std::vector<bool> const*>(obj));
  for (bool element : vec) {
    if (!_innerTypeInfo->IsValidInner(&element, attribs)) {
      return false;
    }
  }
  return true;
}

void VectorTypeInfo<std::vector<bool>>::SetInner(void const* src, void* dst) const {
  const auto& srcVec = *(reinterpret_cast<std::vector<bool> const*>(src));
  auto& dstVec = *(reinterpret_cast<std::vector<bool>*>(dst));
  dstVec = srcVec;
}

void VectorTypeInfo<std::vector<bool>>::SerializeInner(void const* src, picojson::value& dst)
    const {
  const auto& srcVec = *(reinterpret_cast<std::vector<bool> const*>(src));
  const size_t numElements = srcVec.size();

  // Create json array
  dst = picojson::array();
  picojson::array& dstArray = dst.get<picojson::array>();
  dstArray.resize(numElements);

  for (size_t i = 0; i < numElements; ++i) {
    dstArray[i] = static_cast<bool>(srcVec[i]);
  }
}

void VectorTypeInfo<std::vector<bool>>::DeserializeInner(
    picojson::value const& src,
    void* dst,
    DeserializeFlags /*deserializeFlags*/,
    int& /*inOutIssuesDetected*/) const {
  picojson::array const& srcArray = src.get<picojson::array>();
  auto& dstVec = *(reinterpret_cast<std::vector<bool>*>(dst));
  dstVec.reserve(srcArray.size());
  std::transform(
      srcArray.begin(), srcArray.end(), std::back_inserter(dstVec), [](auto& elem) -> auto {
        bool value = false;
        detail::GetJsonValue(elem, value);
        return value;
      });
}

bool VectorTypeInfo<std::vector<bool>>::SerializeToBytesInner(void const* src, StreamWriter& dst)
    const {
  auto const& srcVec = *(static_cast<std::vector<bool> const*>(src));

  // Serialize bool count
  const size_t numBools = srcVec.size();
  if (!serde::bytes::Serialize(numBools, dst)) {
    return false;
  }

  // Write one-byte per bool
  for (size_t i = 0; i < numBools; ++i) {
    bool value = srcVec[i];
    if (!serde::bytes::Serialize(value, dst)) {
      return false;
    }
  }

  return true;
}

bool VectorTypeInfo<std::vector<bool>>::DeserializeFromBytesInner(StreamReader& src, void* dst)
    const {
  auto& dstVec = *(reinterpret_cast<std::vector<bool>*>(dst));

  // Deserialize bool count
  size_t numBools = 0;
  if (!serde::bytes::Deserialize(src, numBools)) {
    return false;
  }

  dstVec.resize(numBools);
  for (size_t i = 0; i < numBools; ++i) {
    bool value = false;
    if (!serde::bytes::Deserialize(src, value)) {
      return false;
    }
    dstVec[i] = value;
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////
// MatrixTypeInfo
/////////////////////////////////////////////////////////////////////////////

MatrixTypeInfo::Layout MatrixTypeInfo::GetLayout(void const* obj) const {
  auto layout = GetLayoutImpl(obj); // Call derived class
  if (layout._leadingDim == 0) {
    // Automatic leading dimension
    layout._leadingDim = _isRowMajor ? layout._numColumns : layout._numRows;
  } else {
    SR_ASSERT(
        (_isRowMajor && layout._leadingDim >= layout._numColumns) ||
            (!_isRowMajor && layout._leadingDim >= layout._numRows),
        "Invalid leading dimension. Must be >= the number of rows (for column-major) or columns (for row-major).");
  }
  return layout;
}

void const* MatrixTypeInfo::GetData(void const* obj) const {
  // Share the non-const implementation. It is still const-correct from the caller's perspective.
  return GetData(const_cast<void*>(obj));
}

// TryResize with some error reporting
static bool TryResizeMatrix(
    MatrixTypeInfo const& ti,
    void* dst,
    MatrixTypeInfo::Layout& dstLayout,
    size_t numRows,
    size_t numColumns) {
  if ((dstLayout._numRows == numRows) && (dstLayout._numColumns == numColumns)) {
    return true; // already the right size
  }
  bool ok = ti.TryResize(dst, numRows, numColumns);
  if (ok) {
    // Fetch the updated layout information
    dstLayout = ti.GetLayout(dst);
    SR_ASSERT(
        dstLayout._numRows == numRows && dstLayout._numColumns == numColumns,
        "TryResize reported success incorrectly. The dimensions were not resized.");
  } else {
    SR_LOG(
        "Incompatible matrix dimensions. Expected (%zu x %zu) but got (%zu x %zu).",
        dstLayout._numRows,
        dstLayout._numColumns,
        numRows,
        numColumns);
  }
  return ok;
}

void MatrixTypeInfo::SetInner(void const* src, void* dst) const {
  SR_ASSERT(
      _innerTypeInfo->IsMemCopySafe(),
      "Expected an inner type that is memcpy safe (e.g. a scalar).");

  auto srcLayout = GetLayout(src);
  auto dstLayout = GetLayout(dst);

  // Resize the destination if necessary
  bool ok = TryResizeMatrix(*this, dst, dstLayout, srcLayout._numRows, srcLayout._numColumns);
  if (!ok) {
    SR_ASSERT(ok, "Incompatible matrix dimensions");
    return;
  }

  if (!(srcLayout._numRows && srcLayout._numColumns)) {
    return; // no values
  }

  // Get the data
  auto const* srcData = static_cast<std::byte const*>(GetData(src));
  auto* dstData = static_cast<std::byte*>(GetData(dst));
  SR_ASSERT(srcData != nullptr, "Nullptr detected on a matrix of non-zero size.");
  SR_ASSERT(dstData != nullptr, "Nullptr detected on a matrix of non-zero size.");

  // Copy values
  size_t dim0 = _isRowMajor ? dstLayout._numRows : dstLayout._numColumns;
  size_t dim1 = _isRowMajor ? dstLayout._numColumns : dstLayout._numRows;
  size_t elemSize = _innerTypeInfo->_sizeInBytes;
  if ((dim1 == srcLayout._leadingDim) && (dim1 == dstLayout._leadingDim)) {
    // Copy all values at once
    std::memcpy(dstData, srcData, dim0 * dim1 * elemSize);
  } else {
    for (size_t i = 0; i < dim0; ++i) {
      auto const* srcVec = srcData + (i * srcLayout._leadingDim * elemSize);
      auto* dstVec = dstData + (i * dstLayout._leadingDim * elemSize);
      // Copy the whole row or column
      std::memcpy(dstVec, srcVec, dim1 * elemSize);
    }
  }
}

void* MatrixTypeInfo::GetElement(void* obj, size_t row, size_t column) const {
  auto layout = GetLayout(obj);
  SR_ASSERT((row < layout._numRows) && (column < layout._numColumns), "Coordinates out-of-bounds");
  auto* data = static_cast<std::byte*>(GetData(obj));
  SR_ASSERT(data != nullptr, "Nullptr detected on a matrix of non-zero size.");
  size_t i = _isRowMajor ? row : column;
  size_t j = _isRowMajor ? column : row;
  size_t elemSize = _innerTypeInfo->_sizeInBytes;
  return data + (i * layout._leadingDim * elemSize) + (j * elemSize);
}

void const* MatrixTypeInfo::GetElement(void const* obj, size_t row, size_t column) const {
  // Share the non-const implementation. It is still const-correct from the caller's perspective.
  return GetElement(const_cast<void*>(obj), row, column);
}

void MatrixTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  obj["innerType"] = AddTypeRef(_innerTypeInfo, dict);
  obj["isRowMajor"] = _isRowMajor;
  obj["isNumRowsDynamic"] = _isNumRowsDynamic;
  obj["isNumColumnsDynamic"] = _isNumColumnsDynamic;
  // Emit dimensions if they are fixed.
  if (_fixedNumRows) {
    obj["numRows"] = static_cast<double>(_fixedNumRows);
  }
  if (_fixedNumColumns) {
    obj["numColumns"] = static_cast<double>(_fixedNumColumns);
  }
}

bool MatrixTypeInfo::IsValidInner(void const* obj, AttributeList const& attribs) const {
  // Could be faster if we handled row-major and column-major differently.
  auto layout = GetLayout(obj);
  for (size_t r = 0; r < layout._numRows; ++r) {
    for (size_t c = 0; c < layout._numColumns; ++c) {
      if (!_innerTypeInfo->IsValidInner(GetElement(obj, r, c), attribs)) {
        return false;
      }
    }
  }
  return true;
}

void MatrixTypeInfo::SerializeInner(void const* src, picojson::value& dst) const {
  // A matrix serializes to JSON as an array of arrays. The outer dimension is the rows and the
  // inner dimension is the columns. Thus matrices are always serialized to JSON in row-major
  // order (how people would write it on paper). Exception: 1D vectors that can't be otherwise, are
  // serialized as a 1D array.
  auto layout = GetLayout(src);
  dst = picojson::array();
  auto& dstOuter = dst.get<picojson::array>();
  if (layout._numRows == 1 && !_isNumRowsDynamic) { // fixed row vector
    dstOuter.resize(layout._numColumns, picojson::array());
    for (size_t i = 0; i < layout._numColumns; ++i) {
      _innerTypeInfo->SerializeInner(GetElement(src, 0, i), dstOuter[i]);
    }
  } else if (layout._numColumns == 1 && !_isNumColumnsDynamic) { // fixed column vector
    dstOuter.resize(layout._numRows, picojson::array());
    for (size_t i = 0; i < layout._numRows; ++i) {
      _innerTypeInfo->SerializeInner(GetElement(src, i, 0), dstOuter[i]);
    }
  } else {
    dstOuter.resize(layout._numRows, picojson::array());
    for (size_t r = 0; r < layout._numRows; ++r) {
      auto& dstInner = dstOuter[r].get<picojson::array>();
      dstInner.resize(layout._numColumns, picojson::value());
      for (size_t c = 0; c < layout._numColumns; ++c) {
        _innerTypeInfo->SerializeInner(GetElement(src, r, c), dstInner[c]);
      }
    }
  }
}

void MatrixTypeInfo::DeserializeInner(
    picojson::value const& src,
    void* dst,
    DeserializeFlags deserializeFlags,
    int& inOutIssuesDetected) const {
  if (!src.is<picojson::array>()) {
    SR_LOG("Failed to deserialize matrix. Expected an array of arrays in JSON.");
    ++inOutIssuesDetected;
    return;
  }

  auto const& srcOuter = src.get<picojson::array>();
  auto dstLayout = GetLayout(dst);
  bool isRowVector = (dstLayout._numRows == 1 && !_isNumRowsDynamic);
  bool isColVector = (dstLayout._numColumns == 1 && !_isNumColumnsDynamic);
  bool srcIs1D = srcOuter.empty() || !srcOuter[0].is<picojson::array>();
  size_t numRows = 0;
  size_t numColumns = 0;
  if (!srcOuter.empty()) {
    if ((isRowVector || isColVector) && srcIs1D) {
      // Vectors are allowed to deserialize from a 1D array. No inner dimension.
      numRows = isRowVector ? 1 : srcOuter.size();
      numColumns = isColVector ? 1 : srcOuter.size();
    } else {
      // All other matrices require nested arrays like, "[[1,2],[3,4]]"
      numRows = srcOuter.size();
      for (size_t i = 0; i < srcOuter.size(); ++i) {
        if (!srcOuter[i].is<picojson::array>()) {
          SR_LOG("Failed to deserialize matrix. Expected an array of arrays in JSON.");
          ++inOutIssuesDetected;
          return;
        }
        auto const& srcInner = srcOuter[i].get<picojson::array>();
        if (i == 0) {
          numColumns = srcInner.size();
        } else if (srcInner.size() != numColumns) {
          SR_LOG("Failed to deserialize matrix. Inconsistent row width.");
          ++inOutIssuesDetected;
          return;
        }
      }
    }
  }

  // Resize the destination if necessary
  bool ok = TryResizeMatrix(*this, dst, dstLayout, numRows, numColumns);
  if (!ok) {
    ++inOutIssuesDetected;
    return;
  }

  if (!(numRows && numColumns)) {
    return; // No values
  }

  if (isRowVector && srcIs1D) {
    for (size_t i = 0; i < numColumns; ++i) {
      auto const& srcElem = srcOuter[i];
      void* dstElem = GetElement(dst, 0, i);
      int issuesDetected = 0;
      _innerTypeInfo->DeserializeInner(srcElem, dstElem, deserializeFlags, issuesDetected);
      if (issuesDetected) {
        inOutIssuesDetected += issuesDetected;
        return;
      }
    }
  } else if (isColVector && srcIs1D) {
    for (size_t i = 0; i < numRows; ++i) {
      auto const& srcElem = srcOuter[i];
      void* dstElem = GetElement(dst, i, 0);
      int issuesDetected = 0;
      _innerTypeInfo->DeserializeInner(srcElem, dstElem, deserializeFlags, issuesDetected);
      if (issuesDetected) {
        inOutIssuesDetected += issuesDetected;
        return;
      }
    }
  } else {
    // Deserialize each value in row-major order
    for (size_t r = 0; r < numRows; ++r) {
      auto const& srcInner = srcOuter[r].get<picojson::array>();
      for (size_t c = 0; c < numColumns; ++c) {
        auto const& srcElem = srcInner[c];
        void* dstElem = GetElement(dst, r, c);
        int issuesDetected = 0;
        _innerTypeInfo->DeserializeInner(srcElem, dstElem, deserializeFlags, issuesDetected);
        if (issuesDetected) {
          inOutIssuesDetected += issuesDetected;
          return;
        }
      }
    }
  }
}

bool MatrixTypeInfo::SerializeToBytesInner(void const* src, StreamWriter& dst) const {
  SR_ASSERT(_innerTypeInfo->IsMemCopySafe(), "Expected an inner type that is memcpy safe.");

  // When serializing to binary, we always use the source matrix's storage direction (unlike JSON
  // serialization). We save that storage direction for validation (at the cost of one byte).
  auto srcLayout = GetLayout(src);

  // Write dimensions, if resizable
  bool ok = true;
  if (_isNumRowsDynamic) {
    ok &= dst.Write(&srcLayout._numRows, sizeof(srcLayout._numRows));
  }
  if (_isNumColumnsDynamic) {
    ok &= dst.Write(&srcLayout._numColumns, sizeof(srcLayout._numColumns));
  }

  if (!(srcLayout._numRows && srcLayout._numColumns)) {
    return true; // No values. We're done.
  }

  size_t dim0 = _isRowMajor ? srcLayout._numRows : srcLayout._numColumns;
  size_t dim1 = _isRowMajor ? srcLayout._numColumns : srcLayout._numRows;
  size_t elemSize = _innerTypeInfo->_sizeInBytes;
  auto const* srcData = static_cast<std::byte const*>(GetData(src));
  SR_ASSERT(srcData != nullptr, "Nullptr detected on a matrix of non-zero size.");
  if (dim1 == srcLayout._leadingDim) {
    // Write all the values at once
    ok &= dst.Write(srcData, dim0 * dim1 * elemSize);
  } else {
    for (size_t i = 0; i < dim0; ++i) {
      auto const* srcVec = srcData + (i * srcLayout._leadingDim * elemSize);
      // Write the whole row or column
      ok &= dst.Write(srcVec, dim1 * elemSize);
    }
  }

  return ok;
}

bool MatrixTypeInfo::DeserializeFromBytesInner(StreamReader& src, void* dst) const {
  SR_ASSERT(_innerTypeInfo->IsMemCopySafe(), "Expected an inner type that is memcpy safe.");

  bool ok = true;
  Layout dstLayout = GetLayout(dst);

  // Read dimensions if they are dynamic
  Layout srcLayout{};
  if (_isNumRowsDynamic) {
    ok = src.Read(&srcLayout._numRows, sizeof(srcLayout._numRows));
  } else {
    srcLayout._numRows = dstLayout._numRows;
  }
  if (_isNumColumnsDynamic) {
    ok &= src.Read(&srcLayout._numColumns, sizeof(srcLayout._numColumns));
  } else {
    srcLayout._numColumns = dstLayout._numColumns;
  }
  if (!ok) {
    return false;
  }

  // Resize destination if necessary
  ok = TryResizeMatrix(*this, dst, dstLayout, srcLayout._numRows, srcLayout._numColumns);
  if (!ok) {
    return false;
  }

  if (!(srcLayout._numRows && srcLayout._numColumns)) {
    return ok; // No values. We're done.
  }

  // Deserialize values
  size_t dim0 = _isRowMajor ? dstLayout._numRows : dstLayout._numColumns;
  size_t dim1 = _isRowMajor ? dstLayout._numColumns : dstLayout._numRows;
  size_t elemSize = _innerTypeInfo->_sizeInBytes;
  auto* dstData = static_cast<std::byte*>(GetData(dst));
  SR_ASSERT(dstData != nullptr, "Nullptr detected on a matrix of non-zero size.");
  if (dim1 == dstLayout._leadingDim) {
    // Read all the values at once
    ok &= src.Read(dstData, dim0 * dim1 * elemSize);
  } else {
    for (size_t i = 0; i < dim0; ++i) {
      auto* dstVec = dstData + (i * dstLayout._leadingDim * elemSize);
      // Read the whole row or column
      ok &= src.Read(dstVec, dim1 * elemSize);
    }
  }

  return ok;
}

/////////////////////////////////////////////////////////////////////////////
// PairTypeInfoImpl
/////////////////////////////////////////////////////////////////////////////

static void const* OffsetPtr(void const* ptr, std::size_t offset) {
  return static_cast<void const*>(reinterpret_cast<uint8_t const*>(ptr) + offset);
}

static void* OffsetPtr(void* ptr, std::size_t offset) {
  return static_cast<void*>(reinterpret_cast<uint8_t*>(ptr) + offset);
}

class PairTypeInfoImpl final : public PairTypeInfo {
  static constexpr const char* kFirst = "key";
  static constexpr const char* kSecond = "value";

  bool IsValidInner(void const* src, AttributeList const& attribs) const override {
    return _infoT->IsValidInner(GetT(src), attribs) && _infoU->IsValidInner(GetU(src), attribs);
  }

  void SetInner(void const* src, void* dst) const override {
    _infoT->SetInner(GetT(src), GetT(dst));
    _infoU->SetInner(GetU(src), GetU(dst));
  }

  void SerializeInner(void const* src, picojson::value& dstJson) const override {
    picojson::value tJson, uJson;

    _infoT->SerializeInner(GetT(src), tJson);
    _infoU->SerializeInner(GetU(src), uJson);

    dstJson[kFirst] = tJson;
    dstJson[kSecond] = uJson;
  }

  void DeserializeInner(
      picojson::value const& srcJson,
      void* dst,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override {
    _infoT->DeserializeInner(srcJson[kFirst], GetT(dst), deserializeFlags, inOutIssuesDetected);
    _infoU->DeserializeInner(srcJson[kSecond], GetU(dst), deserializeFlags, inOutIssuesDetected);
  }

  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    return _infoT->SerializeToBytesInner(GetT(src), dst) &&
        _infoU->SerializeToBytesInner(GetU(src), dst);
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    return _infoT->DeserializeFromBytesInner(src, GetT(dst)) &&
        _infoU->DeserializeFromBytesInner(src, GetU(dst));
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    // only if both pair elements are memcopyable AND there are no padding bytes
    return _infoT->IsMemCopySafe() && _infoU->IsMemCopySafe() &&
        _sizeInBytes == (_infoT->_sizeInBytes + _infoU->_sizeInBytes);
  }

  // clang-format off
  void const* GetT(void const* ptr) const     { return OffsetPtr(ptr, _offsetT); }
  void*       GetT(void* ptr) const           { return OffsetPtr(ptr, _offsetT); }
  void const* GetU(void const* ptr) const     { return OffsetPtr(ptr, _offsetU); }
  void*       GetU(void* ptr) const           { return OffsetPtr(ptr, _offsetU); }
  // clang-format on
};

void PairTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  obj["typeT"] = AddTypeRef(_infoT, dict);
  obj["typeU"] = AddTypeRef(_infoU, dict);
  obj["offsetT"] = static_cast<double>(_offsetT);
  obj["offsetU"] = static_cast<double>(_offsetU);
}

PairTypeInfo* detail::MakePairTypeInfo(
    std::type_info const& stdTypeInfo,
    TypeInfo const* infoT,
    TypeInfo const* infoU,
    size_t sizeInBytes,
    size_t alignment,
    size_t offsetT,
    size_t offsetU) {
  PairTypeInfo* ti = new PairTypeInfoImpl();
  ti->_coreType = SReflect::CoreType::CT_other;
  ti->_infoT = infoT;
  ti->_infoU = infoU;
  ti->_sizeInBytes = sizeInBytes;
  ti->_alignment = alignment;
  ti->_offsetT = offsetT;
  ti->_offsetU = offsetU;
  ti->_name = SReflect::detail::MakeTypeName("pair<", infoT->_name, ",", infoU->_name, ">");
  ti->_nameWithNamespace = SReflect::detail::MakeTypeName(
      "std::pair<", infoT->_nameWithNamespace, ",", infoU->_nameWithNamespace, ">");
  ti->_typeId = SReflect::ComputeTypeId(ti->_nameWithNamespace);
  SReflect::VerifyTypeIdIsUnique(*ti, stdTypeInfo);

  return ti;
}

///////////////////////////////////////////////////////////////////////////
// MapTypeInfo
///////////////////////////////////////////////////////////////////////////

void MapTypeInfo::Enumerate(void const* map, MapTypeInfo::OnEachConst const& callback) const {
  // Implemented using the non-const overload to reduce code bloat in the templates
  OnEach wrapper = [&](void const* key, void* value) { return callback(key, value); };
  Enumerate(const_cast<void*>(map), wrapper);
}

void MapTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  obj["keyType"] = AddTypeRef(_keyTypeInfo, dict);
  obj["valueType"] = AddTypeRef(_valueTypeInfo, dict);
}

bool MapTypeInfo::IsValidInner(void const* src, AttributeList const& attribs) const {
  bool isValid = true;
  Enumerate(src, [&](void const* key, void const* value) {
    isValid = isValid && _keyTypeInfo->IsValidInner(key, attribs);
    isValid = isValid && _valueTypeInfo->IsValidInner(value, attribs);
    return isValid; // Stop enumeration if not valid
  });
  return isValid;
}

void MapTypeInfo::SerializeInner(void const* src, picojson::value& dstJson) const {
  dstJson = picojson::object();

  // Serialize key-value pairs into an array
  Enumerate(src, [&](void const* key, void const* value) {
    picojson::value keyJson, valueJson;
    _keyTypeInfo->SerializeInner(key, keyJson);
    _valueTypeInfo->SerializeInner(value, valueJson);

    // Don't re-serialize strings. Nesting is bad.
    if (keyJson.is<std::string>()) {
      dstJson[keyJson.get<std::string>()] = valueJson;
    } else {
      dstJson[keyJson.serialize()] = valueJson;
    }
    return true; // keep going
  });
}

void MapTypeInfo::DeserializeInner(
    picojson::value const& srcJson,
    void* dst,
    SReflect::DeserializeFlags deserializeFlags,
    int& inOutIssuesDetected) const {
  // Clear previous contents
  Clear(dst);

  void* tempKey = _keyTypeInfo->New();
  SR_DEFER(_keyTypeInfo->Delete(tempKey));
  void* tempValue = _valueTypeInfo->New();
  SR_DEFER(_valueTypeInfo->Delete(tempValue));

  // Populate key-value pairs
  for (auto&& [keyStr, valueJson] : srcJson.get<picojson::object>()) {
    // Convert string key into json
    picojson::value keyJson;
    if (_keyTypeInfo->_coreType == CoreType::CT_string) {
      // The key is a string. Use it directly.
      keyJson = keyStr;
    } else {
      // The key is some other type. Attempt to parse it as a JSON string.
      std::istringstream keyStream(keyStr);
      std::string parseError = picojson::parse(keyJson, keyStream);
      if (!parseError.empty() || keyJson.is<picojson::null>()) {
        SR_LOG(
            "Attempting to parse malformed JSON in map key.\n  JSON: %s\n  error: %s\n",
            keyStr.c_str(),
            parseError.empty() ? "unknown" : parseError.c_str());
        ++inOutIssuesDetected;
      }
    }

    // Deserialize and insert pairs
    _keyTypeInfo->DeserializeInner(keyJson, tempKey, deserializeFlags, inOutIssuesDetected);
    _valueTypeInfo->DeserializeInner(valueJson, tempValue, deserializeFlags, inOutIssuesDetected);
    Insert(dst, tempKey, tempValue);
  }
}

bool MapTypeInfo::SerializeToBytesInner(void const* src, SReflect::StreamWriter& dst) const {
  // Write number of elements
  auto size = static_cast<uint64_t>(GetNumKeys(src));
  if (!dst.Write(&size, sizeof(size))) {
    return false;
  }

  // Write key-value pairs
  bool ok = true;
  Enumerate(src, [&](void const* key, void const* value) {
    ok = ok && _keyTypeInfo->SerializeToBytesInner(key, dst);
    ok = ok && _valueTypeInfo->SerializeToBytesInner(value, dst);
    return ok; // Keep enumerating if OK
  });

  return true;
}

bool MapTypeInfo::DeserializeFromBytesInner(SReflect::StreamReader& src, void* dst) const {
  // Read number of elements
  uint64_t size = 0;
  if (!src.Read(&size, sizeof(size))) {
    return false;
  }

  // Clear previous contents
  Clear(dst);

  void* tempKey = _keyTypeInfo->New();
  SR_DEFER(_keyTypeInfo->Delete(tempKey));
  void* tempValue = _valueTypeInfo->New();
  SR_DEFER(_valueTypeInfo->Delete(tempValue));

  // Read key-value pairs
  bool ok = true;
  for (uint64_t i = 0; ok && i < size; ++i) {
    ok = ok && _keyTypeInfo->DeserializeFromBytesInner(src, tempKey);
    ok = ok && _valueTypeInfo->DeserializeFromBytesInner(src, tempValue);
    ok = ok && Insert(dst, tempKey, tempValue);
  }

  return ok;
}

bool MapTypeInfo::IsMemCopySafe() const {
  return false;
}

void detail::InitMapTypeInfo(
    MapTypeInfo* ti,
    char const* nameWithNamespace,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    TypeInfo const& keyTypeInfo,
    TypeInfo const& valueTypeInfo) {
  ti->_coreType = SReflect::CoreType::CT_map;
  ti->_sizeInBytes = sizeInBytes;
  ti->_alignment = alignment;
  ti->_keyTypeInfo = &keyTypeInfo;
  ti->_valueTypeInfo = &valueTypeInfo;
  ti->_name = MakeTypeName(
      GetTypeNameWithoutNamespace(nameWithNamespace),
      "<",
      keyTypeInfo._name,
      ",",
      valueTypeInfo._name,
      ">");
  ti->_nameWithNamespace = MakeTypeName(
      nameWithNamespace,
      "<",
      keyTypeInfo._nameWithNamespace,
      ",",
      valueTypeInfo._nameWithNamespace,
      ">");

  ti->_typeId = SReflect::ComputeTypeId(ti->_nameWithNamespace);
  SReflect::VerifyTypeIdIsUnique(*ti, rttiTypeInfo);
}

// Implements AppendTemplateArgStr<V> where V is an enum value constant
void detail::AppendTemplateArgStr_EnumValue(
    EnumTypeInfo const& enumInfo,
    uint64_t value,
    bool isEnumClass,
    std::string& out) {
  auto const* enumItem = enumInfo.FindItemByValue(value);
  SR_ASSERT(
      enumItem != nullptr,
      "Attempting to use SR_BeginClassTemplate or SR_BeginStructTemplate with a non-type template parameter, "
      "which is not known to Simple Reflection. Please add an SR_BeginEnum/SR_EndEnum block and make sure you "
      "have an SR_EnumItem line for every relevant item.");
  if (enumItem) {
    if (isEnumClass) {
      // Prepend an enum class/struct value with its namespace and type name.
      out += enumInfo._nameWithNamespace;
      out += "::";
    } else {
      // Prepend a plain enum value with its namespace (if any).
      auto nameWithNamespace = std::string_view(enumInfo._nameWithNamespace);
      auto pos = nameWithNamespace.rfind("::");
      out += (pos == std::string_view::npos) ? "" : nameWithNamespace.substr(0, pos + 2);
    }
    out += enumItem->_name;
  } else {
    // Fallback (in case we keep running after the assertion failure)
    out += std::to_string(static_cast<int64_t>(value));
  }
}

char const* detail::MakeTemplateTypeName(
    const char* structNameWithNamespace,
    const char* formattedArgList) {
  SR_ASSERT(structNameWithNamespace && *structNameWithNamespace, "Missing type name");
  SR_ASSERT(formattedArgList && *formattedArgList, "Missing template argument list");
  std::string args(formattedArgList);
  if (!args.empty() && args[args.length() - 1] == ',') {
    args.pop_back();
  }
  return MakeTypeName(structNameWithNamespace, "<", args.c_str(), ">");
}

char const* detail::EnumToStringImpl(EnumTypeInfo const& ti, uint64_t value) {
  auto const* item = ti.FindItemByValue(value);
  return item ? item->_name : "";
}

/////////////////////////////////////////////////////////////////////////////
// OptionalTypeInfoImpl
/////////////////////////////////////////////////////////////////////////////

void OptionalTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  obj["innerType"] = AddTypeRef(_innerTypeInfo, dict);
}

class OptionalTypeInfoImpl final : public OptionalTypeInfo {
 public:
  using EnsureOptionalValueFn = void* (*)(void* optional);
  using GetOptionalValueFn = void* (*)(void* optional);
  using SetOptionalValueFn = void (*)(void const* srcValue, void* dstOptional);

  // Type-specific accessors
  EnsureOptionalValueFn _ensureOptionalValueFn;
  GetOptionalValueFn _getOptionalValueFn;
  SetOptionalValueFn _setOptionalValueFn;

  // TypeInfo
  bool IsValidInner(void const* srcOptional, AttributeList const& attribs) const override {
    void const* srcValue = GetOptionalValue(srcOptional);
    return srcValue ? _innerTypeInfo->IsValidInner(srcValue, attribs) : true;
  }

  void SetInner(void const* srcOptional, void* dstOptional) const override {
    SetOptionalValue(GetOptionalValue(srcOptional), dstOptional);
  }

  void SerializeInner(void const* srcOptional, picojson::value& dstJson) const override {
    void const* srcValue = GetOptionalValue(srcOptional);
    if (srcValue) {
      _innerTypeInfo->SerializeInner(srcValue, dstJson);
    } else {
      dstJson = picojson::value{}; // null in JSON means "no value"
    }
  }

  void DeserializeInner(
      picojson::value const& srcJson,
      void* dstOptional,
      DeserializeFlags deserializeFlags,
      int& inOutIssuesDetected) const override {
    if (srcJson.is<picojson::null>()) {
      SetOptionalValue(nullptr, dstOptional); // null in JSON means "no value"
    } else {
      // Make sure it has a value, then deserialize it in place
      _innerTypeInfo->DeserializeInner(
          srcJson, EnsureOptionalValue(dstOptional), deserializeFlags, inOutIssuesDetected);
    }
  }

  bool SerializeToBytesInner(void const* srcOptional, StreamWriter& dst) const override {
    void const* srcValue = GetOptionalValue(srcOptional);
    if (srcValue) {
      // Serialize 1 to signal optional has value
      if (!serde::bytes::Serialize(uint8_t(1), dst)) {
        return false;
      }

      return _innerTypeInfo->SerializeToBytesInner(srcValue, dst);
    } else {
      // Serialize 0 to signal optional is nullopt
      return serde::bytes::Serialize(uint8_t(0), dst);
    }
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    uint8_t flag = 0;
    if (!serde::bytes::Deserialize(src, flag)) {
      return false;
    }
    if (flag) {
      return _innerTypeInfo->DeserializeFromBytesInner(src, EnsureOptionalValue(dst));
    } else {
      SetOptionalValue(nullptr, dst);
      return true;
    }
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return false;
  }

  void* EnsureOptionalValue(void* srcOptional) const override {
    return _ensureOptionalValueFn(srcOptional);
  }

  void* GetOptionalValue(void* srcOptional) const override {
    return _getOptionalValueFn(srcOptional);
  }

  void const* GetOptionalValue(void const* srcOptional) const override {
    // Implement using the non-const version to reduce bloat in the templates
    return _getOptionalValueFn(const_cast<void*>(srcOptional));
  }

  void SetOptionalValue(void const* srcValue, void* dstOptional) const override {
    _setOptionalValueFn(srcValue, dstOptional);
  }
};

OptionalTypeInfo* detail::MakeOptionalTypeInfo(
    std::type_info const& stdTypeInfo,
    TypeInfo const& innerTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    void* (*ensureValueFn)(void* optional),
    void* (*getValueFn)(void* optional),
    void (*setValueFn)(void const* srcValue, void* dstOptional)) {
  auto* info = new OptionalTypeInfoImpl;
  info->_coreType = CoreType::CT_optional;
  info->_innerTypeInfo = &innerTypeInfo;
  info->_name = detail::MakeTypeName("optional<", innerTypeInfo._name, ">");
  info->_nameWithNamespace =
      detail::MakeTypeName("std::optional<", innerTypeInfo._nameWithNamespace, ">");
  info->_typeId = ComputeTypeId(info->_nameWithNamespace);
  info->_sizeInBytes = sizeInBytes;
  info->_alignment = alignment;
  info->_ensureOptionalValueFn = ensureValueFn;
  info->_getOptionalValueFn = getValueFn;
  info->_setOptionalValueFn = setValueFn;
  VerifyTypeIdIsUnique(*info, stdTypeInfo);
  return info;
}

/////////////////////////////////////////////////////////////////////////////
// PicojsonValueTypeInfoImpl
/////////////////////////////////////////////////////////////////////////////

class PicojsonValueTypeInfoImpl final : public TypeInfo {
 public:
  using TypeInfo::TypeInfo;

 protected:
  bool IsValidInner(void const* /*src*/, AttributeList const& /*attribs*/) const override {
    return true;
  }

  void SetInner(void const* src, void* dst) const override {
    *reinterpret_cast<picojson::value*>(dst) = *reinterpret_cast<picojson::value const*>(src);
  }

  void SerializeInner(void const* src, picojson::value& dst) const override {
    dst = *reinterpret_cast<picojson::value const*>(src);
  }

  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags /*deserializeFlags*/,
      int& /*inOutIssuesDetected*/) const override {
    *reinterpret_cast<picojson::value*>(dst) = src;
  }

  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    // Convert to std::string
    auto const* value = static_cast<picojson::value const*>(src);
    std::string jsonStr = value->serialize();

    // Serialize std::string into bytes
    return GetFinalTypeInfo(jsonStr).SerializeToBytes(&jsonStr, dst);
  }

  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    // Deserialize std::string from bytes
    std::string jsonStr;
    if (!GetFinalTypeInfo(jsonStr).DeserializeFromBytes(src, &jsonStr)) {
      return false;
    }

    // Parse string into value (may throw)
    auto* value = static_cast<picojson::value*>(dst);
    std::istringstream jsonStream(jsonStr);
    std::string parseError = picojson::parse(*value, jsonStream);
    if (!parseError.empty()) {
      SR_LOG("Attempting to parse malformed JSON.\n  error: %s\n", parseError.c_str());
      return false;
    }
    return true;
  }

  [[nodiscard]] bool IsMemCopySafe() const override {
    return false;
  }
};

TypeInfo* detail::MakePicojsonValueTypeInfo() {
  auto* info = new PicojsonValueTypeInfoImpl;
  info->_coreType = CoreType::CT_other;
  static constexpr const char* kTypeName{"picojson_value"};
  info->_name = kTypeName;
  info->_nameWithNamespace = kTypeName;
  info->_typeId = ComputeTypeId(info->_nameWithNamespace);
  info->_sizeInBytes = sizeof(picojson::value);
  info->_alignment = alignof(picojson::value);
  VerifyTypeIdIsUnique(*info, typeid(picojson::value));
  InitTypeInfoFunctionPointers<picojson::value>(info);
  return info;
}

/////////////////////////////////////////////////////////////////////////////
// VariantTypeInfo
/////////////////////////////////////////////////////////////////////////////

void const* VariantTypeInfo::GetInnerObject(void const* obj) const {
  return GetInnerObject(const_cast<void*>(obj)); // Const-in-const-out
}

void VariantTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  SetField(obj, "innerTypes", picojson::value{AddTypeRefArray(_innerTypes, dict)});
  obj["isMemCopySafe"] = IsMemCopySafe();
}

bool VariantTypeInfo::IsValidInner(void const* src, AttributeList const& attributes) const {
  size_t typeIndex = GetInnerTypeIndex(src);
  SR_ASSERT(typeIndex < _innerTypes.size(), "Invalid type index");
  return _innerTypes[typeIndex]->IsValidInner(GetInnerObject(src), attributes);
}

bool VariantTypeInfo::IsMemCopySafe() const {
  return _isMemCopySafe;
}

void VariantTypeInfo::SetInner(void const* src, void* dst) const {
  size_t typeIndex = GetInnerTypeIndex(src);
  SR_ASSERT(typeIndex < _innerTypes.size(), "Invalid type index");
  bool success = TrySetInnerTypeIndex(dst, typeIndex);
  SR_ASSERT(success, "Failed to set variant type");
  _innerTypes[typeIndex]->SetInner(GetInnerObject(src), GetInnerObject(dst));
}

void VariantTypeInfo::SerializeInner(void const* src, picojson::value& dstJson) const {
  size_t typeIndex = GetInnerTypeIndex(src);
  SR_ASSERT(typeIndex < _innerTypes.size(), "Invalid type index");
  auto const* innerType = _innerTypes[typeIndex];

#ifndef NDEBUG
  // We do not currently include the namespace of the inner type in JSON.
  // Reasons:
  //   - Something like "string" looks nicer than "std::string" in JSON
  //   - JSON files are sometimes read by more than one language. It would be weird if Python code
  //     had to know what to do with a type name like "std::string".
  // Potential Issue:
  //   - The type names could be ambiguous if they only differy by namespace.
  for (size_t i = 0; i < _innerTypes.size(); ++i) {
    for (size_t j = i + 1; j < _innerTypes.size(); ++j) {
      SR_ASSERT(
          0 != strcmp(_innerTypes[i]->_name, _innerTypes[j]->_name),
          "Expected inner types to have unique type names without namespaces.");
    }
  }
#endif

  // Serialize as a JSON object with one field like this:
  //   { "InnerTypeName": innerValue }
  dstJson = picojson::object();
  auto& dstJsonObj = dstJson.get<picojson::object>();
  auto& dstJsonVal = dstJsonObj[innerType->_name];
  innerType->SerializeInner(GetInnerObject(src), dstJsonVal);
}

void VariantTypeInfo::DeserializeInner(
    picojson::value const& srcJson,
    void* dst,
    SReflect::DeserializeFlags deserializeFlags,
    int& inOutIssuesDetected) const {
  // We expect a JSON dictionary with one key
  if (!srcJson.is<picojson::object>()) {
    SR_LOG("Failed to deserialize variant. Expected {} braces.");
    inOutIssuesDetected++;
    return;
  }
  auto const& srcJsonObj = srcJson.get<picojson::object>();
  if (srcJsonObj.size() != 1) {
    SR_LOG("Failed to deserialize variant. Expected a JSON object like { \"TypeName\": value }.");
    ++inOutIssuesDetected;
    return;
  }

  // Find the inner type by name
  auto const& [innerTypeName, innerJsonValue] =
      *srcJsonObj.begin(); // First and only key-value pair
  int innerTypeIndex = -1;
  for (size_t i = 0; i < _innerTypes.size(); ++i) {
    if (innerTypeName == _innerTypes[i]->_name) {
      innerTypeIndex = (int)i; // Found it
      break;
    }
  }
  if (innerTypeIndex == -1) {
    SR_LOG(
        "Failed to deserialize variant. \"%s\" is not one of the supported types.",
        innerTypeName.c_str());
    ++inOutIssuesDetected;
    return;
  }

  // Change the inner type of the runtime object, if necessary.
  auto prevInnerTypeIndex = GetInnerTypeIndex(dst);
  SR_ASSERT(prevInnerTypeIndex < _innerTypes.size(), "Invalid type index");
  if (innerTypeIndex != (int)prevInnerTypeIndex) {
    if (!TrySetInnerTypeIndex(dst, (size_t)innerTypeIndex)) {
      SR_LOG(
          "Failed to deserialize variant. The inner type \"%s\" could not be changed to \"%s\".",
          _innerTypes[prevInnerTypeIndex]->_nameWithNamespace,
          innerTypeName.c_str());
      ++inOutIssuesDetected;
      return;
    }
  }

  // Finally, deserialize the inner object
  _innerTypes[innerTypeIndex]->DeserializeInner(
      innerJsonValue, GetInnerObject(dst), deserializeFlags, inOutIssuesDetected);
}

bool VariantTypeInfo::SerializeToBytesInner(void const* src, SReflect::StreamWriter& dst) const {
  size_t typeIndex = GetInnerTypeIndex(src);
  SR_ASSERT(typeIndex < _innerTypes.size(), "Invalid type index");
  auto const* innerType = _innerTypes[typeIndex];

  // Write the type index. Surely one byte is enough.
  auto typeIndexByte = static_cast<uint8_t>(typeIndex);
  bool ok = dst.Write(&typeIndexByte, sizeof(typeIndexByte));

  // Serialize the inner object
  ok &= innerType->SerializeToBytesInner(GetInnerObject(src), dst);
  return ok;
}

bool VariantTypeInfo::DeserializeFromBytesInner(SReflect::StreamReader& src, void* dst) const {
  // Read the type index (one byte)
  uint8_t innerTypeIndex = 0;
  if (!src.Read(&innerTypeIndex, sizeof(innerTypeIndex))) {
    return false;
  }
  if ((size_t)innerTypeIndex >= _innerTypes.size()) {
    SR_LOG("Failed to deserialize variant. Invalid type index.");
    return false;
  }

  // Change the inner type of the runtime object, if necessary.
  auto prevInnerTypeIndex = GetInnerTypeIndex(dst);
  SR_ASSERT(prevInnerTypeIndex < _innerTypes.size(), "Invalid type index");
  if ((size_t)innerTypeIndex != prevInnerTypeIndex) {
    if (!TrySetInnerTypeIndex(dst, innerTypeIndex)) {
      SR_LOG(
          "Failed to deserialize variant. The inner type \"%s\" could not be changed to \"%s\".",
          _innerTypes[prevInnerTypeIndex]->_nameWithNamespace,
          _innerTypes[innerTypeIndex]->_nameWithNamespace);
      return false;
    }
  }

  // Deserialize the inner object
  return _innerTypes[innerTypeIndex]->DeserializeFromBytesInner(src, GetInnerObject(dst));
}

void detail::InitVariantTypeInfo(
    VariantTypeInfo& ti,
    char const* classNameWithNamespace,
    size_t sizeInBytes,
    size_t alignment,
    bool isTriviallyCopyable) {
  SR_ASSERT(
      !ti._innerTypes.empty(),
      "The _innerType should be initialized before calling this function.");
  SR_ASSERT(
      ti._innerTypes.size() <= (size_t)255,
      "This variant has too many inner types. Binary serialization currently assumes the number can fit in a single byte.");

  // Format the type name with and without namespaces
  std::string name = GetTypeNameWithoutNamespace(classNameWithNamespace);
  std::string nameWithNamespace = classNameWithNamespace;
  name += "<";
  nameWithNamespace += "<";
  bool innerTypesAreMemCopySafe = true;
  for (auto const* innerType : ti._innerTypes) {
    name += innerType->_name;
    name += ",";
    nameWithNamespace += innerType->_nameWithNamespace;
    nameWithNamespace += ",";
    innerTypesAreMemCopySafe &= innerType->IsMemCopySafe();
  }
  name[name.length() - 1] = '>'; // replace the last comma
  nameWithNamespace[nameWithNamespace.length() - 1] = '>';

  // TypeInfo fields
  ti._coreType = SReflect::CoreType::CT_variant;
  ti._alignment = alignment;
  ti._sizeInBytes = sizeInBytes;
  ti._name = MakeTypeName(name.c_str());
  ti._nameWithNamespace = MakeTypeName(nameWithNamespace.c_str());
  ti._typeId = SReflect::ComputeTypeId(ti._nameWithNamespace);
  ti._isMemCopySafe = isTriviallyCopyable && innerTypesAreMemCopySafe;
}

/////////////////////////////////////////////////////////////////////////////
// detail
/////////////////////////////////////////////////////////////////////////////

// Returns true if `attributes` already holds an attribute whose (final) type matches `typeId`.
// Used to enforce one-attribute-per-type so the serialized attribute dictionary is lossless.
static bool AttributeTypeAlreadyPresent(TypeInfo::AttributeList const& attributes, TypeId typeId) {
  for (auto const& [existingTypeId, unusedAttribute] : attributes) {
    (void)unusedAttribute;
    if (existingTypeId == typeId) {
      return true;
    }
  }
  return false;
}

static void AssertAttributeTypeNotPresent(
    TypeInfo::AttributeList const& attributes,
    TypeId typeId) {
  SR_ASSERT(
      !AttributeTypeAlreadyPresent(attributes, typeId),
      "Adding an attribute of the same type more than once");
}

void detail::AddAttribute(TypeInfo& info, Attribute const* a) {
  AssertAttributeTypeNotPresent(info._attributes, a->GetFinalTypeId());
  info._attributes.emplace_back(a->GetFinalTypeId(), a);
}

void detail::AddAttribute(StructTypeInfo& info, Attribute const* a) {
  if (info._fields.empty()) {
    // If the attribute is added before any fields, then it describes the struct.
    AssertAttributeTypeNotPresent(info._attributes, a->GetFinalTypeId());
    info._attributes.emplace_back(a->GetFinalTypeId(), a);
  } else {
    // If the attribute comes after a field, then it describes that field.
    AssertAttributeTypeNotPresent(info._fields.back()->_attributes, a->GetFinalTypeId());
    info._fields.back()->_attributes.emplace_back(a->GetFinalTypeId(), a);
  }
}

void detail::AddField(
    StructTypeInfo& info,
    char const* name,
    size_t offset,
    TypeInfo const& innerType) {
  SR_ASSERT(info.FindField(name) == nullptr, "Adding the same field more than once");
  auto* newField = new FieldTypeInfoImpl;
  newField->_coreType = CoreType::CT_field;
  newField->_innerTypeInfo = &innerType;
  newField->_offset = offset;
  newField->_name = name;
  newField->_nameWithNamespace = name;
  newField->_typeId = ComputeTypeId(newField->_nameWithNamespace);
  info._fields.push_back(newField);
}

void detail::RemoveField(StructTypeInfo& info, char const* name) {
  for (auto it = info._fields.begin(); it != info._fields.end(); ++it) {
    if ((*it)->_name == name) {
      info._fields.erase(it);
      return;
    }
  }
  SR_ASSERT(false, "Field not found");
}

void detail::AddBaseClass(
    StructTypeInfo& derived,
    StructTypeInfo const& base,
    ptrdiff_t baseOffset) {
  SR_ASSERT(baseOffset == 0, "Multiple inheritence of non-empty base classes is not supported!");
  SR_ASSERT(derived._typeIndex != base._typeIndex, "Cannot add a base class to itself.");
#if SR_ASSERT_ENABLE
  for (StructTypeInfo const* baseAncestor : base._baseClasses) {
    SR_ASSERT(baseAncestor->_typeIndex != derived._typeIndex, "Cannot add a base class cycle.");
    for (StructTypeInfo const* derivedAncestor : derived._baseClasses) {
      SR_ASSERT(
          derivedAncestor->_typeIndex != baseAncestor->_typeIndex,
          "Diamond inheritence is not supported");
    }
  }
#endif // SR_ASSERT_ENABLE

  // Import all base classes into a flat list
  derived._baseClasses.reserve(derived._baseClasses.size() + base._baseClasses.size() + 1);
  derived._baseClasses.push_back(&base);
  derived._baseClasses.insert(
      derived._baseClasses.end(), base._baseClasses.begin(), base._baseClasses.end());

  // Import fields by copying the pointers. We don't have to adjust the field offsets
  // because we assume the baseOffset is zero (see verify above).
  derived._fields.insert(derived._fields.end(), base._fields.begin(), base._fields.end());
}

void detail::AddEnumItem(EnumTypeInfo& info, char const* name, uint64_t value) {
  SR_ASSERT(
      info.FindItemByName(name) == nullptr,
      "Attempting to add the same enum value more than once!");
  info._items.emplace_back(name, value);
}

void detail::AddAttribute(EnumTypeInfo& info, Attribute const* a) {
  SR_ASSERT(a != nullptr, "Attribute must not be null.");

  if (info._items.empty()) {
    AssertAttributeTypeNotPresent(info._attributes, a->GetFinalTypeId());
    info._attributes.emplace_back(a->GetFinalTypeId(), a);
    return;
  }

  // Validate PreviouslyKnownAs names before modifying the item
  if (auto const* pna = dynamic_cast<Attribute_PreviouslyKnownAs const*>(a)) {
    for (std::string const& prevName : pna->_previousNames) {
      SR_ASSERT(!prevName.empty(), "Empty string not allowed for attribute PreviouslyKnownAs.");
      SR_ASSERT(
          info.FindItemByName(prevName) == nullptr,
          "Enum alias name clashes with an existing item or alias name.");
    }
  }

  AssertAttributeTypeNotPresent(info._items.back()._attributes, a->GetFinalTypeId());
  info._items.back()._attributes.emplace_back(a->GetFinalTypeId(), a);
}

void StringTypeInfo::SerializeTypeInfoImpl(picojson::value& entry, picojson::value& dict) const {
  TypeInfo::SerializeTypeInfoImpl(entry, dict);
  picojson::object& obj = entry.get<picojson::object>();
  obj["isNullTerminated"] = _isNullTerminated;
  obj["isReadOnly"] = _isReadOnly;
}

namespace {
class StdStringTypeInfo : public StringTypeInfo {
 public:
  // Helpers:
  static std::string& Str(void* obj) {
    return *static_cast<std::string*>(obj);
  }
  static std::string const& Str(void const* obj) {
    return *static_cast<std::string const*>(obj);
  }

  // StringTypeInfo:
  std::string_view GetString(void const* obj) const override {
    return Str(obj);
  }
  bool SetString(void* obj, std::string_view str) const override {
    Str(obj) = str;
    return true;
  }

  // TypeInfo:
  void SetInner(void const* src, void* dst) const override {
    Str(dst) = Str(src);
  }
  bool IsValidInner(void const* /*src*/, AttributeList const& /*attribs*/) const override {
    return true;
  }
  void SerializeInner(void const* src, picojson::value& dst) const override {
    dst = Str(src);
  }
  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      DeserializeFlags /*deserializeFlags*/,
      int& /*inOutIssuesDetected*/) const override {
    detail::GetJsonValue(src, Str(dst));
  }
  bool SerializeToBytesInner(void const* src, StreamWriter& dst) const override {
    return serde::bytes::Serialize(Str(src), dst);
  }
  bool DeserializeFromBytesInner(StreamReader& src, void* dst) const override {
    return serde::bytes::Deserialize(src, Str(dst));
  }
};
} // namespace

static StringTypeInfo* MakeStdStringTypeInfo() {
  auto* ti = new StdStringTypeInfo;
  ti->_isReadOnly = false;
  ti->_isNullTerminated = true;
  ti->_coreType = CoreType::CT_string;
  ti->_sizeInBytes = sizeof(std::string);
  ti->_alignment = alignof(std::string);
  ti->_nameWithNamespace = "std::string";
  ti->_name = "string";
  ti->_typeId = ComputeTypeId(ti->_nameWithNamespace);
  VerifyTypeIdIsUnique(*ti, typeid(std::string));
  InitTypeInfoFunctionPointers<std::string>(ti);
  return ti;
};

EnumTypeInfo* detail::MakeEnum(
    char const* nameWithNamespace,
    std::type_info const& rttiTypeInfo,
    const TypeInfo& innerTypeInfo) {
  EnumTypeInfo* ti = new EnumTypeInfoImpl(std::type_index{rttiTypeInfo});
  ti->_coreType = CoreType::CT_enum;
  ti->_innerTypeInfo = &innerTypeInfo;
  ti->_sizeInBytes = ti->_innerTypeInfo->_sizeInBytes;
  ti->_alignment = ti->_innerTypeInfo->_alignment;
  ti->_nameWithNamespace = nameWithNamespace;
  ti->_name = GetTypeNameWithoutNamespace(nameWithNamespace);
  ti->_typeId = ComputeTypeId(ti->_nameWithNamespace);
  VerifyTypeIdIsUnique(*ti, rttiTypeInfo);
  return ti;
};

static char const* MakeCStyleArrayName(char const* typeName, size_t arraySize) {
  char buffer[kMaxTypeNameLen];
  const int len = snprintf(buffer, sizeof(buffer), "%s[%zu]", typeName, arraySize);
  SR_ASSERT(len > 0 && len < sizeof(buffer), "Type name is too long");
  char* formattedName = new char[len + 1];
  strncpy(formattedName, buffer, size_t(len + 1));
  return formattedName;
}

static char const* MakeTemplateArrayName(
    char const* outerName,
    char const* innerName,
    size_t arraySizeOrZeroIfDynamic) {
  char buffer[kMaxTypeNameLen];
  const int len = arraySizeOrZeroIfDynamic
      ? snprintf(
            buffer, sizeof(buffer), "%s<%s,%zu>", outerName, innerName, arraySizeOrZeroIfDynamic)
      : snprintf(buffer, sizeof(buffer), "%s<%s>", outerName, innerName);
  SR_ASSERT(len > 0 && len < sizeof(buffer), "Type name is too long");
  char* formattedName = new char[len + 1];
  strncpy(formattedName, buffer, size_t(len + 1));
  return formattedName;
}

ArrayTypeInfo* detail::MakeFixedArrayTypeInfoImpl(
    char const* nameWithNamespace,
    bool formatNameAsTemplate,
    TypeInfo const& innerTypeInfo,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    size_t fixedArraySize,
    bool isMemCopySafe) {
  auto* ti = new FixedArrayTypeInfo;
  ti->_coreType = CoreType::CT_array;
  ti->_innerTypeInfo = &innerTypeInfo;
  ti->_sizeInBytes = sizeInBytes;
  ti->_alignment = alignment;
  ti->_numElements = fixedArraySize;
  ti->_isMemCopySafe = isMemCopySafe && innerTypeInfo.IsMemCopySafe();
  if (nameWithNamespace) {
    if (formatNameAsTemplate) {
      // Format as "name<T,N>"
      ti->_nameWithNamespace = MakeTemplateArrayName(
          nameWithNamespace, innerTypeInfo._nameWithNamespace, fixedArraySize);
      ti->_name = MakeTemplateArrayName(
          GetTypeNameWithoutNamespace(nameWithNamespace), innerTypeInfo._name, fixedArraySize);
    } else {
      // Use the provided name with namespace as-is
      ti->_nameWithNamespace = nameWithNamespace;
      // Remove all namespaces from the name, including namespaces on inner types which might exist.
      ti->_name = MakeTypeNameWithoutAnyNamespace(nameWithNamespace);
    }
  } else {
    // Format as "T[N]"
    ti->_nameWithNamespace = MakeCStyleArrayName(innerTypeInfo._nameWithNamespace, fixedArraySize);
    ti->_name = MakeCStyleArrayName(innerTypeInfo._name, fixedArraySize);
  }
  ti->_typeId = ComputeTypeId(ti->_nameWithNamespace);
  VerifyTypeIdIsUnique(*ti, rttiTypeInfo);
  return ti;
}

void detail::InitDynamicArrayTypeInfoImpl(
    ArrayTypeInfo* ti,
    char const* nameWithNamespace,
    bool formatNameAsTemplate,
    TypeInfo const& innerTypeInfo,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment) {
  ti->_coreType = CoreType::CT_array;
  ti->_innerTypeInfo = &innerTypeInfo;
  ti->_sizeInBytes = sizeInBytes;
  ti->_alignment = alignment;
  if (formatNameAsTemplate) {
    // Format as "name<T>"
    ti->_nameWithNamespace =
        MakeTemplateArrayName(nameWithNamespace, innerTypeInfo._nameWithNamespace, 0);
    ti->_name = MakeTemplateArrayName(
        GetTypeNameWithoutNamespace(nameWithNamespace), innerTypeInfo._name, 0);
  } else {
    // Use the provided name as-is
    ti->_nameWithNamespace = nameWithNamespace;
    ti->_name = GetTypeNameWithoutNamespace(nameWithNamespace);
  }
  ti->_typeId = ComputeTypeId(ti->_nameWithNamespace);
  VerifyTypeIdIsUnique(*ti, rttiTypeInfo);
}

SReflect::StructTypeInfo* detail::MakeStructImpl(
    char const* nameWithNamespace,
    std::type_info const& rttiTypeInfo,
    size_t sizeInBytes,
    size_t alignment,
    bool isMemCopySafe) {
  auto* ti = new StructTypeInfoImpl(rttiTypeInfo);
  ti->_coreType = CoreType::CT_struct;
  ti->_sizeInBytes = sizeInBytes;
  ti->_alignment = alignment;
  ti->_nameWithNamespace = nameWithNamespace;
  ti->_name = GetTypeNameWithoutNamespace(nameWithNamespace);
  ti->_typeId = ComputeTypeId(ti->_nameWithNamespace);
  ti->_isMemCopySafe = isMemCopySafe;
  VerifyTypeIdIsUnique(*ti, rttiTypeInfo);
  return ti;
}

SReflect::StructTypeInfo* detail::CloneStructImpl(const SReflect::StructTypeInfo* structInfo) {
  return new StructTypeInfoImpl{*dynamic_cast<const StructTypeInfoImpl*>(structInfo)};
}

void FinalizeStruct(StructTypeInfo* info) {
  for (size_t i = 0; i < info->_fields.size() && info->_isMemCopySafe; ++i) {
    auto* field = info->_fields[i];
    info->_isMemCopySafe &= field->IsMemCopySafe();
  }
}

char const* detail::MakeTypeName(
    char const* a,
    char const* b,
    char const* c,
    char const* d,
    const char* e,
    char const* f) {
  char buffer[kMaxTypeNameLen];
  int len = snprintf(buffer, sizeof(buffer), "%s%s%s%s%s%s", a, b, c, d, e, f);
  SR_ASSERT(len > 0 && len < sizeof(buffer), "Type name is too long");
  char* newName = new char[len + 1];
  strncpy(newName, buffer, size_t(len + 1));
  return newName;
}

TypeInfo const& GetPrimitiveInfo(CoreType type) {
  SR_ASSERT((size_t)type < std::size(kCoreTypeInfo), "Invalid CoreType for GetPrimitiveInfo");
  TypeInfo const* info = kCoreTypeInfo[(size_t)type].primitiveTypeInfo;
  SR_ASSERT(info != nullptr, "Not a primitive type");
  return *info;
}

std::string detail::SaveToJsonString(void const* obj, TypeInfo const& info, bool pretty) {
  picojson::value jsonValue;
  info.Serialize(obj, jsonValue);
  return jsonValue.serialize(pretty);
}

void detail::LoadFromJsonString(
    void* objOut,
    TypeInfo const& info,
    std::string const& jsonStr,
    DeserializeFlags deserializeFlags,
    int& numIssuesOut) {
  std::istringstream jsonStream(jsonStr);
  picojson::value jsonValue;
  std::string parseError = picojson::parse(jsonValue, jsonStream);
  if (!parseError.empty()) {
    SR_LOG("Attempting to parse malformed JSON.\n  error: %s\n", parseError.c_str());
    ++numIssuesOut;
    return;
  }
  info.Deserialize(jsonValue, objOut, deserializeFlags, numIssuesOut);
}

bool detail::SaveToJsonFile(
    void const* obj,
    TypeInfo const& typeInfo,
    char const* filePath,
    bool warnOnFailure) {
  // Attempt to open output file
  std::ofstream outFile(filePath, std::ios::binary);
  if (!outFile.is_open()) {
    if (warnOnFailure) {
      SR_LOG("Failed to open JSON file for writing: %s", filePath);
    }
    return false; // Failed
  }

  // Serialize object into json data
  picojson::value dataVal = picojson::object();
  typeInfo.Serialize(obj, dataVal);

  // Serialize json data to string and write to file
  std::string dataStr = dataVal.serialize(true);
  outFile << dataStr;
  outFile.close();
  return true; // Success
}

bool detail::LoadFromJsonFile(
    void* objOut,
    TypeInfo const& typeInfo,
    char const* filePath,
    DeserializeFlags deserializeFlags,
    int& numIssuesOut) {
  numIssuesOut = 0;

  // Attempt to open file
  std::ifstream inFile(filePath);
  if (!inFile.is_open()) {
    // Failed to open json file
    SR_LOG("Failed to open JSON file for reading: %s", filePath);
    numIssuesOut++;
    return false; // Failed
  }

  // Load data from file into memory
  picojson::value data;
  std::string parseError = picojson::parse(data, inFile);
  inFile.close();

  // Check for parse errors
  if (!parseError.empty()) {
    SR_LOG(
        "Attempting to parse malformed JSON.\n  file: %s\n  error: %s\n",
        filePath,
        parseError.c_str());
    numIssuesOut++;
    return false; // Failed
  }

  // Deserialize data
  typeInfo.Deserialize(data, objOut, deserializeFlags, numIssuesOut);

  // If there were any issues, and user has requested any warnings, log the file that produced the
  // warnings, which is very useful for fixing them.
  auto flags = static_cast<uint32_t>(deserializeFlags);
  if (numIssuesOut > 0 && (flags & static_cast<uint32_t>(DeserializeFlags::MaximumWarnings))) {
    SR_LOG("%d issues detected while deserializing from JSON file: %s", numIssuesOut, filePath);
  }

  return true; // File read was successful, even if there were JSON issues
}

bool detail::ToBytes(void const* src, TypeInfo const& srcInfo, StreamWriter& dst) noexcept {
  return srcInfo.SerializeToBytes(src, dst);
}

bool detail::FromBytes(StreamReader& src, void* dst, TypeInfo const& dstInfo) noexcept {
  return dstInfo.DeserializeFromBytes(src, dst);
}

VecStreamWriter::VecStreamWriter() {
  // Reserve vector. Value chosen arbitrarily.
  dst.reserve(128);
}

VecStreamWriter::VecStreamWriter(size_t capacity) {
  dst.reserve(capacity);
}

bool VecStreamWriter::Write(void const* src, size_t numBytes) {
  auto const* srcBytes = reinterpret_cast<uint8_t const*>(src);
  dst.insert(dst.end(), srcBytes, srcBytes + numBytes);
  return true;
}

Span<const uint8_t> VecStreamWriter::GetBytes() const {
  return dst;
}

size_t VecStreamWriter::GetNumBytesWritten() const {
  return dst.size();
}

SpanStreamWriter::SpanStreamWriter(Span<uint8_t> buffer) : dst(buffer) {}

[[nodiscard]] bool SpanStreamWriter::Write(void const* src, size_t numBytes) {
  // Check size
  size_t dstBytesRemaining = dst.size() - iDst;
  if (numBytes > dstBytesRemaining) {
    return false;
  }

  std::memcpy(dst.data() + iDst, src, numBytes);
  iDst += numBytes;
  return true;
}

Span<const uint8_t> SpanStreamWriter::GetBytes() const {
  return {dst.data(), GetNumBytesWritten()};
}

size_t SpanStreamWriter::GetNumBytesWritten() const {
  return iDst;
}

SpanStreamReader::SpanStreamReader(Span<const uint8_t> buffer) : src(buffer) {}

// StreamReader interface
bool SpanStreamReader::Read(void* dst, size_t numBytes) {
  if (GetNumBytesRemaining() < numBytes) {
    return false;
  }

  std::memcpy(dst, src.data() + iSrc, numBytes);
  iSrc += numBytes;
  return true;
}

// Utilities
bool SpanStreamReader::HasUnreadBytes() const {
  return GetNumBytesRemaining() > 0;
}

size_t SpanStreamReader::GetNumBytesRead() const {
  return iSrc;
}

size_t SpanStreamReader::GetNumBytesRemaining() const {
  return src.size() - GetNumBytesRead();
}

} // namespace SReflect

// Static method
SReflect::StringTypeInfo const& SReflectTypeTraits<std::string>::GetTypeInfo() {
  static auto* s_typeInfo = SReflect::MakeStdStringTypeInfo();
  return *s_typeInfo;
}

#endif // SIMPLE_REFLECTION_ENABLE
