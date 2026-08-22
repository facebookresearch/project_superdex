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

#include <mochi_core/utils/dynamic_string.h>

#if MOCHI_USE_REFLECTION

SR_WARNING_PUSH()
SR_WARNING_IGNORE_MSVC(4459) // declaration of 'last' hides global declaration
#include <picojson/picojson.h> // picojson (third-party)
SR_WARNING_POP()

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mochi {
namespace {

class StringTypeInfo : public SReflect::StringTypeInfo {
 public:
  // Helpers:
  static DynamicString& Str(void* obj) {
    return *static_cast<DynamicString*>(obj);
  }
  static DynamicString const& Str(void const* obj) {
    return *static_cast<DynamicString const*>(obj);
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
    dst = std::string{Str(src)};
  }
  void DeserializeInner(
      picojson::value const& src,
      void* dst,
      SReflect::DeserializeFlags /*deserializeFlags*/,
      int& /*inOutIssuesDetected*/) const override {
    Str(dst) = DynamicString{src.get<std::string>()};
  }
  bool SerializeToBytesInner(void const* src, SReflect::StreamWriter& dst) const override {
    auto const& str = Str(src);
    auto len = static_cast<uint64_t>(str.size());
    bool ok = dst.Write(&len, sizeof(len));
    ok = ok && dst.Write(str.data(), str.size());
    return ok;
  }
  bool DeserializeFromBytesInner(SReflect::StreamReader& src, void* dst) const override {
    uint64_t len = 0;
    bool ok = src.Read(&len, sizeof(len));
    if (ok) {
      // TODO: Expose remaining bytes in SReflect::StreamReader so length-prefixed deserializers can
      // validate lengths before casting and allocating.
      DynamicString str;
      str.resize(static_cast<size_t>(len));
      ok = src.Read(str.data(), static_cast<size_t>(len));
      if (ok) {
        Str(dst) = std::move(str);
      }
    }
    return ok;
  }
};

} // namespace
} // namespace mochi

SReflect::StringTypeInfo const& SReflectTypeTraits<mochi::DynamicString>::GetTypeInfo() {
  static auto* s_typeInfo = [] {
    auto* ti = new mochi::StringTypeInfo;
    ti->_isReadOnly = false;
    ti->_isNullTerminated = true;
    ti->_coreType = SReflect::CoreType::CT_string;
    ti->_sizeInBytes = sizeof(mochi::DynamicString);
    ti->_alignment = alignof(mochi::DynamicString);
    ti->_nameWithNamespace = "mochi::DynamicString";
    ti->_name = "DynamicString";
    ti->_typeId = SReflect::ComputeTypeId(ti->_nameWithNamespace);
    VerifyTypeIdIsUnique(*ti, typeid(mochi::DynamicString));
    InitTypeInfoFunctionPointers<mochi::DynamicString>(ti);
    return ti;
  }();
  return *s_typeInfo;
}

#endif // MOCHI_USE_REFLECTION
