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

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#if !MOCHI_USE_REFLECTION
#error "This file requires MOCHI_USE_REFLECTION=1"
#endif

namespace mochi::net {

//--------------------------------------------------------------------------------------
// MessageDispatcher
//--------------------------------------------------------------------------------------

/**
 * @brief Routes message structs to per-type handlers, keyed by @ref SReflect::TypeId.
 *
 * @tparam Extra Leading argument types passed through to every handler before the message object.
 */
template <class... Extra>
class MessageDispatcher {
 public:
  /**
   * @brief Register a message class to receive via callback.
   *
   * @tparam MessageT The message class/struct type. Must support reflection.
   * @param onReceive Callback function/lambda of the form: void(Extra..., MessageT&&)
   *
   * @note It is illegal to call this function after @ref Dispatch has been called, so register all
   * your message types up front.
   */
  template <class MessageT, class Fn>
  void Register(Fn&& onReceive) {
    static_assert(
        std::is_invocable_v<Fn&, Extra..., MessageT&&>,
        "Register requires a function/lambda with arguments (Extra..., MessageT&&).");
    auto const& typeInfo = SReflect::GetTypeInfo<MessageT>();
    RegisterImpl(typeInfo, [fn = std::forward<Fn>(onReceive)](Extra... extra, void* data) {
      auto* msg = static_cast<MessageT*>(data);
      fn(extra..., std::move(*msg));
    });
  }

  /**
   * @brief Register a message class without a callback.
   *
   * @tparam MessageT The message class/struct type. Must support reflection.
   *
   * @details The type becomes known to @ref TryGetTypeInfo, but @ref Dispatch / @ref DispatchVoid
   * will not fire any handler and will return false for it.
   *
   * @note It is illegal to call this function after @ref Dispatch has been called, so register all
   * your message types up front.
   */
  template <class MessageT>
  void Register() {
    auto const& typeInfo = SReflect::GetTypeInfo<MessageT>();
    RegisterImpl(typeInfo, CallbackFn{});
  }

  /**
   * @brief Dispatch a message to the registered recipient.
   *
   * @tparam MessageT The message class/struct type. Must support reflection.
   * @param extra Extra arguments to pass to the receiver.
   * @param msg The message to pass to the receiver via move semantics.
   * @return True if a callback was fired. False if the message type was not registered, or was
   * registered without a callback.
   */
  template <class MessageT>
  [[nodiscard]] bool Dispatch(Extra... extra, MessageT&& msg) {
    static_assert(
        !std::is_lvalue_reference_v<MessageT>,
        "Dispatch requires an rvalue; use std::move if you have an lvalue.");
    using MessageType = std::decay_t<MessageT>;
    auto const typeId = SReflect::GetTypeId<MessageType>();
    return DispatchVoid(extra..., typeId, &msg);
  }

  /**
   * @brief Unsafe version of @ref Dispatch. Uses @ref SReflect::TypeId and void pointer (must
   * match).
   *
   * @param extra Extra arguments to pass to the receiver.
   * @param typeId Identifies the type of the final message class/struct.
   * @param ptr Address of a mutable message object of the specified type.
   * @return True if a callback was fired. False if the message type was not registered, or was
   * registered without a callback (see the no-argument @ref Register overload).
   *
   * @warning The message object will be passed to the recipient via move semantics. If this
   * function returns true, then the message object will be in a moved-from state.
   */
  [[nodiscard]] bool DispatchVoid(Extra... extra, SReflect::TypeId typeId, void* ptr) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    _registrationComplete = true; // Illegal to register more types after this
    MOCHI_ASSERT_VERBOSE(ptr != nullptr, "Null message");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
    auto it = _registry.find(typeId.value);
    if ((it != _registry.end()) && it->second.callback) {
      MOCHI_ASSERT_VERBOSE(it->second.typeInfo->_typeId == typeId, "Internal error. Type mismatch");
      it->second.callback(extra..., ptr);
      return true;
    }
    return false;
  }

  /** @brief Return true if no message types have been registered. */
  [[nodiscard]] bool IsEmpty() const {
    return _registry.empty();
  }

  /**
   * @brief Return the stored @ref SReflect::TypeInfo for a registered type, or nullptr if not
   * registered.
   */
  SReflect::TypeInfo const* TryGetTypeInfo(SReflect::TypeId typeId) const {
    auto it = _registry.find(typeId.value);
    return it != _registry.end() ? it->second.typeInfo : nullptr;
  }

  /**
   * @brief Return true if a callback was registered for the specified message type.
   */
  bool HasReceiver(SReflect::TypeId typeId) const {
    auto it = _registry.find(typeId.value);
    return it != _registry.end() ? !!it->second.callback : false;
  }

  /**
   * @brief Return a deterministic hash of the reflection metadata for all registered message types.
   */
  [[nodiscard]] uint64_t CalcProtocolVersionHash() const {
    DynamicArray<SReflect::TypeInfo const*> typeInfos;
    typeInfos.reserve(_registry.size());
    for (auto const& entry : _registry) {
      typeInfos.push_back(entry.second.typeInfo);
    }
    std::ranges::sort(typeInfos, [](auto const& a, auto const& b) {
      return std::string_view{a->_nameWithNamespace} < std::string_view{b->_nameWithNamespace};
    });
    std::string json =
        SReflect::TypeInfoListToJson(typeInfos.data(), typeInfos.size(), /*pretty*/ false);
    return SReflect::CalcHash64(json.c_str(), json.size());
  }

 private:
  using CallbackFn = std::function<void(Extra..., void* data)>;
  struct MessageInfo {
    SReflect::TypeInfo const* typeInfo = nullptr;
    CallbackFn callback;
  };

  void RegisterImpl(SReflect::TypeInfo const& typeInfo, CallbackFn&& callback) {
    MOCHI_ASSERT_VERBOSE(
        !_registrationComplete,
        "Illegal to register additional message types after the first message dispatch");
    auto& entry = _registry[typeInfo._typeId.value];
    MOCHI_ASSERT_VERBOSE(
        !entry.callback || !callback, "Overwriting existing message handler. Likely a bug.");
    entry.typeInfo = &typeInfo;
    if (callback) {
      entry.callback = std::move(callback);
    }
  }

  std::unordered_map<uint64_t, MessageInfo> _registry;
#if MOCHI_ASSERT_VERBOSE_ENABLED
  std::atomic<bool> _registrationComplete{false};
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
};

} // namespace mochi::net
