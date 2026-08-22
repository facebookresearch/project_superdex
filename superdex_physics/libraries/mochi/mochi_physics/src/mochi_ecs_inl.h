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

#include "mochi_ecs.h" // back-include for Intellisense

#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi::ecs {

/**************************************************************************
  Manipulating entt::type_list
*/

// Return an entt::type_list by concatenating two or more other type_lists
template <typename... Lists>
constexpr auto CatTypeLists(Lists...) {
  return entt::type_list_cat_t<Lists...>{};
}

// Given one or more entt::type_list, return a type_list of just the unique types.
template <typename... Lists>
constexpr auto GetUniqueTypes(Lists... lists) {
  return entt::type_list_unique_t<decltype(CatTypeLists(lists...))>{};
}

// Given an entt::type_list, return a type_list containing just the const types.
template <typename... ArgsT>
constexpr auto SelectConstTypes(entt::type_list<ArgsT...>) {
  return CatTypeLists(
      std::conditional_t<std::is_const_v<ArgsT>, entt::type_list<ArgsT>, entt::type_list<>>{}...);
}

// Given an entt::type_list, return a type_list containing just the non-const types.
template <typename... ArgsT>
constexpr auto SelectNonConstTypes(entt::type_list<ArgsT...>) {
  return CatTypeLists(
      std::conditional_t<std::is_const_v<ArgsT>, entt::type_list<>, entt::type_list<ArgsT>>{}...);
}

// Given an entt::type_list, return the const version of those same types
template <typename... ArgsT>
constexpr auto AddConst(entt::type_list<ArgsT...>) {
  return entt::type_list<std::add_const_t<ArgsT>...>{};
}

// Return an entt::type_list with all the types in both lists
template <typename... A, typename... B>
constexpr auto GetTypeListIntersection(entt::type_list<A...> a, entt::type_list<B...> /*b*/) {
  return GetUniqueTypes(
      std::conditional_t<
          entt::type_list_contains_v<decltype(a), B>,
          entt::type_list<B>,
          entt::type_list<>>{}...);
}

template <typename T, typename S1 = std::void_t<>, typename S2 = std::void_t<>>
struct UnderlyingValueType {
  using type = std::remove_reference_t<std::remove_pointer_t<T>>;
};
template <typename T>
struct UnderlyingValueType<T* const> {
  using type = T;
};
template <typename T>
struct UnderlyingValueType<std::reference_wrapper<T>> {
  using type = T;
};
template <typename T, typename S1> // specialization for smart pointers with T::operator*
struct UnderlyingValueType<T, S1, std::void_t<decltype(*std::declval<T>())>> {
  using type = typename std::remove_reference_t<decltype(*std::declval<T>())>;
};
template <typename T, typename S2> // specialization for containers with T::begin()
struct UnderlyingValueType<T, std::void_t<decltype(*std::declval<T>().begin())>, S2> {
  using type = typename std::remove_reference_t<decltype(*std::declval<T>().begin())>;
};

template <class T, class... Ts>
struct IsAny : std::disjunction<std::is_same<T, Ts>...> {};

// LooksMutable<T> is std::true_type if we can detect a way in which type T exposes mutable data
// access. Used to enforce the policy that per-entity systems should not receive mutable external
// parameters.
template <typename T>
using LooksMutable = std::integral_constant<
    bool,
    !std::is_const_v<typename UnderlyingValueType<T>::type> &&
        !std::is_same_v<T, typename UnderlyingValueType<T>::type>>;

// Return LookMutable<T> is true_type for any of the arguments. See LooksMutable, above.
template <typename... ArgsT>
constexpr bool LooksLikeMutableDataAccess() {
  return IsAny<std::true_type, LooksMutable<ArgsT>...>::value;
}

/**************************************************************************
  SystemDependency
*/

template <typename T>
inline SystemDependency SystemDependency::Write() {
  return SystemDependency{entt::type_id<T>(), AccessMode::Write};
}

template <typename T>
inline SystemDependency SystemDependency::Read() {
  return SystemDependency{entt::type_id<T>(), AccessMode::Read};
}

/**************************************************************************
  SystemParamInfo
*/

inline bool SystemParamInfo::IsOptional() const {
  return paramType == SystemParamType::ComponentReadOptional ||
      paramType == SystemParamType::ComponentWriteOptional ||
      paramType == SystemParamType::TagOptional ||
      paramType == SystemParamType::GlobalCtxReadOptional ||
      paramType == SystemParamType::GlobalCtxWriteOptional;
}

inline bool SystemParamInfo::IsComponent() const {
  return paramType == SystemParamType::ComponentRead ||
      paramType == SystemParamType::ComponentReadOptional ||
      paramType == SystemParamType::ComponentWriteOptional ||
      paramType == SystemParamType::ComponentWrite;
}

inline bool SystemParamInfo::IsGlobal() const {
  return paramType == SystemParamType::GlobalCtxRead ||
      paramType == SystemParamType::GlobalCtxWrite ||
      paramType == SystemParamType::GlobalCtxReadOptional ||
      paramType == SystemParamType::GlobalCtxWriteOptional;
}

inline bool SystemParamInfo::IsWildcard() const {
  return wildcard != AccessWildcard::None;
}

/**************************************************************************
  SystemInfo
*/

inline void SystemInfo::Build() {
  for (auto& param : parameters) {
    for (auto& dependency : param.dependencies) {
      if (dependency.mode == AccessMode::Write) {
        writes.insert(dependency.type);
      } else if (dependency.mode == AccessMode::Read) {
        reads.insert(dependency.type);
      } else {
        throw std::runtime_error("Dependency has invalid AccessMode!");
      }
    }

    wildcard = std::max(param.wildcard, wildcard);
  }

  // Make sure there are no types in writes that are duplicated in reads
  for (auto t : writes) {
    auto it = reads.find(t);
    if (it != reads.end()) {
      reads.erase(it);
    }
  }

  // If we have a wildcard, clear reads or writes respectively
  if (wildcard >= AccessWildcard::Read) {
    reads.clear();
  }
  if (wildcard >= AccessWildcard::Write) {
    writes.clear();
  }
}

inline SystemInfo SystemInfo::FromParamInfos(std::vector<SystemParamInfo>&& parameters) {
  SystemInfo info;
  info.parameters = std::move(parameters);
  info.Build();
  return info;
}

/**************************************************************************
  System
*/

struct System {
  std::function<void(entt::registry&)> invoke;
  std::function<SystemInfo()> getInfo;

  void operator()(entt::registry& reg) const {
    invoke(reg);
  }

  SystemInfo GetInfo() const {
    return getInfo();
  }
};

/**************************************************************************
  Base Traits for System Parameters
*/

// SystemParamFetchSingleFromRegistry determines how the type T is retrieved from an entity in the
// registry for a given parameter type of a system when a system is called from the invoker. By
// default, SystemParamFetchSingleFromRegistry throws a compile time error if a retrieval
// strategy is not specified.
template <typename T>
struct SystemParamFetchSingleFromRegistry {
  static T Get(entt::registry& /*reg*/, entt::entity /*e*/) {
    static_assert(
        std::is_void_v<T>, "No Matching specialization of SystemParamFetchSingleFromRegistry!");
  }
};

// Same as SystemParamFetchSingleFromRegistry, except that it retrieves the data from a view
// instead of a registry. This is in general more performant.
// This is used as the invoker when the type T has a non-trivial ParamViewIncludeDependencies
// trait.
template <typename T, typename ViewT>
struct SystemParamFetchSingleFromView {
  static T Get(ViewT /*view*/, entt::entity /*e*/) {
    static_assert(
        std::is_void_v<T>, "No Matching specialization of SystemParamFetchSingleFromView!");
  }
};

// Retrieves data globally from the registry without reference to a specific entity.
// This is used as the default fetch when the system is global and not per-entity.
template <typename T>
struct SystemParamFetchGlobal {
  static T Fetch(entt::registry& /*reg*/) {
    static_assert(std::is_void_v<T>, "No Matching specialization of SystemParamFetchGlobal!");
  }
};

// Trait to determine the include dependencies on the iteration view that a per-entity system
// implies
template <typename T>
struct ParamViewIncludeDependencies : entt::type_list<> {};

// Trait to determine the exclude dependencies on the iteration view that a per-entity system
// implies
template <typename T>
struct ParamViewExcludeDependencies : entt::type_list<> {};

// Trait to list all of the dependencies of a system parameter
template <typename T>
struct ParamDependencies : entt::type_list<> {};

// Whether or not this parameter invokes a wildcard and what type of wildcard
template <typename T>
struct ParamWildcardType : std::integral_constant<AccessWildcard, AccessWildcard::None> {};

// Converts a list of types to a system dependency vector in run-time
template <typename T>
struct TypeListToSystemDependencies;
template <typename... T>
struct TypeListToSystemDependencies<entt::type_list<T...>> {
  static std::vector<SystemDependency> Get() {
    return {std::is_const_v<T> ? SystemDependency::Read<T>() : SystemDependency::Write<T>()...};
  }
};

template <typename T>
struct ParamGetDependencies : TypeListToSystemDependencies<typename ParamDependencies<T>::type> {};

template <typename T>
struct ParamGetType : std::integral_constant<SystemParamType, SystemParamType::Unknown> {};

// Gets the information associated with a parameter
template <typename T>
struct ParamGetInfo {
  static SystemParamInfo GetInfo() {
    return SystemParamInfo{
        ParamGetDependencies<T>::Get(), ParamGetType<T>::value, ParamWildcardType<T>::value};
  }
};

template <typename... ArgsT>
struct GetSystemParamInfo {
  static std::vector<SystemParamInfo> Get() {
    return {ParamGetInfo<ArgsT>::GetInfo()...};
  }
};

// Get all access information metadata for a given system
template <typename... ArgsT>
SystemInfo GetSystemInfo(void (* /*system*/)(ArgsT...)) {
  auto infos = GetSystemParamInfo<ArgsT...>::Get();
  return SystemInfo::FromParamInfos(std::move(infos));
}

/**************************************************************************
    System Parameter Trait Specializations
*/

// For a constant pointer, the access mode is read and the component type
// is treated as optional.
template <typename T>
struct SystemParamFetchSingleFromRegistry<T const*> {
  static T const* Get(entt::registry const& reg, entt::entity e) {
    return reg.try_get<T const>(e);
  }
};

template <typename T>
struct SystemParamFetchGlobal<T const*> {
  static T const* Get(entt::registry const& reg) {
    return reg.try_ctx<T const>();
  }
};

template <typename T>
struct ParamGetType<T const*>
    : std::integral_constant<SystemParamType, SystemParamType::ComponentReadOptional> {};
template <typename T>
struct ParamDependencies<T const*> : entt::type_list<T const> {};
template <typename T>
struct ParamViewIncludeDependencies<T const*> : entt::type_list<> {};
template <typename T>
struct ParamViewExcludeDependencies<T const*> : entt::type_list<> {};

// For a constant reference, the access mode is read and the component type
// is not optional
template <typename T>
struct SystemParamFetchSingleFromRegistry<T const&> {
  static T const& Get(entt::registry const& reg, entt::entity e) {
    return reg.template get<T const>(e);
  }
};

template <typename T, typename... IncludeT, typename... ExcludeT>
struct SystemParamFetchSingleFromView<
    T const&,
    ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>> {
  static T const& Get(ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>> view, entt::entity e) {
    return view.template get<T const>(e);
  }
};

template <typename T>
struct SystemParamFetchGlobal<T const&> {
  static T const& Get(entt::registry const& reg) {
    return reg.template ctx<T const>();
  }
};

template <typename T>
struct ParamGetType<T const&>
    : std::integral_constant<SystemParamType, SystemParamType::ComponentRead> {};

template <typename T>
struct ParamDependencies<T const&> : entt::type_list<T const> {};
template <typename T>
struct ParamViewIncludeDependencies<T const&> : entt::type_list<T const> {};
template <typename T>
struct ParamViewExcludeDependencies<T const&> : entt::type_list<> {};

// For a pointer, the access mode is write and the component type
// is optional
template <typename T>
struct SystemParamFetchSingleFromRegistry<T*> {
  static T* Get(entt::registry& reg, entt::entity e) {
    return reg.template try_get<T>(e);
  }
};

template <typename T>
struct SystemParamFetchGlobal<T*> {
  static T* Get(entt::registry& reg) {
    return reg.template try_ctx<T>();
  }
};

template <typename T>
struct ParamGetType<T*>
    : std::integral_constant<SystemParamType, SystemParamType::ComponentWriteOptional> {};

template <typename T>
struct ParamDependencies<T*> : entt::type_list<T> {};
template <typename T>
struct ParamViewIncludeDependencies<T*> : entt::type_list<> {};
template <typename T>
struct ParamViewExcludeDependencies<T*> : entt::type_list<> {};

// For a reference, the access mode is write and the component type
// is not optional
template <typename T>
struct SystemParamFetchSingleFromRegistry<T&> {
  static T& Get(entt::registry& reg, entt::entity e) {
    return reg.template get<T>(e);
  }
};

template <typename T, typename... IncludeT, typename... ExcludeT>
struct SystemParamFetchSingleFromView<T&, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>> {
  static T& Get(ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>> view, entt::entity e) {
    return view.template get<T>(e);
  }
};

template <typename T>
struct SystemParamFetchGlobal<T&> {
  static T& Get(entt::registry& reg) {
    return reg.template ctx<T>();
  }
};

template <typename T>
struct ParamGetType<T&> : std::integral_constant<SystemParamType, SystemParamType::ComponentWrite> {
};

template <typename T>
struct ParamDependencies<T&> : entt::type_list<T> {};
template <typename T>
struct ParamViewIncludeDependencies<T&> : entt::type_list<T> {};
template <typename T>
struct ParamViewExcludeDependencies<T&> : entt::type_list<> {};

// For an OptionalTag, the access mode is read and a boolean flag is
// returned (wrapped in OptionalTag) whose value is whether or not
// the entity has a tag of type T.
template <typename T>
struct SystemParamFetchSingleFromRegistry<OptionalTag<T>> {
  static OptionalTag<T> Get(entt::registry const& reg, entt::entity e) {
    return OptionalTag<T>(reg.all_of<T>(e));
  }
};

template <typename T>
struct ParamGetType<OptionalTag<T>>
    : std::integral_constant<SystemParamType, SystemParamType::TagOptional> {};

template <typename T>
struct ParamDependencies<OptionalTag<T>> : entt::type_list<std::add_const_t<T>> {};

// For an Required, the access mode is read and an empty structure is returned
template <typename... T>
struct SystemParamFetchSingleFromRegistry<Included<T...>> {
  static Included<T...> Get(entt::registry const& /*reg*/, entt::entity /*e*/) {
    return Included<T...>{};
  }
};

template <typename ViewT, typename... T>
struct SystemParamFetchSingleFromView<Included<T...>, ViewT> {
  static Included<T...> Get(ViewT /*view*/, entt::entity /*e*/) {
    return Included<T...>{};
  }
};

template <typename... T>
struct ParamGetType<Included<T...>>
    : std::integral_constant<SystemParamType, SystemParamType::Required> {};

template <typename... T>
struct ParamDependencies<Included<T...>> : entt::type_list<std::add_const_t<T>...> {};
template <typename... T>
struct ParamViewIncludeDependencies<Included<T...>> : entt::type_list<std::add_const_t<T>...> {};
template <typename... T>
struct ParamViewExcludeDependencies<Included<T...>> : entt::type_list<> {};

// For an Excluded, the access mode is read and an empty structure is returned
template <typename... T>
struct SystemParamFetchSingleFromRegistry<Excluded<T...>> {
  static Excluded<T...> Get(entt::registry const& /*reg*/, entt::entity /*e*/) {
    return Excluded<T...>{};
  }
};

template <typename... T>
struct ParamGetType<Excluded<T...>>
    : std::integral_constant<SystemParamType, SystemParamType::Required> {};

template <typename... T>
struct ParamDependencies<Excluded<T...>> : entt::type_list<std::add_const_t<T>...> {};
template <typename... T>
struct ParamViewIncludeDependencies<Excluded<T...>> : entt::type_list<> {};
template <typename... T>
struct ParamViewExcludeDependencies<Excluded<T...>> : entt::type_list<T...> {};

// For a registry, the access mode is write, and we simple return
// a reference to the full registry.
template <>
struct SystemParamFetchSingleFromRegistry<entt::registry&> {
  template <typename RegistryT>
  static inline entt::registry& Get(RegistryT& reg, entt::entity /*e*/) {
    return reg;
  }
};

template <>
struct SystemParamFetchGlobal<entt::registry&> {
  static inline entt::registry& Get(entt::registry& reg) {
    return reg;
  }
};

template <>
struct ParamGetType<entt::registry&>
    : std::integral_constant<SystemParamType, SystemParamType::RegistryFullWrite> {};

template <>
struct ParamWildcardType<entt::registry&>
    : std::integral_constant<AccessWildcard, AccessWildcard::Write> {};

template <>
struct ParamDependencies<entt::registry&> : entt::type_list<> {};
template <>
struct ParamViewIncludeDependencies<entt::registry&> : entt::type_list<> {};
template <>
struct ParamViewExcludeDependencies<entt::registry&> : entt::type_list<> {};

// Same as above, except access is read-only.
template <>
struct SystemParamFetchSingleFromRegistry<entt::registry const&> {
  static inline entt::registry const& Get(entt::registry& reg, entt::entity /*e*/) {
    return reg;
  }
};

template <>
struct SystemParamFetchGlobal<entt::registry const&> {
  static inline entt::registry const& Get(entt::registry& reg) {
    return reg;
  }
};

template <>
struct ParamGetType<entt::registry const&>
    : std::integral_constant<SystemParamType, SystemParamType::RegistryFullRead> {};

template <>
struct ParamWildcardType<entt::registry const&>
    : std::integral_constant<AccessWildcard, AccessWildcard::Read> {};

template <>
struct ParamDependencies<entt::registry const&> : entt::type_list<> {};
template <>
struct ParamViewIncludeDependencies<entt::registry const&> : entt::type_list<> {};
template <>
struct ParamViewExcludeDependencies<entt::registry const&> : entt::type_list<> {};

// If the system requests an entity id, then we simply pass through the current
// entity.
template <>
struct SystemParamFetchSingleFromRegistry<entt::entity> {
  static inline entt::entity Get(entt::registry& /*reg*/, entt::entity e) {
    return e;
  }
};

template <>
struct SystemParamFetchGlobal<entt::entity> {
  template <typename RegistryT>
  static inline entt::entity Get(RegistryT& /*reg*/) {
    static_assert(std::is_void_v<RegistryT>, "Access to entity id not allowed in a global system!");
    return {};
  }
};

template <>
struct ParamGetType<entt::entity>
    : std::integral_constant<SystemParamType, SystemParamType::EntityId> {};

// If the system requests a CtxGlobal, we query the registry context. Note that
// this access is read-only.
template <typename T>
struct SystemParamFetchSingleFromRegistry<CtxGlobal<T const>> {
  static inline CtxGlobal<T const> Get(entt::registry& reg, entt::entity /*e*/) {
    return CtxGlobal<T const>(reg.ctx<T const>());
  }
};

template <typename T>
struct SystemParamFetchGlobal<CtxGlobal<T const>> {
  static inline CtxGlobal<T const> Get(entt::registry& reg) {
    return CtxGlobal<T const>(reg.ctx<T const>());
  }
};

template <typename T>
struct ParamGetType<CtxGlobal<T const>>
    : std::integral_constant<SystemParamType, SystemParamType::GlobalCtxRead> {};

template <typename T>
struct ParamDependencies<CtxGlobal<T const>> : entt::type_list<T const> {};
template <typename T>
struct ParamViewIncludeDependencies<CtxGlobal<T const>> : entt::type_list<> {};
template <typename T>
struct ParamViewExcludeDependencies<CtxGlobal<T const>> : entt::type_list<> {};

// If the system requests a OptionalCtxGlobal, we query the registry context. Note that
// this access is read-only.
template <typename T>
struct SystemParamFetchSingleFromRegistry<OptionalCtxGlobal<T const>> {
  static inline OptionalCtxGlobal<T const> Get(entt::registry& reg, entt::entity /*e*/) {
    return OptionalCtxGlobal<T const>(reg.try_ctx<T const>());
  }
};

template <typename T>
struct SystemParamFetchGlobal<OptionalCtxGlobal<T const>> {
  static inline OptionalCtxGlobal<T const> Get(entt::registry& reg) {
    return OptionalCtxGlobal<T const>(reg.try_ctx<T const>());
  }
};

template <typename T>
struct ParamGetType<OptionalCtxGlobal<T const>>
    : std::integral_constant<SystemParamType, SystemParamType::GlobalCtxReadOptional> {};

template <typename T>
struct ParamDependencies<OptionalCtxGlobal<T const>> : entt::type_list<T const> {};
template <typename T>
struct ParamViewIncludeDependencies<OptionalCtxGlobal<T const>> : entt::type_list<> {};
template <typename T>
struct ParamViewExcludeDependencies<OptionalCtxGlobal<T const>> : entt::type_list<> {};

// Creating a context global in write mode is invalid, otherwise multiple entities
// might write to the same object.
template <typename T>
struct SystemParamFetchSingleFromRegistry<CtxGlobal<T>> {
  template <typename RegistryT>
  static inline CtxGlobal<T> Get(RegistryT& /*reg*/, entt::entity /*e*/) {
    static_assert(
        std::is_void_v<RegistryT>,
        "Per-entity systems are not allowed to write to a CtxGlobal. Please make it const. Example: CtxGlobal<ComponentName const>");
  }
};

template <typename T>
struct SystemParamFetchGlobal<CtxGlobal<T>> {
  static inline CtxGlobal<T> Get(entt::registry& reg) {
    return CtxGlobal<T>(reg.ctx<T>());
  }
};

template <typename T>
struct ParamGetType<CtxGlobal<T>>
    : std::integral_constant<SystemParamType, SystemParamType::GlobalCtxWrite> {};

template <typename T>
struct ParamDependencies<CtxGlobal<T>> : entt::type_list<T> {};
template <typename T>
struct ParamViewIncludeDependencies<CtxGlobal<T>> : entt::type_list<> {};
template <typename T>
struct ParamViewExcludeDependencies<CtxGlobal<T>> : entt::type_list<> {};

// Creating a context global in write mode is invalid, otherwise multiple entities
// might write to the same object.
template <typename T>
struct SystemParamFetchSingleFromRegistry<OptionalCtxGlobal<T>> {
  template <typename RegistryT>
  static inline OptionalCtxGlobal<T> Get(RegistryT& /*reg*/, entt::entity /*e*/) {
    static_assert(
        std::is_void_v<RegistryT>, "Cannot write to a CtxGlobal in a single-entity system!");
  }
};

template <typename T>
struct SystemParamFetchGlobal<OptionalCtxGlobal<T>> {
  static inline OptionalCtxGlobal<T> Get(entt::registry& reg) {
    return OptionalCtxGlobal<T>(reg.try_ctx<T>());
  }
};

template <typename T>
struct ParamGetType<OptionalCtxGlobal<T>>
    : std::integral_constant<SystemParamType, SystemParamType::GlobalCtxWriteOptional> {};

template <typename T>
struct ParamDependencies<OptionalCtxGlobal<T>> : entt::type_list<T> {};
template <typename T>
struct ParamViewIncludeDependencies<OptionalCtxGlobal<T>> : entt::type_list<> {};
template <typename T>
struct ParamViewExcludeDependencies<OptionalCtxGlobal<T>> : entt::type_list<> {};

/**************************************************************************
  Access Verification Metafunctions
*/

// A metafunction for determining whether RequestedT is a valid access a single allowed access
// of type AllowedT. The result is true if RequestedT is either AllowedT or const AllowedT.
// If AllowedT is a WildcardWrite, the result is always true.
// If AllowedT is a WildcardRead, the result is true if RequestedT is const
template <typename RequestedT, typename AllowedT>
struct VerifyAccessToTypeSingle : IsAny<RequestedT, std::add_const_t<AllowedT>, AllowedT> {};
template <typename RequestedT>
struct VerifyAccessToTypeSingle<RequestedT, WildcardRead> : std::is_const<RequestedT> {};
template <typename RequestedT>
struct VerifyAccessToTypeSingle<RequestedT, WildcardWrite> : std::true_type {};
template <>
struct VerifyAccessToTypeSingle<WildcardRead, WildcardRead> : std::true_type {};
template <>
struct VerifyAccessToTypeSingle<WildcardWrite, WildcardRead> : std::false_type {};

// A metafunction for determining whether T is a valid access given the entt::type_list
// TokenT. Simply checks VerifyAccessToTypeSingle for every type in the token
template <typename RequestedT, typename AllowedToken>
struct VerifyAccessToType;

template <typename RequestedT, typename... AllowedT>
struct VerifyAccessToType<RequestedT, entt::type_list<AllowedT...>>
    : std::disjunction<VerifyAccessToTypeSingle<RequestedT, AllowedT>...> {};

// A metafunction for determining whether a requested access token is allowed by
// another access token. Every type in the requested token must be a valid access for this
// to be true.
template <typename RequestedToken, typename AllowedToken>
struct VerifyAccessToken;

template <typename... RequestedT, typename... AllowedT>
struct VerifyAccessToken<entt::type_list<RequestedT...>, entt::type_list<AllowedT...>>
    : std::conjunction<VerifyAccessToType<RequestedT, entt::type_list<AllowedT...>>...> {};

template <typename Requested, typename Allowed>
constexpr bool IsAccessTokenValid_v = VerifyAccessToken<Requested, Allowed>::value;

// Metafunction to determine if a type is accessible via a view
template <typename T, typename ViewT>
struct IsParamAccessibleFromView;

// The ParamViewIncludeDependency list must not be empty, and all types must be a valid access
// with respect to the types of the view.
template <typename T, typename... IncludeT, typename... ExcludeT>
struct IsParamAccessibleFromView<T, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : std::conjunction<
          std::negation<
              std::is_same<typename ParamViewIncludeDependencies<T>::type, entt::type_list<>>>,
          VerifyAccessToken<
              typename ParamViewIncludeDependencies<T>::type,
              entt::type_list<IncludeT...>>> {};

// Metafunction to determine if a type is excluded from a view
template <typename T, typename ViewT>
struct IsParamExcludedFromView;

template <typename T, typename... IncludeT, typename... ExcludeT>
struct IsParamExcludedFromView<T, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : VerifyAccessToken<
          typename ParamViewIncludeDependencies<T>::type,
          entt::type_list<ExcludeT...>> {};

// Return an entt::type_list with all of the component types that the system reads or writes,
// including types referenced within PartialRegistry, View, CtxGlobal, etc...
template <typename... ArgsT>
constexpr auto GetAllParamDependencies() {
  return GetUniqueTypes(typename ParamDependencies<ArgsT>::type{}...);
}

// Return an entt::type_list with all the component types that the system reads, including component
// types used referenced PartialRegistry, View, and CtxGlobal. Does not include WildcardRead.
template <typename... ArgsT>
constexpr auto GetReadParamDependencies() {
  return SelectConstTypes(GetAllParamDependencies<ArgsT...>());
}

// Return an entt::type_list with all the component types that the system writes, including
// component types used referenced PartialRegistry, View, and CtxGlobal. Does not include
// WildcardWrite.
template <typename... ArgsT>
constexpr auto GetWriteParamDependencies() {
  return SelectNonConstTypes(GetAllParamDependencies<ArgsT...>());
}

// Return true if one of the arguments is of type 'entt::registry&' or 'entt::registry const&'
template <typename... ArgsT>
constexpr bool HasRegistryReference() {
  return entt::type_list_contains_v<entt::type_list<ArgsT...>, entt::registry&> ||
      entt::type_list_contains_v<entt::type_list<ArgsT...>, entt::registry const&>;
}

// Deduce the argument types
template <typename... ArgsT>
constexpr bool SystemUsesFullRegistry(void (*)(ArgsT...)) {
  return HasRegistryReference<ArgsT...>();
}

// Return false if the system writes to any component that it also reads. Single entity systems
// should always return true. This ensures that the behavior is order-independent, and safe to
// parallelize.
template <typename... ArgsT>
constexpr bool AreReadsAndWritesIndependent() {
  constexpr auto kReadDeps = GetReadParamDependencies<ArgsT...>();
  constexpr auto kWriteDeps = GetWriteParamDependencies<ArgsT...>();
  constexpr auto kConflicts = GetTypeListIntersection(kReadDeps, AddConst(kWriteDeps));
  return (kConflicts.size == 0);
}

/**************************************************************************
  Partial Registry Access Wrapper
*/

template <typename... AccessT>
class PartialRegistryRead;

template <typename... AccessT>
class PartialRegistryWrite {
 private:
  entt::registry& _registry;

 public:
  using _access_token = entt::type_list<AccessT...>;

  PartialRegistryWrite(entt::registry& reg) : _registry(reg) {}

  template <typename... OtherAccessT>
  PartialRegistryWrite(PartialRegistryWrite<OtherAccessT...> reg) : _registry(reg._registry) {
    static_assert(
        IsAccessTokenValid_v<
            _access_token,
            typename PartialRegistryWrite<OtherAccessT...>::_access_token>,
        "You cannot downcast a partial registry!");
  }

  template <typename Component, typename... Args>
  inline decltype(auto) emplace(entt::entity const entity, Args&&... args) {
    using requested_access = entt::type_list<std::remove_const_t<Component>>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to emplace this component (write-access necessary!)");
    return _registry.template emplace<Component, Args...>(entity, std::forward<Args>(args)...);
  }

  template <typename Component, typename... Args>
  inline decltype(auto) emplace_or_replace(entt::entity const entity, Args&&... args) {
    using requested_access = entt::type_list<std::remove_const_t<Component>>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to emplace this component (write-access necessary!)");
    return _registry.template emplace_or_replace<Component, Args...>(
        entity, std::forward<Args>(args)...);
  }

  inline decltype(auto) destroy(entt::entity const entity) {
    using requested_access = entt::type_list<WildcardWrite>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to emplace this component (WildcardWrite necessary!)");
    return _registry.destroy(entity);
  }

  template <typename It>
  inline decltype(auto) destroy(It first, It last) {
    using requested_access = entt::type_list<WildcardWrite>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to emplace this component (WildcardWrite necessary!)");
    return _registry.template destroy<It>(first, last);
  }

  template <typename Component, typename... Func>
  inline decltype(auto) patch(entt::entity const entity, Func&&... func) {
    using requested_access = entt::type_list<std::remove_const_t<Component>>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to emplace this component (write-access necessary!)");
    return _registry.template patch<Component, Func...>(entity, std::forward<Func>(func)...);
  }

  template <typename Component, typename... Args>
  inline decltype(auto) replace(entt::entity const entity, Args&&... args) {
    using requested_access = entt::type_list<std::remove_const_t<Component>>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to emplace this component (write-access necessary!)");
    return _registry.template replace<Component, Args...>(entity, std::forward<Args>(args)...);
  }

  template <typename Component, typename... Other>
  inline decltype(auto) remove(entt::entity const entity) {
    using requested_access =
        entt::type_list<std::remove_const_t<Component>, std::remove_const_t<Other>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to remove this component (write-access necessary!)");
    return _registry.template remove<Component, Other...>(entity);
  }

  template <typename Component, typename... Other, typename It>
  inline decltype(auto) remove(It first, It last) {
    using requested_access =
        entt::type_list<std::remove_const_t<Component>, std::remove_const_t<Other>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to remove this component (write-access necessary!)");
    return _registry.template remove<Component, Other...>(first, last);
  }

  template <typename Component, typename... Other>
  inline decltype(auto) erase(entt::entity const entity) {
    using requested_access =
        entt::type_list<std::remove_const_t<Component>, std::remove_const_t<Other>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to erase this component (write-access necessary!)");
    return _registry.template erase<Component, Other...>(entity);
  }

  template <typename Component, typename... Other, typename It>
  inline decltype(auto) erase(It first, It last) {
    using requested_access =
        entt::type_list<std::remove_const_t<Component>, std::remove_const_t<Other>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to erase this component (write-access necessary!)");
    return _registry.template erase<Component, Other...>(first, last);
  }

  template <typename... Component>
  [[nodiscard]] inline decltype(auto) all_of(entt::entity const entity) const {
    using requested_access = entt::type_list<std::add_const_t<Component>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to check for this component (read-access necessary!)");
    return _registry.template all_of<Component...>(entity);
  }

  template <typename... Component>
  [[nodiscard]] inline decltype(auto) any_of(entt::entity const entity) const {
    using requested_access = entt::type_list<std::add_const_t<Component>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to check for this component (read-access necessary!)");
    return _registry.template any_of<Component...>(entity);
  }

  template <typename... Component>
  [[nodiscard]] decltype(auto) get([[maybe_unused]] entt::entity const entity) {
    using requested_access = entt::type_list<Component...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this component!");
    return _registry.template get<Component...>(entity);
  }

  template <typename Component, typename... Args>
  [[nodiscard]] decltype(auto) get_or_emplace(entt::entity const entity, Args&&... args) {
    using requested_access =
        entt::type_list<std::remove_const_t<Component>, std::remove_const_t<Args>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this component!");
    return _registry.template get_or_emplace<Component, Args...>(
        entity, std::forward<Args>(args)...);
  }

  template <typename... Component>
  [[nodiscard]] decltype(auto) try_get([[maybe_unused]] entt::entity const entity) {
    using requested_access = entt::type_list<Component...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this component!");
    return _registry.template try_get<Component...>(entity);
  }

  template <typename Component, typename... Other>
  decltype(auto) clear() {
    using requested_access =
        entt::type_list<std::remove_const_t<Component>, std::remove_const_t<Other>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this component!");
    return _registry.template clear<Component, Other...>();
  }

  template <typename Component, typename... Other, typename... Exclude>
  [[nodiscard]] decltype(auto) view(entt::exclude_t<Exclude...> /*param*/ = {}) {
    using requested_access = entt::type_list<Component, Other..., std::add_const_t<Exclude>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to create this view!");
    return _registry.template view<Component, Other...>(entt::exclude_t<Exclude...>());
  }

  template <typename Type, typename... Args>
  Type& set(Args&&... args) {
    using requested_access = entt::type_list<WildcardWrite>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "Adding/Removing items from global context requires WildcardWrite access!");
    return _registry.template set<Type, Args...>(std::forward<Args>(args)...);
  }

  template <typename Type>
  void unset() {
    using requested_access = entt::type_list<WildcardWrite>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "Adding/Removing items from global context requires WildcardWrite access!");
    return _registry.template unset<Type>();
  }

  template <typename Type, typename... Args>
  [[nodiscard]] Type& ctx_or_set(Args&&... args) {
    using requested_access = entt::type_list<WildcardWrite>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "Adding/Removing items from global context requires WildcardWrite access!");
    return _registry.template ctx_or_set<Type, Args...>(std::forward<Args>(args)...);
  }

  template <typename Type>
  [[nodiscard]] decltype(auto) try_ctx() {
    using requested_access = entt::type_list<Type>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this context variable!");
    return _registry.template try_ctx<Type>();
  }

  template <typename Type>
  [[nodiscard]] decltype(auto) ctx() {
    using requested_access = entt::type_list<Type>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this context variable!");
    return _registry.template ctx<Type>();
  }

  template <typename... OtherT>
  friend class PartialRegistryWrite;
  template <typename... OtherT>
  friend class PartialRegistryRead;
};

template <typename... AccessT>
class PartialRegistryRead {
 protected:
  entt::registry const& _registry;

 public:
  using _access_token = entt::type_list<AccessT...>;

  PartialRegistryRead(entt::registry const& reg) : _registry(reg) {}

  template <typename... OtherAccessT>
  PartialRegistryRead(PartialRegistryRead<OtherAccessT...> reg) : _registry(reg._registry) {
    static_assert(
        IsAccessTokenValid_v<
            _access_token,
            typename PartialRegistryRead<OtherAccessT...>::_access_token>,
        "You cannot downcast a partial registry!");
  }

  template <typename... OtherAccessT>
  PartialRegistryRead(PartialRegistryWrite<OtherAccessT...> reg) : _registry(reg._registry) {
    static_assert(
        IsAccessTokenValid_v<
            _access_token,
            typename PartialRegistryWrite<OtherAccessT...>::_access_token>,
        "You cannot downcast a partial registry!");
  }

  template <typename... Component>
  [[nodiscard]] inline decltype(auto) all_of(entt::entity const entity) const {
    using requested_access = entt::type_list<std::add_const_t<Component>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to check for this component (read-access necessary!)");
    return _registry.template all_of<Component...>(entity);
  }

  template <typename... Component>
  [[nodiscard]] inline decltype(auto) any_of(entt::entity const entity) const {
    using requested_access = entt::type_list<std::add_const_t<Component>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have the necessary permission to check for this component (read-access necessary!)");
    return _registry.template any_of<Component...>(entity);
  }

  template <typename... Component>
  [[nodiscard]] decltype(auto) get([[maybe_unused]] entt::entity const entity) const {
    using requested_access = entt::type_list<Component...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this component!");
    return _registry.template get<Component...>(entity);
  }

  template <typename... Component>
  [[nodiscard]] decltype(auto) try_get([[maybe_unused]] entt::entity const entity) const {
    using requested_access = entt::type_list<Component...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this component!");
    return _registry.template try_get<Component...>(entity);
  }

  template <typename Component, typename... Other, typename... Exclude>
  [[nodiscard]] decltype(auto) view(entt::exclude_t<Exclude...> /*param*/ = {}) {
    using requested_access = entt::type_list<Component, Other..., std::add_const_t<Exclude>...>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to create this view!");
    return _registry.template view<Component, Other...>(entt::exclude_t<Exclude...>());
  }

  template <typename Type>
  [[nodiscard]] decltype(auto) try_ctx() const {
    using requested_access = entt::type_list<Type>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this context variable!");
    return _registry.template try_ctx<Type>();
  }

  template <typename Type>
  [[nodiscard]] decltype(auto) ctx() const {
    using requested_access = entt::type_list<Type>;
    static_assert(
        IsAccessTokenValid_v<requested_access, _access_token>,
        "You do not have permission to access this context variable!");
    return _registry.template ctx<Type>();
  }

  template <typename... OtherT>
  friend class PartialRegistryWrite;
  template <typename... OtherT>
  friend class PartialRegistryRead;
};

template <typename... AccessT>
constexpr bool IsAccessConst_v =
    IsAccessTokenValid_v<entt::type_list<AccessT...>, entt::type_list<WildcardRead>>;

/*
    A thin wrapper around entt::registry that verifies that the user actually has the required
    permissions to access the component types given by AccessT. To specify a wildcard use either
    WildcardRead or WildcardWrite as type parameters. Note that this may have severe performance
    implications in the future.
*/
template <typename... AccessT>
using PartialRegistry = std::conditional_t<
    IsAccessConst_v<AccessT...>,
    PartialRegistryRead<AccessT...>,
    PartialRegistryWrite<AccessT...>>;

// Metafunction to assert that the given list of access tokens has no writes
template <typename... AccessT>
struct IsNoWrite : std::conjunction<std::is_const<AccessT>...> {};

template <typename... AccessT>
constexpr bool IsNoWrite_v = IsNoWrite<AccessT...>::value;

// System parameters and fetching for partial registries
template <typename... AccessT>
struct SystemParamFetchSingleFromRegistry<PartialRegistryRead<AccessT...>> {
  static inline PartialRegistryRead<AccessT...> Get(entt::registry& reg, entt::entity /*e*/) {
    static_assert(
        ParamWildcardType<PartialRegistryRead<AccessT...>>::value == AccessWildcard::None,
        "Per-entity sytems are not allowed to use WildCardRead nor WildcardWrite");
    return PartialRegistryRead<AccessT...>(reg);
  }
};

template <typename... AccessT>
struct SystemParamFetchGlobal<PartialRegistryRead<AccessT...>> {
  static inline PartialRegistryRead<AccessT...> Get(entt::registry& reg) {
    return PartialRegistryRead<AccessT...>(reg);
  }
};

template <typename... AccessT>
struct ParamGetType<PartialRegistryRead<AccessT...>>
    : std::integral_constant<SystemParamType, SystemParamType::RegistryPartialRead> {};

template <typename... AccessT>
struct ParamDependencies<PartialRegistryRead<AccessT...>>
    : entt::type_list_diff<
          entt::type_list<AccessT...>,
          entt::type_list<WildcardRead, WildcardWrite>> {};
template <typename... AccessT>
struct ParamViewIncludeDependencies<PartialRegistryRead<AccessT...>> : entt::type_list<> {};
template <typename... AccessT>
struct ParamViewExcludeDependencies<PartialRegistryRead<AccessT...>> : entt::type_list<> {};

template <typename... AccessT>
struct ParamWildcardType<PartialRegistryRead<AccessT...>>
    : std::conditional_t<
          IsAny<WildcardWrite, AccessT...>::value,
          std::integral_constant<AccessWildcard, AccessWildcard::Write>,
          std::conditional_t<
              IsAny<WildcardRead, AccessT...>::value,
              std::integral_constant<AccessWildcard, AccessWildcard::Read>,
              std::integral_constant<AccessWildcard, AccessWildcard::None>>> {};

template <typename... AccessT>
struct SystemParamFetchSingleFromRegistry<PartialRegistryWrite<AccessT...>> {
  static inline PartialRegistryWrite<AccessT...> Get(entt::registry& reg, entt::entity /*e*/) {
    static_assert(
        IsAccessConst_v<AccessT...>,
        "Per-entity systems are not allowed to write to components via PartialRegistry. Please add const. Example: PartialRegistry<ComponentName const>");
    return PartialRegistryWrite<AccessT...>(reg);
  }
};

template <typename... AccessT>
struct SystemParamFetchGlobal<PartialRegistryWrite<AccessT...>> {
  static inline PartialRegistryWrite<AccessT...> Get(entt::registry& reg) {
    return PartialRegistryWrite<AccessT...>(reg);
  }
};

template <typename... AccessT>
struct ParamGetType<PartialRegistryWrite<AccessT...>>
    : std::integral_constant<SystemParamType, SystemParamType::RegistryPartialWrite> {};

template <typename... AccessT>
struct ParamDependencies<PartialRegistryWrite<AccessT...>>
    : entt::type_list_diff<
          entt::type_list<AccessT...>,
          entt::type_list<WildcardRead, WildcardWrite>> {};
template <typename... AccessT>
struct ParamViewIncludeDependencies<PartialRegistryWrite<AccessT...>> : entt::type_list<> {};
template <typename... AccessT>
struct ParamViewExcludeDependencies<PartialRegistryWrite<AccessT...>> : entt::type_list<> {};

template <typename... AccessT>
struct ParamWildcardType<PartialRegistryWrite<AccessT...>>
    : std::conditional_t<
          IsAny<WildcardWrite, AccessT...>::value,
          std::integral_constant<AccessWildcard, AccessWildcard::Write>,
          std::conditional_t<
              IsAny<WildcardRead, AccessT...>::value,
              std::integral_constant<AccessWildcard, AccessWildcard::Read>,
              std::integral_constant<AccessWildcard, AccessWildcard::None>>> {};

/**************************************************************************
    System parameters and fetching for ENTT views
*/
template <typename... AccessT, typename... ExcludeT>
struct SystemParamFetchSingleFromRegistry<
    entt::basic_view<entt::entity, entt::get_t<AccessT...>, entt::exclude_t<ExcludeT...>>> {
  static inline entt::
      basic_view<entt::entity, entt::get_t<AccessT...>, entt::exclude_t<ExcludeT...>>
      Get(entt::registry& reg, entt::entity /*e*/) {
    static_assert(
        IsNoWrite_v<AccessT...>,
        "Per-entity systems are not allowed to write to components via View. Please add const. Example: View<ComponentName const>");
    return reg.view<AccessT...>(entt::exclude_t<ExcludeT...>{});
  }
};

template <typename... AccessT, typename... ExcludeT>
struct SystemParamFetchGlobal<
    entt::basic_view<entt::entity, entt::get_t<AccessT...>, entt::exclude_t<ExcludeT...>>> {
  static inline entt::
      basic_view<entt::entity, entt::get_t<AccessT...>, entt::exclude_t<ExcludeT...>>
      Get(entt::registry& reg) {
    return reg.view<AccessT...>(entt::exclude_t<ExcludeT...>{});
  }
};

template <typename... IncludeT, typename... ExcludeT>
struct ParamGetType<ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : std::integral_constant<SystemParamType, SystemParamType::View> {};

template <typename... IncludeT, typename... ExcludeT>
struct ParamDependencies<ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : entt::type_list<IncludeT..., std::add_const_t<ExcludeT>...> {};
template <typename... IncludeT, typename... ExcludeT>
struct ParamViewIncludeDependencies<ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : entt::type_list<> {};
template <typename... IncludeT, typename... ExcludeT>
struct ParamViewExcludeDependencies<ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : entt::type_list<> {};

// Converts the parameters of a system into a list of types
template <typename... ArgsT>
constexpr entt::type_list<ArgsT...> SystemToTypeList(void (* /*system*/)(ArgsT...)) {
  return entt::type_list<ArgsT...>();
}

// Meta function for extracting types from a type list to be used
// as the inclusion parameters for a view of entities to iterate over in
// InvokeForEach.
template <typename T>
struct IncludeInViewFilter;

template <typename... typesT>
struct IncludeInViewFilter<entt::type_list<typesT...>>
    : entt::type_list_cat<typename ParamViewIncludeDependencies<typesT>::type...> {};

template <typename T>
using IncludeInViewFilter_t = typename IncludeInViewFilter<T>::type;

// Meta function for extracting types from a type list to be used
// as the exclusion parameters for a view of entities to iterate over in
// InvokeForEach.
template <typename T>
struct ExcludeFromViewFilter;

template <typename... typesT>
struct ExcludeFromViewFilter<entt::type_list<typesT...>>
    : entt::type_list_cat<typename ParamViewExcludeDependencies<typesT>::type...> {};

template <typename T>
using ExcludeFromViewFilter_t = typename ExcludeFromViewFilter<T>::type;

// Converts an inclusion type list and exclusion type list into an ENTT view
template <typename includeListT, typename excludeListT>
struct ViewFromTypeLists;

template <typename... IncludeT, typename... ExcludeT>
struct ViewFromTypeLists<entt::type_list<IncludeT...>, entt::type_list<ExcludeT...>> {
  static_assert(sizeof...(IncludeT) != 0, "A view must always include at least one type");
  using type = ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>;

  static auto Get(entt::registry& reg) {
    return reg.view<IncludeT...>(entt::exclude_t<ExcludeT...>{});
  }
};

/**************************************************************************
    System Invocation Utilities
*/

// A helper utility to remove std::reference_wrapper from template parameters.
template <typename T>
struct InvokerUnwrap {
  using result = T;
};

template <typename T>
struct InvokerUnwrap<std::reference_wrapper<T>> {
  using result = T&;
};

// Converts a fetch call with a view and registry into a fetch call with only a registry
template <typename T, typename ViewT>
struct FetchWithViewWrapperRegistry {
  static inline auto Get(entt::registry& reg, ViewT /*view*/, entt::entity e)
      -> decltype(SystemParamFetchSingleFromRegistry<T>::Get(reg, e)) {
    return SystemParamFetchSingleFromRegistry<T>::Get(reg, e);
  }
};

// Converts a fetch call with a view and registry into a fetch call with only a view
template <typename T, typename ViewT>
struct FetchWithViewWrapperView {
  static inline auto Get(entt::registry& /*reg*/, ViewT view, entt::entity e)
      -> decltype(SystemParamFetchSingleFromView<T, ViewT>::Get(view, e)) {
    return SystemParamFetchSingleFromView<T, ViewT>::Get(view, e);
  }
};

template <typename T, typename ViewT>
struct SystemParamFetchSingleWithView;

// Use to redirect a fetch to either a provided view or the registry depending on whether the
// requested object is obtainable from the view provided.
template <typename T, typename... IncludeT, typename... ExcludeT>
struct SystemParamFetchSingleWithView<T, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>
    : std::conditional_t<
          IsParamAccessibleFromView<T, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>::value,
          FetchWithViewWrapperView<T, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>,
          FetchWithViewWrapperRegistry<T, ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>>> {};

// A structure to be used to determine whether an entity would be contained in a given view type.
template <typename ViewT>
struct IsEntityInView;

template <typename... IncludeT, typename... ExcludeT>
struct IsEntityInView<ViewEx<Included<IncludeT...>, Excluded<ExcludeT...>>> {
  static bool Test(entt::registry& reg, entt::entity e) {
    // Entity must have all of the component types in IncludeT and none in ExcludeT
    return reg.all_of<std::remove_const_t<IncludeT>...>(e) &&
        !reg.any_of<std::remove_const_t<ExcludeT>...>(e);
  }
};

// A syntactic sugar class that automatically retrieves the necessary component references
// from the registry to invoke a system on a single entity. Here ExternalT is all of the
// parameters of a system that are not retrieved from the registry, but rather passed in by
// the caller.
template <typename PolicyList, typename... ExternalT>
struct InvokerImpl {
  std::tuple<typename InvokerUnwrap<ExternalT>::result...> _externalParams;

  InvokerImpl(ExternalT... extParams) : _externalParams(extParams...) {}

  // Call this to ensure that you have a legal per-entity system. Some exceptions
  // are allowed for now to handle existing systems in MochiPhysics.
  template <typename... LookupT>
  static constexpr bool EnforcePerEntitySystemPolicies(
      void (* /*system*/)(typename InvokerUnwrap<ExternalT>::result..., LookupT...)) {
    static_assert(
        entt::type_list_contains_v<PolicyList, policy::AllowMutableExternalParams> ||
            !LooksLikeMutableDataAccess<ExternalT...>(),
        "You are not allowed to pass mutable external arguments to a per-entity system because that might "
        "introduce a race condition or order dependency. Please make them const or wrap in std::cref(). If "
        "you are certain that your code is safe, then it is possible to bypass this static_assert by using "
        "ecs::policy::AllowMutableExternalParams.");
    static_assert(
        entt::type_list_contains_v<PolicyList, policy::AllowFullRegistryAccess> ||
            !HasRegistryReference<LookupT...>(),
        "Per-entity systems are not allowed access the whole registry because that might introduce a race "
        "condition or order dependency, and we have no programic way to check. Please use a View or "
        "PartialRegistry instead. If you are certain that your code is safe, then it is possible to bypass "
        "this static_assert by using ecs::policy::AllowFullRegistryAccess.");
    static_assert(
        entt::type_list_contains_v<PolicyList, policy::AllowReadWriteSameComponent> ||
            AreReadsAndWritesIndependent<LookupT...>(),
        "Per-entity systems are not allowed to modify components that another invocation of the system might "
        "read, because that might introduce a race condition or order dependency. Please add 'const' to all "
        "types that you do not intend to modify. If you are certain that your code is safe, then it is possible to bypass "
        "this static_assert by using ecs::policy::AllowReadWriteSameComponent.");
    // Always return true so we can wrap this in a static_assert.
    return true;
  };

  template <typename... LookupT>
  static auto GetViewFromSystem(
      void (* /*system*/)(typename InvokerUnwrap<ExternalT>::result..., LookupT...),
      entt::registry& reg) {
    using IncludeList = IncludeInViewFilter_t<entt::type_list<LookupT...>>;
    static_assert(
        IncludeList::size,
        "Per-entity systems must lookup at least one argument which depends on the entity. "
        "They cannot all be optional argument. Also, they cannot all be things like CtxGlobal, "
        "View, or PartialRegistry, because those things do not change per entity.");
    using ExcludeList = ExcludeFromViewFilter_t<entt::type_list<LookupT...>>;
    return ViewFromTypeLists<IncludeList, ExcludeList>::Get(reg);
  }

  // Return true if it the system can be invoked on the given entity
  template <typename... ArgsT>
  static bool
  CanInvokeOnEntity(void (*system)(ArgsT...), entt::registry& reg, entt::entity entity) {
    using ViewType = decltype(GetViewFromSystem(system, reg));
    return IsEntityInView<ViewType>::Test(reg, entity);
  }

  // Invokes the system as a global system
  template <typename... LookupT>
  inline void InvokeGlobal(
      void (*system)(typename InvokerUnwrap<ExternalT>::result..., LookupT...),
      entt::registry& reg) const {
    // Lookup system parameters from the registry
    auto lookupParams = std::tuple<LookupT...>(SystemParamFetchGlobal<LookupT>::Get(reg)...);

    // Call the system with both the external parameters passed in during initialization and the
    // lookup parameters from the registry.
    std::apply(system, std::tuple_cat(_externalParams, lookupParams));
  }

  // Invokes the system as a single-entity system using a view
  template <typename ViewT, typename... LookupT>
  inline void InvokeOnEntityInView(
      void (*system)(typename InvokerUnwrap<ExternalT>::result..., LookupT...),
      entt::registry& reg,
      ViewT view,
      entt::entity entity) const {
    // This code is used when you invoke a per-entity system using the global registry (all matching
    // entities). Such invocations should be order-independent and parallelizable.
    static_assert(EnforcePerEntitySystemPolicies(decltype(system){}));

    // Lookup system parameters from the registry
    auto lookupParams = std::tuple<LookupT...>(
        SystemParamFetchSingleWithView<LookupT, ViewT>::Get(reg, view, entity)...);

    // Call the system with both the external parameters passed in during initialization and the
    // lookup parameters from the registry.
    std::apply(system, std::tuple_cat(_externalParams, lookupParams));
  }

  // Invokes the system as a single-entity system.
  // Entity must have all expected types emplaced upon it or this will throw an exception.
  template <typename... LookupT>
  inline void InvokeOnEntity(
      void (*system)(typename InvokerUnwrap<ExternalT>::result..., LookupT...),
      entt::registry& reg,
      entt::entity entity) const {
    static_assert(EnforcePerEntitySystemPolicies(decltype(system){}));

    // Lookup system parameters from the registry
    auto lookupParams =
        std::tuple<LookupT...>(SystemParamFetchSingleFromRegistry<LookupT>::Get(reg, entity)...);

    // Call the system with both the external parameters passed in during initialization and the
    // lookup parameters from the registry.
    std::apply(system, std::tuple_cat(_externalParams, lookupParams));
  }
};

// Extends Invoker_Impl with helpers for InvokeForEach and friends
template <typename PolicyList, typename... ExternalT>
struct ForEachInvokerImpl : InvokerImpl<PolicyList, ExternalT...> {
  using BaseClass = InvokerImpl<PolicyList, ExternalT...>;
  using BaseClass::InvokerImpl;

  // Invoke for each entities in the view that satisfies the system's requirements
  template <typename SystemT>
  void InvokeForEachEntityInView(SystemT system, entt::registry& reg) const {
    static_assert(BaseClass::EnforcePerEntitySystemPolicies(SystemT{}));
    auto view = BaseClass::GetViewFromSystem(system, reg);
    for (entt::entity e : view) {
      BaseClass::InvokeOnEntityInView(system, reg, view, e);
    }
  }

  // Invoke for each entity in the set which satisfies the system's requirements
  template <typename SystemT, typename EntitySetT>
  void InvokeForEachEntityInSet(SystemT system, entt::registry& reg, EntitySetT const& entitySet)
      const {
    static_assert(BaseClass::EnforcePerEntitySystemPolicies(SystemT{}));
    for (entt::entity e : entitySet) {
      if (BaseClass::CanInvokeOnEntity(system, reg, e)) {
        BaseClass::InvokeOnEntity(system, reg, e);
      }
    }
  }
};

/**************************************************************************************************
  Invoking Systems
*/

template <typename... Policies, typename SystemT, typename... ExternalT>
inline void
InvokeOnEntity(SystemT system, entt::registry& reg, entt::entity e, ExternalT... extParams) {
  auto invoker = InvokerImpl<entt::type_list<Policies...>, ExternalT...>{extParams...};
  MOCHI_ASSERT_VERBOSE(invoker.CanInvokeOnEntity(system, reg, e));
  invoker.InvokeOnEntity(system, reg, e);
}

template <typename... Policies, typename SystemT, typename... ExternalT>
inline bool
CanInvokeOnEntity(SystemT system, entt::registry& reg, entt::entity e, ExternalT... /*extParams*/) {
  using InvokerT = InvokerImpl<entt::type_list<Policies...>, ExternalT...>;
  return InvokerT::CanInvokeOnEntity(system, reg, e);
}

template <typename... Policies, typename SystemT, typename... ExternalT>
inline bool
TryInvokeOnEntity(SystemT system, entt::registry& reg, entt::entity e, ExternalT... extParams) {
  using InvokerT = InvokerImpl<entt::type_list<Policies...>, ExternalT...>;
  if (InvokerT::CanInvokeOnEntity(system, reg, e)) {
    auto invoker = InvokerT{extParams...};
    invoker.InvokeOnEntity(system, reg, e);
    return true;
  }
  return false;
}

template <typename... Policies, typename SystemT, typename... ExternalT>
inline bool TryScheduleInvokeOnEntity(
    TaskSemaphore sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    entt::entity e,
    ExternalT... extParams) {
  using InvokerT = InvokerImpl<entt::type_list<Policies...>, ExternalT...>;
  if (InvokerT::CanInvokeOnEntity(system, reg, e)) {
    Schedule(sem, debugLabel, [system, &reg, e, invoker = InvokerT{extParams...}]() {
      invoker.InvokeOnEntity(system, reg, e);
    });
    return true;
  }
  return false;
}

template <typename SystemT, typename... ExternalT>
inline void InvokeGlobal(SystemT system, entt::registry& reg, ExternalT... extParams) {
  auto invoker = InvokerImpl<entt::type_list<>, ExternalT...>(extParams...);
  invoker.InvokeGlobal(system, reg);
}

template <typename... Policies, typename SystemT, typename... ExternalT>
inline void InvokeForEachGlobal(SystemT system, entt::registry& reg, ExternalT... extParams) {
  auto invoker = ForEachInvokerImpl<entt::type_list<Policies...>, ExternalT...>{extParams...};
  invoker.InvokeForEachEntityInView(system, reg);
}

template <typename... Policies, typename SystemT, typename... ExternalT>
inline void ScheduleInvokeForEachGlobal(
    TaskSemaphore& sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    ExternalT... extParams) {
  auto invoker = InvokerImpl<entt::type_list<Policies...>, ExternalT...>{extParams...};
  auto view = invoker.GetViewFromSystem(system, reg);
  for (entt::entity e : view) {
    Schedule(sem, debugLabel, [invoker, system, &reg, view, e]() {
      invoker.InvokeOnEntityInView(system, reg, view, e);
    });
  }
}

template <typename... Policies, typename SystemT, typename... ExternalT>
inline void ParallelInvokeForEachGlobal(
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    ExternalT... extParams) {
  TaskSemaphore sem;
  ScheduleInvokeForEachGlobal<Policies...>(sem, debugLabel, system, reg, extParams...);
  sem.Wait();
}

// Force inline to avoid cost when entitySet is empty
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
MOCHI_FORCE_INLINE void InvokeForEach(
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams) {
  if (std::size(entitySet) > 0) {
    auto invoker = ForEachInvokerImpl<entt::type_list<Policies...>, ExternalT...>{extParams...};
    invoker.InvokeForEachEntityInSet(system, reg, entitySet);
  }
}

namespace detail {
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
inline void ScheduleInvokeForEachImpl(
    TaskSemaphore& sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams) {
  auto invoker = InvokerImpl<entt::type_list<Policies...>, ExternalT...>{extParams...};
  static_assert(invoker.EnforcePerEntitySystemPolicies(SystemT{})); // no exceptions
#if MOCHI_ASSERT_VERBOSE_ENABLED
  // Paranoid check to prevent duplicate entities. Note that entitySet does not have to be a set.
  // It could be any forward iterable container.
  std::vector<entt::entity> sorted(std::begin(entitySet), std::end(entitySet));
  std::sort(sorted.begin(), sorted.end());
  MOCHI_ASSERT_VERBOSE(
      std::unique(sorted.begin(), sorted.end()) == sorted.end(),
      "Repeat entities detected in list!");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  for (auto e : entitySet) {
    if (invoker.CanInvokeOnEntity(system, reg, e)) {
      Schedule(sem, debugLabel, [invoker, system, &reg, e]() {
        invoker.InvokeOnEntity(system, reg, e);
      });
    }
  }
}
} // namespace detail

// Force inline to avoid cost when entitySet is empty
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
MOCHI_FORCE_INLINE void ScheduleInvokeForEach(
    TaskSemaphore& sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams) {
  if (std::size(entitySet) > 0) {
    detail::ScheduleInvokeForEachImpl<Policies...>(
        sem, debugLabel, system, reg, entitySet, extParams...);
  }
}

// Force inline to avoid cost when entitySet is empty
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
MOCHI_FORCE_INLINE void ParallelInvokeForEach(
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams) {
  if (std::size(entitySet) > 0) {
    TaskSemaphore sem;
    ScheduleInvokeForEach<Policies...>(sem, debugLabel, system, reg, entitySet, extParams...);
    sem.Wait();
  }
}

/**************************************************************************************************
  Creating System Objects
*/

template <typename SystemT, typename... ExternalT>
inline System
CreatePerEntitySystem(SystemT system, std::string const& name, ExternalT... extParams) {
  auto invoker = ForEachInvokerImpl<entt::type_list<>, ExternalT...>(extParams...);
  return System{
      [system, invoker = std::move(invoker)](entt::registry& reg) {
        invoker.InvokeForEachEntityInView(system, reg);
      },
      [system, name]() {
        auto info = GetSystemInfo(system);
        info.name = name;
        return info;
      }};
}

template <typename SystemT, typename... ExternalT>
inline System CreateSystem(SystemT system, std::string const& name, ExternalT... extParams) {
  auto invoker = InvokerImpl<entt::type_list<>, ExternalT...>(extParams...);
  return System{
      [system, invoker = std::move(invoker)](entt::registry& reg) {
        invoker.InvokeGlobal(system, reg);
      },
      [system, name]() {
        auto info = GetSystemInfo(system);
        info.name = name;
        return info;
      }};
}

} // namespace mochi::ecs
