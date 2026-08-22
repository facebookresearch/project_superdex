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

#include "mochi_ecs_registry.h"

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/no_copy.h>
#include <mochi_core/utils/task_scheduler.h>

MOCHI_WARNING_PUSH(); // Some warning suppression is required for entt
MOCHI_WARNING_IGNORE_MSVC(4307); // warning C4307: '*': integral constant overflow
#include <entt/entity/view.hpp>
MOCHI_WARNING_POP();

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace mochi::ecs {

/**************************************************************************************************
  ECS System Parameter Types
*/

// Used in a per-entity system definition to signal that T is a tag that is required
template <typename... T>
using Included = entt::get_t<T...>;

// Used in a per-entity system to signal that the types T... are excluded from execution
template <typename... T>
using Excluded = entt::exclude_t<T...>;

// Used in a system definition to signal that T is a tag that may or may not be emplaced on the
// entity. This type can be treated as a boolean and its value is whether or not the current entity
// has a tag of type T.
template <typename T>
struct OptionalTag {
  bool hasTag;
  explicit OptionalTag(bool value) : hasTag(value) {}
  operator bool() const {
    return hasTag;
  }
};

// Used in a system definition to signal that T is a tag that should be required on any entity it
// invokes on.
template <typename... T>
using RequiredTag = Included<T...>;

// A view that includes all of the types used in T. To exclude types as well, use ViewEx.
template <typename... T>
using View = entt::basic_view<entt::entity, entt::get_t<T...>, entt::exclude_t<>>;

// A view that allows both inclusion and exclusion of types.
// Usage: ViewEx<Required<ComponentA, ComponentB>, Excluded<ComponentC>>
template <typename RequiredT, typename ExcludedT>
using ViewEx = entt::basic_view<entt::entity, RequiredT, ExcludedT>;

// Used to read/write a global context variable in the registry. If the type T is specified as
// const, then the access mode is read, otherwise it is write. A CtxGlobal object can be treated as
// a pointer to the underlying data.
template <typename T>
struct CtxGlobal {
  T const& value;
  CtxGlobal(T& value) : value(value) {}
  T* operator->() {
    return &value;
  }
};

// Like CtxGlobal but the value may be optional
template <typename T>
struct OptionalCtxGlobal {
  T const* value;
  OptionalCtxGlobal(T* value) : value(value) {}
  T* operator->() {
    return value;
  }
  explicit operator bool() const {
    return value != nullptr;
  }
};

// If you need a PartialRegistry that can read any type of component, then add a parameter of type
// PartialRegistry<WildcardRead> to your system.
struct WildcardRead {};

// If you need a PartialRegistry that can read or write (but not emplace/destroy) any type of
// component, then add a parameter of type PartialRegistry<WildcardWrite> to your system.
struct WildcardWrite {};

/**************************************************************************************************
  ECS System Invocation
*/

/*
Invokes a single-entity system on a given entity.

Usage:
    Suppose we define a system as a pure function as such:

    void System(extype1 et1, ...., extypeN etN, lookuptype1 lt1, ..., lookuptypeM ltM) {
        ...
    }

    Here the objects et1, ..., etN will be passed through by the user, while the objects
    lt1, ..., ltM will be retrieved from the registry automatically following the procedures
    defined in SystemParamTraits<lookuptypei>::FetchSingleEntity.

    To invoke the System, one may use the following code:

    InvokeOnEntity(&System, registry, entity, et1, ..., etN);

    Note that typically, we will want et1, ..., etN to be references or pointers. If they are
    pointers, nothing additional need by done, but if eti should be passed by reference, you must
    use the std::ref (for nonconstant references) or std::cref (for constant references) wrapper
    utilities to signal that the object should be passed by reference and not by value, i.e.,

    InvokeOnEntity(&System, registry, entity, et1, ..., std::ref(eti), ..., etN);
    InvokeOnEntity(&System, registry, entity, et1, ..., std::cref(eti), ..., etN);

    The invoker will expect that the corresponding type in the signature of the System will
    be either a reference type or a const reference type, i.e.,

    void System(extype1 et1, ...., extypei& eti, ..., extypeN etN,
        lookuptype1 lt1, ..., lookuptypeM ltM) {
        ...
    }

    or

    void System(extype1 et1, ...., const extypei& eti, ..., extypeN etN,
        lookuptype1 lt1, ..., lookuptypeM ltM) {
        ...
    }

    respectively. For lookup types, the corresponding rules are as follows:

    lookuptypei is a:
        reference (lookuptypei&) =>
            we retrieve the component lookuptypei for specified entity, write access.
        const reference (const lookuptypei&) =>
            we retrieve the component lookuptypei for specified entity, read access.
        pointer (lookuptypei*) =>
            specifies that the component lookuptypei is optional, nullptr is returned if
            not emplaced. Write access.
        const pointer (const lookuptypei*) =>
            specifies that the component lookuptypei is optional, nullptr is returned if
            not emplaced. Read access.
        optional tag (OptionalTag<lookuptypei>) =>
            specifies that the tag lookuptypei may or may not be emplaced on the entity.
            OptionalTag<lookuptypei> is a boolean-like type that is true if the tag was
            found on the entity.
        const global context (GlobalCtx<const lookuptypei>) =>
            Reads this object from the global context (i.e., registry.ctx<lookuptypei>()).
            For single-entity systems the parameter must be specified as const, as
            otherwise multiple systems might write to the same object. The GlobalCtx
            wrapper can be treated as a pointer.
        global context (GlobalCtx<lookuptypei>) =>
            Illegal for the reason stated above.
        everything else =>
            Illegal.

*/
template <typename... Policies, typename SystemT, typename... ExternalT>
void InvokeOnEntity(SystemT system, entt::registry& reg, entt::entity e, ExternalT... extParams);

/*
Returns true if all the requirements are met for the given system. If so, a call to
TryInvokeOnEntity will actually call it.
*/
template <typename... Policies, typename SystemT, typename... ExternalT>
bool CanInvokeOnEntity(SystemT system, entt::registry& reg, entt::entity e, ExternalT... extParams);

/*
Invokes a single-entity system on a given entity if the requirements are met.
See CanInvokeOnEntity.

Usage:
    Same as InvokeOnEntity

Returns:
    true if invoked

*/
template <typename... Policies, typename SystemT, typename... ExternalT>
bool TryInvokeOnEntity(SystemT system, entt::registry& reg, entt::entity e, ExternalT... extParams);

/*
If the system can be invoked on the entity, then increment the semaphore, schedule an async task,
and return true. Else, return false without scheduling a task.

Usage:
  Same as InvokeOnEntity (except for the first 2 arguments)

Returns:
  true if a task was scheduled

*/
template <typename... Policies, typename SystemT, typename... ExternalT>
inline bool TryScheduleInvokeOnEntity(
    TaskSemaphore sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    entt::entity e,
    ExternalT... extParams);

/*
Invokes a global system on a given registry.

Usage:
   Same as with single system. References and pointers are automatically treated
   as CtxGlobals.
*/
template <typename SystemT, typename... ExternalT>
void InvokeGlobal(SystemT system, entt::registry& reg, ExternalT... extParams);

// Invoke the system for each matching entity in the global registry
template <typename... Policies, typename SystemT, typename... ExternalT>
void InvokeForEachGlobal(SystemT system, entt::registry& reg, ExternalT... extParams);

// Schedule tasks to invoke the system for each matching entity in the global registry
template <typename... Policies, typename SystemT, typename... ExternalT>
void ScheduleInvokeForEachGlobal(
    TaskSemaphore& sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    ExternalT... extParams);

// Invoke the system concurrently for each matching entity in the global registry.
// Wait for completion.
template <typename... Policies, typename SystemT, typename... ExternalT>
void ParallelInvokeForEachGlobal(
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    ExternalT... extParams);

// Invoke the system for each matching entity in a local list/set of entities
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
void InvokeForEach(
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams);

// Schedule tasks to invoke the system for each matching entity in a local list/set of entities
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
void ScheduleInvokeForEach(
    TaskSemaphore& sem,
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams);

// Invoke the system concurrently for each matching entity in a local list/set of entities. Wait for
// completion.
template <typename... Policies, typename SystemT, typename SubsetT, typename... ExternalT>
void ParallelInvokeForEach(
    std::string_view debugLabel,
    SystemT system,
    entt::registry& reg,
    SubsetT const& entitySet,
    ExternalT... extParams);

/**************************************************************************************************
  Policy Exceptions

  Certain data access restrictions are enforced for per-entity systems. The goal is to ensure that
  the system can be invoked concurrently for each matching entity. Even in a single-threaded
  context, it is still important to ensure that behavior never depends on the order in which
  entities are processed.

  If your system violates any of these policies, then your code will not compile and you will see a
  static_assert message in the compiler output. If you are certain that your code is safe, then you
  can bypass the static_assert by specifying one of the following policy exceptions. These
  exceptions should become rare as we get more and more of our code converted to systems with
  explicit inputs and outputs.
*/

namespace policy {

/**
  Per-entity systems usually should not have mutable external parameters, because that may introduce
  a race condition or order dependency. If you are certain that your code is safe, then you can
  bypass the static_assert with this template argument.

  Example:
    InvokeForEach<policy::AllowMutableExternalParams>(&MySystem, reg, e, std::ref(myMutableData));
*/
struct AllowMutableExternalParams {};

/**
  Per-entity systems usually should not have access to the full entt::registry, because that may
  introduce a race condition or order dependency. If you are certain that your code is safe, then
  you can bypass the static_assert with this template argument.

  Example:
    ecs::InvokeForEach<ecs::policy::AllowFullRegistryAccess>(&MySystem, reg);
*/
struct AllowFullRegistryAccess {};

/**
  Per-entity systems usually should not write to a component and read the same component from
  another entity, because that may introduce a race condition or order dependency. If you are
  certain that your code is safe, then you can bypass the static_assert with this template argument.

  Example:
    ecs::InvokeForEach<ecs::policy::AllowReadWriteSameComponent>(&MySystem, reg);
*/
struct AllowReadWriteSameComponent {};

} // namespace policy

/**************************************************************************************************
  ECS System Metadata
*/

enum class AccessWildcard { None = 0, Read = 1, Write = 2 };
enum class AccessMode { NotApplicable = 0, Read = 1, Write = 2 };

enum class SystemParamType {
  Unknown,
  ComponentRead,
  ComponentWrite,
  ComponentReadOptional,
  ComponentWriteOptional,
  GlobalCtxRead,
  GlobalCtxReadOptional,
  GlobalCtxWrite,
  GlobalCtxWriteOptional,
  RegistryPartialRead,
  RegistryPartialWrite,
  RegistryFullRead,
  RegistryFullWrite,
  TagOptional,
  Required,
  Excluded,
  View,
  EntityId
};

struct SystemDependency {
  // The type id of this dependency
  entt::type_info type = entt::type_id<void>();
  // The type of dependency (i.e., read/write)
  AccessMode mode = AccessMode::NotApplicable;

  template <typename T>
  static SystemDependency Write();

  template <typename T>
  static SystemDependency Read();
};

// Meta-data for each parameter of a system
struct SystemParamInfo {
  // The dependencies that this parameter incurs
  std::vector<SystemDependency> dependencies;
  // The type of this parameter
  SystemParamType paramType = SystemParamType::Unknown;
  // Whether this parameter incurs a wildcard
  AccessWildcard wildcard = AccessWildcard::None;

  bool IsOptional() const;
  bool IsComponent() const;
  bool IsGlobal() const;
  bool IsWildcard() const;
};

// Supports unordered containers of entt::type_info
struct TypeInfoHasher {
  std::size_t operator()(entt::type_info const& type) const {
    return type.hash();
  }
};

struct SystemInfo {
  // All of the parameters of a system in order
  std::vector<SystemParamInfo> parameters;

  // All of the types that a system reads from
  std::unordered_set<entt::type_info, TypeInfoHasher> reads;
  // All of the types that a system writes to
  std::unordered_set<entt::type_info, TypeInfoHasher> writes;
  // Whether or not this system has a read/write wildcard
  // If so, then reads and writes will be empty
  AccessWildcard wildcard = AccessWildcard::None;

  // The name of the system
  std::string name;

  // Removes all duplicate entries from this access info
  // And builds reads and writes
  void Build();

  static SystemInfo FromParamInfos(std::vector<SystemParamInfo>&& parameters);
};

} // namespace mochi::ecs

namespace mochi {

/**************************************************************************************
  ECS Component Base Classes
*/

// Base for reference counted components
struct RefCounted : public NoCopy {
  int referenceCount = 0; // Use functions like AddOrIncRefComponent.
};

} // namespace mochi

#include "mochi_ecs_inl.h"
