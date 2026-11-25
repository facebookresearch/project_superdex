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

#include "mochi_scene_console.h"
#include "mochi_actor.h"
#include "mochi_common_components.h"
#include "mochi_constraint_interface.h"
#include "mochi_context.h"
#include "mochi_debug_draw.h"
#include "mochi_ecs_utils.h"
#include "mochi_scene.h"

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/span.h>
#include <picojson/picojson.h>
#include <simple_reflection/simple_reflection_console.h>

#include <algorithm>
#include <charconv>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::dbg;

DynamicArray<std::string> dbg::GetCommandArgs(protocol::CommandRequest const& request) {
  return DynamicArray<std::string>(request.args.begin(), request.args.end());
}

std::optional<uint64_t> dbg::ParseConsoleHandle(std::string_view token) {
  if (token.empty() || token.front() == '@') {
    return std::nullopt;
  }
  bool const negative = token.front() == '-';
  std::string_view const digits = negative || token.front() == '+' ? token.substr(1) : token;
  if (digits.empty() ||
      !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) {
    return std::nullopt;
  }
  uint64_t value = 0;
  std::from_chars_result result;
  if (negative) {
    int64_t signedValue = 0;
    result = std::from_chars(token.data(), token.data() + token.size(), signedValue);
    value = static_cast<uint64_t>(signedValue);
  } else {
    result = std::from_chars(digits.data(), digits.data() + digits.size(), value);
  }
  if (result.ec != std::errc{}) {
    throw std::invalid_argument("Handle identifier is out of range");
  }
  return value;
}

std::string dbg::SanitizeConsoleName(std::string_view name) {
  if (name == "." || name == "..") {
    return {};
  }
  std::string sanitized{name};
  std::ranges::replace_if(
      sanitized,
      [](char c) {
        auto const byte = static_cast<unsigned char>(c);
        return c == '/' || c == '"' || c == '\'' || byte <= ' ' || byte == 0x7F;
      },
      '_');
  if (sanitized.starts_with('[') || sanitized.starts_with('{') || sanitized.starts_with('@')) {
    sanitized[0] = '_';
  }
  return sanitized;
}

static entt::entity ToEntity(Actor const* actor) {
  return GetEntityUnchecked(actor->GetHandle());
}

static bool IsNested(Actor const* actor) {
  return actor->IsNestedLinkActor() || actor->IsNestedSoftActor();
}

// The actor whose children directory lists actor, or an invalid handle for a top-level actor.
static ActorHandle GetParent(Actor const* actor) {
  return IsNested(actor) ? actor->GetArticulatedActor(ErrorAssert{}) : ActorHandle{};
}

static std::string MakeActorDebugName(
    Actor const* actor,
    bool isNested,
    std::unordered_set<std::string> const& siblingNames,
    DebugNameSuffixes& nextSuffixes) {
  std::string_view name = actor->GetName();
  if (isNested) {
    // The name is "<parent name>/<local name>", and local names never contain '/'
    name.remove_prefix(name.rfind('/') + 1);
  }
  std::string base = SanitizeConsoleName(name);
  if (base.empty()) {
    base = SReflect::EnumToString(actor->GetType());
  }
  return MakeUniqueDebugName(base, nextSuffixes, [&](std::string const& candidate) {
    return siblingNames.contains(candidate);
  });
}

void dbg::AssignDebugNames(SceneImpl& scene, ActorDebugNameSuffixes& nextSuffixes) {
  entt::registry& reg = scene.GetRegistry();
  auto const namedActors = reg.view<CDebugName const, CActorInfo const>();
  size_t const numNamed =
      static_cast<size_t>(std::distance(namedActors.begin(), namedActors.end()));
  auto const numActors = static_cast<size_t>(scene.GetNumActors());
  if (numNamed == numActors) {
    return;
  }

  // Taken names by directory, keyed by the parent's handle (invalid for the top-level actors)
  std::unordered_map<ActorHandle, std::unordered_set<std::string>> takenNames;
  DynamicArray<Actor*> unnamed;
  scene.ForEachActor([&](Actor* actor) {
    if (auto const* debugName = reg.try_get<CDebugName const>(ToEntity(actor))) {
      takenNames[GetParent(actor)].insert(debugName->name);
    } else {
      unnamed.push_back(actor);
    }
  });

  // Named in handle order, so the result is deterministic
  std::ranges::sort(unnamed, {}, [](Actor const* actor) { return actor->GetHandle().value; });
  for (Actor const* actor : unnamed) {
    ActorHandle const parent = GetParent(actor);
    auto& siblingNames = takenNames[parent];
    std::string name =
        MakeActorDebugName(actor, parent.IsValid(), siblingNames, nextSuffixes[parent]);
    siblingNames.insert(name);
    reg.emplace<CDebugName>(ToEntity(actor), std::move(name));
  }
  MOCHI_ASSERT_VERBOSE(numNamed + unnamed.size() == numActors, "Every actor has a debug name");
}

static constexpr std::string_view kActorsStep = "actors";
static constexpr std::string_view kConstraintsStep = "constraints";
static constexpr std::string_view kChildrenStep = "children";

// Call fn(debugName, actor) for the scene's top-level actors if parent is null, otherwise for
// the actors nested in parent. Call AssignDebugNames first.
template <class Fn>
static void ForEachActorEntry(SceneImpl& scene, Actor const* parent, Fn const& fn) {
  entt::registry const& reg = scene.GetRegistry();
  auto const visit = [&](Actor* actor) {
    fn(std::string_view{reg.get<CDebugName>(ToEntity(actor)).name}, actor);
  };
  if (parent == nullptr) {
    scene.ForEachActor([&](Actor* actor) {
      if (!IsNested(actor)) {
        visit(actor);
      }
    });
    return;
  }
  for (Span<ActorHandle const> const children :
       {parent->GetNestedLinkActors(ErrorAssert{}), parent->GetNestedSoftActors(ErrorAssert{})}) {
    for (ActorHandle const child : children) {
      Actor* const actor = scene.GetActor(child);
      MOCHI_ASSERT_VERBOSE(actor != nullptr, "A live actor's nested actor handles must resolve.");
      visit(actor);
    }
  }
}

static size_t CountActorEntries(SceneImpl& scene, Actor const* parent) {
  size_t count = 0;
  ForEachActorEntry(scene, parent, [&](std::string_view, Actor const*) { ++count; });
  return count;
}

static Actor* FindActorEntry(SceneImpl& scene, Actor const* parent, std::string_view reference) {
  std::optional<uint64_t> number;
  try {
    number = ParseConsoleHandle(reference);
  } catch (std::invalid_argument const&) {
    return nullptr;
  }
  if (!number && reference.starts_with('@')) {
    reference.remove_prefix(1);
  }
  Actor* found = nullptr;
  ForEachActorEntry(scene, parent, [&](std::string_view name, Actor* actor) {
    if (found == nullptr && (number ? actor->GetHandle().value == *number : name == reference)) {
      found = actor;
    }
  });
  return found;
}

// Sorted by name, like the scenes in the debug server's listing.
static void ListActorEntries(
    SceneImpl& scene,
    Actor const* parent,
    DynamicArray<SReflect::ConsoleListEntry>& outEntries) {
  ForEachActorEntry(scene, parent, [&](std::string_view name, Actor const* actor) {
    char const* const typeName = SReflect::EnumToString(actor->GetType());
    outEntries.push_back(
        {._kind = SReflect::ConsoleEntryKind::Directory,
         ._name = std::string{name},
         ._typeName = typeName,
         ._typeNameWithNamespace = typeName,
         ._value = "(handle=" + std::to_string(actor->GetHandle().value) + ")",
         ._coreType = SReflect::CoreType::CT_struct});
  });
  std::ranges::sort(outEntries, {}, &SReflect::ConsoleListEntry::_name);
}

// A JSON object with an empty object for each actor entry, because actors are not serialized
// recursively.
static std::string FormatActorEntries(SceneImpl& scene, Actor const* parent) {
  picojson::object actors;
  ForEachActorEntry(scene, parent, [&](std::string_view name, Actor const*) {
    actors.emplace(std::string{name}, picojson::object{});
  });
  std::string json = picojson::value{std::move(actors)}.serialize(/*prettify*/ true);
  json.pop_back(); // Pretty JSON ends with a newline
  return json;
}

static SReflect::ConsoleListEntry MakeDirectoryEntry(std::string_view name, size_t containerSize) {
  return {
      ._kind = SReflect::ConsoleEntryKind::Directory,
      ._name = std::string{name},
      ._coreType = SReflect::CoreType::CT_map,
      ._containerSize = containerSize};
}

static DynamicArray<SReflect::ConsoleListEntry> ListConstraintEntries(SceneImpl& scene) {
  DynamicArray<SReflect::ConsoleListEntry> entries;
  scene.ForEachConstraint([&](Constraint* constraint) {
    std::string actors = "(actors=[";
    for (int i = 0; i < constraint->GetNumActors(); ++i) {
      if (i != 0) {
        actors += ", ";
      }
      actors += scene.GetRegistry().get<CDebugName>(ToEntity(constraint->GetActor(i))).name;
    }
    actors += "])";
    entries.push_back({
        ._kind = SReflect::ConsoleEntryKind::Directory,
        ._name = std::to_string(constraint->GetHandle().value),
        ._typeName = SReflect::EnumToString(constraint->GetType()),
        ._value = std::move(actors),
        ._coreType = SReflect::CoreType::CT_struct,
    });
  });
  std::ranges::sort(entries, {}, &SReflect::ConsoleListEntry::_name);
  return entries;
}

static Constraint* FindConstraintEntry(SceneImpl& scene, std::string_view reference) {
  if (auto number = ParseConsoleHandle(reference)) {
    return scene.GetConstraint(ConstraintHandle{*number});
  }
  return nullptr;
}

protocol::CommandReply dbg::ExecuteSceneCommand(
    SceneImpl& scene,
    ActorDebugNameSuffixes& nextSuffixes,
    protocol::CommandRequest const& request) {
  ContextImpl::ScopedShapeHandleRetention retainShapes(
      assert_cast<ContextImpl*>(scene.GetContext()));
  AssignDebugNames(scene, nextSuffixes);
  protocol::CommandReply reply(request);
  reply.path = request.path;
  std::string_view const path = request.path;
  auto const args = GetCommandArgs(request);
  auto const setError = [&](std::string_view message, size_t offset) {
    reply.error = SReflect::detail::FormatConsoleError(message, path, args, 0, offset);
  };

  void* object = &scene;
  SReflect::TypeInfo const* type = &SReflect::GetTypeInfo<SceneImpl>();
  size_t pathStart = request.pathStart;

  // SReflect reports an invalid path below.
  std::vector<SReflect::detail::ConsolePathStep> steps;
  if (!SReflect::detail::SplitConsolePath(path.substr(pathStart), steps)) {
    steps.clear();
  }
  for (auto& step : steps) {
    step.offset += pathStart;
  }

  if (!steps.empty() && steps[0].name == kConstraintsStep) {
    if (steps.size() == 1) {
      auto entries = ListConstraintEntries(scene);
      if (request.isList) {
        reply.listing = std::move(entries);
      } else if (!args.empty()) {
        setError("Cannot set value because \"constraints\" is read-only", steps[0].offset);
      } else {
        picojson::object constraints;
        for (auto const& entry : entries) {
          constraints.emplace(entry._name, picojson::object{});
        }
        std::string json = picojson::value{std::move(constraints)}.serialize(/*prettify*/ true);
        json.pop_back(); // Pretty JSON ends with a newline
        reply.output = json;
      }
      return reply;
    }
    ConstraintInterface* found = nullptr;
    try {
      if (auto* constraint = FindConstraintEntry(scene, steps[1].name)) {
        found = assert_cast<ConstraintInterface*>(constraint);
      }
    } catch (std::invalid_argument const& e) {
      setError(e.what(), steps[1].offset);
      return reply;
    }
    if (found == nullptr) {
      setError(SReflect::ToJsonString(steps[1].name, false) + " not found", steps[1].offset);
      return reply;
    }
    object = dynamic_cast<void*>(found);
    type = &found->GetFinalTypeInfo();
    pathStart = steps.size() > 2 ? steps[2].offset : path.size();
  }

  Actor* actor = nullptr;
  if (!steps.empty() && steps[0].name == kActorsStep) {
    // Walk actors/<name>, then children/<name> in each actor that can own nested actors.
    size_t i = 0;
    do {
      if (i + 1 == steps.size()) {
        if (request.isList) {
          reply.listing.emplace();
          ListActorEntries(scene, actor, *reply.listing);
        } else if (!args.empty()) {
          setError(
              "Cannot set value because " + SReflect::ToJsonString(steps[i].name, false) +
                  " is read-only",
              steps[i].offset);
        } else {
          reply.output = FormatActorEntries(scene, actor);
        }
        return reply;
      }
      actor = FindActorEntry(scene, actor, steps[i + 1].name);
      if (actor == nullptr) {
        setError(
            SReflect::ToJsonString(steps[i + 1].name, false) + " not found", steps[i + 1].offset);
        return reply;
      }
      i += 2;
    } while (i < steps.size() && steps[i].name == kChildrenStep &&
             actor::CanOwnNestedActors(actor->GetType()));
    auto* const actorInterface = assert_cast<ActorInterface*>(actor);
    object = dynamic_cast<void*>(actorInterface);
    type = &actorInterface->GetFinalTypeInfo();
    pathStart = i < steps.size() ? steps[i].offset : path.size();
  } else if (!steps.empty() && steps[0].name == "debugDraw") {
    auto* draw = assert_cast<DebugDrawInternal*>(&scene.GetDebugDraw());
    object = dynamic_cast<void*>(draw);
    type = &draw->GetFinalTypeInfo();
    pathStart = steps.size() > 1 ? steps[1].offset : path.size();
  }

  std::string error;
  if (request.isList) {
    auto listing = SReflect::ConsoleList(object, *type, path, /*params*/ {}, &error, pathStart);
    if (error.empty()) {
      reply.listing = std::move(listing);
      if (pathStart == path.size()) {
        if (object == &scene) {
          reply.listing->push_back(
              MakeDirectoryEntry(kActorsStep, CountActorEntries(scene, nullptr)));
          reply.listing->push_back(
              MakeDirectoryEntry(kConstraintsStep, static_cast<size_t>(scene.GetNumConstraints())));
        } else if (actor != nullptr && actor::CanOwnNestedActors(actor->GetType())) {
          reply.listing->push_back(
              MakeDirectoryEntry(kChildrenStep, CountActorEntries(scene, actor)));
        }
      }
    }
  } else {
    std::string output;
    SReflect::detail::ConsoleExecute(object, *type, path, args, &output, &error, pathStart);
    reply.output = output;
  }
  reply.error = error;
  return reply;
}
