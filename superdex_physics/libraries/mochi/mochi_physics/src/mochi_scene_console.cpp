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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

namespace {

// A built-in directory: the scene's top-level actors, the nested actors of one actor, or the
// scene's constraints.
struct SceneDirectory {
  bool isConstraints = false;
  Actor const* parent = nullptr; // The actor whose nested actors are listed, or null
  SReflect::detail::ConsolePathStep step; // The path step that names the directory
};

// Where a path inside a scene leads: to an error, a built-in directory, or a reflected object.
struct SceneTarget {
  std::string error;
  size_t errorOffset = 0;
  std::optional<SceneDirectory> directory;
  void* object = nullptr;
  SReflect::TypeInfo const* type = nullptr;
  size_t pathStart = 0; // Where the rest of the path, inside object, starts
  Actor const* actor = nullptr; // The actor that object is, if any
};

} // namespace

// Follow the built-in directories at the start of the path, which starts at pathStart. The rest of
// the path is left for SReflect. Call AssignDebugNames first.
static SceneTarget ResolveScenePath(SceneImpl& scene, std::string_view path, size_t pathStart) {
  SceneTarget target;
  target.object = &scene;
  target.type = &SReflect::GetTypeInfo<SceneImpl>();
  target.pathStart = pathStart;
  auto const fail = [&](std::string message, size_t offset) {
    target.error = std::move(message);
    target.errorOffset = offset;
    return target;
  };

  // SReflect reports an invalid path.
  std::vector<SReflect::detail::ConsolePathStep> steps;
  if (!SReflect::detail::SplitConsolePath(path.substr(pathStart), steps)) {
    steps.clear();
  }
  for (auto& step : steps) {
    step.offset += pathStart;
  }
  auto const startAt = [&](size_t i) { return i < steps.size() ? steps[i].offset : path.size(); };

  if (!steps.empty() && steps[0].name == kConstraintsStep) {
    if (steps.size() == 1) {
      target.directory = SceneDirectory{.isConstraints = true, .step = steps[0]};
      return target;
    }
    ConstraintInterface* found = nullptr;
    try {
      if (auto* constraint = FindConstraintEntry(scene, steps[1].name)) {
        found = assert_cast<ConstraintInterface*>(constraint);
      }
    } catch (std::invalid_argument const& e) {
      return fail(e.what(), steps[1].offset);
    }
    if (found == nullptr) {
      return fail(SReflect::ToJsonString(steps[1].name, false) + " not found", steps[1].offset);
    }
    target.object = dynamic_cast<void*>(found);
    target.type = &found->GetFinalTypeInfo();
    target.pathStart = startAt(2);
  } else if (!steps.empty() && steps[0].name == kActorsStep) {
    // Walk actors/<name>, then children/<name> in each actor that can own nested actors.
    Actor* actor = nullptr;
    size_t i = 0;
    do {
      if (i + 1 == steps.size()) {
        target.directory = SceneDirectory{.parent = actor, .step = steps[i]};
        return target;
      }
      actor = FindActorEntry(scene, actor, steps[i + 1].name);
      if (actor == nullptr) {
        return fail(
            SReflect::ToJsonString(steps[i + 1].name, false) + " not found", steps[i + 1].offset);
      }
      i += 2;
    } while (i < steps.size() && steps[i].name == kChildrenStep &&
             actor::CanOwnNestedActors(actor->GetType()));
    auto* const actorInterface = assert_cast<ActorInterface*>(actor);
    target.object = dynamic_cast<void*>(actorInterface);
    target.type = &actorInterface->GetFinalTypeInfo();
    target.pathStart = startAt(i);
    target.actor = actor;
  } else if (!steps.empty() && steps[0].name == "debug") {
    auto const debugger = scene.GetDebugger();
    if (!debugger) {
      return fail("The scene has no debugger client connected", steps[0].offset);
    }
    target.object = dynamic_cast<void*>(debugger.get());
    target.type = &debugger->GetFinalTypeInfo();
    target.pathStart = startAt(1);
  } else if (!steps.empty() && steps[0].name == "debugDraw") {
    auto* draw = assert_cast<DebugDrawInternal*>(&scene.GetDebugDraw());
    target.object = dynamic_cast<void*>(draw);
    target.type = &draw->GetFinalTypeInfo();
    target.pathStart = startAt(1);
  }
  return target;
}

static DynamicArray<SReflect::ConsoleListEntry> ListDirectory(
    SceneImpl& scene,
    SceneDirectory const& directory) {
  if (directory.isConstraints) {
    return ListConstraintEntries(scene);
  }
  DynamicArray<SReflect::ConsoleListEntry> entries;
  ListActorEntries(scene, directory.parent, entries);
  return entries;
}

// The built-in directories inside the object that target reaches, when the path ends there.
static DynamicArray<SReflect::ConsoleListEntry>
ListBuiltInDirectories(SceneImpl& scene, SceneTarget const& target, std::string_view path) {
  DynamicArray<SReflect::ConsoleListEntry> entries;
  if (target.pathStart != path.size()) {
    return entries;
  }
  if (target.object == &scene) {
    entries.push_back(MakeDirectoryEntry(kActorsStep, CountActorEntries(scene, nullptr)));
    entries.push_back(
        MakeDirectoryEntry(kConstraintsStep, static_cast<size_t>(scene.GetNumConstraints())));
  } else if (target.actor != nullptr && actor::CanOwnNestedActors(target.actor->GetType())) {
    entries.push_back(MakeDirectoryEntry(kChildrenStep, CountActorEntries(scene, target.actor)));
  }
  return entries;
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

  SceneTarget const target = ResolveScenePath(scene, path, request.pathStart);
  if (!target.error.empty()) {
    setError(target.error, target.errorOffset);
    return reply;
  }
  if (target.directory) {
    auto entries = ListDirectory(scene, *target.directory);
    if (request.isList) {
      reply.listing = std::move(entries);
    } else if (!args.empty()) {
      auto const& step = target.directory->step;
      setError(
          "Cannot set value because " + SReflect::ToJsonString(step.name, false) + " is read-only",
          step.offset);
    } else {
      // An empty object for each entry, because entries are not serialized recursively
      picojson::object names;
      for (auto const& entry : entries) {
        names.emplace(entry._name, picojson::object{});
      }
      std::string json = picojson::value{std::move(names)}.serialize(/*prettify*/ true);
      json.pop_back(); // Pretty JSON ends with a newline
      reply.output = json;
    }
    return reply;
  }

  std::string error;
  if (request.isList) {
    auto listing = SReflect::ConsoleList(
        target.object, *target.type, path, /*params*/ {}, &error, target.pathStart);
    if (error.empty()) {
      reply.listing = std::move(listing);
      for (auto& entry : ListBuiltInDirectories(scene, target, path)) {
        reply.listing->push_back(std::move(entry));
      }
    }
  } else {
    std::string output;
    SReflect::detail::ConsoleExecute(
        target.object, *target.type, path, args, &output, &error, target.pathStart);
    reply.output = output;
  }
  reply.error = error;
  return reply;
}

// Escape a scene or actor debug name that would be read as a handle number. SanitizeConsoleName
// keeps debug names from starting with '@'.
static std::string ConsoleDirectoryStep(std::string_view debugName) {
  bool isNumber = true;
  try {
    isNumber = ParseConsoleHandle(debugName).has_value();
  } catch (std::invalid_argument const&) {
    // An out-of-range number still reads as a handle number.
  }
  return isNumber ? "@" + std::string{debugName} : std::string{debugName};
}

void dbg::AddDirectoryCandidates(
    SReflect::ConsoleAutoCompleteResult& result,
    Span<SReflect::ConsoleListEntry const> entries,
    std::string_view typed,
    SReflect::ConsoleAutoCompleteParams const& params,
    bool namesAreHandles) {
  std::vector<std::string> names;
  for (auto const& entry : entries) {
    names.push_back(namesAreHandles ? entry._name : ConsoleDirectoryStep(entry._name));
  }
  SReflect::detail::AddConsoleStepCandidates(result, names, typed, params);
}

protocol::AutoCompleteReply dbg::CompleteSceneCommand(
    SceneImpl& scene,
    ActorDebugNameSuffixes& nextSuffixes,
    protocol::AutoCompleteRequest const& request) {
  ContextImpl::ScopedShapeHandleRetention retainShapes(
      assert_cast<ContextImpl*>(scene.GetContext()));
  AssignDebugNames(scene, nextSuffixes);
  protocol::AutoCompleteReply reply(request);
  auto const& completion = request.completion;
  std::string_view const path = completion.path;
  std::string_view const typed = std::string_view{request.command}.substr(completion.offset);
  bool const isStep = completion.target == SReflect::detail::ConsoleCompletionTarget::PathStep;
  SReflect::ConsoleAutoCompleteParams params;
  params._directoriesOnly = request.directoriesOnly;

  SceneTarget const target = ResolveScenePath(scene, path, request.pathStart);
  if (!target.error.empty()) {
    return reply;
  }
  if (target.directory) {
    // A built-in directory takes no argument, because it cannot be set.
    if (isStep) {
      AddDirectoryCandidates(
          reply.result,
          ListDirectory(scene, *target.directory),
          typed,
          params,
          target.directory->isConstraints);
    }
    return reply;
  }
  std::string error;
  if (SReflect::detail::ConsoleAutoComplete(
          target.object,
          *target.type,
          completion,
          typed,
          params,
          reply.result,
          &error,
          target.pathStart) &&
      isStep) {
    AddDirectoryCandidates(
        reply.result, ListBuiltInDirectories(scene, target, path), typed, params);
  }
  return reply;
}
