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
#include "mochi_scene.h"

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>
#include <picojson/picojson.h>
#include <simple_reflection/simple_reflection_console.h>

#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <utility>
#include <vector>

using namespace mochi;
using namespace mochi::dbg;

DynamicArray<std::string> dbg::GetCommandArgs(protocol::CommandRequest const& request) {
  return DynamicArray<std::string>(request.args.begin(), request.args.end());
}

static constexpr std::string_view kActorsStep = "actors";
static constexpr std::string_view kChildrenStep = "children";

// Call fn(name, actor) for the scene's top-level actors if parent is null, otherwise for the actors
// nested in parent, by their local name.
template <class Fn>
static void ForEachActorEntry(SceneImpl& scene, Actor const* parent, Fn const& fn) {
  if (parent == nullptr) {
    scene.ForEachActor([&](Actor* actor) {
      if (!actor->IsNestedLinkActor() && !actor->IsNestedSoftActor()) {
        fn(std::string_view{actor->GetName()}, actor);
      }
    });
    return;
  }
  for (Span<ActorHandle const> const children :
       {parent->GetNestedLinkActors(ErrorAssert{}), parent->GetNestedSoftActors(ErrorAssert{})}) {
    for (ActorHandle const child : children) {
      Actor* const actor = scene.GetActor(child);
      if (actor == nullptr) {
        continue; // Destroying a nested soft actor leaves its handle in the parent
      }
      // The name is "<parent name>/<local name>", and local names never contain '/'
      std::string_view name = actor->GetName();
      name.remove_prefix(name.rfind('/') + 1);
      fn(name, actor);
    }
  }
}

static Actor* FindActorEntry(SceneImpl& scene, Actor const* parent, std::string_view name) {
  Actor* found = nullptr;
  ForEachActorEntry(scene, parent, [&](std::string_view entryName, Actor* actor) {
    if (found == nullptr && entryName == name) {
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
         ._name = SReflect::detail::FormatPathStep(name),
         ._typeName = typeName,
         ._typeNameWithNamespace = typeName});
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

static SReflect::ConsoleListEntry MakeDirectoryEntry(std::string_view name) {
  return {._kind = SReflect::ConsoleEntryKind::Directory, ._name = std::string{name}};
}

protocol::CommandReply dbg::ExecuteSceneCommand(
    SceneImpl& scene,
    protocol::CommandRequest const& request) {
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
  }

  std::string error;
  if (request.isList) {
    auto listing = SReflect::ConsoleList(object, *type, path, /*params*/ {}, &error, pathStart);
    if (error.empty()) {
      reply.listing = std::move(listing);
      if (pathStart == path.size()) {
        if (object == &scene) {
          reply.listing->push_back(MakeDirectoryEntry(kActorsStep));
        } else if (actor != nullptr && actor::CanOwnNestedActors(actor->GetType())) {
          reply.listing->push_back(MakeDirectoryEntry(kChildrenStep));
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
