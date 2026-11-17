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
#include "mochi_scene.h"

#include <simple_reflection/simple_reflection_console.h>

using namespace mochi;
using namespace mochi::dbg;

DynamicArray<std::string> dbg::GetCommandArgs(protocol::CommandRequest const& request) {
  return DynamicArray<std::string>(request.args.begin(), request.args.end());
}

protocol::CommandReply dbg::ExecuteSceneCommand(
    SceneImpl& scene,
    protocol::CommandRequest const& request) {
  protocol::CommandReply reply(request);
  reply.path = request.path;
  auto const args = GetCommandArgs(request);
  std::string output;
  std::string error;
  if (request.isList) {
    auto listing =
        SReflect::ConsoleList(&scene, request.path, /*params*/ {}, &error, request.pathStart);
    if (error.empty()) {
      reply.listing = std::move(listing);
    }
  } else {
    SReflect::detail::ConsoleExecute(
        &scene, request.path, args, &output, &error, request.pathStart);
  }
  reply.output = output;
  reply.error = error;
  return reply;
}
