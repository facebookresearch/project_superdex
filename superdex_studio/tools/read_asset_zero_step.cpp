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

#include "zero_step_output.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>
#include <mochi_physics/cpp_api/mochi_actor.h>
#include <mochi_physics/cpp_api/mochi_context.h>
#include <mochi_physics/cpp_api/mochi_scene.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {
void String(std::string& json, char const* value) {
  json += '"';
  for (unsigned char c : std::string_view(value)) {
    if (c == '"' || c == '\\') {
      json += '\\';
      json += c;
    } else if (c < 0x20) {
      json += mochi::Format("\\u%04x", c);
    } else {
      json += c;
    }
  }
  json += '"';
}

void Number(std::string& json, double value) {
  if (std::isfinite(value)) {
    json += mochi::Format("%.17g", value);
  } else if (std::isnan(value)) {
    json += R"("nan")";
  } else {
    json += value < 0 ? R"("-inf")" : R"("+inf")";
  }
}

template <class Values>
void Array(std::string& json, Values const& values) {
  json += '[';
  bool first = true;
  for (auto const value : values) {
    if (!first) {
      json += ',';
    }
    Number(json, value);
    first = false;
  }
  json += ']';
}

bool Failed(mochi::Error const& error) {
  if (error.IsOK()) {
    return false;
  }
  std::fprintf(stderr, "%s\n", error.GetDescription());
  return true;
}
} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s <scene.mochi_scene>\n", argv[0]);
    return 1;
  }
  auto* const previousBuffer = std::cout.rdbuf(std::cerr.rdbuf());
  MOCHI_DEFER(std::cout.rdbuf(previousBuffer));
  auto const previousLog = mochi::GetLogCallback();
  MOCHI_DEFER(mochi::SetLogCallback(previousLog));
  mochi::SetLogCallback([](mochi::LogChannel channel, char const* message, char const*, int) {
    char const* level = channel == mochi::LogChannel::Error ? "Error"
        : channel == mochi::LogChannel::Warning             ? "Warning"
                                                            : "Info";
    std::fprintf(stderr, "%s: %s", level, message);
  });
  mochi::Context* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("asset_zero_step");
  mochi::Error error;
  std::filesystem::path const path = std::filesystem::absolute(argv[1]);
  auto prefab =
      mochi::prefab::LoadFromFile(path.string(), path.parent_path().string(), context, error);
  if (Failed(error)) {
    return 1;
  }
  auto added = mochi::prefab::AddToScene(prefab, scene, error);
  if (Failed(error)) {
    return 1;
  }
  std::string json = mochi::Format(
      R"({"precision_bytes":%zu,"positive_steps":0,"articulations":[)", sizeof(mochi::real));
  bool firstActor = true;
  for (auto* actor : added.actors) {
    if (actor->GetType() != mochi::ActorType::Articulated) {
      continue;
    }
    int const n = actor->GetNumDofs();
    std::vector<mochi::real> pose(n), velocity(n), forces(n);
    std::vector<mochi::Real2> limits(n);
    actor->GetArticulatedPose({pose.data(), pose.size()}, error);
    actor->GetArticulatedJointVelocities({velocity.data(), velocity.size()}, error);
    actor->GetArticulatedDofLimits({limits.data(), limits.size()}, error);
    actor->GetExternalForces({forces.data(), forces.size()}, error);
    bool const hasController = actor->HasArticulatedPoseController(error);
    auto const links = actor->GetNestedLinkActors(error);
    if (Failed(error)) {
      return 1;
    }
    if (!firstActor) {
      json += ',';
    }
    firstActor = false;
    json += R"({"name":)";
    String(json, actor->GetName());
    json += mochi::Format(
        R"(,"num_dofs":%d,"has_pose_controller":%s,"q":)", n, hasController ? "true" : "false");
    Array(json, pose);
    json += R"(,"qd":)";
    Array(json, velocity);
    json += R"(,"external_forces":)";
    Array(json, forces);
    json += R"(,"limits":[)";
    for (int i = 0; i < n; ++i) {
      if (i != 0) {
        json += ',';
      }
      Array(json, limits[i]);
    }
    json += R"(],"links":[)";
    bool firstLink = true;
    for (auto handle : links) {
      auto* link = scene->GetActor(handle);
      if (link == nullptr) {
        std::fprintf(stderr, "Nested link actor was not found\n");
        return 1;
      }
      auto const transform = link->GetRootTransform();
      auto const mass = link->GetMass(error);
      auto const density = link->GetDensity(error);
      auto const com = link->GetRigidCenterOfMassLocal(error);
      auto const inertia = link->GetRigidMomentOfInertiaLocal(error);
      if (Failed(error)) {
        return 1;
      }
      if (!firstLink) {
        json += ',';
      }
      firstLink = false;
      json += R"({"name":)";
      String(json, link->GetName());
      json += mochi::Format(R"(,"is_static":%s,"mass_kg":)", link->IsStatic() ? "true" : "false");
      Number(json, mass);
      json += R"(,"density_kg_m3":)";
      Number(json, density);
      json += R"(,"com_local_m":)";
      Array(json, com);
      json += R"(,"inertia_com_kg_m2_real6":)";
      Array(json, inertia);
      json += R"(,"world_translation":)";
      Array(json, transform.GetTranslation());
      json +=
          mochi::Format(R"(,"collider_type_enum":%d})", static_cast<int>(link->GetColliderType()));
    }
    json += "]}";
  }
  json += "]}\n";
  return superdex::studio::PublishZeroStepJson(json, error);
}
