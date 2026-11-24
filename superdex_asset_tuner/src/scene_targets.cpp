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

#include "superdex_asset_tuner/scene_targets.h"

#include "scene_utils.h"
#include "superdex_asset_tuner/tuner_runner.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/defer.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/superdex_robotics.h>

#include <cmath>
#include <optional>
#include <string_view>

namespace superdex::tuner {

mochi::DynamicArray<mochi::Actor*> details::GetTopLevelActors(mochi::Scene& scene) {
  DynamicArray<mochi::Actor*> all(static_cast<size_t>(scene.GetNumActors()));
  mochi::ErrorLog error;
  scene.GetActors(mochi::MakeSpan(all), error);
  DynamicArray<mochi::Actor*> topLevel;
  for (mochi::Actor* actor : all) {
    if (!actor->IsNestedLinkActor()) {
      topLevel.push_back(actor);
    }
  }
  return topLevel;
}

std::optional<mochi::Aabb> details::WorldBounds(mochi::Actor const& actor) {
  // Planes report huge or non-finite boxes.
  constexpr real kMaxBoundedExtent = 1e6f;
  std::optional<mochi::Aabb> bounds;
  auto const include = [&](mochi::Actor const& part) {
    // A fresh error per part: an articulated actor without a skin has no box of its own, and that
    // failure must not empty the boxes of the parts after it.
    mochi::Error error;
    mochi::Aabb const box = part.GetAabbWorld(error);
    Real3 const min = box.GetMin();
    Real3 const max = box.GetMax();
    if (!error.IsOK() || min == max) {
      return; // no box, or a shapeless link (a point at its origin)
    }
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(min[d]) || !std::isfinite(max[d]) ||
          std::abs(min[d]) >= kMaxBoundedExtent || std::abs(max[d]) >= kMaxBoundedExtent) {
        return;
      }
    }
    bounds = bounds ? mochi::EncloseShapes(*bounds, box) : box;
  };
  include(actor);
  if (actor.GetType() == mochi::ActorType::Articulated) {
    mochi::Scene const* scene = actor.GetScene();
    mochi::Error error;
    for (Span<mochi::ActorHandle const> const nested :
         {actor.GetNestedLinkActors(error), actor.GetNestedSoftActors(error)}) {
      for (mochi::ActorHandle const handle : nested) {
        if (mochi::Actor const* part = scene->GetActor(handle)) {
          include(*part);
        }
      }
    }
  }
  return bounds;
}

bool details::IsBounded(mochi::Actor const& actor) {
  return WorldBounds(actor).has_value();
}

mochi::DynamicArray<details::FloatInputController> details::FindFloatInputControllers(
    robotics::BotScene& bots) {
  DynamicArray<FloatInputController> found;
  robotics::RoboticsContext* const robotics = bots.GetRoboticsContext();
  if (robotics == nullptr) {
    return found;
  }
  for (robotics::BotEntry const& entry : bots.GetBotScenePrefab().bots) {
    robotics::Bot* const bot = bots.GetBot(entry.name);
    if (bot == nullptr) {
      continue; // declared but not built
    }
    for (robotics::ControllerHandle const handle : bot->GetControllerHandles()) {
      robotics::ControllerBase* const controller = robotics->GetController(handle);
      // A controller without a float input drives the bot on its own, if at all.
      if (controller != nullptr && controller->GetActor() != nullptr &&
          controller->GetInputSize() > 0) {
        found.push_back(
            {.name = entry.name + "/" + DynamicString(controller->GetName()),
             .controller = controller});
      }
    }
  }
  return found;
}

namespace {

std::string_view InputKindName(robotics::ControllerInputKind kind) {
  switch (kind) {
    case robotics::ControllerInputKind::JointPositions:
      return "JointPositions";
    case robotics::ControllerInputKind::Pose:
      return "Pose";
    case robotics::ControllerInputKind::Custom:
      return "Custom";
    case robotics::ControllerInputKind::None:
    case robotics::ControllerInputKind::Count:
      break;
  }
  return "None";
}

// The joint each entry of a joint-position input sets.
void DescribeInputJoints(robotics::ControllerBase const& controller, ControllerTarget& target) {
  mochi::Actor const* const actor = controller.GetActor();
  if (actor == nullptr ||
      controller.GetInputKind() != robotics::ControllerInputKind::JointPositions) {
    return;
  }
  mochi::ErrorLog error;
  mochi::ArticulatedShapeInfo const info = actor->GetArticulatedShapeInfo(error);
  for (int const dof : controller.GetControlledDofs()) {
    target.inputJoints.emplace_back();
    InputJoint& joint = target.inputJoints.back();
    for (size_t j = 0; j < info.jointNames.size(); ++j) {
      if (info.dofInfo[j].offset != dof || info.dofInfo[j].GetSize() != 1) {
        continue;
      }
      joint.name = info.jointNames[j];
      joint.type = info.jointTypes[j];
      // Limits are stored along the joint's axis.
      Real3 const axis = mochi::Normalize(info.jointAxes[j]);
      joint.min = mochi::Dot(info.jointMinLimits[j], axis);
      joint.max = mochi::Dot(info.jointMaxLimits[j], axis);
      if (!mochi::IsFinite(joint.min) || !mochi::IsFinite(joint.max)) {
        joint.min = joint.max = 0;
      }
      break;
    }
  }
}

} // namespace

bool CollectControllerTargets(
    robotics::BotScene& bots,
    DynamicArray<ControllerTarget>& outTargets,
    std::string& outError) {
  outTargets.clear();
  for (details::FloatInputController const& found : details::FindFloatInputControllers(bots)) {
    outTargets.emplace_back();
    ControllerTarget& target = outTargets.back();
    target.name = found.name;
    target.type = DynamicString(found.controller->GetTypeName());
    target.inputKind = DynamicString(InputKindName(found.controller->GetInputKind()));
    target.input.resize(static_cast<size_t>(found.controller->GetInputSize()));
    mochi::Error error;
    found.controller->GetCurrentInput(mochi::MakeSpan(target.input), error);
    if (!error.IsOK()) {
      outError = "Controller '" + std::string(std::string_view(found.name)) +
          "': " + error.GetDescription();
      return false;
    }
    DescribeInputJoints(*found.controller, target);
    if (mochi::Actor const* const actor = found.controller->GetActor()) {
      mochi::TransformRT const root = actor->GetRootTransform();
      target.rootPosition = root.GetTranslation();
      target.rootRotation = root.GetRotation().ToRotationVector();
    }
  }
  return true;
}

SceneTargets CollectSceneTargets(mochi::Scene& scene) {
  SceneTargets targets;
  mochi::ErrorLog error;
  for (mochi::Actor* actor : details::GetTopLevelActors(scene)) {
    switch (actor->GetType()) {
      case mochi::ActorType::Rigid: {
        RigidTarget rigid;
        rigid.name = actor->GetName();
        rigid.isStatic = actor->IsStatic();
        if (!rigid.isStatic) {
          rigid.mass = actor->GetMass(error);
          rigid.inertia = MeanDiagonalInertia(actor->GetRigidMomentOfInertiaLocal(error));
          rigid.centerOfMassLocal = actor->GetRigidCenterOfMassLocal(error);
        }
        targets.rigid.push_back(std::move(rigid));
        break;
      }
      case mochi::ActorType::Articulated: {
        mochi::ArticulatedShapeInfo const info = actor->GetArticulatedShapeInfo(error);
        ArticulatedTarget articulated;
        articulated.name = actor->GetName();
        articulated.numDofs = actor->GetNumDofs();
        articulated.links.assign(info.linkNames.begin(), info.linkNames.end());
        for (mochi::ActorHandle const link : actor->GetNestedLinkActors(error)) {
          // Links with no inertia (e.g. massless dummies) report no center of mass or inertia.
          mochi::Error comError;
          mochi::Error massError;
          mochi::Actor const* linkActor = scene.GetActor(link);
          Real3 com{0.0f, 0.0f, 0.0f};
          real mass = 0;
          real inertia = 0;
          if (linkActor != nullptr) {
            com = linkActor->GetRigidCenterOfMassLocal(comError);
            mass = linkActor->GetMass(massError);
            inertia = MeanDiagonalInertia(linkActor->GetRigidMomentOfInertiaLocal(massError));
          }
          articulated.linkCentersOfMass.push_back(comError.IsOK() ? com : Real3{0.0f, 0.0f, 0.0f});
          articulated.linkMasses.push_back(massError.IsOK() ? mass : real{0});
          articulated.linkInertias.push_back(massError.IsOK() ? inertia : real{0});
        }
        for (size_t i = 0; i < info.jointNames.size(); ++i) {
          JointTarget joint;
          joint.name = info.jointNames[i];
          joint.dofOffset = info.dofInfo[i].offset;
          joint.dofCount = info.dofInfo[i].transSize + info.dofInfo[i].rotSize;
          articulated.joints.push_back(std::move(joint));
        }
        targets.articulated.push_back(std::move(articulated));
        break;
      }
      case mochi::ActorType::None:
      case mochi::ActorType::Soft:
      case mochi::ActorType::Shell:
      case mochi::ActorType::Rod:
      case mochi::ActorType::Count:
        break;
    }
  }
  return targets;
}

namespace {

// A dynamic rigid body, or link number `link` of an articulated actor, found by its target name.
struct BodyTarget {
  RigidTarget const* rigid = nullptr;
  ArticulatedTarget const* articulated = nullptr;
  size_t link = 0;
};

std::optional<BodyTarget> FindBodyTarget(SceneTargets const& targets, std::string_view target) {
  for (RigidTarget const& rigid : targets.rigid) {
    if (!rigid.isStatic && std::string_view(rigid.name) == target) {
      return BodyTarget{.rigid = &rigid};
    }
  }
  // "actor/link", split at the last '/' (actor names may contain '/' from nested prefabs).
  size_t const slash = target.rfind('/');
  if (slash == std::string_view::npos) {
    return std::nullopt;
  }
  for (ArticulatedTarget const& articulated : targets.articulated) {
    if (std::string_view(articulated.name) != target.substr(0, slash)) {
      continue;
    }
    for (size_t i = 0; i < articulated.links.size(); ++i) {
      if (std::string_view(articulated.links[i]) == target.substr(slash + 1)) {
        return BodyTarget{.articulated = &articulated, .link = i};
      }
    }
  }
  return std::nullopt;
}

} // namespace

std::optional<Real3> FindCenterOfMassLocal(SceneTargets const& targets, std::string_view target) {
  std::optional<BodyTarget> const body = FindBodyTarget(targets, target);
  if (!body) {
    return std::nullopt;
  }
  if (body->rigid != nullptr) {
    return body->rigid->centerOfMassLocal;
  }
  auto const& coms = body->articulated->linkCentersOfMass;
  return body->link < coms.size() ? std::optional<Real3>(coms[body->link]) : std::nullopt;
}

std::optional<DriveScale> FindDriveScale(SceneTargets const& targets, std::string_view target) {
  std::optional<BodyTarget> const body = FindBodyTarget(targets, target);
  if (!body) {
    return std::nullopt;
  }
  if (body->rigid != nullptr) {
    return MakeDriveScale(body->rigid->mass, body->rigid->inertia, 0);
  }
  ArticulatedTarget const& articulated = *body->articulated;
  if (body->link >= articulated.linkMasses.size() ||
      body->link >= articulated.linkInertias.size()) {
    return std::nullopt;
  }
  real total = 0;
  for (real const mass : articulated.linkMasses) {
    total += mass;
  }
  return MakeDriveScale(
      articulated.linkMasses[body->link], articulated.linkInertias[body->link], total);
}

bool DescribeTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    SceneTargets& outTargets,
    std::string& outError) {
  mochi::Context* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  return DescribeTuner(spec, tunerFilePath, context, outTargets, outError);
}

bool DescribeTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    SceneTargets& outTargets,
    std::string& outError) {
  if (context == nullptr) {
    outError = "DescribeTuner needs a Mochi context.";
    return false;
  }
  mochi::Scene* scene = context->CreateScene("superdex_asset_tuner_describe");
  MOCHI_DEFER(context->DestroyScene(scene));
  TunerBots bots;
  std::optional<real> groundHeight;
  DynamicArray<mochi::Actor*> prefabActors;
  if (!BuildTunerScene(
          spec, tunerFilePath, context, *scene, bots, groundHeight, prefabActors, outError)) {
    return false;
  }
  outTargets = CollectSceneTargets(*scene);
  outTargets.groundHeight = groundHeight;
  robotics::BotScene* const botScene = bots.Get();
  return botScene == nullptr ||
      CollectControllerTargets(*botScene, outTargets.controllers, outError);
}

} // namespace superdex::tuner
