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

#include <superdex_recorder/physics_recorder.h>

#include <superdex_recorder/series.h>

#include <mochi_core/utils/log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

constexpr std::string_view kActorArchetype = "superdex.archetypes.PhysicsActor";
constexpr std::string_view kContactArchetype = "superdex.archetypes.PhysicsContacts";
constexpr std::string_view kFrameArchetype = "superdex.archetypes.PhysicsFrame";

ComponentField ActorField(std::string component) {
  return {std::string{kActorArchetype}, std::move(component)};
}

// Recorded in this order under superdex.physics.contact_param_names, with the Python field names.
constexpr std::array<std::string_view, 14> kContactParamNames = {
    "penalty_coefficient",
    "penalty_smoothing_half_distance",
    "penalty_threshold_default",
    "penalty_threshold_extra_padding",
    "colliding_penalty_threshold_padding",
    "friction_with_collider_normal",
    "max_alignment_normals",
    "viscous_friction_coefficient",
    "coulomb_friction_coefficient",
    "friction_falloff_vel",
    "normal_viscous_damping_coefficient",
    "distance_error_bound",
    "obj_scale",
    "colliding_penalty_length_scale",
};

// Mirrors mochi::ContactParams so that a new field fails to compile until it is recorded.
struct RecordedContactParamsLayout {
  mochi::real penaltyCoefficient;
  mochi::real penaltySmoothingHalfDistance;
  mochi::real penaltyThresholdDefault;
  mochi::real penaltyThresholdExtraPadding;
  mochi::real collidingPenaltyThresholdPadding;
  bool frictionWithColliderNormal;
  mochi::real maxAlignmentNormals;
  mochi::real viscousFrictionCoefficient;
  mochi::real coulombFrictionCoefficient;
  mochi::real frictionFalloffVel;
  mochi::real normalViscousDampingCoefficient;
  mochi::real distanceErrorBound;
  mochi::real objScale;
  mochi::real collidingPenaltyLengthScale;
};
static_assert(
    sizeof(mochi::ContactParams) == sizeof(RecordedContactParamsLayout),
    "mochi::ContactParams changed; update kContactParamNames and ContactParamValues.");

std::array<double, kContactParamNames.size()> ContactParamValues(mochi::ContactParams const& p) {
  return {
      p.penaltyCoefficient,
      p.penaltySmoothingHalfDistance,
      p.penaltyThresholdDefault,
      p.penaltyThresholdExtraPadding,
      p.collidingPenaltyThresholdPadding,
      p.frictionWithColliderNormal ? 1.0 : 0.0,
      p.maxAlignmentNormals,
      p.viscousFrictionCoefficient,
      p.coulombFrictionCoefficient,
      p.frictionFalloffVel,
      p.normalViscousDampingCoefficient,
      p.distanceErrorBound,
      p.objScale,
      p.collidingPenaltyLengthScale,
  };
}

ComponentField FrameField(std::string component) {
  return {std::string{kFrameArchetype}, std::move(component)};
}

ComponentField ContactField(std::string component) {
  return {std::string{kContactArchetype}, std::move(component)};
}

std::string_view ActorTypeName(mochi::ActorType type) {
  switch (type) {
    case mochi::ActorType::Rigid:
      return "rigid";
    case mochi::ActorType::Articulated:
      return "articulated";
    case mochi::ActorType::Soft:
      return "soft";
    case mochi::ActorType::None:
    case mochi::ActorType::Shell:
    case mochi::ActorType::Rod:
    case mochi::ActorType::Count:
      break;
  }
  return "unrecorded";
}

std::string_view JointTypeName(mochi::ArticulatedJointType type) {
  switch (type) {
    case mochi::ArticulatedJointType::Free:
      return "free";
    case mochi::ArticulatedJointType::Prismatic:
      return "prismatic";
    case mochi::ArticulatedJointType::Revolute:
      return "revolute";
    case mochi::ArticulatedJointType::Spherical:
      return "spherical";
    case mochi::ArticulatedJointType::Hard:
      return "hard";
    case mochi::ArticulatedJointType::Cycle:
      return "cycle";
    case mochi::ArticulatedJointType::Count:
      return "invalid";
  }
  return "invalid";
}

template <typename Values>
mochi::Span<mochi::real const> AsSpan(Values const& values) {
  return {values.data(), static_cast<size_t>(values.size())};
}

std::array<uint8_t, 4> ColorForId(std::string_view id) {
  uint32_t hash = 2166136261U;
  for (unsigned char const character : id) {
    hash = (hash ^ character) * 16777619U;
  }
  // Mix related IDs before interpreting the hash bytes as RGB.
  hash ^= hash >> 16;
  hash *= 0x85ebca6bU;
  hash ^= hash >> 13;
  hash *= 0xc2b2ae35U;
  hash ^= hash >> 16;
  return {
      static_cast<uint8_t>(96U + (hash & 0x7fU)),
      static_cast<uint8_t>(96U + ((hash >> 8) & 0x7fU)),
      static_cast<uint8_t>(96U + ((hash >> 16) & 0x7fU)),
      255U};
}

std::string const* FindSpawnName(PhysicsRecorderConfig const& config, mochi::Actor const& actor) {
  auto const entry = std::ranges::find_if(config.spawnNames, [&](auto const& candidate) {
    return candidate.first == actor.GetHandle();
  });
  return entry == config.spawnNames.end() ? nullptr : &entry->second;
}

// Actor IDs are entity path segments, where `.` reads better than an escaped `\/`.
std::string WithoutSlashes(std::string name) {
  std::ranges::replace(name, '/', '.');
  return name;
}

std::string UniqueActorId(
    PhysicsRecorderConfig const& config,
    mochi::Actor const& actor,
    std::unordered_set<std::string>& ids) {
  char const* const name = actor.GetName();
  std::string base = name == nullptr || *name == '\0' ? "actor" : WithoutSlashes(name);
  if (std::string const* const spawnName = FindSpawnName(config, actor)) {
    auto const spawnBodies = std::count_if(
        config.spawnNames.begin(), config.spawnNames.end(), [&](auto const& candidate) {
          return candidate.second == *spawnName;
        });
    base = spawnBodies == 1 ? WithoutSlashes(*spawnName) : WithoutSlashes(*spawnName) + "." + base;
  }
  std::string id = base;
  for (uint32_t suffix = 1; !ids.insert(id).second; ++suffix) {
    id = base + "_" + std::to_string(suffix);
  }
  return id;
}

mochi::Actor* ResolveActor(mochi::Scene& scene, mochi::ActorHandle handle, mochi::Error& error) {
  mochi::Actor* const actor = scene.GetActor(handle);
  MOCHI_ERROR_IF(actor == nullptr, error, "A recorded actor was destroyed.");
  return actor;
}

} // namespace

PhysicsRecorder::Link::Link(std::string name, TransformSeries& transformSeries)
    : name{std::move(name)}, transformSeries{transformSeries} {}

PhysicsRecorder::Actor::Actor(
    std::string id,
    std::string dataPath,
    mochi::ActorHandle handle,
    ScalarSeries& stateSeries,
    TransformSeries& transformSeries)
    : id{std::move(id)},
      dataPath{std::move(dataPath)},
      handle{handle},
      stateSeries{stateSeries},
      transformSeries{transformSeries} {}

PhysicsRecorder::PhysicsRecorder(Episode& episode, mochi::Scene& scene, mochi::Error& error)
    : PhysicsRecorder{episode, scene, PhysicsRecorderConfig{}, error} {}

PhysicsRecorder::PhysicsRecorder(
    Episode& episode,
    mochi::Scene& scene,
    PhysicsRecorderConfig const& config,
    mochi::Error& error)
    : _episode{episode},
      _scene{scene},
      _config{config},
      _contactRegistrationTimeSec{scene.GetTotalSimulationTime()} {
  MOCHI_ERROR_IF_NOT(
      !_config.recordContactForces ||
          (_config.contactForceVisualizationScaleMetersPerNewton > mochi::real{} &&
           std::isfinite(_config.contactForceVisualizationScaleMetersPerNewton)),
      error,
      "Contact force visualization scale must be finite and positive.");
  MOCHI_ERROR_IF(
      std::ranges::any_of(
          _config.spawnNames, [](auto const& entry) { return entry.second.empty(); }),
      error,
      "Spawn names must not be empty.");
  MOCHI_ERROR_RETURN(error);
  _frameSeries = _episode.CreateScalarSeries(
      "data/physics/frames",
      {
          FrameField("superdex.physics.step_dt_sec"),
      },
      error);
  std::vector<std::string> const contactParamNames{
      kContactParamNames.begin(), kContactParamNames.end()};
  _episode.WriteConstantTexts(
      "data/physics/actors",
      ActorField("superdex.physics.contact_param_names"),
      contactParamNames,
      error);
  MOCHI_ERROR_RETURN(error);

  std::unordered_set<std::string> ids;
  std::vector<char const*> unrecordedActorNames;
  std::vector<char const*> contactlessActorNames;
  scene.ForEachActor([&](mochi::Actor* actor) {
    if (!error.IsOK() || actor == nullptr || actor->IsNestedLinkActor()) {
      return;
    }
    // A nested soft actor moves with its articulated actor, which is recorded.
    if (actor->IsNestedSoftActor()) {
      contactlessActorNames.push_back(actor->GetName());
      return;
    }
    if (actor->GetType() != mochi::ActorType::Rigid &&
        actor->GetType() != mochi::ActorType::Articulated &&
        actor->GetType() != mochi::ActorType::Soft) {
      unrecordedActorNames.push_back(actor->GetName());
      return;
    }
    if (actor->GetType() == mochi::ActorType::Soft) {
      contactlessActorNames.push_back(actor->GetName());
    }
    RegisterActor(UniqueActorId(_config, *actor, ids), *actor, error);
  });
  MOCHI_ERROR_RETURN(error);
  if (_config.recordContactForces) {
    // Contacts with these actors would have no recorded body to index.
    contactlessActorNames.insert(
        contactlessActorNames.end(), unrecordedActorNames.begin(), unrecordedActorNames.end());
    for (char const* const name : contactlessActorNames) {
      MOCHI_LOG_ERROR("Contacts with actor %s cannot be recorded.", name);
    }
    MOCHI_ERROR_IF(
        !contactlessActorNames.empty(),
        error,
        "Contact recording supports only rigid and articulated actors.");
    MOCHI_ERROR_RETURN(error);
  }
  for (char const* const name : unrecordedActorNames) {
    MOCHI_LOG_WARNING(
        "Not recording actor %s: only rigid, articulated and soft actors are recorded, so this "
        "episode cannot be restored.",
        name);
  }

  if (_config.recordContactForces) {
    _contactCollector = std::make_unique<ContactCollector>(scene, std::move(_contactBodies), error);
    MOCHI_ERROR_RETURN(error);
    _contactSeries = _episode.CreateScalarSeries(
        "data/physics/contacts",
        {
            ContactField("superdex.physics.contact_body_a_indices"),
            ContactField("superdex.physics.contact_body_b_indices"),
            ContactField("superdex.physics.contact_position_a_world_xyz"),
            ContactField("superdex.physics.contact_position_b_world_xyz"),
            ContactField("superdex.physics.contact_force_on_a_world_xyz"),
        },
        error);
    _contactArrowSeries = _episode.CreateArrow3DSeries("world/contact_forces", error);
    std::vector<std::string> bodyPaths;
    for (auto const& body : _contactCollector->GetBodies()) {
      bodyPaths.push_back(body.path);
    }
    _episode.WriteConstantTexts(
        "data/physics/contacts",
        ContactField("superdex.physics.contact_body_paths"),
        bodyPaths,
        error);
    _episode.WriteConstantText(
        "data/physics/contacts",
        ContactField("superdex.physics.contact_force_source"),
        "simulation",
        error);
    std::array<double, 1> const visualizationScale{
        static_cast<double>(_config.contactForceVisualizationScaleMetersPerNewton)};
    _episode.WriteConstantScalars(
        "data/physics/contacts",
        ContactField("superdex.physics.contact_force_visual_scale_m_per_n"),
        visualizationScale,
        error);
  }
}

PhysicsRecorder::~PhysicsRecorder() {
  // Mochi computes a registered query's result at every step until it is cancelled.
  for (auto const& registered : _actors) {
    if (!registered.queries.empty()) {
      if (mochi::Actor* const actor = _scene.GetActor(registered.handle)) {
        for (mochi::QueryHandle const query : registered.queries) {
          actor->CancelQuery(query);
        }
      }
    }
  }
}

void PhysicsRecorder::RegisterActor(std::string id, mochi::Actor& actor, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::string const encodedId = detail::EncodeEntityPathSegment(id);
  std::string const dataPath = "data/physics/actors/" + encodedId;
  std::string const entityPath = "world/" + encodedId;

  bool const hasPoseController =
      actor.GetType() == mochi::ActorType::Articulated && actor.HasArticulatedPoseController(error);
  MOCHI_ERROR_RETURN(error);
  auto* const transformSeries = _episode.CreateTransformSeries(entityPath, error);
  std::vector<ComponentField> fields{
      ActorField("superdex.physics.root_translation_xyz"),
      ActorField("superdex.physics.root_quaternion_xyzw"),
  };
  if (actor.GetType() == mochi::ActorType::Rigid) {
    fields.push_back(ActorField("superdex.physics.linear_velocity_xyz"));
    fields.push_back(ActorField("superdex.physics.angular_velocity_xyz"));
    if (!actor.IsStatic()) {
      fields.push_back(ActorField("superdex.physics.com_translation_xyz"));
    }
  } else if (actor.GetType() == mochi::ActorType::Soft) {
    fields.push_back(ActorField("superdex.physics.displacements"));
    fields.push_back(ActorField("superdex.physics.node_velocities"));
  } else {
    fields.push_back(ActorField("superdex.physics.joint_positions"));
    fields.push_back(ActorField("superdex.physics.joint_velocities"));
    if (hasPoseController) {
      fields.push_back(ActorField("superdex.physics.controller_target_positions"));
    }
  }
  auto* const stateSeries = _episode.CreateScalarSeries(dataPath, fields, error);
  MOCHI_ERROR_RETURN(error);
  if (transformSeries == nullptr || stateSeries == nullptr) {
    MOCHI_ERROR_SET(error, "Failed to create actor recording series.");
    return;
  }

  Actor registered{std::move(id), dataPath, actor.GetHandle(), *stateSeries, *transformSeries};

  if (actor.GetType() == mochi::ActorType::Articulated) {
    auto const linkHandles = actor.GetNestedLinkActors(error);
    auto const shapeInfo = actor.GetArticulatedShapeInfo(error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF_NOT(
        linkHandles.size() == shapeInfo.linkNames.size(),
        error,
        "Articulated actor link handles and names have inconsistent sizes.");
    size_t const numLinks = linkHandles.size();
    MOCHI_ERROR_IF_NOT(
        shapeInfo.jointTypes.size() >= numLinks && shapeInfo.jointAxes.size() >= numLinks &&
            shapeInfo.jointFromChildLink.size() >= numLinks &&
            shapeInfo.parentLinkFromJoint.size() >= numLinks,
        error,
        "Articulated actor joint metadata is incomplete.");
    MOCHI_ERROR_RETURN(error);
    // Each link is recorded relative to its parent link, so a link on a fixed joint is recorded
    // once.
    auto const linkPaths =
        detail::NestedLinkPaths(shapeInfo.parents, shapeInfo.linkNames, std::nullopt, error);
    // The per-joint arrays end with the cycle joints, which have no DoFs and no child link of
    // their own.
    registered.linkKinematics.emplace(
        shapeInfo.parents,
        shapeInfo.jointTypes.subspan(0, numLinks),
        shapeInfo.jointAxes.subspan(0, numLinks),
        shapeInfo.jointFromChildLink.subspan(0, numLinks),
        shapeInfo.parentLinkFromJoint.subspan(0, numLinks),
        error);
    MOCHI_ERROR_RETURN(error);

    registered.links.reserve(linkHandles.size());
    for (size_t i = 0; i < linkHandles.size(); ++i) {
      mochi::Actor* const linkActor = _scene.GetActor(linkHandles[i]);
      if (linkActor == nullptr) {
        MOCHI_ERROR_SET(error, "Articulated actor has an invalid link.");
        return;
      }
      std::string const linkName = std::string{shapeInfo.linkNames[i]};
      std::string const linkPath = entityPath + "/links/" + linkPaths[i];
      auto* const linkSeries = _episode.CreateTransformSeries(linkPath, error);
      MOCHI_ERROR_RETURN(error);
      if (linkSeries == nullptr) {
        MOCHI_ERROR_SET(error, "Failed to create link recording series.");
        return;
      }
      _episode.WriteConstantText(
          linkPath, ActorField("superdex.physics.link_name"), linkName, error);
      WriteGeometry(linkPath, *linkActor, error);
      WritePhysicalParameters(linkPath, *linkActor, error);
      MOCHI_ERROR_RETURN(error);
      RegisterContactBody(linkPath, *linkActor);
      MOCHI_ERROR_RETURN(error);
      registered.links.emplace_back(linkName, *linkSeries);
    }

    // A skinned articulation keeps its contact skin's parameters on the root actor. The root's
    // mass and density aren't written: they describe the whole articulation, not one body.
    mochi::Error skinContactError;
    mochi::ContactParams const skinContact = actor.GetContactParams(skinContactError);
    if (skinContactError.IsOK()) {
      _episode.WriteConstantScalars(
          dataPath,
          ActorField("superdex.physics.contact_params"),
          ContactParamValues(skinContact),
          error);
    }

    MOCHI_ERROR_IF_NOT(
        shapeInfo.jointNames.size() == shapeInfo.jointTypes.size() &&
            shapeInfo.jointNames.size() == shapeInfo.dofInfo.size(),
        error,
        "Articulated actor joint metadata has inconsistent sizes.");
    MOCHI_ERROR_RETURN(error);

    std::vector<std::string> jointNames;
    std::vector<std::string> jointTypes;
    std::vector<double> jointDofOffsets;
    std::vector<double> jointTranslationDofCounts;
    std::vector<double> jointRotationDofCounts;
    jointNames.reserve(shapeInfo.jointNames.size());
    jointTypes.reserve(shapeInfo.jointTypes.size());
    jointDofOffsets.reserve(shapeInfo.dofInfo.size());
    jointTranslationDofCounts.reserve(shapeInfo.dofInfo.size());
    jointRotationDofCounts.reserve(shapeInfo.dofInfo.size());
    for (size_t i = 0; i < shapeInfo.jointNames.size(); ++i) {
      jointNames.emplace_back(shapeInfo.jointNames[i]);
      jointTypes.emplace_back(JointTypeName(shapeInfo.jointTypes[i]));
      jointDofOffsets.push_back(static_cast<double>(shapeInfo.dofInfo[i].offset));
      jointTranslationDofCounts.push_back(static_cast<double>(shapeInfo.dofInfo[i].transSize));
      jointRotationDofCounts.push_back(static_cast<double>(shapeInfo.dofInfo[i].rotSize));
    }
    if (!jointNames.empty()) {
      std::string const jointLayoutPath = dataPath + "/joint_layout";
      _episode.WriteConstantTexts(
          jointLayoutPath, ActorField("superdex.physics.joint_names"), jointNames, error);
      _episode.WriteConstantTexts(
          jointLayoutPath, ActorField("superdex.physics.joint_types"), jointTypes, error);
      _episode.WriteConstantScalars(
          jointLayoutPath,
          ActorField("superdex.physics.joint_dof_offsets"),
          jointDofOffsets,
          error);
      _episode.WriteConstantScalars(
          jointLayoutPath,
          ActorField("superdex.physics.joint_translation_dof_counts"),
          jointTranslationDofCounts,
          error);
      _episode.WriteConstantScalars(
          jointLayoutPath,
          ActorField("superdex.physics.joint_rotation_dof_counts"),
          jointRotationDofCounts,
          error);
      MOCHI_ERROR_RETURN(error);
    }

    int const numDofs = actor.GetNumDofs();
    registered.positions.resize(numDofs);
    registered.velocities.resize(numDofs);
    registered.hasPoseController = hasPoseController;
    if (hasPoseController) {
      registered.controllerTargets.resize(numDofs);
    }
  } else if (actor.GetType() == mochi::ActorType::Soft) {
    WritePhysicalParameters(dataPath, actor, error);
    auto const mesh = actor.GetSurfaceMesh();
    MOCHI_ERROR_IF(
        mesh.nodesPerElement != 3, error, "Soft actor surface mesh is not a triangle mesh.");
    MOCHI_ERROR_RETURN(error);
    // A negative index wraps past the vertex count, which CreateDeformableMeshSeries rejects.
    std::vector<uint32_t> const triangles(mesh.connectivity.begin(), mesh.connectivity.end());
    registered.surfaceSeries = _episode.CreateDeformableMeshSeries(
        entityPath + "/surface",
        mesh.coordinates.size() / 3,
        triangles,
        ColorForId(entityPath),
        error);
  } else {
    WriteGeometry(entityPath, actor, error);
    WritePhysicalParameters(dataPath, actor, error);
    RegisterContactBody(entityPath, actor);
  }
  MOCHI_ERROR_RETURN(error);

  char const* const runtimeName = actor.GetName();
  _episode.WriteConstantText(
      dataPath, ActorField("superdex.physics.actor_id"), registered.id, error);
  _episode.WriteConstantText(
      dataPath,
      ActorField("superdex.physics.runtime_name"),
      runtimeName == nullptr ? "" : runtimeName,
      error);
  if (std::string const* const spawnName = FindSpawnName(_config, actor)) {
    _episode.WriteConstantText(
        dataPath, ActorField("superdex.physics.spawn_name"), *spawnName, error);
  }
  _episode.WriteConstantText(
      dataPath, ActorField("superdex.physics.actor_type"), ActorTypeName(actor.GetType()), error);
  std::array<double, 1> const isStatic{actor.IsStatic() ? 1.0 : 0.0};
  std::array<double, 1> const numDofs{static_cast<double>(actor.GetNumDofs())};
  _episode.WriteConstantScalars(
      dataPath, ActorField("superdex.physics.is_static"), isStatic, error);
  _episode.WriteConstantScalars(dataPath, ActorField("superdex.physics.num_dofs"), numDofs, error);
  MOCHI_ERROR_RETURN(error);
  if (registered.surfaceSeries != nullptr) {
    // Mochi computes surface vertices only for registered queries. Computing them now also covers
    // a Record before the first step.
    for (mochi::QueryType const query :
         {mochi::QueryType::SurfaceNodePositions, mochi::QueryType::SurfaceNodeNormals}) {
      registered.queries.push_back(actor.RegisterQueryAndCompute(query, error));
    }
  }
  _actors.push_back(std::move(registered));
}

void PhysicsRecorder::WritePhysicalParameters(
    std::string_view path,
    mochi::Actor const& actor,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Links without a collider are massless frames with nothing to reapply, and static bodies have
  // no mass.
  mochi::Error contactError;
  mochi::ContactParams const contact = actor.GetContactParams(contactError);
  if (!contactError.IsOK()) {
    return;
  }
  _episode.WriteConstantScalars(
      path, ActorField("superdex.physics.contact_params"), ContactParamValues(contact), error);
  if (actor.IsStatic()) {
    return;
  }
  mochi::Error massError;
  std::array<double, 1> const density{static_cast<double>(actor.GetDensity(massError))};
  std::array<double, 1> const mass{static_cast<double>(actor.GetMass(massError))};
  if (massError.IsOK()) {
    _episode.WriteConstantScalars(path, ActorField("superdex.physics.density"), density, error);
    _episode.WriteConstantScalars(path, ActorField("superdex.physics.mass"), mass, error);
  }
}

void PhysicsRecorder::RegisterContactBody(std::string path, mochi::Actor& actor) {
  if (_config.recordContactForces) {
    _contactBodies.push_back({std::move(path), actor.GetHandle()});
  }
}

void PhysicsRecorder::RecordContacts(SampleTime time, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (_contactCollector == nullptr || _contactSeries == nullptr || _contactArrowSeries == nullptr ||
      time.simulationTimeSec <= _contactRegistrationTimeSec) {
    return;
  }

  _contactColumns.Clear();
  _contactCollector->Collect(_contactColumns, error);
  MOCHI_ERROR_RETURN(error);
  size_t const numContacts = _contactColumns.bodyA.size();
  std::vector<mochi::Real3> arrowOrigins(numContacts);
  std::vector<mochi::Real3> arrowVectors(numContacts);
  for (size_t i = 0; i < numContacts; ++i) {
    for (size_t axis = 0; axis < 3; ++axis) {
      arrowOrigins[i][axis] = _contactColumns.positionA[3 * i + axis];
      arrowVectors[i][axis] = _contactColumns.forceOnA[3 * i + axis] *
          _config.contactForceVisualizationScaleMetersPerNewton;
    }
  }

  std::array<mochi::Span<mochi::real const>, 5> const fields{
      AsSpan(_contactColumns.bodyA),
      AsSpan(_contactColumns.bodyB),
      AsSpan(_contactColumns.positionA),
      AsSpan(_contactColumns.positionB),
      AsSpan(_contactColumns.forceOnA),
  };
  _contactSeries->Append(time, fields, error);
  _contactArrowSeries->Append(time, arrowOrigins, arrowVectors, error);
}

void PhysicsRecorder::WriteGeometry(
    std::string_view entityPath,
    mochi::Actor const& actor,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const mesh = actor.GetSurfaceMesh();
  if (mesh.nodesPerElement != 3 || mesh.coordinates.empty() || mesh.connectivity.empty()) {
    return;
  }

  std::vector<float> vertices;
  vertices.reserve(mesh.coordinates.size());
  std::ranges::transform(mesh.coordinates, std::back_inserter(vertices), [](mochi::real value) {
    return static_cast<float>(value);
  });

  std::vector<uint32_t> triangles;
  triangles.reserve(mesh.connectivity.size());
  MOCHI_ERROR_IF_NOT(
      std::ranges::all_of(mesh.connectivity, [](int index) { return index >= 0; }),
      error,
      "Physics mesh contains a negative vertex index.");
  MOCHI_ERROR_RETURN(error);
  std::ranges::transform(mesh.connectivity, std::back_inserter(triangles), [](int index) {
    return static_cast<uint32_t>(index);
  });
  _episode.WriteMesh(entityPath, vertices, triangles, ColorForId(entityPath), error);
}

void PhysicsRecorder::Record(SampleTime time, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (_frameSeries == nullptr) {
    MOCHI_ERROR_SET(error, "Physics frame series is unavailable.");
    return;
  }
  std::array<mochi::real, 1> const stepDuration{static_cast<mochi::real>(_scene.GetLastTimeStep())};
  std::array<mochi::Span<mochi::real const>, 1> const frameFields{AsSpan(stepDuration)};
  _frameSeries->Append(time, frameFields, error);
  MOCHI_ERROR_RETURN(error);

  for (auto& registered : _actors) {
    mochi::Actor* const actor = ResolveActor(_scene, registered.handle, error);
    MOCHI_ERROR_RETURN(error);
    if (actor->IsStatic() && registered.stateSeries.GetNumSamples() != 0) {
      continue;
    }

    auto const worldFromActor = actor->GetRootTransform();
    auto const rootTranslation = worldFromActor.GetTranslation();
    auto const& rootRotation = worldFromActor.GetRotation();
    std::array<mochi::real, 3> const translation{
        rootTranslation[0], rootTranslation[1], rootTranslation[2]};
    std::array<mochi::real, 4> const quaternion{
        rootRotation.data[0], rootRotation.data[1], rootRotation.data[2], rootRotation.data[3]};
    registered.transformSeries.Append(time, worldFromActor, error);
    MOCHI_ERROR_RETURN(error);

    if (actor->GetType() == mochi::ActorType::Rigid) {
      auto const linearVelocity = actor->GetLinearVelocity(error);
      auto const angularVelocity = actor->GetAngularVelocity(error);
      // The root transform is rounded from the simulated center-of-mass transform, so restoring
      // the state exactly needs the center of mass itself.
      mochi::Real3 comTranslation;
      if (!actor->IsStatic()) {
        comTranslation = actor->GetCenterOfMassTransform(error).GetTranslation();
      }
      MOCHI_ERROR_RETURN(error);
      std::array<mochi::Span<mochi::real const>, 5> const fields{
          AsSpan(translation),
          AsSpan(quaternion),
          AsSpan(linearVelocity),
          AsSpan(angularVelocity),
          AsSpan(comTranslation),
      };
      size_t const numFields = actor->IsStatic() ? fields.size() - 1 : fields.size();
      registered.stateSeries.Append(
          time, mochi::Span<mochi::Span<mochi::real const> const>(fields.data(), numFields), error);
      MOCHI_ERROR_RETURN(error);
      continue;
    }

    if (actor->GetType() == mochi::ActorType::Soft) {
      std::array<mochi::Span<mochi::real const>, 4> const fields{
          AsSpan(translation),
          AsSpan(quaternion),
          actor->GetDisplacements(error),
          actor->GetNodeVelocitiesLocal(error),
      };
      MOCHI_ERROR_RETURN(error);
      registered.stateSeries.Append(time, fields, error);
      auto const surfacePositions = actor->GetSurfaceMeshNodePositionsLocal(error);
      auto const surfaceNormals = actor->GetSurfaceMeshNodeNormalsLocal(error);
      registered.surfaceSeries->Append(time, surfacePositions, surfaceNormals, error);
      MOCHI_ERROR_RETURN(error);
      continue;
    }

    actor->GetArticulatedPose(registered.positions, error);
    actor->GetArticulatedJointVelocities(registered.velocities, error);
    MOCHI_ERROR_RETURN(error);
    if (registered.hasPoseController) {
      actor->GetArticulatedTargetPose(registered.controllerTargets, error);
      MOCHI_ERROR_RETURN(error);
    }
    std::array<mochi::Span<mochi::real const>, 5> const fields{
        AsSpan(translation),
        AsSpan(quaternion),
        AsSpan(registered.positions),
        AsSpan(registered.velocities),
        AsSpan(registered.controllerTargets),
    };
    size_t const numFields = registered.hasPoseController ? fields.size() : fields.size() - 1;
    registered.stateSeries.Append(
        time, mochi::Span<mochi::Span<mochi::real const> const>(fields.data(), numFields), error);
    MOCHI_ERROR_RETURN(error);

    registered.linkKinematics->Compute(registered.positions, error);
    MOCHI_ERROR_RETURN(error);
    auto const parentFromLinks = registered.linkKinematics->GetParentFromLinks();
    for (size_t i = 0; i < registered.links.size(); ++i) {
      registered.links[i].transformSeries.Append(time, parentFromLinks[i], error);
      MOCHI_ERROR_RETURN(error);
    }
  }
  RecordContacts(time, error);
}

} // namespace superdex::recorder
