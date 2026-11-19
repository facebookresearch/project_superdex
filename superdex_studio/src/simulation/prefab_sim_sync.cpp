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

#include "simulation/prefab_sim_sync.h"

#include "simulation/deformable_surface_queries.h"

#include <mochi_core/utils/basic_utils.h>

#include <tuple>
#include <utility>

namespace superdex::studio {

void PrefabSimSync::CaptureStagedSurfaces(SceneStage const& stage) {
  _stagedSurfaces.clear();
  for (StagedActor const& staged : stage.GetActors()) {
    _stagedSurfaces.try_emplace(staged.name, staged.deformableSurfaces);
  }
}

bool PrefabSimSync::Track(mochi::Span<mochi::Actor* const> actors) {
  _actors.clear();
  _deformableQueryState.clear();
  mochi::ErrorLog e;
  for (auto* actor : actors) {
    _actors.push_back(actor->GetHandle());
  }
  // Register per-soft-actor node queries so the deforming meshes can be read each step. Kept until
  // the actors are torn down (cancelled in Untrack).
  for (auto* actor : actors) {
    // Soft actors expose their deforming meshes; articulated actors expose a surface only when they
    // carry a skin (folded into the compound entity, sharing the actor's handle). Both are read the
    // same way and emitted as DeformableMeshUpdates each step, so they share this query list. Rod
    // actors too: their visual mesh or contact skin through the same queries, their centerline from
    // their pose.
    bool const isSoft = actor->GetType() == mochi::ActorType::Soft;
    bool const isRod = actor->GetType() == mochi::ActorType::Rod;
    bool const isSkinnedArticulated =
        actor->GetType() == mochi::ActorType::Articulated && !actor->GetSurfaceMesh().IsEmpty();
    if (!isSoft && !isRod && !isSkinnedArticulated) {
      continue;
    }
    DeformableQueryState state;
    state.actor = actor->GetHandle();
    state.isRod = isRod;
    char const* const name = actor->GetName();
    std::string const actorName = name ? name : "";
    // The skin has no independent actor, so it is identified by its skin staged name to match the
    // deforming mesh SceneStage staged for it (ApplyDeformableMeshUpdates matches by name).
    state.name = isSkinnedArticulated ? SkinStagedName(actorName) : actorName;
    // Query the meshes the staged actor was built from, so each query feeds its staged mesh.
    auto const staged = _stagedSurfaces.find(state.name);
    if (staged == _stagedSurfaces.end()) {
      MOCHI_LOG_ERROR(
          "PrefabSimSync::Track: no staged actor named '%s'; skipping surface queries.",
          state.name.c_str());
      continue;
    }
    state.surfaces = staged->second;
    std::tie(state.positions, state.normals) =
        RegisterDeformableSurfaceQueries(*actor, state.surfaces.rendered, e);
    if (state.surfaces.collision != state.surfaces.rendered) {
      std::tie(state.collisionPositions, state.collisionNormals) =
          RegisterDeformableSurfaceQueries(*actor, state.surfaces.collision, e);
    }
    if (!e.IsOK()) {
      // Abort setup rather than storing a half-registered actor whose per-frame mesh update would
      // read nothing.
      MOCHI_LOG_ERROR("Failed to register actor surface queries");
      state.CancelQueries(*actor);
      return false;
    }
    _deformableQueryState.push_back(std::move(state));
  }
  return true;
}

void PrefabSimSync::DeformableQueryState::CancelQueries(mochi::Actor& owner) const {
  owner.CancelQuery(positions);
  owner.CancelQuery(normals);
  owner.CancelQuery(collisionPositions);
  owner.CancelQuery(collisionNormals);
}

void PrefabSimSync::Untrack(mochi::Scene* scene) {
  for (auto const& state : _deformableQueryState) {
    if (auto* actor = scene->GetActor(state.actor)) {
      state.CancelQueries(*actor);
    }
  }
}

void PrefabSimSync::Produce(mochi::Scene* scene) {
  mochi::ErrorLog e;
  auto& data = _simData.GetProducerData();
  data.actorTransforms.clear();
  data.actorNames.clear();
  data.skinnedPoseUpdates.clear();
  // Build the ordered transform list to match StagePrefab's staged-actor order: articulated
  // actors expand to their nested link transforms; rigid, soft and rod actors push their root
  // transform. Soft and rod actors are included here so their world transform is applied by name
  // and their row appears in the Render Scene Stage debug view; their deformed meshes are
  // additionally applied via the per-vertex mesh-update channel below (ApplyDeformableMeshUpdates).
  for (auto const& handle : _actors) {
    auto* actor = scene->GetActor(handle);
    if (!actor) {
      continue;
    }
    if (actor->GetType() == mochi::ActorType::Articulated) {
      auto const links = actor->GetNestedLinkActors(e);
      std::vector<mochi::TransformRT> linkTransforms(links.size());
      actor->GetArticulatedLinkTransforms(mochi::MakeSpan(linkTransforms), e);
      for (int i = 0; i < mochi::isize(linkTransforms); ++i) {
        data.actorTransforms.push_back(linkTransforms[i]);
        char const* const linkName = scene->GetActor(links[i])->GetName();
        data.actorNames.emplace_back(linkName ? linkName : "");
      }
      // If this articulated actor carries a skin, add a row for it (matched by skin staged
      // name) so its transform is applied and its row appears in the Render Scene Stage debug
      // view; its deformed surface is additionally applied via the mesh-update channel below.
      if (!actor->GetSurfaceMesh().IsEmpty()) {
        char const* const actorName = actor->GetName();
        data.actorTransforms.push_back(actor->GetRootTransform());
        data.actorNames.push_back(SkinStagedName(actorName ? actorName : ""));
        // Pose the skin's render model from this actor's link world transforms (nested-link
        // order == GLB joint order). ApplySkinnedPose no-ops if the skin has no render model.
        SkinnedPoseUpdate poseUpdate;
        poseUpdate.name = SkinStagedName(actorName ? actorName : "");
        poseUpdate.linkWorldTransforms.assign(linkTransforms.begin(), linkTransforms.end());
        data.skinnedPoseUpdates.push_back(std::move(poseUpdate));
      }
    } else if (
        actor->GetType() == mochi::ActorType::Rigid || actor->GetType() == mochi::ActorType::Soft ||
        actor->GetType() == mochi::ActorType::Rod) {
      data.actorTransforms.push_back(actor->GetRootTransform());
      char const* const name = actor->GetName();
      data.actorNames.emplace_back(name ? name : "");
    }
  }
  // Soft actors: overwrite per-vertex mesh deformations (matched to staged meshes by name),
  // converting local positions/normals to renderer space. The DeformableMeshUpdate entries and
  // their position/normal buffers are kept resident across steps and resized/overwritten in place
  // to avoid per-step reallocation; the vector is trimmed to the number of live soft actors so the
  // consumer sees exactly the valid updates.
  data.deformableMeshUpdates.resize(_deformableQueryState.size());
  int updateCount = 0;
  for (auto const& state : _deformableQueryState) {
    auto* actor = scene->GetActor(state.actor);
    if (!actor) {
      continue;
    }
    DeformableMeshUpdate& update = data.deformableMeshUpdates[updateCount++];
    update.name = state.name;
    update.worldTransform = actor->GetRootTransform();
    ReadDeformableSurfaceNodes(
        *actor, state.surfaces.rendered, _editorToRenderer, update.positions, update.normals, e);
    if (state.surfaces.collision != state.surfaces.rendered) {
      ReadDeformableSurfaceNodes(
          *actor,
          state.surfaces.collision,
          _editorToRenderer,
          update.collisionPositions,
          update.collisionNormals,
          e);
    } else {
      update.collisionPositions.clear();
      update.collisionNormals.clear();
    }
    if (state.isRod) {
      // Its own error: a failure elsewhere this step must not drop the rod's pose.
      mochi::ErrorLog rodError;
      ReadRodPose(*actor, update.rodPose, rodError);
      // A centerline has no node queries, so its slot's positions come from the pose. Only a rod
      // without a visual mesh or contact skin draws its centerline in the render slot, and then
      // also in the shape slot.
      if (state.surfaces.rendered == DeformableSurface::Centerline) {
        ConvertRodPoseNodes(update.rodPose, _editorToRenderer, update.positions);
      } else if (state.surfaces.collision == DeformableSurface::Centerline) {
        ConvertRodPoseNodes(update.rodPose, _editorToRenderer, update.collisionPositions);
      }
    } else {
      update.rodPose.nodes.clear();
      update.rodPose.frameAxes.clear();
    }
  }
  // Trim any trailing entries left over from soft actors that were missing this step (rare;
  // reuse keeps the common all-present case allocation-free).
  data.deformableMeshUpdates.resize(updateCount);
  _simData.Produce();
}

bool PrefabSimSync::Apply(
    SceneStage& stage,
    mochi::CoordinateSpaceConverter const& editorToRenderer) {
  if (!_simData.Consume()) {
    return false;
  }
  auto const& data = _simData.GetConsumerData();
  // Rigid, soft, and articulated actors are matched by name here. Soft actors additionally have
  // their deformed surface mesh applied by ApplyDeformableMeshUpdates below.
  stage.ApplyWorldTransforms(
      mochi::MakeConstSpan(data.actorNames),
      mochi::MakeConstSpan(data.actorTransforms),
      editorToRenderer);
  stage.ApplyDeformableMeshUpdates(
      mochi::MakeConstSpan(data.deformableMeshUpdates), editorToRenderer);
  stage.ApplySkinnedPose(mochi::MakeConstSpan(data.skinnedPoseUpdates), editorToRenderer);
  return true;
}

void PrefabSimSync::Reset() {
  _deformableQueryState.clear();
  _simData.Consume();
  _actors.clear();
}

} // namespace superdex::studio
