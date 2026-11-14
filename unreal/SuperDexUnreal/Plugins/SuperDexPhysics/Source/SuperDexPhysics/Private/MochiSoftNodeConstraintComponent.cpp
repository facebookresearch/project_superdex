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

#include "MochiSoftNodeConstraintComponent.h"
#include "MochiConversions.h"

#include <mochi_core/geometry/geometry_utils.h>

UMochiSoftNodeConstraintComponent::UMochiSoftNodeConstraintComponent() {
  bWantsOnUpdateTransform = true;
}

bool UMochiSoftNodeConstraintComponent::GetConstrainedNodeIndices(
    TArray<int>& OutNodeIndices) const {
  if (!AreConstraintsCreated() || !Nodes->bReady.load(std::memory_order_acquire)) {
    return false;
  }
  OutNodeIndices = Nodes->Indices;
  return true;
}

double UMochiSoftNodeConstraintComponent::GetSaturationScale() const {
  return FUnrealToMochiConverter().MeasureToMochi(1.0);
}

void UMochiSoftNodeConstraintComponent::OnUpdateTransform(
    EUpdateTransformFlags UpdateTransformFlags,
    ETeleportType Teleport) {
  Super::OnUpdateTransform(UpdateTransformFlags, Teleport);
  if (Type == EMochiSoftNodeConstraintType::NodePosition) {
    const mochi::Real3 position =
        FUnrealToMochiConverter().TranslationToMochi(GetComponentLocation());
    ForEachConstraint(
        [position](mochi::Constraint& C) { C.SetTargetPosition(position, mochi::ErrorLog{}); });
  }
}

EMochiBodyState UMochiSoftNodeConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody body;
  mochi::ActorHandle actor;
  const EMochiBodyState state = ResolveBody(Body, body, OutRequest.Scene, actor);
  if (state != EMochiBodyState::Ready) {
    return state;
  }
  // Mochi chooses the nodes for Closest, and InsideRigidBounds finds them on the Mochi thread.
  Nodes = std::make_shared<FNodes>();
  const bool bNodeByIndex = Type == EMochiSoftNodeConstraintType::NodePosition ||
      NodeSelection == EMochiSoftNodeSelection::Index;
  if (bNodeByIndex) {
    Nodes->Indices = {NodeIndex};
    Nodes->bReady = true;
  }
  const FUnrealToMochiConverter u2m;

  if (Type == EMochiSoftNodeConstraintType::NodePosition) {
    mochi::DeformableNodePositionConstraintParams params;
    FillCommonParams(params);
    params.actor = actor;
    params.nodeIndex = NodeIndex;
    params.position = u2m.TranslationToMochi(GetComponentLocation());
    OutRequest.CreateOne(params, &mochi::Scene::CreateDeformableNodePositionConstraint);
    return EMochiBodyState::Ready;
  }

  FMochiConstraintBody otherBody;
  mochi::ActorHandle otherActor;
  UMochiSceneObject* otherScene = nullptr;
  const EMochiBodyState otherState = ResolveBody(OtherBody, otherBody, otherScene, otherActor);
  if (otherState != EMochiBodyState::Ready) {
    return otherState;
  }
  if (otherScene != OutRequest.Scene) {
    LogError(TEXT("the bodies are not in the same Mochi scene."));
    return EMochiBodyState::Failed;
  }
  if (!bAllowContact) {
    OutRequest.NoContactPairs.emplace_back(actor, otherActor);
  }

  if (Type == EMochiSoftNodeConstraintType::NodeToNode) {
    if (NodeSelection == EMochiSoftNodeSelection::InsideRigidBounds) {
      LogError(TEXT("Inside Rigid Bounds only applies to Node to Rigid."));
      return EMochiBodyState::Failed;
    }
    mochi::DeformableNodeToDeformableNodeConstraintParams params;
    FillCommonParams(params);
    params.actorA = actor;
    params.actorB = otherActor;
    params.nodeIndexA = NodeIndex;
    params.nodeIndexB = OtherNodeIndex;
    params.findClosest = NodeSelection == EMochiSoftNodeSelection::Closest;
    OutRequest.CreateOne(params, &mochi::Scene::CreateDeformableNodeToDeformableNodeConstraint);
    return EMochiBodyState::Ready;
  }

  mochi::DeformableNodeToRigidConstraintParams params;
  FillCommonParams(params);
  params.deformableActor = actor;
  params.rigidActor = otherActor;
  params.deformableNodeIndex = NodeIndex;
  params.findClosest = NodeSelection == EMochiSoftNodeSelection::Closest;
  params.fixToDeformablePos = bFixToDeformablePosition;
  params.rigidLocalPos = otherBody.LocalPositionToMochi(RigidLocalPosition);
  if (NodeSelection != EMochiSoftNodeSelection::InsideRigidBounds) {
    OutRequest.CreateOne(params, &mochi::Scene::CreateDeformableNodeToRigidConstraint);
    return EMochiBodyState::Ready;
  }
  OutRequest.Create = [params, nodes = Nodes](mochi::Scene* Scene) {
    mochi::ErrorLog e;
    std::vector<mochi::ConstraintHandle> constraints;
    mochi::Actor* deformable = Scene->GetActor(params.deformableActor);
    const mochi::Actor* rigid = Scene->GetActor(params.rigidActor);
    if (deformable == nullptr || rigid == nullptr) {
      return constraints;
    }
    // The nodes inside the rigid body's world bounding box, in the deformable body's frame.
    const mochi::Obb rigidBoundsLocal = mochi::TransformShape(
        Invert(deformable->GetRootTransform()), mochi::GetObb(rigid->GetAabbWorld(e)));
    const mochi::QueryHandle query =
        deformable->RegisterQueryAndCompute(mochi::QueryType::NodePositions, e);
    deformable->QueryNodesInVolumeLocal(
        rigidBoundsLocal,
        false,
        [&](int Node, mochi::Real3 /*Position*/) {
          mochi::DeformableNodeToRigidConstraintParams nodeParams = params;
          nodeParams.deformableNodeIndex = Node;
          nodeParams.findClosest = false;
          // Each node ties where it is; one shared rigid point would collapse the region.
          nodeParams.fixToDeformablePos = true;
          if (const mochi::Constraint* constraint =
                  Scene->CreateDeformableNodeToRigidConstraint(nodeParams, e)) {
            constraints.push_back(constraint->GetHandle());
            nodes->Indices.Add(Node);
          }
        },
        e);
    deformable->CancelQuery(query);
    nodes->bReady.store(true, std::memory_order_release);
    return constraints;
  };
  return EMochiBodyState::Ready;
}
