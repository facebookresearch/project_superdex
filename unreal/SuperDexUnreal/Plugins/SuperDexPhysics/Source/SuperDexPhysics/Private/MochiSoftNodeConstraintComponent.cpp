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
    const mochi::Real3 Position =
        FUnrealToMochiConverter().TranslationToMochi(GetComponentLocation());
    ForEachConstraint(
        [Position](mochi::Constraint& C) { C.SetTargetPosition(Position, mochi::ErrorLog{}); });
  }
}

EMochiBodyState UMochiSoftNodeConstraintComponent::BuildConstraints(
    FConstraintRequest& OutRequest) {
  FMochiConstraintBody ResolvedBody;
  mochi::ActorHandle Actor;
  const EMochiBodyState State = ResolveBody(Body, ResolvedBody, OutRequest.Scene, Actor);
  if (State != EMochiBodyState::Ready) {
    return State;
  }
  // Mochi chooses the nodes for Closest, and InsideRigidBounds finds them on the Mochi thread.
  Nodes = std::make_shared<FNodes>();
  const bool bNodeByIndex = Type == EMochiSoftNodeConstraintType::NodePosition ||
      NodeSelection == EMochiSoftNodeSelection::Index;
  if (bNodeByIndex) {
    Nodes->Indices = {NodeIndex};
    Nodes->bReady = true;
  }
  const FUnrealToMochiConverter U2m;

  if (Type == EMochiSoftNodeConstraintType::NodePosition) {
    mochi::DeformableNodePositionConstraintParams Params;
    FillCommonParams(Params);
    Params.actor = Actor;
    Params.nodeIndex = NodeIndex;
    Params.position = U2m.TranslationToMochi(GetComponentLocation());
    OutRequest.CreateOne(Params, &mochi::Scene::CreateDeformableNodePositionConstraint);
    return EMochiBodyState::Ready;
  }

  FMochiConstraintBody ResolvedOtherBody;
  mochi::ActorHandle OtherActor;
  UMochiSceneObject* OtherScene = nullptr;
  const EMochiBodyState OtherState =
      ResolveBody(OtherBody, ResolvedOtherBody, OtherScene, OtherActor);
  if (OtherState != EMochiBodyState::Ready) {
    return OtherState;
  }
  if (OtherScene != OutRequest.Scene) {
    LogError(TEXT("the bodies are not in the same Mochi scene."));
    return EMochiBodyState::Failed;
  }
  if (!bAllowContact) {
    OutRequest.NoContactPairs.emplace_back(Actor, OtherActor);
  }

  if (Type == EMochiSoftNodeConstraintType::NodeToNode) {
    if (NodeSelection == EMochiSoftNodeSelection::InsideRigidBounds) {
      LogError(TEXT("Inside Rigid Bounds only applies to Node to Rigid."));
      return EMochiBodyState::Failed;
    }
    mochi::DeformableNodeToDeformableNodeConstraintParams Params;
    FillCommonParams(Params);
    Params.actorA = Actor;
    Params.actorB = OtherActor;
    Params.nodeIndexA = NodeIndex;
    Params.nodeIndexB = OtherNodeIndex;
    Params.findClosest = NodeSelection == EMochiSoftNodeSelection::Closest;
    OutRequest.CreateOne(Params, &mochi::Scene::CreateDeformableNodeToDeformableNodeConstraint);
    return EMochiBodyState::Ready;
  }

  mochi::DeformableNodeToRigidConstraintParams Params;
  FillCommonParams(Params);
  Params.deformableActor = Actor;
  Params.rigidActor = OtherActor;
  Params.deformableNodeIndex = NodeIndex;
  Params.findClosest = NodeSelection == EMochiSoftNodeSelection::Closest;
  Params.fixToDeformablePos = bFixToDeformablePosition;
  Params.rigidLocalPos = ResolvedOtherBody.LocalPositionToMochi(RigidLocalPosition);
  if (NodeSelection != EMochiSoftNodeSelection::InsideRigidBounds) {
    OutRequest.CreateOne(Params, &mochi::Scene::CreateDeformableNodeToRigidConstraint);
    return EMochiBodyState::Ready;
  }
  OutRequest.Create = [Params, NodeIndices = Nodes](mochi::Scene* Scene) {
    mochi::ErrorLog E;
    std::vector<mochi::ConstraintHandle> Constraints;
    mochi::Actor* Deformable = Scene->GetActor(Params.deformableActor);
    const mochi::Actor* Rigid = Scene->GetActor(Params.rigidActor);
    if (Deformable == nullptr || Rigid == nullptr) {
      return Constraints;
    }
    // The nodes inside the rigid body's world bounding box, in the deformable body's frame.
    const mochi::Obb RigidBoundsLocal = mochi::TransformShape(
        Invert(Deformable->GetRootTransform()), mochi::GetObb(Rigid->GetAabbWorld(E)));
    const mochi::QueryHandle Query =
        Deformable->RegisterQueryAndCompute(mochi::QueryType::NodePositions, E);
    Deformable->QueryNodesInVolumeLocal(
        RigidBoundsLocal,
        false,
        [&](int Node, mochi::Real3 /*Position*/) {
          mochi::DeformableNodeToRigidConstraintParams NodeParams = Params;
          NodeParams.deformableNodeIndex = Node;
          NodeParams.findClosest = false;
          // Each node ties where it is; one shared rigid point would collapse the region.
          NodeParams.fixToDeformablePos = true;
          if (const mochi::Constraint* Constraint =
                  Scene->CreateDeformableNodeToRigidConstraint(NodeParams, E)) {
            Constraints.push_back(Constraint->GetHandle());
            NodeIndices->Indices.Add(Node);
          }
        },
        E);
    Deformable->CancelQuery(Query);
    NodeIndices->bReady.store(true, std::memory_order_release);
    return Constraints;
  };
  return EMochiBodyState::Ready;
}
