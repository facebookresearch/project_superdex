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

#pragma once

#include "CoreMinimal.h"
#include "MochiConstraintComponentBase.h"

#include <atomic>
#include <memory>

#include "MochiSoftNodeConstraintComponent.generated.h"

UENUM(BlueprintType)
enum class EMochiSoftNodeConstraintType : uint8 {
  // Pulls a node of a deformable body to this component's location (Mochi DeformableNodePosition).
  NodePosition UMETA(DisplayName = "Node Position"),
  // Ties a node of a deformable body to a rigid body (Mochi DeformableNodeToRigid).
  NodeToRigid UMETA(DisplayName = "Node to Rigid"),
  // Ties a node of one deformable body to a node of another (Mochi
  // DeformableNodeToDeformableNode).
  NodeToNode UMETA(DisplayName = "Node to Node"),
};

UENUM(BlueprintType)
enum class EMochiSoftNodeSelection : uint8 {
  // The node with the given index.
  Index,
  // The boundary node Mochi finds closest: to the rigid point for Node to Rigid, or between the
  // two bodies for Node to Node.
  Closest,
  // Every node inside the rigid body's bounding box, each with its own constraint. Node to Rigid
  // only.
  InsideRigidBounds UMETA(DisplayName = "Inside Rigid Bounds"),
};

// Ties nodes of a deformable body (a soft, shell or rod component) to this component's location, a
// rigid body, or another deformable body. Saturation is in [cm].
UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiSoftNodeConstraintComponent : public UMochiConstraintComponentBase {
  GENERATED_BODY()

 public:
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  EMochiSoftNodeConstraintType Type = EMochiSoftNodeConstraintType::NodeToRigid;
  // The deformable body, by name: "cloth", or "inner/cloth" inside a child actor.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint")
  FString Body;
  // The rigid body (Node to Rigid) or the other deformable body (Node to Node).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type != EMochiSoftNodeConstraintType::NodePosition",
           EditConditionHides))
  FString OtherBody;
  // Whether the two bodies can touch once constrained. Contact stays disabled after the
  // constraints are destroyed.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type != EMochiSoftNodeConstraintType::NodePosition",
           EditConditionHides))
  bool bAllowContact = false;
  // How the node is chosen.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition = "Type != EMochiSoftNodeConstraintType::NodePosition",
           EditConditionHides))
  EMochiSoftNodeSelection NodeSelection = EMochiSoftNodeSelection::Index;
  // The node of Body.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Constraint", meta = (ClampMin = "0"))
  int32 NodeIndex = 0;
  // The node of OtherBody (Node to Node).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (ClampMin = "0",
           EditCondition = "Type == EMochiSoftNodeConstraintType::NodeToNode",
           EditConditionHides))
  int32 OtherNodeIndex = 0;
  // Whether the node ties to the rigid body where the node is when the constraint is created,
  // rather than at RigidLocalPosition.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition =
               "Type == EMochiSoftNodeConstraintType::NodeToRigid && NodeSelection != EMochiSoftNodeSelection::InsideRigidBounds",
           EditConditionHides))
  bool bFixToDeformablePosition = true;
  // The point on the rigid body [cm], in its local frame: where the node ties, or the point the
  // closest node is found from.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Constraint",
      meta =
          (EditCondition =
               "Type == EMochiSoftNodeConstraintType::NodeToRigid && NodeSelection != EMochiSoftNodeSelection::InsideRigidBounds",
           EditConditionHides))
  FVector RigidLocalPosition = FVector::ZeroVector;

  // Get the indices of the nodes of Body this component constrains, once the constraints exist.
  // Not known when Mochi chooses the node (Closest).
  UFUNCTION(BlueprintCallable, Category = "Mochi Constraint")
  bool GetConstrainedNodeIndices(TArray<int>& OutNodeIndices) const;

  UMochiSoftNodeConstraintComponent();

 protected:
  virtual EMochiBodyState BuildConstraints(FConstraintRequest& OutRequest) override;
  virtual double GetSaturationScale() const override;
  virtual void OnUpdateTransform(EUpdateTransformFlags UpdateTransformFlags, ETeleportType Teleport)
      override;

 private:
  // Nodes constrained, written by the Mochi thread when it creates the constraints.
  struct FNodes {
    TArray<int> Indices;
    std::atomic<bool> bReady = false;
  };
  std::shared_ptr<FNodes> Nodes = std::make_shared<FNodes>();
};
