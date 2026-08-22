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

#include <mochi_physics/src/mochi_contact.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {

constexpr int kNumPartitions = 3;

template <TimeStep kTimeStep>
using CActiveCollisionsAsync = CActiveCollisions<ContactType::Async, kTimeStep>;

using CPotentialCollidersAsync = CPotentialColliders<ContactType::Async>;

entt::entity Entity(int value) {
  return static_cast<entt::entity>(value);
}

PotentialColliderData PotentialCollider(int entity) {
  return {Entity(entity)};
}

template <TimeStep kTimeStep>
CActiveCollisionsAsync<kTimeStep> MakeActiveCollisions(
    std::initializer_list<int> colliderEntities) {
  CPotentialCollidersAsync potentialColliders;
  for (int colliderEntity : colliderEntities) {
    potentialColliders.push_back(PotentialCollider(colliderEntity));
  }

  CActiveCollisionsAsync<kTimeStep> activeCollisions;
  activeCollisions.SetUp(potentialColliders, kNumPartitions);
  return activeCollisions;
}

template <TimeStep kTimeStep>
void ValidateActiveCollisions(
    CActiveCollisionsAsync<kTimeStep>& activeCollisions,
    std::initializer_list<int> colliderEntities) {
  CPotentialCollidersAsync potentialColliders;
  for (int colliderEntity : colliderEntities) {
    potentialColliders.push_back(PotentialCollider(colliderEntity));
  }

  activeCollisions.SetUp(potentialColliders, kNumPartitions);
  EXPECT_EQ(isize(colliderEntities) * kNumPartitions, isize(activeCollisions));

  int activeCollisionIndex = 0;
  for (int colliderEntity : colliderEntities) {
    for (int partitionId = 0; partitionId < kNumPartitions; ++partitionId) {
      EXPECT_EQ(Entity(colliderEntity), activeCollisions[activeCollisionIndex].colliderEntity);
      EXPECT_EQ(
          partitionId, activeCollisions[activeCollisionIndex].collisionResult.collidingPartitionId);
      ++activeCollisionIndex;
    }
  }
}

} // namespace

TEST(MochiStageStartContact, AddMissingStageStartCollisions) {
  auto stageStartCollisions = MakeActiveCollisions<TimeStep::StageStart>({10, 20});
  auto currentCollisions = MakeActiveCollisions<TimeStep::Current>({10, 30});

  ValidateActiveCollisions(stageStartCollisions, {10, 20});
  ValidateActiveCollisions(currentCollisions, {10, 30});

  stageStartCollisions[4].collisionResult.sampleIndices.push_back(7);

  AddMissingStageStartCollisions(stageStartCollisions, currentCollisions);

  ValidateActiveCollisions(currentCollisions, {10, 20, 30});
}
