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

#include <mochi_physics/utils/articulated_actor_params_utils.h>

#include "mochi_hdf5.h"

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/hdf5_utils.h>
#include <mochi_core/utils/transform_rt.h>

#include <array>

namespace mochi {

ArticulatedActorParams LoadArticulatedActorParams(
    Context* /*context*/,
    std::string_view articulatedShapePath,
    Span<ShapeHandle const> linkShapes,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(articulatedShapePath.empty(), error, "Empty file path");
  MOCHI_ERROR_RETURN(error, {});

  // Read file into memory
  auto fileBytes = ReadFileBytes(articulatedShapePath, error);
  MOCHI_ERROR_RETURN(error, {});

  MOCHI_ERROR_IF(
      !hdf5::LooksLikeHDF5(fileBytes),
      error,
      "Articulated shape file is not in HDF5 (.mochi.h5) format.");
  MOCHI_ERROR_RETURN(error, {});

  // Determine link/joint counts up front; the rest of the parsing depends on them.
  int const numLinks = hdf5::ReadMeshTransformsBytesBodyCount(fileBytes, error);
  int const numJoints = hdf5::ReadMeshTransformsBytesJointCount(fileBytes, error);
  MOCHI_ERROR_IF_NOT(numLinks >= 0 && numJoints >= 0, error, "Invalid number of links and joints.");
  MOCHI_ERROR_IF(
      numJoints < numLinks,
      error,
      "Number of joints must be greater than or equal to the number of links (one joint per link plus cycle joints).");
  MOCHI_ERROR_RETURN(error, {});

  int const numCycles = numJoints - numLinks;

  // Allocate raw output arrays.
  DynamicArray<int> linkParentList(numLinks);
  DynamicArray<Quaternion> rotationList(numLinks);
  DynamicArray<Real3> translationList(numLinks);
  DynamicArray<Real3> jointAxisList(numJoints);
  DynamicArray<Real3> jointXYZList(numJoints);
  DynamicArray<ArticulatedJointType> jointTypeList(numJoints);
  DynamicArray<ArticulatedCycleJoint> cycleJointList(numCycles);
  bool hasJointLimits = false;
  DynamicArray<Real3> minJointLimits(numJoints);
  DynamicArray<Real3> maxJointLimits(numJoints);
  bool hasJointNames = false;
  DynamicArray<std::array<char, hdf5::kMaxMeshTransformsNameLength>> jointNamesFixed(numJoints);
  bool hasLinkNames = false;
  DynamicArray<std::array<char, hdf5::kMaxMeshTransformsNameLength>> linkNamesFixed(numLinks);

  hdf5::ReadMeshTransformsBytes(
      fileBytes,
      rotationList,
      translationList,
      linkParentList,
      jointTypeList,
      cycleJointList,
      jointAxisList,
      jointXYZList,
      hasJointLimits,
      minJointLimits,
      maxJointLimits,
      hasJointNames,
      jointNamesFixed,
      hasLinkNames,
      linkNamesFixed,
      error);
  MOCHI_ERROR_RETURN(error, {});

  MOCHI_ERROR_IF(
      isize(linkShapes) != numLinks,
      error,
      "linkShapes size does not match the number of links in the articulated shape.");
  MOCHI_ERROR_RETURN(error, {});

  // The legacy file format stores the joint anchor in writer-side coordinates; mochi negates it
  // when reading. Match that behavior so the resulting parentLinkFromJoint matches the legacy
  // ArticulatedShapeInfo path.
  for (Real3& xyz : jointXYZList) {
    xyz = -xyz;
  }

  // jointMinLimits / jointMaxLimits are either empty (no joint has limits) or sized
  // numLinks + numCycles. Leaving ArticulatedJointParams::{min,max}Limit unset (the default)
  // is the correct "no limit" sentinel.
  ArticulatedActorParams params;
  params.joints.resize(numLinks);
  params.links.resize(numLinks);

  for (int i = 0; i < numLinks; ++i) {
    int const parent = linkParentList[i];
    MOCHI_ERROR_IF(parent < -1 || parent >= numLinks, error, "Parent link index out-of-range.");
    MOCHI_ERROR_RETURN(error, {});
    TransformRT const rootFromLink{rotationList[i], translationList[i]};
    TransformRT const rootFromParent = (parent >= 0)
        ? TransformRT{rotationList[parent], translationList[parent]}
        : TransformRT::Identity();
    // The legacy file stored only the joint anchor translation in jointXYZList[i] (the joint
    // position in the link frame); parentJointFromLink is its inverse (link relative to the
    // parent joint). The rest of the joint transform lives in parentLinkFromJoint.
    TransformRT const parentJointFromLink{-jointXYZList[i]};

    auto& joint = params.joints[i];
    if (hasJointNames) {
      joint.name = jointNamesFixed[i].data();
    }
    joint.type = jointTypeList[i];
    joint.axis = jointAxisList[i];
    joint.parentLinkFromJoint = Invert(rootFromParent) * rootFromLink * Invert(parentJointFromLink);
    if (hasJointLimits && minJointLimits[i] != -kInf3) {
      joint.minLimit = minJointLimits[i];
    }
    if (hasJointLimits && maxJointLimits[i] != kInf3) {
      joint.maxLimit = maxJointLimits[i];
    }

    auto& link = params.links[i];
    if (hasLinkNames) {
      link.name = linkNamesFixed[i].data();
    }
    link.parentLink = parent;
    link.parentJointFromLink = parentJointFromLink;
    link.shape = linkShapes[i];
  }

  // Cycle joints (closed kinematic loops). Per-joint info arrays carry the cycle joint
  // anchor in the trailing entries [numLinks, numLinks + numCycles).
  params.cycles.resize(numCycles);
  for (int i = 0; i < numCycles; ++i) {
    params.cycles[i].parentLink = cycleJointList[i].parent;
    params.cycles[i].childLink = cycleJointList[i].child;
    // As with the per-link parentJointFromLink above, jointFromChildLink is the inverse of the
    // legacy joint anchor (the child link relative to the cycle joint).
    params.cycles[i].jointFromChildLink = TransformRT{-jointXYZList[numLinks + i]};
  }

  return params;
}

} // namespace mochi
