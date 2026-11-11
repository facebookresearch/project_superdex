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

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/cpp_api/mochi_context.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "superdex_robotics/core/loader.h"

using namespace mochi;
using namespace superdex::robotics;

static constexpr real kUnitVectorTolerance = 5e-6_r;

static int AlphaNumCompare(std::string_view l, std::string_view r) {
  auto IsDigit = [](char const c) { return c >= '0' && c <= '9'; };
  auto li = l.begin();
  auto ri = r.begin();
  while (li != l.end() && ri != r.end()) {
    bool const lDigit = IsDigit(*li);
    bool const rDigit = IsDigit(*ri);
    if (lDigit && rDigit) {
      unsigned long lNum = 0;
      while (li != l.end() && IsDigit(*li)) {
        lNum = lNum * 10 + (*li - '0');
        ++li;
      }
      unsigned long rNum = 0;
      while (ri != r.end() && IsDigit(*ri)) {
        rNum = rNum * 10 + (*ri - '0');
        ++ri;
      }
      if (lNum != rNum) {
        return lNum < rNum ? -1 : 1;
      }
    } else if (lDigit) {
      return -1;
    } else if (rDigit) {
      return 1;
    } else {
      if (*li != *ri) {
        return *li < *ri ? -1 : 1;
      }
      ++li;
      ++ri;
    }
  }
  if (ri != r.end()) {
    return -1;
  }
  if (li != l.end()) {
    return 1;
  }
  return 0;
}

// Returns the number of DOFs for a bot joint. Revolute/Prismatic/Spherical
// contribute via GetJointTypeNumDofs; Hard/Cycle contribute 0. Free
// technically contributes 6, but Free joints are only allowed on the root
// of Mochi Bots and are stripped from defaultPose, therefore 0 here.
static int BotJointNumDofs(ArticulatedJointType type) {
  if (type == ArticulatedJointType::Free) {
    return 0;
  }
  auto const dofs = articulated::GetJointTypeNumDofs(type);
  return dofs.first + dofs.second;
}

// Returns the expected number of DOFs for a BotPrefab congurent with
// the size of defaultPose. This does NOT include the first 6 DOFs of
// bots with a Free world joint.
static int ComputeExpectedNumDofs(BotPrefab const& botPrefab) {
  int numDofs = 0;
  for (auto const& joint : botPrefab.joints) {
    numDofs += BotJointNumDofs(joint.type);
  }
  return numDofs;
}

// Remap (and prune) transmission/tendon joint & link references through an
// old->new index map. oldToNew[oldIdx] == kIndexNone marks a joint/link that was
// removed: any transmission or tendon referencing it is dropped, mirroring how
// cycles and contact overrides are dropped on removal. Because joint i travels
// with link i (one joint per link at the same index), the same map applies to
// both joint indices (linearTransmission jointIndices, LinearJoint routing) and
// link indices (Waypoint routing). Pre-existing out-of-range indices are left
// untouched so Validate can still surface malformed data.
static void RemapTransmissionAndTendonIndices(
    BotPrefab& botPrefab,
    DynamicArray<int> const& oldToNew) {
  auto isRemoved = [&](int idx) {
    return idx >= 0 && idx < isize(oldToNew) && oldToNew[idx] == kIndexNone;
  };
  auto remap = [&](int& idx) {
    if (idx >= 0 && idx < isize(oldToNew) && oldToNew[idx] != kIndexNone) {
      idx = oldToNew[idx];
    }
  };

  auto& transmissions = botPrefab.linearTransmissions;
  transmissions.erase(
      std::remove_if(
          transmissions.begin(),
          transmissions.end(),
          [&](BotLinearTransmissionPrefab const& t) {
            return std::any_of(t.jointIndices.begin(), t.jointIndices.end(), isRemoved);
          }),
      transmissions.end());
  for (auto& t : transmissions) {
    for (int& jointIdx : t.jointIndices) {
      remap(jointIdx);
    }
  }

  auto& tendons = botPrefab.spatialTendons;
  tendons.erase(
      std::remove_if(
          tendons.begin(),
          tendons.end(),
          [&](BotSpatialTendonPrefab const& s) {
            return std::any_of(
                s.routingElements.begin(), s.routingElements.end(), [&](RoutingElement const& e) {
                  return isRemoved(e.index);
                });
          }),
      tendons.end());
  for (auto& s : tendons) {
    for (auto& e : s.routingElements) {
      remap(e.index);
    }
  }
}

int superdex::robotics::AddLink(
    BotPrefab& botPrefab,
    int parentLinkIdx,
    std::string_view baseName,
    Error& error) {
  MOCHI_ERROR_IF(
      parentLinkIdx < 0 || parentLinkIdx >= isize(botPrefab.links),
      error,
      "Parent link index is out of bounds.");
  MOCHI_ERROR_RETURN(error, kIndexNone);
  // Find insertion index: after iLink and all its descendants
  struct Local {
    static int FindLastDescendant(BotPrefab const& params, int idx) {
      int last = idx;
      for (int child : params.links[idx]._childrenIndices) {
        last = std::max(last, FindLastDescendant(params, child));
      }
      return last;
    }
  };
  int newLinkIdx = Local::FindLastDescendant(botPrefab, parentLinkIdx) + 1;
  // Generate a unique name
  DynamicString baseNameD(baseName);
  // Note: std::unordered_set<DynamicString> does not compile on some platforms
  std::set<DynamicString> existingNames;
  for (auto const& link : botPrefab.links) {
    existingNames.insert(link.name);
  }
  DynamicString name = baseNameD;
  int suffix = 1;
  while (existingNames.count(name) > 0) {
    name = baseNameD + "_" + DynamicString(std::to_string(suffix++));
  }
  // Create the new link and joint
  BotLinkPrefab newLink;
  newLink.name = name;
  newLink.parentLink = parentLinkIdx;
  BotJointPrefab newJoint;
  newJoint.type = ArticulatedJointType::Hard;
  newJoint.name = newLink.name + "_joint";
  // Update parentLink for links at or after the insertion point
  for (auto& link : botPrefab.links) {
    if (link.parentLink >= newLinkIdx) {
      ++link.parentLink;
    }
  }
  // Shift transmission/tendon joint & link references at or after the insertion
  // point up by one to match the link/joint insertion below. The map is indexed
  // by old indices, so it must be built while botPrefab.links still has its
  // pre-insertion size. This is a pure shift (bijection, no kIndexNone), so
  // nothing is dropped; the freshly-inserted link is not yet referenced.
  DynamicArray<int> insertShift(isize(botPrefab.links));
  for (int i = 0; i < isize(insertShift); ++i) {
    insertShift[i] = i < newLinkIdx ? i : i + 1;
  }
  RemapTransmissionAndTendonIndices(botPrefab, insertShift);
  // Insert link at newLinkIdx by appending and rotating into place
  botPrefab.links.push_back(std::move(newLink));
  for (int i = isize(botPrefab.links) - 1; i > newLinkIdx; --i) {
    std::swap(botPrefab.links[i], botPrefab.links[i - 1]);
  }
  // Insert joint at newLinkIdx by appending and rotating into place
  botPrefab.joints.push_back(std::move(newJoint));
  for (int i = isize(botPrefab.joints) - 1; i > newLinkIdx; --i) {
    std::swap(botPrefab.joints[i], botPrefab.joints[i - 1]);
  }
  return newLinkIdx;
}

void superdex::robotics::RemoveLinkAndDescendants(
    BotPrefab& botPrefab,
    int linkIdx,
    bool keepParentJoint,
    int& outNewParentIdx,
    BotJointPrefab& outConnectingJoint,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Find linkToReplace by name; error if not found or is root or OOB
  MOCHI_ERROR_IF(linkIdx == kIndexNone, error, "Link to replace not found");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(linkIdx == 0, error, "Cannot replace root link");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      linkIdx < 0 || linkIdx >= botPrefab.links.size(), error, "Link index is out of bounds");
  MOCHI_ERROR_RETURN(error);
  // Save the parent link index and connecting joint
  int const parentLinkIdx = botPrefab.links[linkIdx].parentLink;
  outConnectingJoint = botPrefab.joints[linkIdx];
  // Save connecting joint's default positions using _dofIndices
  int const connectingJointIdx = linkIdx;
  DynamicArray<real> connectingJointDefaultPositions;
  for (size_t iDof = 0; iDof < botPrefab._dofIndices.size(); ++iDof) {
    if (botPrefab._dofIndices[iDof] == connectingJointIdx) {
      if (iDof < botPrefab.defaultPose.size()) {
        connectingJointDefaultPositions.push_back(botPrefab.defaultPose[iDof]);
      } else {
        connectingJointDefaultPositions.push_back(0_r);
      }
    }
  }
  // Mark link and descendants for removal
  DynamicArray<bool> toRemove(botPrefab.links.size(), false);
  toRemove[linkIdx] = true;
  for (int i = linkIdx + 1; i < isize(botPrefab.links); ++i) {
    if (toRemove[botPrefab.links[i].parentLink]) {
      toRemove[i] = true;
    }
  }
  // Build new links/joints arrays with remapped indices
  DynamicArray<int> indexMap(botPrefab.links.size(), kIndexNone);
  DynamicArray<BotLinkPrefab> newLinks;
  DynamicArray<BotJointPrefab> newJoints;
  for (int i = 0; i < isize(botPrefab.links); ++i) {
    if (!toRemove[i]) {
      indexMap[i] = isize(newLinks);
      BotLinkPrefab link = botPrefab.links[i];
      if (link.parentLink != kIndexNone) {
        link.parentLink = indexMap[link.parentLink];
      }
      newLinks.push_back(link);
      newJoints.push_back(botPrefab.joints[i]); // joint i goes with link i
    }
  }
  // Drop cycles that reference removed links and remap surviving valid endpoints.
  // Preserve pre-existing malformed indices so validation can report them.
  auto& cycles = botPrefab.cycles;
  cycles.erase(
      std::remove_if(
          cycles.begin(),
          cycles.end(),
          [&](ArticulatedCycleJointParams const& cycle) {
            return (cycle.parentLink >= 0 && cycle.parentLink < isize(indexMap) &&
                    indexMap[cycle.parentLink] == kIndexNone) ||
                (cycle.childLink >= 0 && cycle.childLink < isize(indexMap) &&
                 indexMap[cycle.childLink] == kIndexNone);
          }),
      cycles.end());
  for (auto& cycle : cycles) {
    if (cycle.parentLink >= 0 && cycle.parentLink < isize(indexMap)) {
      cycle.parentLink = indexMap[cycle.parentLink];
    }
    if (cycle.childLink >= 0 && cycle.childLink < isize(indexMap)) {
      cycle.childLink = indexMap[cycle.childLink];
    }
  }
  // Remap surviving transmission/tendon joint & link references and drop any
  // transmission/tendon that referenced a removed joint or link (indexMap entry
  // == kIndexNone), mirroring the cycle handling above.
  RemapTransmissionAndTendonIndices(botPrefab, indexMap);
  // Determine which joints are being deleted
  std::unordered_set<int> jointsToDelete;
  for (int i = 0; i < isize(botPrefab.links); ++i) {
    if (toRemove[i]) {
      jointsToDelete.insert(i);
    }
  }
  // Rebuild defaultJointPositions for surviving DOFs
  DynamicArray<real> newDefaultJointPositions;
  for (size_t iDof = 0; iDof < botPrefab._dofIndices.size(); ++iDof) {
    int const jointIdx = botPrefab._dofIndices[iDof];
    if (jointsToDelete.count(jointIdx) == 0) {
      if (iDof < botPrefab.defaultPose.size()) {
        newDefaultJointPositions.push_back(botPrefab.defaultPose[iDof]);
      } else {
        newDefaultJointPositions.push_back(0_r);
      }
    }
  }
  botPrefab.links = std::move(newLinks);
  botPrefab.joints = std::move(newJoints);
  botPrefab.defaultPose = std::move(newDefaultJointPositions);
  if (keepParentJoint) {
    // Append the connecting joint's default positions
    // (the joint itself will be pushed by the caller via AttachLink/AttachBot)
    for (real const pos : connectingJointDefaultPositions) {
      botPrefab.defaultPose.push_back(pos);
    }
  }
  // Return remapped parent index
  outNewParentIdx = indexMap[parentLinkIdx];
}

void superdex::robotics::RemoveLinkAndDescendants(BotPrefab& botPrefab, int linkIdx, Error& error) {
  MOCHI_ERROR_RETURN(error);
  int newParentIdx = 0;
  BotJointPrefab connectingJoint;
  bool keepParentJoint = false;
  RemoveLinkAndDescendants(
      botPrefab, linkIdx, keepParentJoint, newParentIdx, connectingJoint, error);
}

void superdex::robotics::ApplyMod(
    BotPrefab& botPrefab,
    AttachLink const& mod,
    IBotLoader const& /*unused*/,
    bool /*validate*/,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  int const parentLinkIdx = FindLinkIndexByName(botPrefab, mod.parentLinkName);
  MOCHI_ERROR_IF(parentLinkIdx == kIndexNone, error, "Parent link not found");
  MOCHI_ERROR_RETURN(error);
  BotLinkPrefab newLink = mod.link;
  newLink.parentLink = parentLinkIdx;
  newLink._parentFromLink = mod.joint.parentLinkFromJoint;
  newLink._index = isize(botPrefab.links);
  newLink._childrenIndices.clear();
  botPrefab.joints.push_back(mod.joint);
  botPrefab.links.push_back(newLink);
}

void superdex::robotics::ApplyMod(
    BotPrefab& botPrefab,
    AttachBot const& mod,
    IBotLoader const& loader,
    bool validate,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  int const parentLinkIdx = FindLinkIndexByName(botPrefab, mod.parentLinkName);
  MOCHI_ERROR_IF(parentLinkIdx == kIndexNone, error, "Parent link not found");
  MOCHI_ERROR_RETURN(error);
  // Load child bot recursively (supports nested combos)
  BotPrefab child = LoadBotPrefab(mod.path, loader, validate, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      child.scale != 1_r, error, "BotPrefab::scale is not supported for bots attached by mods");
  MOCHI_ERROR_RETURN(error);
  if (child.links.empty()) {
    MOCHI_LOG_WARNING("BuiltBot: Child bot '%s' has no links", mod.path.c_str());
    MOCHI_ERROR_SET(error, "Child bot has no links");
    MOCHI_ERROR_RETURN(error);
  }
  // A bot supports at most one skin, so a skinned child cannot be attached onto a bot that already
  // has one.
  MOCHI_ERROR_IF(
      child.skin.has_value() && botPrefab.skin.has_value(),
      error,
      "Cannot attach a skinned bot onto a bot that already has a skin (a bot supports at most one skin).");
  MOCHI_ERROR_RETURN(error);
  // Record current link count for index offset
  int const linkOffset = isize(botPrefab.links);
  // Append child joints, replacing child's world joint (index 0) with connecting joint
  for (int iJoint = 0; iJoint < isize(child.joints); ++iJoint) {
    BotJointPrefab prefixedJoint;
    if (iJoint == 0) {
      prefixedJoint = mod.joint;
    } else {
      prefixedJoint = child.joints[iJoint];
      prefixedJoint.name = mod.prefix + child.joints[iJoint].name;
    }
    botPrefab.joints.push_back(prefixedJoint);
  }
  // Append child links with prefixed names and remapped indices
  for (int iLink = 0; iLink < isize(child.links); ++iLink) {
    BotLinkPrefab prefixedLink = child.links[iLink];
    prefixedLink.name = mod.prefix + child.links[iLink].name;
    if (iLink == 0) {
      // Root link of child: attach to parent link
      prefixedLink.parentLink = parentLinkIdx;
      prefixedLink._parentFromLink = mod.joint.parentLinkFromJoint;
    } else {
      // Non-root link: offset parent index
      prefixedLink.parentLink = child.links[iLink].parentLink + linkOffset;
    }
    // Remap children indices (will be rebuilt by RebuildBotData anyway)
    prefixedLink._childrenIndices.clear();
    for (int childIdx : child.links[iLink]._childrenIndices) {
      prefixedLink._childrenIndices.push_back(childIdx + linkOffset);
    }
    // Update _index
    prefixedLink._index = linkOffset + iLink;
    botPrefab.links.push_back(prefixedLink);
  }
  // Append default joint positions from child
  for (real pos : child.defaultPose) {
    botPrefab.defaultPose.push_back(pos);
  }
  // Append child contact overrides with prefixed link names. A party that names the child's skin
  // (the child's own top-level actor name) is re-pointed to this bot's top-level actor name,
  // because the child skin is adopted below and referenced by the composed bot's name (see
  // GetBotSkinName).
  std::string_view const childSkinName =
      child.skin.has_value() ? GetBotSkinName(child) : std::string_view{};
  for (auto const& over : child.contactOverrides) {
    BotContactOverride prefixed = over;
    if (!childSkinName.empty() && std::string_view(over.linkA) == childSkinName) {
      prefixed.linkA = botPrefab.name;
    } else {
      prefixed.linkA = mod.prefix + over.linkA;
    }
    if (!childSkinName.empty() && std::string_view(over.linkB) == childSkinName) {
      prefixed.linkB = botPrefab.name;
    } else {
      prefixed.linkB = mod.prefix + over.linkB;
    }
    botPrefab.contactOverrides.push_back(std::move(prefixed));
  }
  // Append child cycle joints, offsetting valid child-local link indices into
  // the merged link array. Preserve malformed indices for validation.
  int const childNumLinks = isize(child.links);
  for (auto const& cycle : child.cycles) {
    ArticulatedCycleJointParams offsetCycle = cycle;
    if (cycle.parentLink >= 0 && cycle.parentLink < childNumLinks) {
      offsetCycle.parentLink += linkOffset;
    }
    if (cycle.childLink >= 0 && cycle.childLink < childNumLinks) {
      offsetCycle.childLink += linkOffset;
    }
    botPrefab.cycles.push_back(offsetCycle);
  }
  // Adopt the child's skin (the base has none; verified above). Record the composed link name each
  // skin bone binds to (_skinBoneLinks) so the skin's baked bone indices can be remapped onto the
  // reordered composed skeleton in BuildArticulatedActorParams. Link names are stable across the
  // RebuildBotData re-sort, so this mapping survives composition.
  if (child.skin.has_value()) {
    botPrefab.skin = child.skin;
    botPrefab._skinBoneLinks.clear();
    if (!child._skinBoneLinks.empty()) {
      // Nested combo: the child already carries a bone->link mapping; extend the prefix.
      for (auto const& name : child._skinBoneLinks) {
        botPrefab._skinBoneLinks.push_back(mod.prefix + name);
      }
    } else {
      // Leaf child: skin bone b is identity with the child's link order (child.links is in bone
      // order), so bone b binds to child link b.
      for (int iLink = 0; iLink < childNumLinks; ++iLink) {
        botPrefab._skinBoneLinks.push_back(mod.prefix + child.links[iLink].name);
      }
    }
    // Forward the names of the links the skin stands in for, prefixed to match the merged link
    // names.
    for (auto* links : {&botPrefab.skin->nonCollidingLinks, &botPrefab.skin->hiddenLinks}) {
      if (links->has_value()) {
        for (auto& name : **links) {
          name = mod.prefix + name;
        }
      }
    }
  }
}

void superdex::robotics::ApplyMod(
    BotPrefab& botPrefab,
    ReplaceLink const& mod,
    IBotLoader const& loader,
    bool validate,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  int newParentIdx = 0;
  BotJointPrefab connectingJoint;
  auto linkIdx = FindLinkIndexByName(botPrefab, mod.linkToReplace);
  MOCHI_ERROR_IF(linkIdx == kIndexNone, error, "Link to replace not found");
  MOCHI_ERROR_RETURN(error);
  RemoveLinkAndDescendants(botPrefab, linkIdx, true, newParentIdx, connectingJoint, error);
  MOCHI_ERROR_RETURN(error);
  AttachLink attachMod;
  attachMod.link = mod.link;
  attachMod.parentLinkName = botPrefab.links[newParentIdx].name;
  attachMod.joint = connectingJoint;
  ApplyMod(botPrefab, attachMod, loader, validate, error);
}

void superdex::robotics::ApplyMod(
    BotPrefab& botPrefab,
    ReplaceLinkWithBot const& mod,
    IBotLoader const& loader,
    bool validate,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  int newParentIdx = 0;
  BotJointPrefab connectingJoint;
  auto linkIdx = FindLinkIndexByName(botPrefab, mod.linkToReplace);
  MOCHI_ERROR_IF(linkIdx == kIndexNone, error, "Link to replace not found");
  RemoveLinkAndDescendants(botPrefab, linkIdx, true, newParentIdx, connectingJoint, error);
  MOCHI_ERROR_RETURN(error);
  AttachBot attachMod;
  attachMod.path = mod.path;
  attachMod.prefix = mod.prefix;
  attachMod.parentLinkName = botPrefab.links[newParentIdx].name;
  attachMod.joint = connectingJoint;
  ApplyMod(botPrefab, attachMod, loader, validate, error);
}

void superdex::robotics::ApplyMod(
    BotPrefab& botPrefab,
    AttachSkin const& mod,
    IBotLoader const& /*loader*/,
    bool /*validate*/,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // A bot supports at most one skin.
  MOCHI_ERROR_IF(
      botPrefab.skin.has_value(),
      error,
      "Cannot attach a skin to a bot that already has one (a bot supports at most one skin).");
  MOCHI_ERROR_RETURN(error);
  botPrefab.skin = mod.skin;
  // Record the link each skin bone binds to by NAME: identity, i.e. skin bone b -> the bot's link b
  // in its current order. Names are stable across RebuildBotData's re-sort and nested composition,
  // so the skin's baked bone indices remap correctly in BuildArticulatedActorParams. This mirrors
  // the AttachBot "leaf child" branch and reduces to the base-bot identity skin path when the link
  // order is unchanged.
  botPrefab._skinBoneLinks.clear();
  botPrefab._skinBoneLinks.reserve(isize(botPrefab.links));
  for (auto const& link : botPrefab.links) {
    botPrefab._skinBoneLinks.push_back(link.name);
  }
}

BotPrefab superdex::robotics::BuildBot(
    ModBotPrefab const& buildParams,
    IBotLoader const& loader,
    bool validate,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  // A mod bot without a base starts from an empty bot -- newly created mod bots have no base
  // assigned yet.
  BotPrefab botPrefab;
  if (!buildParams.base.empty()) {
    botPrefab = LoadBotPrefab(buildParams.base, loader, validate, error);
    MOCHI_ERROR_IF(
        botPrefab.scale != 1_r,
        error,
        "BotPrefab::scale is not supported for bots composed with mods");
    MOCHI_ERROR_RETURN(error, {});
    if (validate) {
      Validate(botPrefab, nullptr, error);
      MOCHI_ERROR_RETURN(error, {});
    }
  }
  // Rebuild to populate _dofIndices before any mods
  RebuildBotData(botPrefab, error);
  MOCHI_ERROR_RETURN(error, {});
  // A base authored with its own skin (a standalone skinned .superdex_bot) has no _skinBoneLinks:
  // its baked bone indices match its own link order. Mods below add, remove and re-sort links, and
  // nothing would record that the mapping used to be identity -- the skin would then silently bind
  // each bone to whatever link ends up at that index. Capture the mapping now, while it still
  // holds. ApplyMod(AttachSkin) does the same for a skin attached later.
  if (botPrefab.skin.has_value() && botPrefab._skinBoneLinks.empty()) {
    botPrefab._skinBoneLinks.reserve(isize(botPrefab.links));
    for (auto const& link : botPrefab.links) {
      botPrefab._skinBoneLinks.push_back(link.name);
    }
  }
  // Set the combo name
  botPrefab.name = buildParams.name;
  // Process each modification
  for (int iMod = 0; iMod < buildParams.modifications.size(); ++iMod) {
    auto const& mod = buildParams.modifications[iMod];
    std::visit(
        [&](auto const& m) {
          if (m.enabled) {
            ApplyMod(botPrefab, m, loader, validate, error);
          }
        },
        mod);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to apply modification [%d]", iMod);
    }
    MOCHI_ERROR_RETURN(error, {});
    // Validate after each mod
    if (validate) {
      Validate(botPrefab, nullptr, error);
      MOCHI_ERROR_RETURN(error, {});
    }
    // Rebuild after each mod to keep _dofIndices in sync for subsequent mods
    RebuildBotData(botPrefab, error);
    if (!error.IsOK()) {
      MOCHI_LOG_ERROR("Failed to build bot after modification [%d]", iMod);
    }
    MOCHI_ERROR_RETURN(error, {});
  }
  // Apply the recipe's own contact-filter overrides on top of the composed bot. These are matched
  // by party name (link names, or the skin party == the composed bot name), so RebuildBotData's
  // name-keyed prune keeps the ones whose parties exist and drops any dangling references.
  if (!buildParams.contactOverrides.empty()) {
    auto& composed = botPrefab.contactOverrides;
    for (auto const& over : buildParams.contactOverrides) {
      // AttachBot may already have contributed a prefixed child override for this same pair. Drop
      // it so the recipe's entry is the only one: readers that stop at the first match would
      // otherwise report the child's value, while spawning applies every entry in order and ends up
      // with the recipe's -- the editor matrix and the spawned bot would disagree.
      std::string_view const a(over.linkA);
      std::string_view const b(over.linkB);
      composed.erase(
          std::remove_if(
              composed.begin(),
              composed.end(),
              [&](BotContactOverride const& existing) {
                std::string_view const ea(existing.linkA);
                std::string_view const eb(existing.linkB);
                return (ea == a && eb == b) || (ea == b && eb == a);
              }),
          composed.end());
      composed.push_back(over);
    }
    RebuildBotData(botPrefab, error);
    MOCHI_ERROR_RETURN(error, {});
  }
  return botPrefab;
}

// clang-format off
#define REPORT_BOT_ISSUE(results, ...)                                        \
  do {                                                                        \
    auto _msg = Format(__VA_ARGS__);                                          \
    if (!(results) || !(results)->suppressWarnings) {                         \
      MOCHI_LOG_WARNING("Validate: %s", _msg.c_str());                        \
    }                                                                         \
    if (results) {                                                            \
      (results)->botIssues.push_back(DynamicString(std::move(_msg)));         \
    }                                                                         \
  } while (0)

#define REPORT_LINK_ISSUE(results, idx, ...)                                  \
  do {                                                                        \
    auto _msg = Format(__VA_ARGS__);                                          \
    if (!(results) || !(results)->suppressWarnings) {                         \
      MOCHI_LOG_WARNING("Validate: %s", _msg.c_str());                        \
    }                                                                         \
    if (results) {                                                            \
      (results)->linkIssues[idx].push_back(DynamicString(std::move(_msg)));   \
    }                                                                         \
  } while (0)

#define REPORT_JOINT_ISSUE(results, idx, ...)                                 \
  do {                                                                        \
    auto _msg = Format(__VA_ARGS__);                                          \
    if (!(results) || !(results)->suppressWarnings) {                         \
      MOCHI_LOG_WARNING("Validate: %s", _msg.c_str());                        \
    }                                                                         \
    if (results) {                                                            \
      (results)->jointIssues[idx].push_back(DynamicString(std::move(_msg)));  \
    }                                                                         \
  } while (0)
// clang-format on

void superdex::robotics::Validate(
    BotPrefab const& botPrefab,
    ValidateResults* results,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  int const numLinks = isize(botPrefab.links);
  int const numJoints = isize(botPrefab.joints);

  if (results) {
    results->botIssues.clear();
    results->linkIssues.clear();
    results->linkIssues.resize(numLinks);
    results->jointIssues.clear();
    results->jointIssues.resize(numJoints);
  }

  // Structural validation
  if (botPrefab.name.empty()) {
    REPORT_BOT_ISSUE(results, "bot has no name");
  }
  MOCHI_ERROR_IF(botPrefab.name.empty(), error, "Invalid BotPrefab; see log warnings for details");
  if (numLinks == 0) {
    REPORT_BOT_ISSUE(results, "bot has no links");
  }
  MOCHI_ERROR_IF(numLinks == 0, error, "Invalid BotPrefab; see log warnings for details");
  if (numJoints != numLinks) {
    REPORT_BOT_ISSUE(
        results, "expected %d joints for %d links, got %d", numLinks, numLinks, numJoints);
  }
  MOCHI_ERROR_IF(numJoints != numLinks, error, "Invalid BotPrefab; see log warnings for details");
  MOCHI_ERROR_RETURN(error);

  // Experimental uniform scale
  bool const validScale = IsFinite(botPrefab.scale) && botPrefab.scale > 0_r;
  if (!validScale) {
    REPORT_BOT_ISSUE(results, "scale must be positive and finite, got %g", botPrefab.scale);
  }
  bool const unsupportedScale = botPrefab.scale != 1_r &&
      (!botPrefab.linearTransmissions.empty() || !botPrefab.spatialTendons.empty());
  if (unsupportedScale) {
    REPORT_BOT_ISSUE(results, "scale is not supported for bots with transmissions or tendons");
  }
  MOCHI_ERROR_IF(
      !validScale || unsupportedScale, error, "Invalid BotPrefab; see log warnings for details");
  MOCHI_ERROR_RETURN(error);

  // Name uniqueness
  std::unordered_set<std::string_view> names;
  for (int i = 0; i < numLinks; ++i) {
    if (botPrefab.links[i].name.empty()) {
      REPORT_LINK_ISSUE(results, i, "link [%d] has an empty name", i);
    }
    MOCHI_ERROR_IF(
        botPrefab.links[i].name.empty(), error, "Invalid BotPrefab; see log warnings for details");
    if (botPrefab.joints[i].name.empty()) {
      REPORT_JOINT_ISSUE(results, i, "joint [%d] has an empty name", i);
    }
    MOCHI_ERROR_IF(
        botPrefab.joints[i].name.empty(), error, "Invalid BotPrefab; see log warnings for details");
    if (!names.insert(std::string_view(botPrefab.links[i].name)).second) {
      REPORT_LINK_ISSUE(
          results, i, "duplicate name '%s' (link [%d])", botPrefab.links[i].name.c_str(), i);
      MOCHI_ERROR_SET(error, "Invalid BotPrefab; see log warnings for details");
    }
    if (!names.insert(std::string_view(botPrefab.joints[i].name)).second) {
      REPORT_JOINT_ISSUE(
          results, i, "duplicate name '%s' (joint [%d])", botPrefab.joints[i].name.c_str(), i);
      MOCHI_ERROR_SET(error, "Invalid BotPrefab; see log warnings for details");
    }
  }
  MOCHI_ERROR_RETURN(error);

  // The skin is referenced in contact overrides by the bot's (top-level actor's) name, so no link
  // may share that name -- otherwise a skin<->link override reference would be ambiguous.
  if (botPrefab.skin.has_value()) {
    std::string_view const skinName = GetBotSkinName(botPrefab);
    for (int i = 0; i < numLinks; ++i) {
      if (std::string_view(botPrefab.links[i].name) == skinName) {
        REPORT_LINK_ISSUE(
            results,
            i,
            "link [%d] name '%s' collides with the bot name (used to reference the skin)",
            i,
            botPrefab.links[i].name.c_str());
        MOCHI_ERROR_SET(error, "Invalid BotPrefab; see log warnings for details");
      }
    }
  }
  MOCHI_ERROR_RETURN(error);

  // Parent index & topology validation
  if (botPrefab.links[0].parentLink != kIndexNone) {
    REPORT_LINK_ISSUE(
        results,
        0,
        "root link has parentLink %d, expected kIndexNone",
        botPrefab.links[0].parentLink);
  }
  MOCHI_ERROR_IF(
      botPrefab.links[0].parentLink != kIndexNone,
      error,
      "Invalid BotPrefab; see log warnings for details");
  for (int i = 1; i < numLinks; ++i) {
    int const parent = botPrefab.links[i].parentLink;
    if (parent < 0 || parent >= numLinks) {
      REPORT_LINK_ISSUE(results, i, "link [%d] has out-of-range parentLink %d", i, parent);
    }
    MOCHI_ERROR_IF(
        parent < 0 || parent >= numLinks, error, "Invalid BotPrefab; see log warnings for details");
    if (parent == i) {
      REPORT_LINK_ISSUE(results, i, "link [%d] is its own parent", i);
    }
    MOCHI_ERROR_IF(parent == i, error, "Invalid BotPrefab; see log warnings for details");
  }
  MOCHI_ERROR_RETURN(error);

  // Cycle detection: walk parent chain; if steps exceed numLinks, a cycle exists
  for (int i = 1; i < numLinks; ++i) {
    int current = botPrefab.links[i].parentLink;
    for (int step = 0; step < numLinks; ++step) {
      if (current == kIndexNone) {
        break;
      }
      current = botPrefab.links[current].parentLink;
    }
    if (current != kIndexNone) {
      REPORT_LINK_ISSUE(results, i, "circular parent chain detected starting at link [%d]", i);
    }
    MOCHI_ERROR_IF(current != kIndexNone, error, "Invalid BotPrefab; see log warnings for details");
  }
  MOCHI_ERROR_RETURN(error);

  // Joint parameter validation
  for (int i = 0; i < numJoints; ++i) {
    auto const& joint = botPrefab.joints[i];
    // Root joint (joint[0], connecting the root link to the world) may only be
    // Free or Hard. All non-root joints must not be Free.
    if (i == 0) {
      if (joint.type != ArticulatedJointType::Free && joint.type != ArticulatedJointType::Hard) {
        REPORT_JOINT_ISSUE(
            results, i, "root joint [%d] '%s' must be Free or Hard", i, joint.name.c_str());
        MOCHI_ERROR_SET(error, "Invalid BotPrefab; see log warnings for details");
      }
    } else if (joint.type == ArticulatedJointType::Free) {
      REPORT_JOINT_ISSUE(
          results, i, "non-root joint [%d] '%s' must not be Free", i, joint.name.c_str());
      MOCHI_ERROR_SET(error, "Invalid BotPrefab; see log warnings for details");
    }
    if (joint.type == ArticulatedJointType::Revolute ||
        joint.type == ArticulatedJointType::Prismatic) {
      if (!IsFinite(joint.axis)) {
        REPORT_JOINT_ISSUE(
            results, i, "joint [%d] '%s' has non-finite axis", i, joint.name.c_str());
      }
      MOCHI_ERROR_IF(
          !IsFinite(joint.axis), error, "Invalid BotPrefab; see log warnings for details");
      real const axisMagSqr = NormSqr(joint.axis);
      if (axisMagSqr == 0_r) {
        REPORT_JOINT_ISSUE(results, i, "joint [%d] '%s' has zero axis", i, joint.name.c_str());
      }
      MOCHI_ERROR_IF(axisMagSqr == 0_r, error, "Invalid BotPrefab; see log warnings for details");
      if (joint.minLimit.has_value() && joint.maxLimit.has_value()) {
        real const minProj = Dot(*joint.minLimit, joint.axis);
        real const maxProj = Dot(*joint.maxLimit, joint.axis);
        bool const unlimited = minProj == -std::numeric_limits<real>::infinity() &&
            maxProj == std::numeric_limits<real>::infinity();
        if (!unlimited) {
          if (!IsFinite(*joint.minLimit)) {
            REPORT_JOINT_ISSUE(
                results, i, "joint [%d] '%s' has non-finite minLimit", i, joint.name.c_str());
          }
          MOCHI_ERROR_IF(
              !IsFinite(*joint.minLimit), error, "Invalid BotPrefab; see log warnings for details");
          if (!IsFinite(*joint.maxLimit)) {
            REPORT_JOINT_ISSUE(
                results, i, "joint [%d] '%s' has non-finite maxLimit", i, joint.name.c_str());
          }
          MOCHI_ERROR_IF(
              !IsFinite(*joint.maxLimit), error, "Invalid BotPrefab; see log warnings for details");
          if (minProj == 0_r && maxProj == 0_r) {
            REPORT_JOINT_ISSUE(
                results, i, "joint [%d] '%s' has zero limits along axis", i, joint.name.c_str());
          }
          MOCHI_ERROR_IF(
              minProj == 0_r && maxProj == 0_r,
              error,
              "Invalid BotPrefab; see log warnings for details");
          if (minProj > maxProj) {
            REPORT_JOINT_ISSUE(
                results,
                i,
                "joint [%d] '%s' has minLimit > maxLimit along axis",
                i,
                joint.name.c_str());
          }
          MOCHI_ERROR_IF(
              minProj > maxProj, error, "Invalid BotPrefab; see log warnings for details");
        }
      }
    }
    if (joint.friction.viscous < 0_r) {
      REPORT_JOINT_ISSUE(
          results,
          i,
          "joint [%d] '%s' has negative damping coefficient %g",
          i,
          joint.name.c_str(),
          static_cast<double>(joint.friction.viscous));
    }
    MOCHI_ERROR_IF(
        joint.friction.viscous < 0_r, error, "Invalid BotPrefab; see log warnings for details");
    if (joint.inertia.has_value() && joint.inertia.value() < 0_r) {
      REPORT_JOINT_ISSUE(
          results,
          i,
          "joint [%d] '%s' has negative inertia coefficient %g",
          i,
          joint.name.c_str(),
          static_cast<double>(joint.inertia.value()));
    }
    MOCHI_ERROR_IF(
        joint.inertia.has_value() && joint.inertia.value() < 0_r,
        error,
        "Invalid BotPrefab; see log warnings for details");
    if (joint.friction.coulomb < 0_r) {
      REPORT_JOINT_ISSUE(
          results,
          i,
          "joint [%d] '%s' has negative coulomb friction coefficient %g",
          i,
          joint.name.c_str(),
          static_cast<double>(joint.friction.coulomb));
    }
    MOCHI_ERROR_IF(
        joint.friction.coulomb < 0_r, error, "Invalid BotPrefab; see log warnings for details");
    if (joint.limitStiffness < 0_r) {
      REPORT_JOINT_ISSUE(
          results,
          i,
          "joint [%d] '%s' has negative limit stiffness coefficient %g",
          i,
          joint.name.c_str(),
          static_cast<double>(joint.limitStiffness));
    }
    MOCHI_ERROR_IF(
        joint.limitStiffness < 0_r, error, "Invalid BotPrefab; see log warnings for details");
    if (joint.limitDamping < 0_r) {
      REPORT_JOINT_ISSUE(
          results,
          i,
          "joint [%d] '%s' has negative limit damping coefficient %g",
          i,
          joint.name.c_str(),
          static_cast<double>(joint.limitDamping));
    }
    MOCHI_ERROR_IF(
        joint.limitDamping < 0_r, error, "Invalid BotPrefab; see log warnings for details");
  }
  MOCHI_ERROR_RETURN(error);

  // Link parameter validation
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = botPrefab.links[i];
    if (!link.shapeFile.empty()) {
      bool const hasDensity = link.density.has_value();
      bool const hasMass = link.mass.has_value();
      bool const hasCOM = link.centerOfMass.has_value();
      bool const hasMOI = link.momentOfInertia.has_value();
      // Valid combinations:
      //   (a) all null
      //   (b) density only
      //   (c) mass only
      //   (d) mass+COM+MOI
      //   (e) density+COM+MOI
      bool const allNull = !hasDensity && !hasMass && !hasCOM && !hasMOI;
      bool const densityOnly = hasDensity && !hasMass && !hasCOM && !hasMOI;
      bool const massOnly = !hasDensity && hasMass && !hasCOM && !hasMOI;
      bool const explicitMass = !hasDensity && hasMass && hasCOM && hasMOI;
      bool const explicitDensity = hasDensity && !hasMass && hasCOM && hasMOI;
      if (!allNull && !densityOnly && !massOnly && !explicitMass && !explicitDensity) {
        REPORT_LINK_ISSUE(
            results,
            i,
            "link [%d] '%s' has invalid mass property combination",
            i,
            link.name.c_str());
      }
      MOCHI_ERROR_IF(
          !allNull && !densityOnly && !massOnly && !explicitMass && !explicitDensity,
          error,
          "Invalid BotPrefab; see log warnings for details");
      if (hasDensity) {
        if (!IsFinite(link.density.value())) {
          REPORT_LINK_ISSUE(
              results, i, "link [%d] '%s' has non-finite density", i, link.name.c_str());
        }
        MOCHI_ERROR_IF(
            !IsFinite(link.density.value()),
            error,
            "Invalid BotPrefab; see log warnings for details");
        if (link.density.value() <= 0_r) {
          REPORT_LINK_ISSUE(
              results,
              i,
              "link [%d] '%s' has non-positive density %g",
              i,
              link.name.c_str(),
              static_cast<double>(link.density.value()));
        }
        MOCHI_ERROR_IF(
            link.density.value() <= 0_r, error, "Invalid BotPrefab; see log warnings for details");
      }
      if (hasMass) {
        if (!IsFinite(link.mass.value())) {
          REPORT_LINK_ISSUE(results, i, "link [%d] '%s' has non-finite mass", i, link.name.c_str());
        }
        MOCHI_ERROR_IF(
            !IsFinite(link.mass.value()), error, "Invalid BotPrefab; see log warnings for details");
        if (link.mass.value() <= 0_r) {
          REPORT_LINK_ISSUE(
              results,
              i,
              "link [%d] '%s' has non-positive mass %g",
              i,
              link.name.c_str(),
              static_cast<double>(link.mass.value()));
        }
        MOCHI_ERROR_IF(
            link.mass.value() <= 0_r, error, "Invalid BotPrefab; see log warnings for details");
      }
    } else if (link.mass && link.centerOfMass && link.momentOfInertia) {
      // Mirrors mochi's ValidateLink: a shapeless link has no volume, so only a full explicit
      // inertia applies to it (other mass properties are ignored). Its mass may be zero (a massless
      // frame) but not negative, and the inertia must be finite.
      bool const invalid = !IsFinite(link.mass.value()) || link.mass.value() < 0_r ||
          !IsFinite(link.centerOfMass.value()) || !IsFinite(link.momentOfInertia.value());
      if (invalid) {
        REPORT_LINK_ISSUE(
            results,
            i,
            "link [%d] '%s' has a negative mass or a non-finite inertia",
            i,
            link.name.c_str());
      }
      MOCHI_ERROR_IF(invalid, error, "Invalid BotPrefab; see log warnings for details");
    }
  }

  // Cycle joint validation. Mirrors mochi's ValidateCycleJoint.
  for (int i = 0; i < isize(botPrefab.cycles); ++i) {
    auto const& cycle = botPrefab.cycles[i];
    if (cycle.childLink < 0 || cycle.childLink >= numLinks) {
      REPORT_BOT_ISSUE(results, "cycle [%d] has out-of-range childLink %d", i, cycle.childLink);
    }
    MOCHI_ERROR_IF(
        cycle.childLink < 0 || cycle.childLink >= numLinks,
        error,
        "Invalid BotPrefab; see log warnings for details");
    if (cycle.parentLink < 0 || cycle.parentLink >= numLinks) {
      REPORT_BOT_ISSUE(results, "cycle [%d] has out-of-range parentLink %d", i, cycle.parentLink);
    }
    MOCHI_ERROR_IF(
        cycle.parentLink < 0 || cycle.parentLink >= numLinks,
        error,
        "Invalid BotPrefab; see log warnings for details");
    if (cycle.parentLink == cycle.childLink) {
      REPORT_BOT_ISSUE(
          results,
          "cycle [%d] parentLink and childLink must differ (both %d)",
          i,
          cycle.parentLink);
    }
    MOCHI_ERROR_IF(
        cycle.parentLink == cycle.childLink,
        error,
        "Invalid BotPrefab; see log warnings for details");
    if (!IsFinite(cycle.jointFromChildLink) ||
        !NearEqual(1_r, Norm(cycle.jointFromChildLink.GetRotation()), kUnitVectorTolerance)) {
      REPORT_BOT_ISSUE(results, "cycle [%d] has invalid jointFromChildLink transform", i);
    }
    MOCHI_ERROR_IF(
        !IsFinite(cycle.jointFromChildLink) ||
            !NearEqual(1_r, Norm(cycle.jointFromChildLink.GetRotation()), kUnitVectorTolerance),
        error,
        "Invalid BotPrefab; see log warnings for details");
    if (!IsFinite(cycle.stiffness) || cycle.stiffness < 0_r) {
      REPORT_BOT_ISSUE(
          results,
          "cycle [%d] has non-finite or negative stiffness %g",
          i,
          static_cast<double>(cycle.stiffness));
    }
    MOCHI_ERROR_IF(
        !IsFinite(cycle.stiffness) || cycle.stiffness < 0_r,
        error,
        "Invalid BotPrefab; see log warnings for details");
  }
  MOCHI_ERROR_RETURN(error);

  // defaultPose size check. defaultPose holds one entry per DOF; each joint
  // contributes a fixed number of DOFs based on its type.
  int const expectedNumDofs = ComputeExpectedNumDofs(botPrefab);
  if (!botPrefab.defaultPose.empty() && isize(botPrefab.defaultPose) > expectedNumDofs) {
    REPORT_BOT_ISSUE(
        results,
        "defaultPose size (%d) exceeds expected number of DOFs (%d)",
        isize(botPrefab.defaultPose),
        expectedNumDofs);
  }
  MOCHI_ERROR_IF(
      !botPrefab.defaultPose.empty() && isize(botPrefab.defaultPose) > expectedNumDofs,
      error,
      "Invalid BotPrefab; see log warnings for details");

  MOCHI_ERROR_RETURN(error);
}

void superdex::robotics::RebuildBotData(BotPrefab& botPrefab, Error& error) {
  MOCHI_ERROR_RETURN(error);
  SortBotPrefab(botPrefab, error);
  MOCHI_ERROR_RETURN(error);
  int const numLinks = isize(botPrefab.links);
  int const numJoints = isize(botPrefab.joints);
  // Validate: joints should be numLinks (one joint per link)
  if (numLinks > 0 && numJoints != numLinks) {
    MOCHI_LOG_WARNING(
        "RebuildBotData: expected %d joints for %d links, got %d", numLinks, numLinks, numJoints);
    MOCHI_ERROR_SET(error, "RebuildBotData: invalid number of links and joints");
    MOCHI_ERROR_RETURN(error);
  }
  // Clear children indices for all links
  for (auto& link : botPrefab.links) {
    link._childrenIndices.clear();
  }
  for (int iLink = 0; iLink < numLinks; ++iLink) {
    auto& link = botPrefab.links[iLink];
    link._index = iLink;

    if (iLink == 0) {
      // Root link
      link.parentLink = kIndexNone;
      link._parentFromLink = TransformRT::Identity();
      link._rootFromLink = TransformRT::Identity();
    } else {
      // Non-root link: get transform from joint
      auto const& joint = botPrefab.joints[iLink];
      link._parentFromLink = joint.parentLinkFromJoint;

      // Add this link to parent's children
      int const iParent = link.parentLink;
      if (iParent < 0 || iParent >= numLinks) {
        MOCHI_LOG_WARNING("RebuildBotData: link %d has invalid parent index %d", iLink, iParent);
        MOCHI_ERROR_SET(error, "RebuildBotData: invalid parent index");
        MOCHI_ERROR_RETURN(error);
      }
      botPrefab.links[iParent]._childrenIndices.push_back(iLink);

      // Compute link-to-root: parent's linkToRoot * this linkToParent
      link._rootFromLink = botPrefab.links[iParent]._rootFromLink * link._parentFromLink;
    }
  }

  // Update DOF indices and count. Revolute/Prismatic joints contribute 1 DOF
  // (1 entry in _dofIndices). Spherical joints contribute RigidSize::kDRot = 3
  // DOFs (3 entries pointing at the same joint index). Hard, Cycle, and Free
  // joints contribute no DOFs here (Free is the base/world joint; bot space always
  // excludes its 6 DOFs, which BuildArticulatedPoseFromBotPose zero-fills).
  botPrefab._dofIndices.clear();
  for (int iJoint = 0; iJoint < numJoints; ++iJoint) {
    auto const& joint = botPrefab.joints[iJoint];
    if (joint.type == ArticulatedJointType::Revolute ||
        joint.type == ArticulatedJointType::Prismatic) {
      botPrefab._dofIndices.push_back(iJoint);
    } else if (joint.type == ArticulatedJointType::Spherical) {
      for (int i = 0; i < RigidSize::kDRot; ++i) {
        botPrefab._dofIndices.push_back(iJoint);
      }
    }
  }
  botPrefab._numDofs = isize(botPrefab._dofIndices);
  // Update default joint positions
  botPrefab.defaultPose.resize(botPrefab._numDofs);
  int i = 0;
  for (auto& dofIdx : botPrefab._dofIndices) {
    auto const& joint = botPrefab.joints[dofIdx];
    if (joint.type == ArticulatedJointType::Prismatic ||
        joint.type == ArticulatedJointType::Revolute) {
      if (joint.minLimit.has_value() && IsFinite(*joint.minLimit)) {
        real minLimit = Dot(*joint.minLimit, joint.axis);
        if (botPrefab.defaultPose[i] < minLimit) {
          botPrefab.defaultPose[i] = minLimit;
        }
      }
      if (joint.maxLimit.has_value() && IsFinite(*joint.maxLimit)) {
        real maxLimit = Dot(*joint.maxLimit, joint.axis);
        if (botPrefab.defaultPose[i] > maxLimit) {
          botPrefab.defaultPose[i] = maxLimit;
        }
      }
    }
    ++i;
  }
  // Drop contact overrides that reference links no longer present in the prefab.
  // Link names are stable across SortBotPrefab, so a missing name indicates the
  // referenced link was removed (e.g. via RemoveLinkAndDescendants/ReplaceLink).
  std::unordered_set<std::string_view> linkNames;
  linkNames.reserve(numLinks);
  for (auto const& link : botPrefab.links) {
    linkNames.insert(std::string_view(link.name));
  }
  // The skin is a valid contact-override party, referenced by the bot's (top-level actor's) name,
  // so keep skin overrides when a skin is present and prune them when it is absent (mirrors
  // link-name pruning).
  if (botPrefab.skin.has_value()) {
    linkNames.insert(GetBotSkinName(botPrefab));
  }
  auto& overrides = botPrefab.contactOverrides;
  overrides.erase(
      std::remove_if(
          overrides.begin(),
          overrides.end(),
          [&](BotContactOverride const& f) {
            return linkNames.count(std::string_view(f.linkA)) == 0 ||
                linkNames.count(std::string_view(f.linkB)) == 0;
          }),
      overrides.end());
  // NOTE: Cycle joints are intentionally NOT pruned here. Their link references
  // are remapped in SortBotPrefab, but invalid cycles (out-of-range or
  // parent == child) are left in place so the user's edits are not silently
  // discarded (e.g. transiently selecting the same link in the editor). Invalid
  // cycles are surfaced by Validate and rejected by actor creation instead.
}

void superdex::robotics::SortBotPrefab(BotPrefab& botPrefab, Error& error) {
  MOCHI_ERROR_RETURN(error);
  int const numLinks = isize(botPrefab.links);
  if (numLinks <= 1) {
    return;
  }
  MOCHI_ERROR_IF(
      botPrefab.links[0].parentLink != kIndexNone,
      error,
      "SortBotPrefab: links[0] is not root (parentLink != kIndexNone)");
  MOCHI_ERROR_IF(
      isize(botPrefab.joints) != numLinks,
      error,
      "SortBotPrefab: expected same number of joints and links");
  MOCHI_ERROR_RETURN(error);
  // Validate parent indices
  for (int i = 1; i < numLinks; ++i) {
    MOCHI_ERROR_IF(
        botPrefab.links[i].parentLink < 0 || botPrefab.links[i].parentLink >= numLinks,
        error,
        "SortBotPrefab: link has invalid parentLink");
  }
  MOCHI_ERROR_RETURN(error);
  // Build children lists from parentLink
  DynamicArray<DynamicArray<int>> children;
  children.resize(numLinks);
  for (int i = 1; i < numLinks; ++i) {
    children[botPrefab.links[i].parentLink].push_back(i);
  }
  // Sort children at each node using alphanumeric comparison on link names
  for (auto& childList : children) {
    std::sort(childList.begin(), childList.end(), [&](int a, int b) {
      return AlphaNumCompare(botPrefab.links[a].name, botPrefab.links[b].name) < 0;
    });
  }
  // DFS pre-order traversal to produce newOrder (newOrder[newIdx] = oldIdx)
  DynamicArray<int> newOrder;
  newOrder.reserve(numLinks);
  DynamicArray<int> stack;
  stack.push_back(0);
  while (!stack.empty()) {
    int const idx = stack.back();
    stack.pop_back();
    newOrder.push_back(idx);
    for (int i = isize(children[idx]) - 1; i >= 0; --i) {
      stack.push_back(children[idx][i]);
    }
  }
  // DFS visits fewer than numLinks nodes only if the link graph is disconnected
  // (e.g. a parent cycle not reachable from root). Reject before indexing below.
  MOCHI_ERROR_IF(
      isize(newOrder) != numLinks,
      error,
      "SortBotPrefab: link graph is not a tree rooted at link 0 (cycle or disconnected component).");
  MOCHI_ERROR_RETURN(error);
  // Build inverse map: oldToNew[oldIdx] = newIdx
  DynamicArray<int> oldToNew(numLinks, 0);
  for (int newIdx = 0; newIdx < numLinks; ++newIdx) {
    oldToNew[newOrder[newIdx]] = newIdx;
  }
  // Build old-joint-to-old-DOF-start map. Stores the index of the first DOF
  // for each joint in defaultPose (kIndexNone if the joint contributes 0 DOFs).
  DynamicArray<int> oldJointToDof(numLinks, kIndexNone);
  int dofCount = 0;
  for (int i = 0; i < numLinks; ++i) {
    auto const& joint = botPrefab.joints[i];
    int const jointDofs = BotJointNumDofs(joint.type);
    if (jointDofs > 0) {
      oldJointToDof[i] = dofCount;
      dofCount += jointDofs;
    }
  }
  // Apply permutation to links and joints
  DynamicArray<BotLinkPrefab> newLinks;
  DynamicArray<BotJointPrefab> newJoints;
  newLinks.reserve(numLinks);
  newJoints.reserve(numLinks);
  for (int newIdx = 0; newIdx < numLinks; ++newIdx) {
    int const oldIdx = newOrder[newIdx];
    newLinks.push_back(botPrefab.links[oldIdx]);
    newJoints.push_back(botPrefab.joints[oldIdx]);
  }
  // Remap parentLink
  for (int i = 1; i < numLinks; ++i) {
    newLinks[i].parentLink = oldToNew[newLinks[i].parentLink];
  }
  // Remap valid cycle link references through the same permutation. Cycles are
  // not reordered, and malformed references remain present for validation.
  auto remapCycleLink = [&](int& linkIdx) {
    if (linkIdx >= 0 && linkIdx < numLinks) {
      linkIdx = oldToNew[linkIdx];
    }
  };
  for (auto& cycle : botPrefab.cycles) {
    remapCycleLink(cycle.parentLink);
    remapCycleLink(cycle.childLink);
  }
  // Remap transmission/tendon joint & link references through the same
  // permutation. oldToNew is a bijection here, so nothing is dropped.
  RemapTransmissionAndTendonIndices(botPrefab, oldToNew);
  // Reorder defaultPose based on new DOF order. Each joint contributes 1
  // (Revolute/Prismatic) or RigidSize::kDRot (Spherical) consecutive entries.
  DynamicArray<real> newDefaultPose;
  newDefaultPose.reserve(botPrefab.defaultPose.size());
  for (int newIdx = 0; newIdx < numLinks; ++newIdx) {
    int const oldIdx = newOrder[newIdx];
    auto const& joint = botPrefab.joints[oldIdx];
    int const jointDofs = BotJointNumDofs(joint.type);
    int const oldDofStart = oldJointToDof[oldIdx];
    for (int k = 0; k < jointDofs; ++k) {
      if (oldDofStart != kIndexNone && oldDofStart + k < isize(botPrefab.defaultPose)) {
        newDefaultPose.push_back(botPrefab.defaultPose[oldDofStart + k]);
      } else {
        newDefaultPose.push_back(0_r);
      }
    }
  }
  // Assign permuted data back
  botPrefab.links = std::move(newLinks);
  botPrefab.joints = std::move(newJoints);
  botPrefab.defaultPose = std::move(newDefaultPose);
}

/* Scales joint friction, joint inertia, joint limit stiffness/damping and cycle stiffness so the
 * scaled bot responds like the nominal one on the same time scale (dynamic similarity).
 * Stiffness, damping and inertia scale by @ref details::JointGainScale, Coulomb and stiction
 * forces/torques by @ref details::JointEffortScale, and velocity thresholds of translational joints
 * by s. Free and hard joints are left unchanged; cycle stiffness scales by s^3. */
static void ScaleJointDynamics(ArticulatedActorParams& params, real scale) {
  if (scale == 1_r) {
    return;
  }
  for (auto& joint : params.joints) {
    bool const translational = joint.type == ArticulatedJointType::Prismatic;
    bool const rotational = joint.type == ArticulatedJointType::Revolute ||
        joint.type == ArticulatedJointType::Spherical;
    if (!translational && !rotational) {
      continue;
    }
    real const gainScale = superdex::robotics::details::JointGainScale(joint.type, scale);
    real const effortScale = superdex::robotics::details::JointEffortScale(joint.type, scale);
    joint.friction.viscous *= gainScale;
    joint.friction.coulomb *= effortScale;
    joint.friction.stictionExtra *= effortScale;
    if (translational) {
      joint.friction.falloffVel *= scale;
      joint.friction.stribeckVel *= scale;
    }
    if (joint.inertia.has_value()) {
      *joint.inertia *= gainScale;
    }
    joint.limitStiffness *= gainScale;
    joint.limitDamping *= gainScale;
  }
  for (auto& cycle : params.cycles) {
    cycle.stiffness *= scale * scale * scale;
  }
}

/* A bot-space pose for a bot instantiated at BotPrefab::scale: prismatic DOFs are distances and
 * scale, rotational DOFs are angles and do not. */
static DynamicArray<real>
ScaleBotPose(BotPrefab const& botPrefab, Span<real const> pose, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  DynamicArray<real> scaled(pose.begin(), pose.end());
  if (botPrefab.scale == 1_r) {
    return scaled;
  }
  MOCHI_ERROR_IF(
      isize(pose) != isize(botPrefab._dofIndices),
      error,
      "Pose size does not match the bot's DOF count (call RebuildBotData first)");
  MOCHI_ERROR_RETURN(error, {});
  for (int dof = 0; dof < isize(scaled); ++dof) {
    if (botPrefab.joints[botPrefab._dofIndices[dof]].type == ArticulatedJointType::Prismatic) {
      scaled[dof] *= botPrefab.scale;
    }
  }
  return scaled;
}

ArticulatedActorParams superdex::robotics::BuildArticulatedActorParams(
    BotPrefab const& botPrefab,
    IBotLoader const& loader,
    Context* context,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  real const scale = botPrefab.scale;
  MOCHI_ERROR_IF_NOT(
      IsFinite(scale) && scale > 0_r, error, "BotPrefab::scale must be positive and finite");
  // Validate reports this too, but CreateBot does not validate.
  MOCHI_ERROR_IF(
      scale != 1_r && (!botPrefab.linearTransmissions.empty() || !botPrefab.spatialTendons.empty()),
      error,
      "BotPrefab::scale is not supported for bots with linear transmissions or spatial tendons");
  MOCHI_ERROR_RETURN(error, {});

  // Joints
  int const numJoints = isize(botPrefab.joints);
  DynamicArray<ArticulatedJointParams> jointParams(numJoints);
  for (int i = 0; i < numJoints; ++i) {
    auto const& joint = botPrefab.joints[i];
    jointParams[i].name = joint.name;
    jointParams[i].type = joint.type;
    jointParams[i].parentLinkFromJoint = joint.parentLinkFromJoint;
    jointParams[i].axis = joint.axis;
    jointParams[i].friction = joint.friction;
    jointParams[i].inertia = joint.inertia;
    jointParams[i].minLimit = joint.minLimit;
    jointParams[i].maxLimit = joint.maxLimit;
    jointParams[i].limitStiffness = joint.limitStiffness;
    jointParams[i].limitDamping = joint.limitDamping;
  }

  // Links
  int const numLinks = isize(botPrefab.links);
  DynamicArray<ArticulatedLinkParams> linkParams(numLinks);
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = botPrefab.links[i];
    ShapeHandle linkShape;
    if (!link.shapeFile.empty()) {
      linkShape = loader.LoadShape(
          link.shapeFile,
          link.shapeScale * scale,
          TransformRT(link.shapeRotation, link.shapeTranslation * scale),
          context,
          error);
    }
    MOCHI_ERROR_RETURN(error, {});
    linkParams[i].name = link.name;
    linkParams[i].parentLink = link.parentLink;
    linkParams[i].parentJointFromLink = link.parentJointFromLink;
    linkParams[i].shape = linkShape;
    linkParams[i].layer = link.layer;
    linkParams[i].colliderType = link.colliderType;
    linkParams[i].contact = link.contact;
    linkParams[i].hasGravity = link.hasGravity;
    linkParams[i].density = link.density;
    linkParams[i].mass = link.mass;
    linkParams[i].centerOfMass = link.centerOfMass;
    linkParams[i].momentOfInertia = link.momentOfInertia;
    linkParams[i].boundaryElementType = link.boundaryElementType;
    linkParams[i].boundarySubsampling = link.boundarySubsampling;
  }

  ArticulatedActorParams articulatedParams;
  articulatedParams.name = botPrefab.name;
  articulatedParams.worldFromRoot = botPrefab.worldFromRoot;
  articulatedParams.joints = std::move(jointParams);
  articulatedParams.links = std::move(linkParams);
  articulatedParams.cycles = botPrefab.cycles;

  // Skin (optional). The skin has no baked transform (see ArticulatedSkinPrefab), so the shape is
  // loaded with only BotPrefab::scale baked in.
  if (botPrefab.skin.has_value() && botPrefab.skin->shapeFile.empty() &&
      botPrefab.skin->renderModelFile.empty()) {
    // A skin with only a render model is a valid render-only skin; one with neither is not.
    MOCHI_LOG_WARNING(
        "BuiltBot: bot '%s' has a skin with no shape or render model file; instantiating without a "
        "skin.",
        botPrefab.name.c_str());
  }
  if (botPrefab.skin.has_value() && !botPrefab.skin->shapeFile.empty()) {
    ShapeHandle skinShape;
    if (botPrefab._skinBoneLinks.empty()) {
      // Identity bone->link mapping: the skin's baked bone indices already match the link order, so
      // the cached file shape can be used directly.
      skinShape = loader.LoadShape(
          botPrefab.skin->shapeFile, {scale, scale, scale}, TransformRT{}, context, error);
      MOCHI_ERROR_RETURN(error, {});
    } else {
      // A composed (mod) bot re-sorts links, so the skin's baked bone indices (which index link
      // frames positionally) must be remapped onto the final link order. Resolve each skin bone's
      // target link by name (names are stable across the sort), load the skin model, offset its
      // skinning indices, and bake a fresh shape from the remapped model data.
      std::unordered_map<std::string_view, int> linkIndexByName;
      linkIndexByName.reserve(botPrefab.links.size());
      for (int iLink = 0; iLink < isize(botPrefab.links); ++iLink) {
        linkIndexByName.emplace(std::string_view(botPrefab.links[iLink].name), iLink);
      }
      // _skinBoneLinks carries one entry per link at the time the skin was attached, which is
      // generally more than the skin has bones for. An entry that no longer resolves is only a
      // problem if a bone actually references it, so record -1 here and fail below only for the
      // bones the skin really uses -- otherwise removing an unrelated, unweighted link would fail
      // the whole build.
      DynamicArray<int> boneRemap;
      boneRemap.reserve(isize(botPrefab._skinBoneLinks));
      for (auto const& linkName : botPrefab._skinBoneLinks) {
        auto it = linkIndexByName.find(std::string_view(linkName));
        boneRemap.push_back(it == linkIndexByName.end() ? -1 : it->second);
      }
      ModelData skinModel = loader.LoadModelData(botPrefab.skin->shapeFile, error);
      MOCHI_ERROR_RETURN(error, {});
      if (!skinModel.mesh.has_value() || !skinModel.mesh->skinning.has_value()) {
        MOCHI_LOG_WARNING(
            "BuiltBot: skin model '%s' has no skinning data to remap.",
            botPrefab.skin->shapeFile.c_str());
        MOCHI_ERROR_SET(error, "Skin model has no skinning data to remap.");
        MOCHI_ERROR_RETURN(error, {});
      }
      // The skin mesh's rest coordinates are authored in the standalone (sub-)bot's own root frame,
      // where that root link is actor link 0. In the composed articulation the top-level actor root
      // is a different link, and the skinning solver resolves rest coordinates in that top-level
      // root frame. Re-express the rest coordinates by baking the transform of the skin's origin
      // link (skin bone 0's link) relative to the composed root, evaluated at the zero pose -- the
      // same reference pose at which the actor captures its reference bone frames. For a standalone
      // bot this transform is identity, which is why the branch above bakes only the scale.
      DynamicArray<TransformRT> rootFromLink;
      DynamicArray<real> const zeroPose(botPrefab._numDofs, 0_r);
      ComputeLinkTransformsFromPose(
          botPrefab, zeroPose, rootFromLink, LinkTransformSpace::RootFromParent, error);
      MOCHI_ERROR_RETURN(error, {});
      if (boneRemap.empty() || boneRemap[0] < 0) {
        MOCHI_LOG_WARNING(
            "BuiltBot: skin origin link '%s' is not present in the built bot.",
            boneRemap.empty() ? "<none>" : botPrefab._skinBoneLinks[0].c_str());
        MOCHI_ERROR_SET(error, "Skin origin link is not present in the built bot.");
        MOCHI_ERROR_RETURN(error, {});
      }
      TransformRT const rootFromSkinOrigin = rootFromLink[boneRemap[0]];
      model_utils::BakeTransform(
          skinModel,
          {scale, scale, scale},
          TransformRT(
              rootFromSkinOrigin.GetRotation(), rootFromSkinOrigin.GetTranslation() * scale),
          error);
      MOCHI_ERROR_RETURN(error, {});
      for (int& boneIndex : skinModel.mesh->skinning->indices) {
        if (boneIndex < 0 || boneIndex >= isize(boneRemap)) {
          MOCHI_LOG_WARNING(
              "BuiltBot: skin bone index %d out of range for bone->link mapping (size %d).",
              boneIndex,
              isize(boneRemap));
          MOCHI_ERROR_SET(error, "Skin bone index out of range for the skin's bone->link mapping.");
          MOCHI_ERROR_RETURN(error, {});
        }
        if (boneRemap[boneIndex] < 0) {
          MOCHI_LOG_WARNING(
              "BuiltBot: skin bone references link '%s' not present in the built bot.",
              botPrefab._skinBoneLinks[boneIndex].c_str());
          MOCHI_ERROR_SET(error, "Skin bone references a link not present in the built bot.");
          MOCHI_ERROR_RETURN(error, {});
        }
        boneIndex = boneRemap[boneIndex];
      }
      skinShape = context->CreateModelShape(skinModel, error);
      MOCHI_ERROR_RETURN(error, {});
    }
    ArticulatedSkinParams skinParams = *botPrefab.skin; // slice base ArticulatedSkinParams fields
    skinParams.shape = skinShape;
    articulatedParams.skin.emplace(std::move(skinParams));
  }

  prefab::ScaleArticulatedActorParams(articulatedParams, scale, error);
  MOCHI_ERROR_RETURN(error, {});
  ScaleJointDynamics(articulatedParams, scale);

  return articulatedParams;
}

DynamicArray<real> superdex::robotics::BuildArticulatedPoseFromBotPose(
    BotPrefab const& botPrefab,
    Span<real const> srcPose,
    int numActorDofs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (botPrefab.joints.empty()) {
    MOCHI_ERROR_SET(error, "Bot must have at least one joint");
    return {};
  }
  // TODO: either handle the other joint types for the base or invalidate them in Validate
  bool const hardWorldRoot = botPrefab.joints[0].type == ArticulatedJointType::Hard;
  bool const freeWorldRoot = botPrefab.joints[0].type == ArticulatedJointType::Free;
  if (!hardWorldRoot && !freeWorldRoot) {
    MOCHI_ERROR_SET(error, "Bot root joint must be Hard or Free")
    return {};
  }
  int const numBaseDofs = hardWorldRoot ? 0 : 6;
  int const expectedPoseSize = numActorDofs - numBaseDofs;
  int const poseSize = isize(srcPose);
  if (poseSize != expectedPoseSize) {
    MOCHI_LOG_ERROR(
        "Pose size [%d] does not match expected size [%d].", poseSize, expectedPoseSize);
    MOCHI_ERROR_SET(error, "Pose size does not match expected size.");
    return {};
  }
  DynamicArray<real> pose;
  pose.resize_noinit(numActorDofs);
  for (int i = 0; i < numBaseDofs; ++i) {
    pose[i] = 0_r;
  }
  for (int i = 0; i < poseSize; ++i) {
    pose[i + numBaseDofs] = srcPose[i];
  }
  return pose;
}

std::string_view superdex::robotics::GetBotSkinName(BotPrefab const& botPrefab) {
  if (!botPrefab.skin.has_value()) {
    return {};
  }
  // The skin shares the bot's top-level articulated actor, so it is referenced by that actor's
  // name (BotPrefab::name, which BuildArticulatedActorParams assigns to
  // ArticulatedActorParams::name).
  return std::string_view(botPrefab.name);
}

static std::unordered_map<std::string_view, int> MakeLinkIndexByName(
    Span<DynamicString const> linkNames) {
  std::unordered_map<std::string_view, int> linkIndexByName;
  linkIndexByName.reserve(linkNames.size());
  for (int i = 0; i < isize(linkNames); ++i) {
    linkIndexByName.emplace(std::string_view(linkNames[i]), i);
  }
  return linkIndexByName;
}

// Flag the entries of linkNames that names lists; names that match no link are ignored.
static DynamicArray<bool> FlagNamedLinks(
    Span<DynamicString const> names,
    Span<DynamicString const> linkNames) {
  DynamicArray<bool> flags(isize(linkNames), false);
  auto const linkIndexByName = MakeLinkIndexByName(linkNames);
  for (auto const& name : names) {
    auto const it = linkIndexByName.find(std::string_view(name));
    if (it != linkIndexByName.end()) {
      flags[it->second] = true;
    }
  }
  return flags;
}

DynamicArray<int> superdex::robotics::ComputeSkinBoneLinkIndices(
    BotPrefab const& botPrefab,
    Span<DynamicString const> linkNames) {
  DynamicArray<int> boneLinks;
  if (!botPrefab.skin.has_value()) {
    return boneLinks;
  }
  if (botPrefab._skinBoneLinks.empty()) {
    boneLinks.resize_noinit(isize(linkNames));
    std::iota(boneLinks.begin(), boneLinks.end(), 0);
    return boneLinks;
  }
  auto const linkIndexByName = MakeLinkIndexByName(linkNames);
  boneLinks.reserve(isize(botPrefab._skinBoneLinks));
  for (auto const& name : botPrefab._skinBoneLinks) {
    auto const it = linkIndexByName.find(std::string_view(name));
    boneLinks.push_back(it != linkIndexByName.end() ? it->second : kIndexNone);
  }
  return boneLinks;
}

DynamicArray<bool> superdex::robotics::ComputeSkinCoveredLinks(
    BotPrefab const& botPrefab,
    Span<DynamicString const> linkNames,
    SkinningData const* skinning) {
  // A skin without a collision shape is not part of the physics actor, so it covers nothing.
  if (!botPrefab.skin.has_value() || botPrefab.skin->shapeFile.empty()) {
    return DynamicArray<bool>(isize(linkNames), false);
  }
  if (botPrefab.skin->nonCollidingLinks.has_value()) {
    return FlagNamedLinks(MakeConstSpan(*botPrefab.skin->nonCollidingLinks), linkNames);
  }
  DynamicArray<bool> covered(isize(linkNames), false);
  if (skinning == nullptr) {
    return covered;
  }
  DynamicArray<int> const boneLinks = ComputeSkinBoneLinkIndices(botPrefab, linkNames);
  int const numInfluences = Min(isize(skinning->indices), isize(skinning->weights));
  for (int i = 0; i < numInfluences; ++i) {
    int const bone = skinning->indices[i];
    // Skins pad unused influence slots with bone 0 at weight 0, which must not cover the root link.
    if (skinning->weights[i] <= 0_r || bone < 0 || bone >= isize(boneLinks)) {
      continue;
    }
    int const link = boneLinks[bone];
    if (link != kIndexNone) {
      covered[link] = true;
    }
  }
  return covered;
}

DynamicArray<bool> superdex::robotics::ComputeSkinHiddenLinks(
    BotPrefab const& botPrefab,
    Span<DynamicString const> linkNames) {
  if (!botPrefab.skin.has_value() || botPrefab.skin->renderModelFile.empty()) {
    return DynamicArray<bool>(isize(linkNames), false);
  }
  if (botPrefab.skin->hiddenLinks.has_value()) {
    return FlagNamedLinks(MakeConstSpan(*botPrefab.skin->hiddenLinks), linkNames);
  }
  DynamicArray<bool> hidden(isize(linkNames), false);
  for (int const link : ComputeSkinBoneLinkIndices(botPrefab, linkNames)) {
    if (link != kIndexNone) {
      hidden[link] = true;
    }
  }
  return hidden;
}

static void ApplyBotContactOverrides(
    BotPrefab const& botPrefab,
    Span<BotContactOverride const> contactOverrides,
    Actor* actor,
    Scene* scene,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (contactOverrides.empty()) {
    return;
  }
  // Map prefab link names -> prefab link index -> nested link actor handle.
  // Handles are returned in the same order as the link params passed to
  // CreateArticulatedActorNew, which matches botPrefab.links. Mochi may
  // rename the nested actors, so do not rely on Actor::GetName().
  Span<ActorHandle const> const linkActors = actor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);
  if (isize(linkActors) != isize(botPrefab.links)) {
    MOCHI_ERROR_SET(
        error,
        "ApplyBotContactOverrides: nested link actor count does not match bot prefab link count");
    return;
  }
  std::unordered_map<std::string_view, int> nameToIndex;
  nameToIndex.reserve(botPrefab.links.size());
  for (int i = 0; i < isize(botPrefab.links); ++i) {
    nameToIndex.emplace(std::string_view(botPrefab.links[i].name), i);
  }
  // A skin party resolves to the articulated actor's own compound entity handle (the skin shares
  // the top-level actor). Skin<->link contact collides by default, so a disable override here opts
  // a specific skin<->link pair out.
  std::string_view const skinName =
      botPrefab.skin.has_value() ? GetBotSkinName(botPrefab) : std::string_view{};
  ActorHandle const skinHandle = actor->GetHandle();
  auto resolveParty = [&](DynamicString const& party, ActorHandle& outHandle) -> bool {
    std::string_view const name(party);
    if (!skinName.empty() && name == skinName) {
      outHandle = skinHandle;
      return true;
    }
    auto const it = nameToIndex.find(name);
    if (it == nameToIndex.end()) {
      return false;
    }
    outHandle = linkActors[it->second];
    return true;
  };
  for (auto const& over : contactOverrides) {
    ActorHandle handleA;
    ActorHandle handleB;
    if (!resolveParty(over.linkA, handleA) || !resolveParty(over.linkB, handleB)) {
      MOCHI_LOG_WARNING(
          "ApplyBotContactOverrides: contact override references unknown link or skin '%s' or '%s'",
          over.linkA.c_str(),
          over.linkB.c_str());
      continue;
    }
    scene->EnableActorContactSymmetric(
        handleA, handleB, /*enable*/ over.enable, IncludeNestedActors::No, error);
    MOCHI_ERROR_RETURN(error);
  }
}

static Actor* AddToSceneImpl(
    BotPrefab const& botPrefab,
    Scene* scene,
    IBotLoader const& loader,
    Span<real const> pose,
    Span<BotContactOverride const> contactOverrides,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  auto* context = scene->GetContext();
  auto params = BuildArticulatedActorParams(botPrefab, loader, context, error);
  auto* actor = scene->CreateArticulatedActor(params, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  DynamicArray<real> const scaledPose = ScaleBotPose(botPrefab, pose, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  DynamicArray<real> actorPose =
      BuildArticulatedPoseFromBotPose(botPrefab, scaledPose, actor->GetNumDofs(), error);
  MOCHI_ERROR_RETURN(error, nullptr);
  actor->SetArticulatedPoseFromJoints(actorPose, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  // Skin<->link contact is enabled by default (Mochi's contact matrix collides by default). A
  // BotContactOverride that names the skin opts a specific skin<->link pair out -- e.g. the skin's
  // own links, which it wraps and would otherwise fight the joints (a skinned double pendulum
  // seizing near 90 degrees).
  ApplyBotContactOverrides(botPrefab, contactOverrides, actor, scene, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  return actor;
}

Actor* superdex::robotics::AddToScene(
    BotPrefab const& botPrefab,
    Scene* scene,
    IBotLoader const& loader,
    Error& error) {
  return AddToSceneImpl(
      botPrefab, scene, loader, botPrefab.defaultPose, botPrefab.contactOverrides, error);
}

Actor* superdex::robotics::AddToScene(BotPrefab const& botPrefab, Scene* scene, Error& error) {
  return AddToSceneImpl(
      botPrefab, scene, FileBotLoader{}, botPrefab.defaultPose, botPrefab.contactOverrides, error);
}

int superdex::robotics::FindLinkIndexByName(BotPrefab const& bot, std::string_view name) {
  for (int i = 0; i < isize(bot.links); ++i) {
    if (bot.links[i].name == name) {
      return i;
    }
  }
  return kIndexNone;
}

DynamicArray<int> superdex::robotics::FindLeafLinkIndices(BotPrefab const& botPrefab) {
  int const numLinks = isize(botPrefab.links);
  DynamicArray<bool> isParent(numLinks, false);
  for (auto const& link : botPrefab.links) {
    if (link.parentLink >= 0 && link.parentLink < numLinks) {
      isParent[link.parentLink] = true;
    }
  }
  DynamicArray<int> leafIndices;
  for (int i = 0; i < numLinks; ++i) {
    if (!isParent[i]) {
      leafIndices.push_back(i);
    }
  }
  return leafIndices;
}

// Motion of a joint in its own frame for the bot-space DOFs starting at pose[inOutDofIndex], which
// is advanced past them. As Mochi's articulated::ComputeJointTransform, with axes normalized as
// Mochi does when it creates the actor, so a prismatic position is along the unit axis [m].
static TransformRT
ComputeBotJointMotion(BotJointPrefab const& joint, Span<real const> pose, int& inOutDofIndex) {
  switch (joint.type) {
    case ArticulatedJointType::Revolute:
      return TransformRT(Quaternion::FromAxisAngle(joint.axis, pose[inOutDofIndex++]));
    case ArticulatedJointType::Prismatic:
      return TransformRT(Normalize(joint.axis) * pose[inOutDofIndex++]);
    case ArticulatedJointType::Spherical: {
      static_assert(RigidSize::kDRot == 3, "Spherical joint pose layout assumes 3 rotation DOFs.");
      Real3 const rotVec{pose[inOutDofIndex], pose[inOutDofIndex + 1], pose[inOutDofIndex + 2]};
      inOutDofIndex += RigidSize::kDRot;
      return TransformRT(Quaternion::FromRotationVector(rotVec));
    }
    case ArticulatedJointType::Free:
    case ArticulatedJointType::Hard:
    case ArticulatedJointType::Cycle:
    case ArticulatedJointType::Count:
      break;
  }
  return TransformRT::Identity();
}

void superdex::robotics::ComputeLinkTransformsFromPose(
    BotPrefab const& botPrefab,
    Span<real const> pose,
    DynamicArray<TransformRT>& outLinkTransforms,
    LinkTransformSpace transformSpace,
    Error& error,
    int finalLinkIndex) {
  MOCHI_ERROR_RETURN(error);
  int const numLinks = isize(botPrefab.links);
  MOCHI_ERROR_IF(numLinks == 0, error, "botPrefab has no links");
  MOCHI_ERROR_IF(isize(botPrefab.joints) != numLinks, error, "Joint count not equal to link count");
  MOCHI_ERROR_IF(isize(pose) != botPrefab._numDofs, error, "pose size does not match DOF count");
  MOCHI_ERROR_RETURN(error);
  int const iFinal = finalLinkIndex < 0 ? numLinks : Min(finalLinkIndex + 1, numLinks);
  outLinkTransforms.resize(iFinal);
  outLinkTransforms[0] = TransformRT::Identity();
  int iDof = 0;
  for (int iLink = 1; iLink < iFinal; ++iLink) {
    auto const& link = botPrefab.links[iLink];
    auto const& joint = botPrefab.joints[iLink];
    MOCHI_ASSERT_VERBOSE(
        link.parentLink >= 0 && link.parentLink < iLink, "Links must follow their parents");
    // As Mochi's articulated::ComputeTransformsFromReducedPose: parentFromLink =
    // parentLinkFromJoint * motion * jointFromChildLink.
    outLinkTransforms[iLink] = joint.parentLinkFromJoint *
        ComputeBotJointMotion(joint, pose, iDof) * link.parentJointFromLink;
    if (transformSpace == LinkTransformSpace::RootFromParent) {
      // Mochi: parent * child (opposite of Unreal's child * parent)
      outLinkTransforms[iLink] = outLinkTransforms[link.parentLink] * outLinkTransforms[iLink];
    }
  }
  MOCHI_ASSERT_VERBOSE(iFinal < numLinks || iDof == botPrefab._numDofs, "Stale _numDofs");
}

// Angle [rad] of the rotation of a unit quaternion with the given vector and scalar parts, in
// [0, pi]. atan2 keeps small angles accurate, unlike acos of the scalar part.
static real RotationAngle(Real3 const& vectorPart, real scalarPart) {
  return 2_r * std::atan2(Norm(vectorPart), Abs(scalarPart));
}

// What a joint's motion holds that the joint can't produce (see LinkPoseResidual).
static LinkPoseResidual ComputeJointResidual(
    BotJointPrefab const& joint,
    TransformRT const& motion) {
  Quaternion const& q = motion.GetRotation();
  Real3 const vectorPart{q.data[0], q.data[1], q.data[2]};
  Real3 const translation = motion.GetTranslation();
  switch (joint.type) {
    case ArticulatedJointType::Revolute: {
      // The swing left after the nearest rotation about the axis, whose quaternion is the axis
      // component of the vector part with the scalar part.
      Real3 const axis = Normalize(joint.axis);
      real const sinHalf = Dot(vectorPart, axis);
      return {
          RotationAngle(
              vectorPart - axis * sinHalf, std::sqrt(sinHalf * sinHalf + q.data[3] * q.data[3])),
          Norm(translation)};
    }
    case ArticulatedJointType::Prismatic: {
      Real3 const axis = Normalize(joint.axis);
      return {
          RotationAngle(vectorPart, q.data[3]), Norm(translation - axis * Dot(translation, axis))};
    }
    case ArticulatedJointType::Spherical:
      return {0_r, Norm(translation)};
    case ArticulatedJointType::Hard:
      return {RotationAngle(vectorPart, q.data[3]), Norm(translation)};
    case ArticulatedJointType::Free:
    case ArticulatedJointType::Cycle:
    case ArticulatedJointType::Count:
      break;
  }
  return {};
}

void superdex::robotics::ComputePoseFromLinkTransforms(
    BotPrefab const& botPrefab,
    Span<TransformRT const> linkTransforms,
    DynamicArray<real>& outPose,
    LinkTransformSpace transformSpace,
    Error& error,
    Span<real const> referencePose,
    DynamicArray<LinkPoseResidual>* outResiduals) {
  // Below this squared norm of the twist quaternion (a swing within about 2e-5 rad of a half turn
  // off a revolute axis), the revolute angle is undetermined.
  static constexpr real kDegenerateTwistSq = 1e-10_r;
  MOCHI_ERROR_RETURN(error);
  int const numLinks = isize(botPrefab.links);
  MOCHI_ERROR_IF(numLinks == 0, error, "botPrefab has no links");
  MOCHI_ERROR_IF(isize(botPrefab.joints) != numLinks, error, "Joint count not equal to link count");
  MOCHI_ERROR_IF(
      isize(linkTransforms) != numLinks, error, "Link transform count not equal to link count");
  MOCHI_ERROR_IF(
      !referencePose.empty() && isize(referencePose) != botPrefab._numDofs,
      error,
      "referencePose size does not match DOF count");
  MOCHI_ERROR_RETURN(error);
  outPose.resize(botPrefab._numDofs);
  if (outResiduals != nullptr) {
    *outResiduals = DynamicArray<LinkPoseResidual>(numLinks, LinkPoseResidual{});
  }
  int iDof = 0;
  for (int iLink = 1; iLink < numLinks; ++iLink) {
    auto const& link = botPrefab.links[iLink];
    auto const& joint = botPrefab.joints[iLink];
    // Mochi: parent * child (opposite of Unreal's child * parent)
    TransformRT const parentFromLink = transformSpace == LinkTransformSpace::RootFromParent
        ? Invert(linkTransforms[link.parentLink]) * linkTransforms[iLink]
        : linkTransforms[iLink];
    // ComputeLinkTransformsFromPose: parentFromLink = parentLinkFromJoint * motion *
    // parentJointFromLink.
    TransformRT const motion =
        Invert(joint.parentLinkFromJoint) * parentFromLink * Invert(link.parentJointFromLink);
    if (outResiduals != nullptr) {
      (*outResiduals)[iLink] = ComputeJointResidual(joint, motion);
    }
    Quaternion const& q = motion.GetRotation();
    switch (joint.type) {
      case ArticulatedJointType::Revolute: {
        // The twist of the motion about the axis, 2 atan2(v . axis, w): the nearest rotation about
        // the axis. Both vanish when every angle is equally near (a half turn about a
        // perpendicular axis); the angle is then the reference (or 0) rather than whatever
        // rounding gives. Of the angles 2 pi apart giving the same rotation, take the nearest to
        // the reference, else the one in [-pi, pi].
        real const sinHalf = Dot(Real3{q.data[0], q.data[1], q.data[2]}, Normalize(joint.axis));
        real const cosHalf = q.data[3];
        real const reference = referencePose.empty() ? 0_r : referencePose[iDof];
        real const angle = sinHalf * sinHalf + cosHalf * cosHalf > kDegenerateTwistSq
            ? 2_r * std::atan2(sinHalf, cosHalf)
            : reference;
        outPose[iDof++] = angle + 2_r * kPI * std::round((reference - angle) / (2_r * kPI));
        break;
      }
      case ArticulatedJointType::Prismatic:
        // Translation of the motion along the unit axis.
        outPose[iDof++] = Dot(motion.GetTranslation(), Normalize(joint.axis));
        break;
      case ArticulatedJointType::Spherical: {
        // Rotation vector of norm <= pi, as Mochi's articulated::ConvertPoseToDofs.
        Real3 const rotVec = q.ToRotationVector();
        outPose[iDof++] = rotVec[0];
        outPose[iDof++] = rotVec[1];
        outPose[iDof++] = rotVec[2];
        break;
      }
      case ArticulatedJointType::Free:
      case ArticulatedJointType::Hard:
      case ArticulatedJointType::Cycle:
      case ArticulatedJointType::Count:
        break;
    }
  }
  MOCHI_ASSERT_VERBOSE(iDof == botPrefab._numDofs, "Stale _numDofs");
}

// Computes and bakes mass properties (centerOfMass, momentOfInertia, and mass for density-only
// links) for a single link from its mesh geometry. No-op for links without a shape, links already
// fully specified, or links with neither mass nor density.
static void BakeLinkMassProperties(
    BotLinkPrefab& link,
    Scene* scene,
    Context* context,
    IBotLoader const& loader,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Skip links without a shape (cannot compute from geometry).
  if (link.shapeFile.empty()) {
    return;
  }
  // Skip links that already have full inertial specification.
  if (link.centerOfMass.has_value() && link.momentOfInertia.has_value()) {
    return;
  }
  // Skip links that have neither mass nor density (nothing to anchor computation).
  if (!link.mass.has_value() && !link.density.has_value()) {
    return;
  }

  // Load the link's shape.
  ShapeHandle const shape = loader.LoadShape(
      link.shapeFile,
      link.shapeScale,
      TransformRT(link.shapeRotation, link.shapeTranslation),
      context,
      error);
  MOCHI_ERROR_RETURN(error);

  // Create a temporary rigid actor - the engine computes COM/MOI from the mesh.
  RigidActorParams params;
  params.shape = shape;
  params.density = link.density;
  params.mass = link.mass;
  params.boundaryElementType = link.boundaryElementType;
  auto* actor = scene->CreateRigidActor(params, error);
  MOCHI_ERROR_RETURN(error);

  // Read back computed mass properties.
  link.centerOfMass = actor->GetRigidCenterOfMassLocal(error);
  MOCHI_ERROR_RETURN(error);
  link.momentOfInertia = actor->GetRigidMomentOfInertiaLocal(error);
  MOCHI_ERROR_RETURN(error);
  // Bake density into mass (mass = density * volume) and clear density.
  if (!link.mass.has_value()) {
    link.mass = actor->GetMass(error);
    MOCHI_ERROR_RETURN(error);
  }
  link.density.reset();

  scene->DestroyActor(actor);
}

void superdex::robotics::BakeMassProperties(
    BotPrefab& botPrefab,
    Context* context,
    IBotLoader const& loader,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  auto* scene = context->CreateScene("BakeMassProperties");
  if (scene == nullptr) {
    MOCHI_ERROR_SET(error, "Failed to create scene for BakeMassProperties");
    return;
  }
  MOCHI_DEFER(context->DestroyScene(scene));

  for (auto& link : botPrefab.links) {
    BakeLinkMassProperties(link, scene, context, loader, error);
    MOCHI_ERROR_RETURN(error);
  }
}

void superdex::robotics::BakeMassProperties(
    ModBotPrefab& modBotPrefab,
    Context* context,
    IBotLoader const& loader,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  auto* scene = context->CreateScene("BakeMassProperties");
  if (scene == nullptr) {
    MOCHI_ERROR_SET(error, "Failed to create scene for BakeMassProperties");
    return;
  }
  MOCHI_DEFER(context->DestroyScene(scene));

  // Bake the inertial properties of the link introduced by each AttachLink / ReplaceLink mod.
  for (auto& mod : modBotPrefab.modifications) {
    if (auto* attach = std::get_if<AttachLink>(&mod)) {
      BakeLinkMassProperties(attach->link, scene, context, loader, error);
    } else if (auto* replace = std::get_if<ReplaceLink>(&mod)) {
      BakeLinkMassProperties(replace->link, scene, context, loader, error);
    }
    MOCHI_ERROR_RETURN(error);
  }
}

bool superdex::robotics::IsContactImplicitlyDisabled(
    BotPrefab const& botPrefab,
    int linkA,
    int linkB) {
  int const numLinks = isize(botPrefab.links);
  if (linkA < 0 || linkA >= numLinks || linkB < 0 || linkB >= numLinks || linkA == linkB) {
    return false;
  }
  // Mochi only manages contact between shape-bearing link actors.
  if (botPrefab.links[linkA].shapeFile.empty() || botPrefab.links[linkB].shapeFile.empty()) {
    return false;
  }
  // Mirror Scene::CreateArticulatedActor's DisableContactForAdjacentActors DFS:
  // starting at linkA, traverse undirected link graph; we may pass *through* a node only
  // if it has no shape, or the edge entering it is a Hard joint. The destination
  // (linkB) counts as reached upon arrival via any edge.
  auto hasShape = [&](int idx) { return !botPrefab.links[idx].shapeFile.empty(); };
  auto edgeIsHard = [&](int child) {
    // The joint at index child is the joint connecting child to its parent.
    return botPrefab.joints[child].type == ArticulatedJointType::Hard;
  };
  DynamicArray<bool> visited(numLinks, false);
  DynamicArray<int> stack;
  stack.push_back(linkA);
  visited[linkA] = true;
  while (!stack.empty()) {
    int const current = stack.back();
    stack.pop_back();
    // Collect neighbors via the parent/children topology.
    auto visitNeighbor = [&](int neighbor, bool edgeHard) -> bool {
      if (visited[neighbor]) {
        return false;
      }
      visited[neighbor] = true;
      if (neighbor == linkB) {
        return true; // reached destination
      }
      // Continue past neighbor only if the edge is hard OR neighbor has no shape.
      if (edgeHard || !hasShape(neighbor)) {
        stack.push_back(neighbor);
      }
      return false;
    };
    // Parent edge: edge entering current from its parent uses joint[current].type.
    int const parent = botPrefab.links[current].parentLink;
    if (parent != kIndexNone) {
      if (visitNeighbor(parent, edgeIsHard(current))) {
        return true;
      }
    }
    // Child edges: edge from current to child uses joint[child].type.
    for (int child : botPrefab.links[current]._childrenIndices) {
      if (visitNeighbor(child, edgeIsHard(child))) {
        return true;
      }
    }
  }
  return false;
}

// Iterate the DOFs of bp in pose order; for each one call fn(jointType, lo_opt, hi_opt)
// and append its result to the returned pose. Sets err and returns {} if the produced size
// does not match bp._numDofs.
template <typename Fn>
static DynamicArray<real> BuildPoseImpl(BotPrefab const& bp, char const* name, Error& err, Fn fn) {
  MOCHI_ERROR_RETURN(err, {});
  DynamicArray<real> pose;
  pose.reserve(bp._numDofs);
  int const numJoints = isize(bp.joints);
  for (int iJoint = 0; iJoint < numJoints; ++iJoint) {
    auto const& joint = bp.joints[iJoint];
    if (joint.type == ArticulatedJointType::Revolute ||
        joint.type == ArticulatedJointType::Prismatic) {
      std::optional<real> lo;
      std::optional<real> hi;
      if (joint.minLimit.has_value() && IsFinite(*joint.minLimit)) {
        lo = Dot(*joint.minLimit, joint.axis);
      }
      if (joint.maxLimit.has_value() && IsFinite(*joint.maxLimit)) {
        hi = Dot(*joint.maxLimit, joint.axis);
      }
      pose.push_back(fn(joint.type, lo, hi));
    } else if (joint.type == ArticulatedJointType::Spherical) {
      // Spherical pose is a 3-component rotation vector (axis-angle).
      static_assert(RigidSize::kDRot == 3, "Spherical joint pose layout assumes 3 rotation DOFs.");
      for (int k = 0; k < RigidSize::kDRot; ++k) {
        std::optional<real> lo;
        std::optional<real> hi;
        if (joint.minLimit.has_value() && IsFinite(*joint.minLimit)) {
          lo = (*joint.minLimit)[k];
        }
        if (joint.maxLimit.has_value() && IsFinite(*joint.maxLimit)) {
          hi = (*joint.maxLimit)[k];
        }
        pose.push_back(fn(joint.type, lo, hi));
      }
    }
    // Hard / Cycle / Free contribute no entries to the pose.
  }
  if (isize(pose) != bp._numDofs) {
    MOCHI_LOG_ERROR(
        "%s: produced pose size [%d] does not match _numDofs [%d]. "
        "Call RebuildBotData first.",
        name,
        isize(pose),
        bp._numDofs);
    MOCHI_ERROR_SET(err, "Produced pose size does not match _numDofs.");
    return {};
  }
  return pose;
}

DynamicArray<real>
superdex::robotics::MakeBotPose(BotPrefab const& botPrefab, MakeBotPoseType type, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  constexpr real kTwoPi = 2_r * kPI;
  constexpr real kPrismaticDefaultHalfRange = 1_r;
  auto fallbackLo = [](ArticulatedJointType jt) {
    return jt == ArticulatedJointType::Prismatic ? -kPrismaticDefaultHalfRange : 0_r;
  };
  auto fallbackHi = [](ArticulatedJointType jt) {
    return jt == ArticulatedJointType::Prismatic ? kPrismaticDefaultHalfRange : kTwoPi;
  };
  switch (type) {
    case MakeBotPoseType::Zero:
      return BuildPoseImpl(
          botPrefab,
          "MakeBotPose(Zero)",
          error,
          [](ArticulatedJointType, std::optional<real> lo, std::optional<real> hi) -> real {
            real v = 0_r;
            if (lo.has_value()) {
              v = std::max(v, *lo);
            }
            if (hi.has_value()) {
              v = std::min(v, *hi);
            }
            return v;
          });
    case MakeBotPoseType::Min:
      return BuildPoseImpl(
          botPrefab,
          "MakeBotPose(Min)",
          error,
          [&](ArticulatedJointType jt, std::optional<real> lo, std::optional<real>) -> real {
            return lo.value_or(fallbackLo(jt));
          });
    case MakeBotPoseType::Max:
      return BuildPoseImpl(
          botPrefab,
          "MakeBotPose(Max)",
          error,
          [&](ArticulatedJointType jt, std::optional<real>, std::optional<real> hi) -> real {
            return hi.value_or(fallbackHi(jt));
          });
    case MakeBotPoseType::Mid:
      return BuildPoseImpl(
          botPrefab,
          "MakeBotPose(Mid)",
          error,
          [](ArticulatedJointType, std::optional<real> lo, std::optional<real> hi) -> real {
            if (lo.has_value() && hi.has_value()) {
              return (*lo + *hi) * 0.5_r;
            }
            real v = 0_r;
            if (lo.has_value()) {
              v = std::max(v, *lo);
            }
            if (hi.has_value()) {
              v = std::min(v, *hi);
            }
            return v;
          });
    case MakeBotPoseType::Random: {
      std::mt19937 rng{std::random_device{}()};
      return BuildPoseImpl(
          botPrefab,
          "MakeBotPose(Random)",
          error,
          [&](ArticulatedJointType jt, std::optional<real> lo, std::optional<real> hi) -> real {
            real l = lo.value_or(fallbackLo(jt));
            real h = hi.value_or(fallbackHi(jt));
            if (l > h) {
              std::swap(l, h);
            }
            std::uniform_real_distribution<real> dist(l, h);
            return dist(rng);
          });
    }
  }
  MOCHI_ERROR_SET(error, "MakeBotPose: invalid MakeBotPoseType");
  return {};
}

DynamicArray<real> superdex::robotics::GetEffortLimitsPerDof(
    BotPrefab const& botPrefab,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Walk _dofIndices (bot-space DOF order) exactly as RebuildBotData builds defaultPose, so a
  // Spherical joint's single effortLimit repeats across its 3 DOF slots. Values keep their
  // BotJointPrefab::effortLimit semantics (< 0 unbounded, 0 non-actuated, > 0 finite); finite
  // limits are scaled for BotPrefab::scale.
  DynamicArray<real> limits;
  limits.resize(isize(botPrefab._dofIndices));
  for (int dof = 0; dof < isize(botPrefab._dofIndices); ++dof) {
    int const jointIndex = botPrefab._dofIndices[dof];
    MOCHI_ERROR_IF(
        jointIndex < 0 || jointIndex >= isize(botPrefab.joints),
        error,
        "GetEffortLimitsPerDof: _dofIndices is stale or unrebuilt (call RebuildBotData first)");
    MOCHI_ERROR_RETURN(error, {});
    BotJointPrefab const& joint = botPrefab.joints[jointIndex];
    limits[dof] = joint.effortLimit > 0_r ? joint.effortLimit *
            superdex::robotics::details::JointEffortScale(joint.type, botPrefab.scale)
                                          : joint.effortLimit;
  }
  return limits;
}
