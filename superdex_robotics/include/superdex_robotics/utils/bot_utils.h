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
#include <superdex_physics.h>
#include <superdex_robotics/superdex_robotics.h>

namespace superdex::robotics {

struct IBotLoader;

/**
 * @brief Add a new link and connecting joint as a child to an existing link.
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] parentLinkIdx The parent link to attach the new child link to.
 * @param[in] baseName The base name used for naming the new link and joint; actual names may
 * differ.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API int
AddLink(BotPrefab& botPrefab, int parentLinkIdx, std::string_view baseName, superdex::Error& error);

/**
 * @brief Remove a link and all of its descendants from a bot.
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] linkIdx Index of the link to remove.
 * @param[in] keepParentJoint If true, the parent joint of the remove link will be kept.
 * @param[out] outNewParentIdx Index of the removed link's parent after removal.
 * @param[out] outConnectingJoint Joint that connected the removed link to its parent.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void RemoveLinkAndDescendants(
    BotPrefab& botPrefab,
    int linkIdx,
    bool keepParentJoint,
    int& outNewParentIdx,
    BotJointPrefab& outConnectingJoint,
    superdex::Error& error);

/**
 * @brief Remove a link, its parent joint, and all of its descendants from a bot.
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] linkIdx Index of the link to remove.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void RemoveLinkAndDescendants(BotPrefab& botPrefab, int linkIdx, superdex::Error& error);

/**
 * @brief Apply an @ref AttachLink modification to a bot, resolving parent link by name.
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] mod The @ref AttachLink modification to apply.
 * @param[in] loader Bot loader.
 * @param[in] validate Unused.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void ApplyMod(
    BotPrefab& botPrefab,
    AttachLink const& mod,
    IBotLoader const& loader,
    bool validate,
    superdex::Error& error);

/**
 * @brief Apply an @ref AttachBot modification to a bot, resolving parent link by name.
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] mod The @ref AttachBot modification to apply.
 * @param[in] loader Bot loader.
 * @param[in] validate Validate the attached bot when it is loaded.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void ApplyMod(
    BotPrefab& botPrefab,
    AttachBot const& mod,
    IBotLoader const& loader,
    bool validate,
    superdex::Error& error);

/**
 * @brief Apply a @ref ReplaceLink modification to a bot.
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] mod The @ref ReplaceLink modification to apply.
 * @param[in] loader Bot loader.
 * @param[in] validate Unused.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void ApplyMod(
    BotPrefab& botPrefab,
    ReplaceLink const& mod,
    IBotLoader const& loader,
    bool validate,
    superdex::Error& error);

/**
 * @brief Apply a @ref ReplaceLinkWithBot modification to a bot.
 *
 * @param[in,out] botPrefab  Bot parameters to modify.
 * @param[in] mod The @ref ReplaceLinkWithBot modification to apply.
 * @param[in] loader Bot loader.
 * @param[in] validate Validate the attached bot when it is loaded.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void ApplyMod(
    BotPrefab& botPrefab,
    ReplaceLinkWithBot const& mod,
    IBotLoader const& loader,
    bool validate,
    superdex::Error& error);

/**
 * @brief Build a @ref ModBotPrefab recipe into a flat @ref BotPrefab by resolving base and
 * child paths. Recursively calls @ref IBotLoader loading methods for each path reference,
 * supporting nested bots.
 *
 * @param buildParams The ModBotPrefab recipe to build the bot form.
 * @param loader The IBotLoader implementation used to load nested bots.
 * @param validate If true, validation checks will be performed after each modification is applied.
 * If a modification loads another bot, it will be validated as well. Note that is entirely possible
 * that BuildBot could result in a valid bot even if some intermediates are invalid. However, it is
 * best practice to validate the entire chain of modifications.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
[[nodiscard]] MOCHI_API BotPrefab BuildBot(
    ModBotPrefab const& buildParams,
    IBotLoader const& loader,
    bool validate,
    superdex::Error& error);

/// Structured output from @ref Validate collecting per-element issues.
struct ValidateResults {
  /// If true, suppress MOCHI_LOG_WARNING calls during validation.
  bool suppressWarnings = false;

  /// Structural issues not tied to a specific link or joint.
  DynamicArray<DynamicString> botIssues;

  /// Per-link issues, indexed by link index (sized to numLinks by @ref Validate).
  DynamicArray<DynamicArray<DynamicString>> linkIssues;

  /// Per-joint issues, indexed by joint index (sized to numJoints by @ref Validate).
  DynamicArray<DynamicArray<DynamicString>> jointIssues;
};

/**
 * @brief Check a bot for errors (e.g. mismatch link/joint array sizes, missing names, etc.)
 *
 * @param[in] botPrefab The bot to check.
 * @param[out] results Optional structured output collecting validation issues. Can be nullptr.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void
Validate(BotPrefab const& botPrefab, ValidateResults* results, superdex::Error& error);

/**
 * @brief Rebuild computed internal link data from the raw link and joint definitions.
 *
 * @details First sorts the incoming data via @ref SortBotPrefab to ensure canonical ordering
 * (parents before children, siblings in alphanumeric order), then recomputes all internal data
 *
 * @param[in,out] botPrefab The bot parameters to rebuild (modified in place).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void RebuildBotData(BotPrefab& botPrefab, superdex::Error& error);

/**
 * @brief Sort bot link/joint/defaultPose data so that children follow their parents and siblings
 * are ordered by alphanumeric (natural) comparison of link names.
 *
 * @details Performs a DFS pre-order traversal of the link tree, visiting siblings in alphanumeric
 * order. The resulting permutation is applied to @ref BotPrefab::links, @ref BotPrefab::joints,
 * and @ref BotPrefab::defaultPose. Does NOT recompute internal fields; the caller should follow
 * with a call to @ref RebuildBotData.
 *
 * @note Called automatically by @ref RebuildBotData.
 *
 * @param[in,out] botPrefab Bot parameters to sort (modified in place).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void SortBotPrefab(BotPrefab& botPrefab, superdex::Error& error);

/**
 * @brief Build @ref ArticulatedActorParams from a @ref BotPrefab for constructing an articulated
 * actor. Shapes are loaded via the provided @ref IBotLoader, allowing custom asset resolution
 * (e.g., loading from Unreal Engine assets instead of the filesystem).
 *
 * @param[in] botPrefab Bot parameters describing the articulation.
 * @param[in] loader Bot loader used to load shape assets.
 * @param[in] context Mochi context (used for loading link shapes).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Constructed @ref ArticulatedActorParams ready for actor creation.
 */
[[nodiscard]] MOCHI_API ArticulatedActorParams BuildArticulatedActorParams(
    BotPrefab const& botPrefab,
    IBotLoader const& loader,
    Context* context,
    superdex::Error& error);

/**
 * @brief Create an articulated actor from @ref BotPrefab and add it to a scene. Shapes are loaded
 * via the provided @ref IBotLoader, allowing custom asset resolution (e.g., loading from Unreal
 * Engine assets instead of the filesystem).
 *
 * @param[in] botPrefab Bot parameters describing the articulation.
 * @param[in] scene Target scene.
 * @param[in] loader Bot loader used to load shape assets.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Pointer to the created @ref Actor, or nullptr on failure.
 */
[[nodiscard]] MOCHI_API Actor* AddToScene(
    BotPrefab const& botPrefab,
    Scene* scene,
    IBotLoader const& loader,
    superdex::Error& error);

/**
 * @brief Create an articulated actor from @ref BotPrefab and add it to a scene. Convenience
 * overload that uses @ref FileBotLoader for filesystem-based shape loading.
 *
 * @param[in] botPrefab  Bot parameters describing the articulation.
 * @param[in] scene Target scene.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Pointer to the created @ref Actor, or nullptr on failure.
 */
[[nodiscard]] MOCHI_API Actor*
AddToScene(BotPrefab const& botPrefab, Scene* scene, superdex::Error& error);

/**
 * @brief Find the index of a link by name.
 *
 * @param[in] bot Bot parameters to search.
 * @param[in] name Name of the link to find.
 * @return Index of the link, or @ref kIndexNone if not found.
 */
int MOCHI_API FindLinkIndexByName(BotPrefab const& bot, std::string_view name);

/**
 * @brief Find the indices of all leaf links (links that are not referenced as a parent by any
 * other link).
 *
 * @details A link is a leaf iff no other link lists it as its @ref BotLinkPrefab::parentLink.
 * Computed directly from @ref BotLinkPrefab::parentLink, so it is valid on a freshly built prefab
 * without requiring internal child-index data to be populated.
 *
 * @param[in] botPrefab Bot parameters to search.
 * @return Leaf link indices in ascending order.
 */
[[nodiscard]] MOCHI_API DynamicArray<int> FindLeafLinkIndices(BotPrefab const& botPrefab);

/**
 * @brief Compute link transforms for a given pose (joint configuration).
 *
 * @param[in] botPrefab Bot parameters describing the articulation structure.
 * @param[in] pose Array of joint angles (revolute) or linear positions (prismatic), in bot space:
 * one entry per bot-space DOF, excluding any root-joint DOFs.
 * @param[out] outLinkTransforms Output array for computed transforms (will be resized).
 * @param[in] transformSpace Whether to output link-to-parent or link-to-root transforms.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @param[in] finalLinkIndex If >= 0, only compute up to this link index; -1 computes all.
 */
void MOCHI_API ComputeLinkTransformsFromPose(
    BotPrefab const& botPrefab,
    Span<real const> pose,
    DynamicArray<TransformRT>& outLinkTransforms,
    LinkTransformSpace transformSpace,
    superdex::Error& error,
    int finalLinkIndex = -1);

/**
 * @brief Compute and bake mass properties (centerOfMass, momentOfInertia) for links that specify
 * mass or density but are missing explicit inertial data.
 *
 * @details For each link with a shape and mass/density but without centerOfMass or momentOfInertia,
 * creates a temporary rigid actor to compute inertial properties from the mesh geometry. For
 * density-only links, also computes mass = density * volume.
 *
 * @param[in,out] botPrefab Bot parameters to modify (mass properties written in place).
 * @param[in] context Mochi context (used for shape loading and actor creation).
 * @param[in] loader Bot loader used to load shape assets.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void BakeMassProperties(
    BotPrefab& botPrefab,
    Context* context,
    IBotLoader const& loader,
    superdex::Error& error);

/**
 * @brief Compute and bake mass properties for the links introduced by a mod bot recipe.
 *
 * @details Applies the same per-link baking as the @ref BotPrefab overload to the embedded link of
 * every @ref AttachLink and @ref ReplaceLink modification. Other modification types are left
 * untouched.
 *
 * @param[in,out] modBotPrefab Mod bot recipe to modify (mass properties written in place).
 * @param[in] context Mochi context (used for shape loading and actor creation).
 * @param[in] loader Bot loader used to load shape assets.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void BakeMassProperties(
    ModBotPrefab& modBotPrefab,
    Context* context,
    IBotLoader const& loader,
    superdex::Error& error);

/**
 * @brief Check whether contact between two links is implicitly disabled by Mochi's articulated
 * actor creation, mirroring the rule in @ref Scene::CreateArticulatedActor.
 *
 * @details Contact is implicitly disabled between two links iff both have a shape file
 * (@ref BotLinkPrefab::shapeFile) AND they are connected by a path through the link tree where
 * every intermediate node is either shape-less, or the edge entering it is a @ref
 * ArticulatedJointType::Hard joint. Direct parent-child pairs are a special case.
 *
 * Returns false if either link is out of range or shape-less.
 *
 * @param[in] botPrefab Bot describing the articulation.
 * @param[in] linkA First link index.
 * @param[in] linkB Second link index.
 * @return True if contact is implicitly disabled by Mochi's default rule.
 */
[[nodiscard]] MOCHI_API bool
IsContactImplicitlyDisabled(BotPrefab const& botPrefab, int linkA, int linkB);

/**
 * @brief Convert a bot-space pose to an articulated-actor pose.
 *
 * @details Bot DOFs never include the root joint's, whether or not the articulation has any: a
 * Free root contributes 6 actor DOFs that bot space excludes, and a Hard root contributes none.
 * The two spaces therefore share an order and differ only by that leading offset, so this copies
 * @p srcPose to @c pose[i + numBaseDofs] and zero-fills the base DOFs, producing the actor-DOF
 * layout @ref Actor::SetArticulatedPoseFromJoints expects. It never permutes.
 *
 * @param[in] botPrefab Bot describing the articulation.
 * @param[in] srcPose Bot-space pose (must have one entry per bot-space DOF).
 * @param[in] numActorDofs The actor's DOF count (@ref Actor::GetNumDofs).
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Pose in actor-DOF layout, sized to @p numActorDofs.
 */
[[nodiscard]] MOCHI_API DynamicArray<real> BuildArticulatedPoseFromBotPose(
    BotPrefab const& botPrefab,
    Span<real const> srcPose,
    int numActorDofs,
    superdex::Error& error);

/// The kind of pose @ref MakeBotPose should produce.
enum class MakeBotPoseType {
  Zero, ///< All DOFs set to 0, then clamped to the joint's limits if present.
  Min, ///< @c minLimit when present; otherwise the Random-fallback low (revolute/spherical 0,
       ///< prismatic -1).
  Max, ///< @c maxLimit when present; otherwise the Random-fallback high (revolute/spherical 2pi,
       ///< prismatic +1).
  Mid, ///< Midpoint `(lo+hi)/2` when both limits exist; otherwise 0 clamped to whichever side
       ///< exists.
  Random, ///< Uniform sample within `[lo, hi]`; fallback range when a limit is missing is
          ///< [0, 2pi] for Revolute/Spherical and [-1, +1] for Prismatic.
};

/**
 * @brief Produce a pose sized to the bot-space DOF count.
 *
 * @details For Revolute/Prismatic joints the scalar limit is computed via @c Dot(*limit, axis);
 * for Spherical joints limits are treated per-component. See @ref MakeBotPoseType for the
 * per-DOF rule each kind applies.
 *
 * @param[in] botPrefab The bot. @ref RebuildBotData should have been called first.
 * @param[in] type Which kind of pose to produce.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Pose vector with one entry per DOF, in the same order as @ref BotPrefab::defaultPose.
 */
[[nodiscard]] MOCHI_API DynamicArray<real>
MakeBotPose(BotPrefab const& botPrefab, MakeBotPoseType type, superdex::Error& error);

/**
 * @brief Extract a per-DOF effort-limit array from a bot: one entry per DOF in bot-space order
 * (bot-space is mochi's DOF space excluding the base link's free DOFs, if present; a Spherical
 * joint's @c effortLimit repeats across its 3 DOFs).
 *
 * @details Values pass through verbatim per @ref BotJointPrefab::effortLimit semantics: a negative
 * value (default @ref kEffortUnbounded) is unbounded, @c 0 is non-actuated, and a positive value is
 * a finite limit. Errors only if the prefab has not been rebuilt with @ref RebuildBotData. To
 * source limits from a bot file, call @ref LoadBotPrefabFromFile first and pass the resulting
 * prefab.
 *
 * @param[in] botPrefab A bot whose data has been rebuilt with @ref RebuildBotData.
 * @param[in,out] error Error status.
 * @return Per-DOF effort limits, or an empty array on error.
 */
[[nodiscard]] MOCHI_API DynamicArray<real> GetEffortLimitsPerDof(
    BotPrefab const& botPrefab,
    superdex::Error& error);
} // namespace superdex::robotics
