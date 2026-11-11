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
 * @brief Apply an @ref AttachSkin modification to a bot, setting its deformable skin.
 *
 * The skin's baked bone indices are treated as identity-by-name against the bot's current links
 * (recorded in @ref BotPrefab::_skinBoneLinks), so they survive later re-sorts and nested
 * composition. Errors if the bot already has a skin (a bot supports at most one).
 *
 * @param[in,out] botPrefab Bot parameters to modify.
 * @param[in] mod The @ref AttachSkin modification to apply.
 * @param[in] loader Bot loader (unused).
 * @param[in] validate Unused.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
MOCHI_API void ApplyMod(
    BotPrefab& botPrefab,
    AttachSkin const& mod,
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
 * @brief Resolve the name used to reference a bot's skin in contact overrides.
 *
 * The skin has no independent actor in Mochi: it is folded into the bot's top-level articulated
 * actor and shares that actor's handle. It is therefore referenced by the top-level actor's name,
 * which is @ref BotPrefab::name.
 *
 * @param[in] botPrefab Bot to query.
 * @return @ref BotPrefab::name when the bot has a skin, or an empty view when it has no skin.
 */
[[nodiscard]] MOCHI_API std::string_view GetBotSkinName(BotPrefab const& botPrefab);

/**
 * @brief Map each bone of a bot's skin to the index of the link it binds to.
 *
 * @details For a composed bot, @ref BotPrefab::_skinBoneLinks names the link each skin bone binds
 * to. When it is empty (a standalone skinned bot), skin bone b binds to link b.
 *
 * @param[in] botPrefab Bot whose skin to map.
 * @param[in] linkNames Link names, in the order the returned indices refer to.
 * @return One link index per skin bone, or @ref kIndexNone for a bone whose link is not in
 * @p linkNames. When @ref BotPrefab::_skinBoneLinks is non-empty, entry b corresponds to its entry
 * b, which names the unresolved link. Empty if the bot has no skin.
 */
[[nodiscard]] MOCHI_API DynamicArray<int> ComputeSkinBoneLinkIndices(
    BotPrefab const& botPrefab,
    Span<DynamicString const> linkNames);

/**
 * @brief Find the links whose collision a bot's skin stands in for. A collision view or exporter
 * should omit their rigid collision geometry.
 *
 * @details Mirrors @ref ArticulatedSkinParams::nonCollidingLinks. When set, the skin covers exactly
 * the links it names. When unset, no link collides and the skin is the bot's only collider, so it
 * covers every link its collision mesh puts a positive weight on, through
 * @ref ComputeSkinBoneLinkIndices. For the links whose render models the skin's render model stands
 * in for, see @ref ComputeSkinHiddenLinks.
 *
 * @param[in] botPrefab Bot whose skin to query.
 * @param[in] linkNames Link names, in the order of the returned flags.
 * @param[in] skinning Skinning of the skin's collision mesh (@ref
 * ArticulatedSkinPrefab::shapeFile), or nullptr if it is unavailable. Only read when
 * @ref ArticulatedSkinParams::nonCollidingLinks is unset.
 * @return One flag per entry of @p linkNames, true for a covered link. All false if the bot has no
 * skin or its skin has no collision shape.
 */
[[nodiscard]] MOCHI_API DynamicArray<bool> ComputeSkinCoveredLinks(
    BotPrefab const& botPrefab,
    Span<DynamicString const> linkNames,
    SkinningData const* skinning);

/**
 * @brief Find the links whose render models a bot's skin render model stands in for. A renderer or
 * exporter should hide them wherever it draws the skin's render model.
 *
 * @details Mirrors @ref ArticulatedSkinPrefab::hiddenLinks. When set, the skin hides exactly the
 * links it names. When unset, the skin hides every link it is skinned over, through
 * @ref ComputeSkinBoneLinkIndices: every link of a standalone skinned bot, or every link of the bot
 * the skin was attached with in a composed bot.
 *
 * @param[in] botPrefab Bot whose skin to query.
 * @param[in] linkNames Link names, in the order of the returned flags.
 * @return One flag per entry of @p linkNames, true for a hidden link. All false if the bot has no
 * skin or its skin has no render model.
 */
[[nodiscard]] MOCHI_API DynamicArray<bool> ComputeSkinHiddenLinks(
    BotPrefab const& botPrefab,
    Span<DynamicString const> linkNames);

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
 * @details Composes each link as Mochi's articulated body does: parent-from-link =
 * joint.parentLinkFromJoint * joint motion * link.parentJointFromLink, where the joint motion is a
 * rotation of the revolute angle about the joint axis, a translation of the prismatic position
 * along the unit joint axis, or the rotation of a spherical joint's rotation vector. Any finite
 * revolute angle works, including several turns. Transforms are in nominal (unscaled) bot space:
 * @ref BotPrefab::scale is not applied. The root link's transform is the identity.
 *
 * @param[in] botPrefab Bot parameters describing the articulation structure, prepared by
 * @ref RebuildBotData (not rechecked here, as this is meant to run every frame).
 * @param[in] pose Joint angles (revolute) [rad], linear positions (prismatic) [m] and rotation
 * vectors (spherical) [rad], in bot space: one entry per bot-space DOF, excluding any root-joint
 * DOFs.
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
 * @brief What a link's transform holds that its joint can't produce: the part
 * @ref ComputePoseFromLinkTransforms discards.
 */
struct LinkPoseResidual {
  /** Angle [rad] of the rotation left after the joint's DOFs: the swing off a revolute axis, the
   * whole rotation of a prismatic or Hard joint, 0 for a spherical joint. */
  real rotation = 0;
  /** Length, in nominal bot units, of the translation left after the joint's DOFs: the whole
   * translation of a rotational or Hard joint, the part off the axis of a prismatic joint. Only
   * meaningful when the link transforms carry translations. */
  real translation = 0;
};

/**
 * @brief Compute the pose (joint configuration) of given link transforms: the inverse of
 * @ref ComputeLinkTransformsFromPose.
 *
 * @details Each joint's DOFs are read from its link's transform relative to the link's parent as
 * given (in either space), measured in the joint frame, and limited to what the joint can do: a
 * revolute angle is the twist of the joint motion about the joint axis (the nearest rotation about
 * the axis), a prismatic position is the joint motion's translation along the unit joint axis, and
 * a spherical joint is the rotation vector of the joint motion (norm <= pi). Anything else (swing
 * off a revolute axis, translation of a rotational joint, rotation or off-axis translation of a
 * prismatic joint, any motion of a Hard joint) is discarded, and measured in @p outResiduals if
 * given, so transforms that come from @ref ComputeLinkTransformsFromPose round-trip, and an error
 * in one link's transform only affects its own joint. Translations only matter for prismatic
 * joints, so a bot without them can be posed from link rotations alone. In RootFromParent space
 * only relative transforms are used, so link-to-world transforms work too.
 *
 * A link's transform can't tell revolute angles 2 pi apart. Each is the one nearest to its entry in
 * @p referencePose if given, such as the previous frame's pose, so a joint tracked through several
 * turns stays continuous; else the one in [-pi, pi].
 *
 * @param[in] botPrefab Bot parameters, as for @ref ComputeLinkTransformsFromPose.
 * @param[in] linkTransforms One transform per link, with unit rotations, in nominal bot space, in
 * @p transformSpace.
 * @param[out] outPose Output pose (will be resized): one entry per bot-space DOF, excluding any
 * root-joint DOFs, as @ref ComputeLinkTransformsFromPose takes it.
 * @param[in] transformSpace Whether @p linkTransforms are link-to-parent or link-to-root.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @param[in] referencePose Empty, or a pose (one entry per bot-space DOF) whose revolute angles
 * choose the turn of the output's.
 * @param[out] outResiduals Null, or resized to one entry per link: what each link's transform
 * holds that its joint can't produce (zero for the root link). Transforms that fit the bot only
 * leave rounding error; larger residuals mean they don't fit it.
 */
void MOCHI_API ComputePoseFromLinkTransforms(
    BotPrefab const& botPrefab,
    Span<TransformRT const> linkTransforms,
    DynamicArray<real>& outPose,
    LinkTransformSpace transformSpace,
    superdex::Error& error,
    Span<real const> referencePose = {},
    DynamicArray<LinkPoseResidual>* outResiduals = nullptr);

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
 * @details Values follow @ref BotJointPrefab::effortLimit semantics: a negative value (default
 * @ref kEffortUnbounded) is unbounded, @c 0 is non-actuated, and a positive value is a finite
 * limit. Finite limits are those of the instantiated bot: they are scaled by
 * @ref details::JointEffortScale for @ref BotPrefab::scale. Errors only if the prefab has not been
 * rebuilt with @ref RebuildBotData. To
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

namespace details {

/* @brief Factor by which @ref BotPrefab::scale scales a joint's stiffness, damping and inertia for
 * dynamic similarity (lengths scaled at constant density, time scale unchanged): scale^3 for a
 * prismatic joint, scale^5 for a rotational one. */
[[nodiscard]] inline real JointGainScale(ArticulatedJointType type, real scale) {
  real const s3 = scale * scale * scale;
  return type == ArticulatedJointType::Prismatic ? s3 : s3 * scale * scale;
}

/* @brief Factor by which @ref BotPrefab::scale scales a joint's forces (prismatic, scale^4) or
 * torques (rotational, scale^5) for dynamic similarity. */
[[nodiscard]] inline real JointEffortScale(ArticulatedJointType type, real scale) {
  real const s4 = scale * scale * scale * scale;
  return type == ArticulatedJointType::Prismatic ? s4 : s4 * scale;
}

} // namespace details
} // namespace superdex::robotics
