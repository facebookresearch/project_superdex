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

#include "editors/bot_editor_contact.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/defer.h>

#include <string_view>

namespace superdex::studio {

BotContactFilterBuilder::BotContactFilterBuilder(superdex::robotics::BotPrefab& prefab)
    : numLinks(mochi::isize(prefab.links)),
      implicitDisabledMask(numLinks * numLinks, false),
      _prefab(prefab) {
  // Mirror the rule the articulated-actor pipeline applies on creation.
  for (int a = 0; a < numLinks; ++a) {
    for (int b = a + 1; b < numLinks; ++b) {
      bool const off = superdex::robotics::IsContactImplicitlyDisabled(prefab, a, b);
      implicitDisabledMask[a * numLinks + b] = off;
      implicitDisabledMask[b * numLinks + a] = off;
    }
  }
}

bool BotContactFilterBuilder::IsImplicitlyDisabled(int linkA, int linkB) const {
  return implicitDisabledMask[linkA * numLinks + linkB];
}

bool BotContactFilterBuilder::DefaultEnabled(int linkA, int linkB) const {
  return !IsImplicitlyDisabled(linkA, linkB);
}

int BotContactFilterBuilder::FindFilterIndex(int linkA, int linkB) const {
  std::string_view const na(_prefab.links[linkA].name);
  std::string_view const nb(_prefab.links[linkB].name);
  auto const& filters = _prefab.contactOverrides;
  for (int i = 0; i < mochi::isize(filters); ++i) {
    std::string_view const fa(filters[i].linkA);
    std::string_view const fb(filters[i].linkB);
    if ((fa == na && fb == nb) || (fa == nb && fb == na)) {
      return i;
    }
  }
  return -1;
}

bool BotContactFilterBuilder::HasOverride(int linkA, int linkB) const {
  return FindFilterIndex(linkA, linkB) != -1;
}

bool BotContactFilterBuilder::IsEnabled(int linkA, int linkB) const {
  int const idx = FindFilterIndex(linkA, linkB);
  return idx != -1 ? _prefab.contactOverrides[idx].enable : DefaultEnabled(linkA, linkB);
}

void BotContactFilterBuilder::SetFilter(int linkA, int linkB, bool enable) {
  auto& filters = _prefab.contactOverrides;
  int const idx = FindFilterIndex(linkA, linkB);
  if (enable == DefaultEnabled(linkA, linkB)) {
    // Matches the implicit default: drop any redundant override.
    if (idx != -1) {
      filters.erase(filters.begin() + idx);
    }
    return;
  }
  if (idx != -1) {
    filters[idx].enable = enable;
    return;
  }
  superdex::robotics::BotContactOverride nf;
  nf.linkA = _prefab.links[linkA].name;
  nf.linkB = _prefab.links[linkB].name;
  nf.enable = enable;
  filters.push_back(std::move(nf));
}

void BotContactFilterBuilder::SetAll(bool enable) {
  auto& filters = _prefab.contactOverrides;
  filters.clear();
  for (int a = 0; a < numLinks; ++a) {
    for (int b = a + 1; b < numLinks; ++b) {
      if (enable != DefaultEnabled(a, b)) {
        superdex::robotics::BotContactOverride nf;
        nf.linkA = _prefab.links[a].name;
        nf.linkB = _prefab.links[b].name;
        nf.enable = enable;
        filters.push_back(std::move(nf));
      }
    }
  }
}

BotContactProbe::BotContactProbe(
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::Context* context,
    superdex::robotics::IBotLoader const& loader,
    mochi::Error& error)
    : context(context), prefab(botPrefab) {
  using namespace mochi;
  MOCHI_ERROR_RETURN(error);
  if (context == nullptr) {
    MOCHI_ERROR_SET(error, "BotContactProbe: null context");
    return;
  }
  scene = context->CreateScene("BotContactProbe");
  if (scene == nullptr) {
    MOCHI_ERROR_SET(error, "BotContactProbe: failed to create scene");
    return;
  }

  SolverParams sp = scene->GetSolverParams();
  sp.nonLinearSolver.maxIter = 1;
  // Random poses can produce deep penetrations that make the Newton solver report
  // "Solution explosion detected". Contact points are still populated, so silence it.
  sp.nonLinearSolver.verbosity = VerbosityLevel::Silent;
  sp.linearSolver.verbosity = VerbosityLevel::Silent;
  scene->SetSolverParams(sp, error);
  MOCHI_ERROR_RETURN(error);
  // No need to fall under gravity for a single-step contact query.
  scene->SetGravity({0_r, 0_r, 0_r});

  // Build the bot once at its default pose (its contactOverrides are applied here);
  // the queries below are pose-independent and the actor is re-posed per query.
  actor = superdex::robotics::AddToScene(prefab, scene, loader, error);
  MOCHI_ERROR_RETURN(error);
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "BotContactProbe: failed to add bot to scene");
    return;
  }

  Span<ActorHandle const> const nestedLinkActors = actor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);
  int const numLinks = isize(nestedLinkActors);
  linkActors.resize(numLinks);
  queries.resize(numLinks);
  handleToIndex.reserve(numLinks);
  // Register a ContactPoints query on every link actor that supports it (shape-less
  // links and other unsupported actor types are skipped).
  for (int i = 0; i < numLinks; ++i) {
    linkActors[i] = nestedLinkActors[i];
    handleToIndex.emplace(nestedLinkActors[i].value, i);
    Actor* linkActor = scene->GetActor(nestedLinkActors[i]);
    if (linkActor == nullptr || !linkActor->IsQuerySupported(QueryType::ContactPoints)) {
      continue;
    }
    queries[i] = linkActor->RegisterQuery(QueryType::ContactPoints, error);
    MOCHI_ERROR_RETURN(error);
  }
}

BotContactProbe::~BotContactProbe() {
  // Destroying the scene also destroys the actor and cancels its queries.
  if (context != nullptr && scene != nullptr) {
    context->DestroyScene(scene);
  }
}

bool BotContactProbe::IsValid() const {
  return actor != nullptr;
}

mochi::DynamicArray<BotLinkPair> BotContactProbe::DetectCollidingLinks(
    mochi::Span<mochi::real const> pose,
    mochi::Error& error) {
  using namespace mochi;
  MOCHI_ERROR_RETURN(error, {});
  if (!IsValid()) {
    MOCHI_ERROR_SET(error, "BotContactProbe::DetectCollidingLinks: invalid probe");
    return {};
  }

  // Cheap part: re-pose the existing actor and step. No actor/shape rebuild.
  DynamicArray<real> actorPose =
      superdex::robotics::BuildArticulatedPoseFromBotPose(prefab, pose, actor->GetNumDofs(), error);
  MOCHI_ERROR_RETURN(error, {});
  actor->SetArticulatedPoseFromJoints(actorPose, error);
  MOCHI_ERROR_RETURN(error, {});

  // Step with a tiny positive dt so contacts are actually computed. Mochi's
  // SceneImpl::Step skips the contact solve entirely when dt <= 0 (it only refreshes
  // queries), so a literal 0 produces no contact points.
  scene->Step(1e-9);

  // Collect deduplicated colliding (linkA, linkB) pairs with linkA < linkB.
  int const numLinks = isize(linkActors);
  std::set<std::pair<int, int>> seen;
  for (int i = 0; i < numLinks; ++i) {
    if (!queries[i].IsValid()) {
      continue;
    }
    Actor const* linkActor = scene->GetActor(linkActors[i]);
    if (linkActor == nullptr) {
      continue;
    }
    Span<ContactPoint const> const pts = linkActor->GetContactPointsWorld(error);
    MOCHI_ERROR_RETURN(error, {});
    for (ContactPoint const& cp : pts) {
      auto itA = handleToIndex.find(cp.actorA.value);
      auto itB = handleToIndex.find(cp.actorB.value);
      if (itA == handleToIndex.end() || itB == handleToIndex.end()) {
        continue; // contact involves an actor outside our bot
      }
      int a = itA->second;
      int b = itB->second;
      if (a == b) {
        continue; // skip self-contact
      }
      if (a > b) {
        std::swap(a, b);
      }
      seen.emplace(a, b);
    }
  }
  DynamicArray<BotLinkPair> result;
  result.reserve(seen.size());
  for (auto const& p : seen) {
    result.push_back(BotLinkPair{p.first, p.second});
  }
  return result;
}

void BotContactEstimator::Start(
    mochi::Context* context,
    superdex::robotics::IBotLoader const& loader,
    superdex::robotics::BotPrefab const& prefab,
    int iterations) {
  // Join any previous worker first (defensive).
  Stop();

  // Seed the candidate set: all link pairs with shape on both sides that aren't
  // already implicitly disabled by Mochi's articulated-actor rule.
  std::set<std::pair<int, int>> initialCandidates;
  int const N = mochi::isize(prefab.links);
  for (int a = 0; a < N; ++a) {
    if (prefab.links[a].shapeFile.empty()) {
      continue;
    }
    for (int b = a + 1; b < N; ++b) {
      if (prefab.links[b].shapeFile.empty()) {
        continue;
      }
      if (superdex::robotics::IsContactImplicitlyDisabled(prefab, a, b)) {
        continue;
      }
      initialCandidates.emplace(a, b);
    }
  }

  // Reset shared state.
  _cancel.store(false);
  _finished.store(false);
  _completed.store(0);
  _total.store(iterations);
  {
    std::lock_guard<std::mutex> lock(_resultMutex);
    _candidates = std::move(initialCandidates);
    _lastPose.clear();
    _originalPose = prefab.defaultPose;
  }
  _running.store(true);

  // The worker copies the prefab so it can run independently of main-thread
  // edits. Context and loader live for the lifetime of the studio. The estimator
  // detects raw collisions, so drop any existing contact filters on the copy.
  superdex::robotics::BotPrefab prefabCopy = prefab;
  prefabCopy.contactOverrides.clear();

  _thread = std::thread(
      [this, prefabCopy = std::move(prefabCopy), iterations, context, &loader]() mutable {
        // This worker did not create the Context, so bind it for the thread's
        // lifetime before touching any scene/actor API and unbind on every exit
        // path (the probe's scene teardown still needs the binding, so unbind
        // runs after the probe is destroyed).
        if (context != nullptr) {
          context->BindThisThread();
        }
        MOCHI_DEFER(if (context != nullptr) { context->UnbindThisThread(); });

        // Build the collision probe once (scene + actor + per-link queries) and
        // reuse it for every sampled pose. This amortizes the expensive shape
        // loading and actor construction that otherwise dominate per-pose cost;
        // each sample then only re-poses the actor and steps.
        mochi::ErrorLog setupErr;
        BotContactProbe probe(prefabCopy, context, loader, setupErr);
        if (!probe.IsValid()) {
          _finished.store(true);
          return;
        }

        // Evaluates one sampled pose: re-poses the probe's actor, detects colliding
        // link pairs, prunes them from the candidate set, and reports whether the
        // candidate set is now empty. A fresh ErrorLog per sample ensures a failure
        // on one pose neither aborts the worker (ErrorLog, not ErrorAssert) nor
        // short-circuits later samples.
        auto evaluatePose = [&](superdex::robotics::MakeBotPoseType type) -> bool {
          mochi::ErrorLog err;
          mochi::DynamicArray<mochi::real> pose =
              superdex::robotics::MakeBotPose(prefabCopy, type, err);
          auto colliding = probe.DetectCollidingLinks(mochi::MakeConstSpan(pose), err);
          std::lock_guard<std::mutex> lock(_resultMutex);
          _lastPose = pose;
          for (auto const& p : colliding) {
            _candidates.erase({p.linkA, p.linkB});
          }
          return _candidates.empty();
        };

        // Run the deterministic Min/Max/Mid/Zero poses first so the user
        // gets guaranteed coverage of the joint-limit extremes and rest pose
        // regardless of how few random iterations they request.
        for (auto type :
             {superdex::robotics::MakeBotPoseType::Min,
              superdex::robotics::MakeBotPoseType::Max,
              superdex::robotics::MakeBotPoseType::Mid,
              superdex::robotics::MakeBotPoseType::Zero}) {
          if (_cancel.load()) {
            break;
          }
          if (evaluatePose(type)) {
            _finished.store(true);
            return;
          }
        }
        for (int i = 0; i < iterations; ++i) {
          if (_cancel.load()) {
            break;
          }
          bool const noCandidates = evaluatePose(superdex::robotics::MakeBotPoseType::Random);
          _completed.store(i + 1);
          if (noCandidates) {
            break;
          }
        }
        _finished.store(true);
      });
}

void BotContactEstimator::Stop() {
  _cancel.store(true);
  if (_thread.joinable()) {
    _thread.join();
  }
  _running.store(false);
  _finished.store(false);
}

mochi::DynamicArray<mochi::real> BotContactEstimator::GetLatestPose() const {
  std::lock_guard<std::mutex> lock(_resultMutex);
  return _lastPose;
}

BotContactEstimator::Result BotContactEstimator::TakeResult() {
  Result result;
  result.cancelled = _cancel.load();
  {
    std::lock_guard<std::mutex> lock(_resultMutex);
    result.survivors = std::move(_candidates);
    result.originalPose = std::move(_originalPose);
  }
  Stop();
  return result;
}

} // namespace superdex::studio
