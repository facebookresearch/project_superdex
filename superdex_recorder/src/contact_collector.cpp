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

#include <superdex_recorder/contact_collector.h>

#include <mochi_core/utils/log.h>

#include <algorithm>
#include <utility>

namespace superdex::recorder {
namespace {

struct Contact {
  size_t bodyA;
  size_t bodyB;
  mochi::Real3 positionA;
  mochi::Real3 positionB;
  mochi::Real3 forceOnA;
};

void AppendXyz(std::vector<mochi::real>& output, mochi::Real3 const& value) {
  output.insert(output.end(), value.begin(), value.end());
}

} // namespace

void ContactColumns::Clear() {
  frameLengths.clear();
  bodyA.clear();
  bodyB.clear();
  positionA.clear();
  positionB.clear();
  forceOnA.clear();
}

ContactCollector::ContactCollector(
    mochi::Scene& scene,
    std::vector<ContactBody> bodies,
    mochi::Error& error)
    : _scene{scene}, _bodies{std::move(bodies)} {
  std::ranges::sort(
      _bodies, [](ContactBody const& lhs, ContactBody const& rhs) { return lhs.path < rhs.path; });
  _queries.resize(_bodies.size());
  for (size_t i = 0; i < _bodies.size(); ++i) {
    _indices.emplace(_bodies[i].handle, i);
    mochi::Actor* const actor = _scene.GetActor(_bodies[i].handle);
    MOCHI_ERROR_IF(actor == nullptr, error, "Invalid contact body.");
    MOCHI_ERROR_RETURN(error);
    if (actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
      _queries[i] = actor->RegisterQuery(mochi::QueryType::ContactPoints, error);
      MOCHI_ERROR_RETURN(error);
    }
  }
}

ContactCollector::~ContactCollector() {
  for (size_t i = 0; i < _bodies.size(); ++i) {
    mochi::Actor* const actor = _scene.GetActor(_bodies[i].handle);
    if (actor != nullptr && _queries[i].IsValid()) {
      actor->CancelQuery(_queries[i]);
    }
  }
}

std::vector<ContactBody> const& ContactCollector::GetBodies() const {
  return _bodies;
}

size_t ContactCollector::FindBody(mochi::ActorHandle handle, mochi::Error& error) const {
  auto const index = _indices.find(handle);
  if (index == _indices.end()) {
    mochi::Actor const* const actor = _scene.GetActor(handle);
    MOCHI_LOG_ERROR(
        "Contact references actor %s, which is not a recorded contact body.",
        actor == nullptr ? "(destroyed)" : actor->GetName());
    MOCHI_ERROR_SET(error, "Contact references an actor outside the recorded contact bodies.");
    return 0;
  }
  return index->second;
}

void ContactCollector::Collect(ContactColumns& output, mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error);
  std::vector<Contact> contacts;
  for (size_t bodyIndex = 0; bodyIndex < _bodies.size(); ++bodyIndex) {
    if (!_queries[bodyIndex].IsValid()) {
      continue;
    }
    mochi::Actor* const actor = _scene.GetActor(_bodies[bodyIndex].handle);
    MOCHI_ERROR_IF(actor == nullptr, error, "A recorded actor was destroyed.");
    MOCHI_ERROR_RETURN(error);
    for (auto const& contact : actor->GetContactPointsWorld(error)) {
      MOCHI_ERROR_RETURN(error);
      size_t const bodyAIndex = FindBody(contact.actorA, error);
      size_t const bodyBIndex = FindBody(contact.actorB, error);
      MOCHI_ERROR_RETURN(error);
      // Both bodies report a contact between them, so keep it on the lower queried index only.
      size_t ownerIndex = bodyAIndex;
      if (!_queries[ownerIndex].IsValid() ||
          (_queries[bodyBIndex].IsValid() && bodyBIndex < ownerIndex)) {
        ownerIndex = bodyBIndex;
      }
      if (bodyIndex != ownerIndex) {
        continue;
      }

      bool const preservesOrientation = bodyAIndex <= bodyBIndex;
      contacts.push_back(
          {preservesOrientation ? bodyAIndex : bodyBIndex,
           preservesOrientation ? bodyBIndex : bodyAIndex,
           preservesOrientation ? contact.posA : contact.posB,
           preservesOrientation ? contact.posB : contact.posA,
           preservesOrientation ? contact.force : -contact.force});
    }
  }
  std::ranges::stable_sort(contacts, [](Contact const& lhs, Contact const& rhs) {
    return std::pair{lhs.bodyA, lhs.bodyB} < std::pair{rhs.bodyA, rhs.bodyB};
  });

  output.frameLengths.push_back(static_cast<uint32_t>(contacts.size()));
  for (auto const& contact : contacts) {
    output.bodyA.push_back(static_cast<mochi::real>(contact.bodyA));
    output.bodyB.push_back(static_cast<mochi::real>(contact.bodyB));
    AppendXyz(output.positionA, contact.positionA);
    AppendXyz(output.positionB, contact.positionB);
    AppendXyz(output.forceOnA, contact.forceOnA);
  }
}

} // namespace superdex::recorder
