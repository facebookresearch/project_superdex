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

#include "reqrep_helper.h"

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/reflection.h>

#include <chrono>
#include <functional>
#include <utility>

using namespace mochi;
using namespace net;

MessageClient::ReqRepHelper::ReqRepHelper(MessageClient& client) : _client(client) {}

MessageClient::ReqRepHelper::~ReqRepHelper() {
  std::lock_guard lock(_mutex);
  MOCHI_ASSERT(_waiters.empty(), "ReqRepHelper destroyed while requests are still pending");
}

void MessageClient::ReqRepHelper::SendAndAwaitReply(
    RequestMessage& request,
    ReplyMessage& outReply,
    double timeoutSeconds,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  // If we are not connected, then don't even try to send.
  auto const status = _client.GetStatus();
  if (status != SocketStatus::Connected) {
    MOCHI_ERROR_SET(error, "No connection");
    return;
  }

  request.requestId = _nextId++;

  Key const key{outReply.GetFinalTypeId().value, request.requestId};
  Waiter waiter{outReply};
  {
    std::lock_guard lock(_mutex);
    [[maybe_unused]] auto const [_, inserted] = _waiters.emplace(key, &waiter);
    MOCHI_ASSERT_VERBOSE(inserted, "Duplicate request/reply waiter");
  }

  if (!_client.Send(request)) {
    MOCHI_ERROR_SET(error, "Send failed");

    std::lock_guard lock(_mutex);
    EraseWaiter(key, waiter);
    return;
  }

  auto const timeout = std::chrono::duration<double>(timeoutSeconds);

  std::unique_lock lock(_mutex);
  bool const completed =
      waiter.cv.wait_for(lock, timeout, [&] { return waiter.hasReply || waiter.canceled; });
  EraseWaiter(key, waiter);
  MOCHI_ERROR_IF(!completed, error, "Timeout");
  MOCHI_ERROR_IF(!waiter.hasReply, error, "Canceled");
}

void MessageClient::ReqRepHelper::DispatchReply(ReplyMessage&& msg) {
  Key const key{msg.GetFinalTypeId().value, msg.requestId};
  std::lock_guard lock(_mutex);
  auto const it = _waiters.find(key);
  if (it == _waiters.end()) {
    return;
  }
  it->second->Store(std::move(msg));
  it->second->cv.notify_one();
}

void MessageClient::ReqRepHelper::Cancel() {
  std::lock_guard lock(_mutex);
  for (auto& [_, waiter] : _waiters) {
    waiter->canceled = true;
    waiter->cv.notify_one();
  }
  _waiters.clear();
}

size_t MessageClient::ReqRepHelper::KeyHash::operator()(Key const& key) const {
  size_t const h1 = std::hash<uint64_t>{}(key.typeId);
  size_t const h2 = std::hash<uint64_t>{}(key.messageId);
  return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
}

MessageClient::ReqRepHelper::Waiter::Waiter(ReplyMessage& replyIn) : reply(replyIn) {}

void MessageClient::ReqRepHelper::Waiter::Store(ReplyMessage&& msg) {
  SReflect::TypeInfo const& typeInfo = reply.GetFinalTypeInfo();
  MOCHI_ASSERT_VERBOSE(msg.GetFinalTypeId().value == typeInfo._typeId.value, "Reply type mismatch");
  typeInfo.Set(&msg, &reply);
  hasReply = true;
}

void MessageClient::ReqRepHelper::EraseWaiter(Key const& key, Waiter const& waiter) {
  auto const it = _waiters.find(key);
  if ((it != _waiters.end()) && (it->second == &waiter)) {
    _waiters.erase(it);
  }
}
