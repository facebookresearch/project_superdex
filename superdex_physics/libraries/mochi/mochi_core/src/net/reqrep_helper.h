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

#include <mochi_core/net/message.h>
#include <mochi_core/net/message_client.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/no_copy.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace mochi::net {

//--------------------------------------------------------------------------------------
// ReqRepHelper: private request/reply machinery for MessageClient::SendAndAwaitReply.
//--------------------------------------------------------------------------------------

struct MessageClient::ReqRepHelper : public NoCopy {
  explicit ReqRepHelper(MessageClient& client);

  ~ReqRepHelper();

  void SendAndAwaitReply(
      RequestMessage& request,
      ReplyMessage& outReply,
      double timeoutSeconds,
      Error& error);

  void DispatchReply(ReplyMessage&& msg);

  void Cancel();

 private:
  struct Key {
    uint64_t typeId = 0;
    uint64_t messageId = 0;
    bool operator==(Key const& other) const = default;
  };

  struct KeyHash {
    size_t operator()(Key const& key) const;
  };

  struct Waiter {
    explicit Waiter(ReplyMessage& replyIn);

    void Store(ReplyMessage&& msg);

    ReplyMessage& reply;
    std::condition_variable cv;
    bool hasReply = false;
    bool canceled = false;
  };

  void EraseWaiter(Key const& key, Waiter const& waiter);

  MessageClient& _client;
  std::mutex _mutex;
  std::unordered_map<Key, Waiter*, KeyHash> _waiters;
  std::atomic<uint64_t> _nextId = 1;
};

} // namespace mochi::net
