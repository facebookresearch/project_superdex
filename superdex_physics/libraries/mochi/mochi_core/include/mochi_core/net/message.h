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

#include <mochi_core/utils/reflection.h>

#include <cstdint>

#if !MOCHI_USE_REFLECTION
#error "This file requires MOCHI_USE_REFLECTION=1"
#endif

namespace mochi::net {

/**
 * @brief Base class for all messages used with @ref MessageServer and @ref MessageClient.
 *
 * @see ReplyMessage, MessageClient::SendAndAwaitReply
 */
struct Message : SReflect::BaseObject {
  MOCHI_STRUCT(mochi::net::Message);
};

/**
 * @brief Base class for request messages (used to request a @ref ReplyMessage).
 */
struct RequestMessage : Message {
  uint64_t requestId = 0;

  MOCHI_STRUCT_BEGIN(mochi::net::RequestMessage)
  MOCHI_BASE_CLASS(Message)
  MOCHI_FIELD(requestId)
  MOCHI_STRUCT_END()
};

/**
 * @brief Base class for reply messages (used to respond to a @ref RequestMessage).
 *
 * @see RequestMessage, MessageClient::SendAndAwaitReply
 */
struct ReplyMessage : Message {
  ReplyMessage() = default;
  explicit ReplyMessage(RequestMessage const& request) : requestId(request.requestId) {}

  uint64_t requestId = 0; ///< Copy of the @ref RequestMessage::requestId

  MOCHI_STRUCT_BEGIN(mochi::net::ReplyMessage)
  MOCHI_BASE_CLASS(Message)
  MOCHI_FIELD(requestId)
  MOCHI_STRUCT_END()
};

} // namespace mochi::net
