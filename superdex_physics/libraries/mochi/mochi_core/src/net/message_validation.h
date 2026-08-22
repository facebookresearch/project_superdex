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
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/reflection.h>

namespace mochi::net {

inline void ValidateMessageType(
    [[maybe_unused]] SReflect::StructTypeInfo const& typeInfo,
    bool isRequest,
    bool isReply) {
  // Validation:
  if (isRequest) {
    MOCHI_ASSERT_VERBOSE(
        typeInfo.IsSameOrDerivedFrom(SReflect::GetTypeId<RequestMessage>()),
        "Please add MOCHI_BASE_CLASS(net::RequestMessage) to your message class declaration.");
  } else if (isReply) {
    MOCHI_ASSERT_VERBOSE(
        typeInfo.IsSameOrDerivedFrom(SReflect::GetTypeId<ReplyMessage>()),
        "Please add MOCHI_BASE_CLASS(net::ReplyMessage) to your message class declaration.");
  } else {
    MOCHI_ASSERT_VERBOSE(
        typeInfo.IsSameOrDerivedFrom(SReflect::GetTypeId<Message>()),
        "Please add MOCHI_BASE_CLASS(net::Message) to your message class declaration.");
  }
}

} // namespace mochi::net
