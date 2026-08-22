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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/reflection.h>

#include <cstddef>
#include <cstdint>
#include <functional>

#if !MOCHI_USE_REFLECTION
#error "This file requires MOCHI_USE_REFLECTION=1"
#endif

namespace mochi::net {

// Forwards
struct Message;

/**
 * @brief Serialize a message into the wire format shared by @ref MessageClient and @ref
 * MessageServer.
 *
 * @param[in] msg Message to serialize. Its most-derived type is resolved via reflection.
 * @param[out] outBytes Destination buffer
 */
void SerializeMessage(Message const& msg, DynamicArray<uint8_t>& outBytes);

/**
 * @brief Deserialize a message from the wire format.
 *
 * @param[in] buffer Address of the buffer to read
 * @param[in] size Size of the buffer to read [bytes]
 * @param[in] tryGetTypeInfo Function to look up the SReflect::TypeInfo.
 * @param[in,out] error Check @ref Error::IsOK for success.
 * @return The address of a newly allocated @ref Message object.
 *
 * @note Must call @ref DeleteMessage when you are done with it.
 */
Message* DeserializeMessage(
    void const* buffer,
    size_t size,
    std::function<SReflect::TypeInfo const*(SReflect::TypeId)> const& tryGetTypeInfo,
    Error& error);

/**
 * @brief Delete a message that was returned via @ref DeserializeMessage.
 *
 * @param msg @ref Message address to delete
 */
void DeleteMessage(Message* msg);

} // namespace mochi::net
