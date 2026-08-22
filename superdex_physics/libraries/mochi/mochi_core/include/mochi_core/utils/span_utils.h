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

#include <mochi_core/utils/span.h>

namespace mochi {

/// @brief Cast Span object with scalar entries to a Span object with a different
/// scalar type.
///
/// @tparam ToScalar The target scalar type for the conversion (must be non-const).
/// @tparam FromScalar The source scalar type from the input Span object.
/// @tparam ToSz The size type of the destination Span.
/// @tparam FromSz The size type of the source Span.
///
/// @param src The source Span containing elements to be cast.
/// @param dst The destination Span where cast elements will be stored.
///
/// @note The function does a straight copy if the source and destination scalar types are the same.
template <typename ToScalar, typename FromScalar, typename ToSz, typename FromSz>
void StaticCast(Span<FromScalar const, FromSz> src, Span<ToScalar, ToSz> dst);

} // namespace mochi

#include "span_utils_inl.h"
