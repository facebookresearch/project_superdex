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

// Encode and decode teleop messages with nanopb, straight from and into the caller's containers:
// repeated fields are callbacks, so there is no size cap and no fixed-size message struct.
// Header-only, with no dependencies beyond nanopb and the C++ standard library.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "pb_decode.h"
#include "pb_encode.h"
#include "teleop.pb.h"

namespace teleop_msgs {

using Header = superdex_teleop_v1_Header;
using Pose = superdex_teleop_v1_Pose;

// Full protobuf type names, for sample encodings such as "application/protobuf;<name>".
inline constexpr const char* kDoubleArrayTypeName = "superdex.teleop.v1.DoubleArray";
inline constexpr const char* kPoseArrayTypeName = "superdex.teleop.v1.PoseArray";

namespace detail {

template <typename T>
struct Span {
  const T* data;
  size_t count;
};

inline bool EncodeDoubles(pb_ostream_t* stream, const pb_field_t* field, void* const* arg) {
  const auto& values = *static_cast<const Span<double>*>(*arg);
  if (values.count == 0) {
    return true;
  }
  // Packed, as proto3 encoders write repeated scalars.
  if (!pb_encode_tag(stream, PB_WT_STRING, field->tag) ||
      !pb_encode_varint(stream, values.count * sizeof(double))) {
    return false;
  }
  for (size_t i = 0; i < values.count; ++i) {
    if (!pb_encode_fixed64(stream, &values.data[i])) {
      return false;
    }
  }
  return true;
}

inline bool EncodePoses(pb_ostream_t* stream, const pb_field_t* field, void* const* arg) {
  const auto& poses = *static_cast<const Span<Pose>*>(*arg);
  for (size_t i = 0; i < poses.count; ++i) {
    if (!pb_encode_tag_for_field(stream, field) ||
        !pb_encode_submessage(stream, superdex_teleop_v1_Pose_fields, &poses.data[i])) {
      return false;
    }
  }
  return true;
}

// nanopb calls this once per value, for packed and unpacked encodings alike.
template <typename TSink>
bool DecodeDouble(pb_istream_t* stream, const pb_field_t* /*field*/, void** arg) {
  double value = 0;
  if (!pb_decode_fixed64(stream, &value)) {
    return false;
  }
  (*static_cast<TSink*>(*arg))(value);
  return true;
}

template <typename TSink>
bool DecodePose(pb_istream_t* stream, const pb_field_t* /*field*/, void** arg) {
  Pose pose = superdex_teleop_v1_Pose_init_zero;
  if (!pb_decode(stream, superdex_teleop_v1_Pose_fields, &pose)) {
    return false;
  }
  (*static_cast<TSink*>(*arg))(pose);
  return true;
}

} // namespace detail

// Writes one DoubleArray to stream. Pass a PB_OSTREAM_SIZING stream to get the encoded size.
inline bool
EncodeDoubleArray(pb_ostream_t* stream, const Header& header, const double* values, size_t count) {
  detail::Span<double> span{values, count};
  superdex_teleop_v1_DoubleArray msg = superdex_teleop_v1_DoubleArray_init_zero;
  msg.has_header = true;
  msg.header = header;
  msg.data.funcs.encode = &detail::EncodeDoubles;
  msg.data.arg = &span;
  return pb_encode(stream, superdex_teleop_v1_DoubleArray_fields, &msg);
}

inline bool
EncodePoseArray(pb_ostream_t* stream, const Header& header, const Pose* poses, size_t count) {
  detail::Span<Pose> span{poses, count};
  superdex_teleop_v1_PoseArray msg = superdex_teleop_v1_PoseArray_init_zero;
  msg.has_header = true;
  msg.header = header;
  msg.poses.funcs.encode = &detail::EncodePoses;
  msg.poses.arg = &span;
  return pb_encode(stream, superdex_teleop_v1_PoseArray_fields, &msg);
}

// Calls onValue(double) for each value in order. header may be null.
template <typename TSink>
bool DecodeDoubleArray(const uint8_t* bytes, size_t numBytes, Header* header, TSink&& onValue) {
  using TSinkValue = typename std::remove_reference<TSink>::type;
  superdex_teleop_v1_DoubleArray msg = superdex_teleop_v1_DoubleArray_init_zero;
  msg.data.funcs.decode = &detail::DecodeDouble<TSinkValue>;
  msg.data.arg = &onValue;
  pb_istream_t stream = pb_istream_from_buffer(bytes, numBytes);
  if (!pb_decode(&stream, superdex_teleop_v1_DoubleArray_fields, &msg)) {
    return false;
  }
  if (header != nullptr) {
    *header = msg.header;
  }
  return true;
}

// Calls onPose(const Pose&) for each pose in order. header may be null.
template <typename TSink>
bool DecodePoseArray(const uint8_t* bytes, size_t numBytes, Header* header, TSink&& onPose) {
  using TSinkValue = typename std::remove_reference<TSink>::type;
  superdex_teleop_v1_PoseArray msg = superdex_teleop_v1_PoseArray_init_zero;
  msg.poses.funcs.decode = &detail::DecodePose<TSinkValue>;
  msg.poses.arg = &onPose;
  pb_istream_t stream = pb_istream_from_buffer(bytes, numBytes);
  if (!pb_decode(&stream, superdex_teleop_v1_PoseArray_fields, &msg)) {
    return false;
  }
  if (header != nullptr) {
    *header = msg.header;
  }
  return true;
}

} // namespace teleop_msgs
