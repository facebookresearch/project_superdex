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

/** Full protobuf type names, for sample encodings such as "application/protobuf;<name>". */
inline constexpr const char* DoubleArrayTypeName = "superdex.teleop.v1.DoubleArray";
inline constexpr const char* PoseArrayTypeName = "superdex.teleop.v1.PoseArray";

namespace detail {

template <typename T>
struct Span {
  const T* Data;
  size_t Count;
};

inline bool EncodeDoubles(pb_ostream_t* Stream, const pb_field_t* Field, void* const* Arg) {
  const Span<double>& Values = *static_cast<const Span<double>*>(*Arg);
  if (Values.Count == 0) {
    return true;
  }
  // Packed, as proto3 encoders write repeated scalars.
  if (!pb_encode_tag(Stream, PB_WT_STRING, Field->tag) ||
      !pb_encode_varint(Stream, Values.Count * sizeof(double))) {
    return false;
  }
  for (size_t i = 0; i < Values.Count; ++i) {
    if (!pb_encode_fixed64(Stream, &Values.Data[i])) {
      return false;
    }
  }
  return true;
}

inline bool EncodePoses(pb_ostream_t* Stream, const pb_field_t* Field, void* const* Arg) {
  const Span<Pose>& Poses = *static_cast<const Span<Pose>*>(*Arg);
  for (size_t i = 0; i < Poses.Count; ++i) {
    if (!pb_encode_tag_for_field(Stream, Field) ||
        !pb_encode_submessage(Stream, superdex_teleop_v1_Pose_fields, &Poses.Data[i])) {
      return false;
    }
  }
  return true;
}

/** nanopb calls this once per value, for packed and unpacked encodings alike. */
template <typename TSink>
bool DecodeDouble(pb_istream_t* Stream, const pb_field_t* /*field*/, void** Arg) {
  double Value = 0;
  if (!pb_decode_fixed64(Stream, &Value)) {
    return false;
  }
  (*static_cast<TSink*>(*Arg))(Value);
  return true;
}

template <typename TSink>
bool DecodePose(pb_istream_t* Stream, const pb_field_t* /*field*/, void** Arg) {
  Pose DecodedPose = superdex_teleop_v1_Pose_init_zero;
  if (!pb_decode(Stream, superdex_teleop_v1_Pose_fields, &DecodedPose)) {
    return false;
  }
  (*static_cast<TSink*>(*Arg))(DecodedPose);
  return true;
}

} // namespace detail

/**
 * Writes one DoubleArray to Stream. Pass a PB_OSTREAM_SIZING stream to get the encoded size.
 *
 * @param Stream The output stream to write, or a sizing stream.
 * @param HeaderValue The header copied into the message.
 * @param Values The values to encode.
 * @param Count The number of values available at Values.
 * @return True if the message was encoded successfully.
 */
inline bool EncodeDoubleArray(
    pb_ostream_t* Stream,
    const Header& HeaderValue,
    const double* Values,
    size_t Count) {
  detail::Span<double> Span{Values, Count};
  superdex_teleop_v1_DoubleArray Msg = superdex_teleop_v1_DoubleArray_init_zero;
  Msg.has_header = true;
  Msg.header = HeaderValue;
  Msg.data.funcs.encode = &detail::EncodeDoubles;
  Msg.data.arg = &Span;
  return pb_encode(Stream, superdex_teleop_v1_DoubleArray_fields, &Msg);
}

/**
 * Writes one PoseArray to the output or sizing stream.
 *
 * @param Stream The output stream to write, or a sizing stream.
 * @param HeaderValue The header copied into the message.
 * @param Poses The poses to encode without coordinate conversion.
 * @param Count The number of poses available at Poses.
 * @return True if the message was encoded successfully.
 */
inline bool
EncodePoseArray(pb_ostream_t* Stream, const Header& HeaderValue, const Pose* Poses, size_t Count) {
  detail::Span<Pose> Span{Poses, Count};
  superdex_teleop_v1_PoseArray Msg = superdex_teleop_v1_PoseArray_init_zero;
  Msg.has_header = true;
  Msg.header = HeaderValue;
  Msg.poses.funcs.encode = &detail::EncodePoses;
  Msg.poses.arg = &Span;
  return pb_encode(Stream, superdex_teleop_v1_PoseArray_fields, &Msg);
}

/**
 * Calls OnValue(double) for each value in order. HeaderValue may be null.
 *
 * @param Bytes The encoded message bytes.
 * @param NumBytes The number of bytes available at Bytes.
 * @param HeaderValue Receives the decoded header on success; this decoder does not assign it on
 * failure.
 * @param OnValue The callback invoked as values are decoded, including before a later failure.
 * @return True if the whole message decoded successfully.
 */
template <typename TSink>
bool DecodeDoubleArray(
    const uint8_t* Bytes,
    size_t NumBytes,
    Header* HeaderValue,
    TSink&& OnValue) {
  using TSinkValue = typename std::remove_reference<TSink>::type;
  superdex_teleop_v1_DoubleArray Msg = superdex_teleop_v1_DoubleArray_init_zero;
  Msg.data.funcs.decode = &detail::DecodeDouble<TSinkValue>;
  Msg.data.arg = &OnValue;
  pb_istream_t Stream = pb_istream_from_buffer(Bytes, NumBytes);
  if (!pb_decode(&Stream, superdex_teleop_v1_DoubleArray_fields, &Msg)) {
    return false;
  }
  if (HeaderValue != nullptr) {
    *HeaderValue = Msg.header;
  }
  return true;
}

/**
 * Calls OnPose(const Pose&) for each pose in order. HeaderValue may be null.
 *
 * @param Bytes The encoded message bytes.
 * @param NumBytes The number of bytes available at Bytes.
 * @param HeaderValue Receives the decoded header on success; this decoder does not assign it on
 * failure.
 * @param OnPose The callback invoked as poses are decoded, including before a later failure.
 * @return True if the whole message decoded successfully.
 */
template <typename TSink>
bool DecodePoseArray(const uint8_t* Bytes, size_t NumBytes, Header* HeaderValue, TSink&& OnPose) {
  using TSinkValue = typename std::remove_reference<TSink>::type;
  superdex_teleop_v1_PoseArray Msg = superdex_teleop_v1_PoseArray_init_zero;
  Msg.poses.funcs.decode = &detail::DecodePose<TSinkValue>;
  Msg.poses.arg = &OnPose;
  pb_istream_t Stream = pb_istream_from_buffer(Bytes, NumBytes);
  if (!pb_decode(&Stream, superdex_teleop_v1_PoseArray_fields, &Msg)) {
    return false;
  }
  if (HeaderValue != nullptr) {
    *HeaderValue = Msg.header;
  }
  return true;
}

} // namespace teleop_msgs
