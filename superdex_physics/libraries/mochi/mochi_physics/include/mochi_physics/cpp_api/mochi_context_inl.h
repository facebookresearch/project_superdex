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

#include "mochi_context.h" // Reverse include for intellisense

namespace mochi {

inline ShapeHandle
Context::LoadShapeFromFile(std::string_view filePath, Real3 const& bakeScale, Error& error) {
  return LoadShapeFromFile(filePath, bakeScale, TransformRT::Identity(), error);
}

inline ShapeHandle Context::LoadShapeFromFile(
    std::string_view filePath,
    TransformRT const& bakeTransform,
    Error& error) {
  return LoadShapeFromFile(filePath, Real3{1_r, 1_r, 1_r}, bakeTransform, error);
}

inline ShapeHandle Context::LoadShapeFromBytes(
    Span<char const> fileData,
    MeshFileType format,
    Real3 const& bakeScale,
    Error& error) {
  return LoadShapeFromBytes(fileData, format, bakeScale, TransformRT::Identity(), error);
}

inline ShapeHandle Context::LoadShapeFromBytes(
    Span<char const> fileData,
    MeshFileType format,
    TransformRT const& bakeTransform,
    Error& error) {
  return LoadShapeFromBytes(fileData, format, Real3{1_r, 1_r, 1_r}, bakeTransform, error);
}

inline ShapeHandle Context::LoadShapeFromBytes(Span<char const> fileData, Error& error) {
  return LoadShapeFromBytes(fileData, MeshFileType::Legacy, error);
}

inline ShapeHandle
Context::LoadShapeFromBytes(Span<char const> fileData, Real3 const& bakeScale, Error& error) {
  return LoadShapeFromBytes(
      fileData, MeshFileType::Legacy, bakeScale, TransformRT::Identity(), error);
}

inline ShapeHandle Context::LoadShapeFromBytes(
    Span<char const> fileData,
    TransformRT const& bakeTransform,
    Error& error) {
  return LoadShapeFromBytes(
      fileData, MeshFileType::Legacy, Real3{1_r, 1_r, 1_r}, bakeTransform, error);
}

inline ShapeHandle Context::LoadShapeFromBytes(
    Span<char const> fileData,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error) {
  return LoadShapeFromBytes(fileData, MeshFileType::Legacy, bakeScale, bakeTransform, error);
}

inline void Context::EnableLogChannel(LogChannel channel, bool enable) {
  // Enable the log channel in the calling library or executable.
  mochi::EnableLogChannel(channel, enable);

  // Also enable inside the mochi_physics library. This is not redundant if mochi_physics is
  // compiled as a DLL on Windows because static variables are not shared across DLL boundaries.
  mochi::Context::EnableLogChannelInternal(channel, enable);
}

inline void Context::SetLogCallback(LogFn callback) {
  // Set the log callback in the calling library or executable.
  mochi::SetLogCallback(callback);

  // Also set it inside the mochi_physics library. This is not redundant if mochi_physics is
  // compiled as a DLL on Windows because static variables are not dynamically linked on Windows.
  mochi::Context::SetLogCallbackInternal(callback);
}

inline void Context::SetAssertionFailureCallback(OnAssertFn callback) {
  // Set the assertion failure callback in the calling library or executable.
  mochi::SetAssertionFailureCallback(callback);

  // Also set it inside the mochi_physics library. This is not redundant if mochi_physics is
  // compiled as a DLL on Windows because static variables are not dynamically linked on Windows.
  mochi::Context::SetAssertionFailureCallbackInternal(callback);
}

} // namespace mochi
