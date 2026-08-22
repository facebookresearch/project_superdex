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

#include "../utils/mochi_physics_macros.h"
#include "mochi_handle.h"
#include "mochi_structs.h"

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <string_view>

namespace mochi {

class Actor;
class AsyncScene;
class DebugServer;
class IKSolver;
class Scene;

class Context {
 public:
  //----------------------------------------------------------------------------------------------
  // Concurrency APIs
  //----------------------------------------------------------------------------------------------

  virtual void BindThisThread() = 0;

  virtual void UnbindThisThread() = 0;

  virtual void SetIsSingleThreaded(bool isSingleThreaded) = 0;

  [[nodiscard]] virtual bool IsSingleThreaded() const = 0;

  [[nodiscard]] virtual int GetNumThreads() const = 0;

  //----------------------------------------------------------------------------------------------
  // Debugging APIs
  //----------------------------------------------------------------------------------------------

  static void EnableLogChannel(LogChannel channel, bool enable);

  [[nodiscard]] MOCHI_API static bool IsLogChannelEnabled(LogChannel channel);

  static void SetLogCallback(LogFn callback);

  [[nodiscard]] MOCHI_API static LogFn GetLogCallback();

  static void SetAssertionFailureCallback(OnAssertFn callback);

  [[nodiscard]] MOCHI_API static OnAssertFn GetAssertionFailureCallback();

  //----------------------------------------------------------------------------------------------
  // Shape APIs
  //----------------------------------------------------------------------------------------------

  [[nodiscard]] virtual ShapeHandle LoadShapeFromFile(std::string_view filePath, Error& error) = 0;

  [[nodiscard]] ShapeHandle
  LoadShapeFromFile(std::string_view filePath, Real3 const& bakeScale, Error& error);

  [[nodiscard]] ShapeHandle
  LoadShapeFromFile(std::string_view filePath, TransformRT const& bakeTransform, Error& error);

  [[nodiscard]] virtual ShapeHandle LoadShapeFromFile(
      std::string_view filePath,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle
  LoadShapeFromBytes(Span<char const> fileData, MeshFileType format, Error& error) = 0;

  [[nodiscard]] ShapeHandle LoadShapeFromBytes(
      Span<char const> fileData,
      MeshFileType format,
      Real3 const& bakeScale,
      Error& error);

  [[nodiscard]] ShapeHandle LoadShapeFromBytes(
      Span<char const> fileData,
      MeshFileType format,
      TransformRT const& bakeTransform,
      Error& error);

  [[nodiscard]] virtual ShapeHandle LoadShapeFromBytes(
      Span<char const> fileData,
      MeshFileType format,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Error& error) = 0;

  [[nodiscard]] ShapeHandle LoadShapeFromBytes(Span<char const> fileData, Error& error);

  [[nodiscard]] ShapeHandle
  LoadShapeFromBytes(Span<char const> fileData, Real3 const& bakeScale, Error& error);

  [[nodiscard]] ShapeHandle
  LoadShapeFromBytes(Span<char const> fileData, TransformRT const& bakeTransform, Error& error);

  [[nodiscard]] ShapeHandle LoadShapeFromBytes(
      Span<char const> fileData,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Error& error);

  [[nodiscard]] virtual ShapeHandle CreateMeshShape(MeshData const& mesh, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle CreateMeshShape(MeshDataView const& mesh, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle CreateModelShape(ModelData const& model, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle CreateModelShape(ModelDataView const& model, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle
  CreateTetMeshShape(Span<real const> coordinates, Span<int const> connectivity, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle
  CreateTriMeshShape(Span<real const> coordinates, Span<int const> connectivity, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle
  CreateSphereShape(Real3 const& center, real radius, Error& error) = 0;

  [[nodiscard]] virtual ShapeHandle
  CreatePlaneShape(Real3 const& normal, real distance, Error& error) = 0;

  [[nodiscard]] virtual MeshDataView GetShapeMesh(ShapeHandle shape, Error& error) const = 0;
  [[nodiscard]] virtual MeshDataView GetShapeSurfaceMesh(ShapeHandle shape, Error& error) const = 0;
  [[nodiscard]] virtual MeshDataView GetShapeVisualMesh(ShapeHandle shape, Error& error) const = 0;

  [[nodiscard]] virtual Aabb GetShapeAabb(ShapeHandle shape, Error& error) const = 0;

  [[nodiscard]] virtual ArticulatedShapeInfo GetArticulatedShapeInfo(
      ShapeHandle shape,
      Error& error) const = 0;

  virtual void ReleaseShape(ShapeHandle shape) = 0;

  [[nodiscard]] virtual int GetNumShapes() const = 0;

  virtual void EnableFileCache(bool enable) = 0;

  [[nodiscard]] virtual bool IsFileCacheEnabled() const = 0;

  virtual void ClearFileCache() = 0;

  virtual void ClearFileFromCache(std::string_view filePath) = 0;

  //----------------------------------------------------------------------------------------------
  // Scene APIs
  //----------------------------------------------------------------------------------------------

  [[nodiscard]] virtual Scene* CreateScene(std::string_view name) = 0;

  virtual void DestroyScene(Scene* scene) = 0;

  virtual Scene* GetScene(SceneHandle handle) = 0;

  virtual Scene const* GetScene(SceneHandle handle) const = 0;

  [[nodiscard]] virtual bool IsValidScene(Scene const* scene) const = 0;

  //----------------------------------------------------------------------------------------------
  // AsyncScene APIs
  //----------------------------------------------------------------------------------------------

  [[nodiscard]] virtual AsyncScene* CreateAsyncScene(std::string_view name, Error& error) = 0;

  [[nodiscard]] virtual AsyncScene* CreateAsyncScenePaused(std::string_view name, Error& error) = 0;

  virtual void DestroyAsyncScene(AsyncScene* scene) = 0;

  [[nodiscard]] virtual bool IsValidAsyncScene(AsyncScene const* scene) const = 0;

  //----------------------------------------------------------------------------------------------
  // DebugServer APIs
  //----------------------------------------------------------------------------------------------

  virtual DebugServer& GetDebugServer() = 0;
  virtual DebugServer const& GetDebugServer() const = 0;

 protected:
  MOCHI_API static void EnableLogChannelInternal(LogChannel channel, bool enable);
  MOCHI_API static void SetLogCallbackInternal(LogFn callback);
  MOCHI_API static void SetAssertionFailureCallbackInternal(OnAssertFn callback);

  // Please use DestroyContext. Do not delete the pointer directly.
  virtual ~Context() = default;
};

[[nodiscard]] MOCHI_API Context* CreateContext(int numWorkerThreads = -1);

MOCHI_API void DestroyContext(Context* context);

} // namespace mochi

#include "mochi_context_inl.h"
