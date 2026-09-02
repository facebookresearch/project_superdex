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

#include "mochi_async_scene.h"
#include "mochi_ik.h"
#include "mochi_newton_euler_terms.h"
#if MOCHI_USE_OSC
#include "internal/mochi_operational_space_controller.h"
#endif
#include "mochi_scene.h"
#include "mochi_shape.h"

#include <mochi_core/utils/task_scheduler.h>
#include <mochi_physics/mochi_physics.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mochi {

// Forward
namespace dbg {
class DebugServerInternal;
} // namespace dbg
struct ModelData;
struct ExperimentalModelData;

/**
  ContextImpl Implementation

    - Implements the mochi::Context virtual API.
    - Owns all runtime state including the collection of Scenes.
    - The user is expected to create exactly one instance of this class.
*/
class ContextImpl final : public Context {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ContextImpl);

 public:
  ContextImpl();
  ~ContextImpl() override;

  // Used by CreateContext and DestroyContext
  void Initialize();
  void PreShutDown();
  void ShutDown();

  // mochi::Context API:
  ShapeHandle LoadShapeFromFile(std::string_view filePath, Error& error) override {
    return LoadShapeFromFile(filePath, Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), error);
  }
  ShapeHandle LoadShapeFromFile(
      std::string_view filePath,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Error& error) override;

  ShapeHandle LoadShapeFromBytes(Span<char const> fileData, MeshFileType format, Error& error)
      override {
    return LoadShapeFromBytes(
        fileData, format, Real3{1_r, 1_r, 1_r}, TransformRT::Identity(), error);
  }
  ShapeHandle LoadShapeFromBytes(
      Span<char const> fileData,
      MeshFileType format,
      Real3 const& bakeScale,
      TransformRT const& bakeTransform,
      Error& error) override;

  ShapeHandle CreateMeshShape(MeshData const& mesh, Error& error) override;
  ShapeHandle CreateMeshShape(MeshDataView const& mesh, Error& error) override;
  ShapeHandle CreateModelShape(ModelData const& model, Error& error) override;
  ShapeHandle CreateModelShape(ModelDataView const& model, Error& error) override;

  ShapeHandle CreateTetMeshShape(
      Span<real const> coordinates,
      Span<int const> connectivity,
      Error& error) override;
  ShapeHandle CreateTriMeshShape(
      Span<real const> coordinates,
      Span<int const> connectivity,
      Error& error) override;
  ShapeHandle CreateSphereShape(Real3 const& center, real radius, Error& error) override;
  ShapeHandle CreatePlaneShape(Real3 const& normal, real distance, Error& error) override;
  MeshDataView GetShapeMesh(ShapeHandle shape, Error& error) const override;
  MeshDataView GetShapeSurfaceMesh(ShapeHandle shape, Error& error) const override;
  MeshDataView GetShapeVisualMesh(ShapeHandle shape, Error& error) const override;
  Aabb GetShapeAabb(ShapeHandle shape, Error& error) const override;
  [[nodiscard]] ArticulatedShapeInfo GetArticulatedShapeInfo(ShapeHandle shape, Error& error)
      const override;
  void ReleaseShape(ShapeHandle shape) override;
  int GetNumShapes() const override;

  void EnableFileCache(bool enable) override;
  bool IsFileCacheEnabled() const override;
  void ClearFileCache() override;
  void ClearFileFromCache(std::string_view filePath) override;

  Scene* CreateScene(std::string_view name) override;
  void DestroyScene(Scene* scene) override;
  Scene* GetScene(SceneHandle handle) override;
  Scene const* GetScene(SceneHandle handle) const override;
  bool IsValidScene(Scene const* scene) const override;
  experimental::IKSolver* CreateIKSolver(Scene* scene, Error& error);
  void DestroyIKSolver(experimental::IKSolver* solver);
  bool IsValidIKSolver(experimental::IKSolver const* solver) const;
#if MOCHI_USE_OSC
  experimental::OperationalSpaceController*
  CreateOperationalSpaceController(Actor* robot, int linkId, Error& error);
  void DestroyOperationalSpaceController(experimental::OperationalSpaceController* controller);
  bool IsValidOperationalSpaceController(
      experimental::OperationalSpaceController const* controller) const;
#endif
  experimental::NewtonEulerTerms* CreateNewtonEulerTerms(Actor* robot, Error& error);
  void DestroyNewtonEulerTerms(experimental::NewtonEulerTerms* newtonEulerTerms);
  bool IsValidNewtonEulerTerms(experimental::NewtonEulerTerms const* newtonEulerTerms) const;
  AsyncScene* CreateAsyncScene(std::string_view name, Error& error) override;
  AsyncScene* CreateAsyncScenePaused(std::string_view name, Error& error) override;
  void DestroyAsyncScene(AsyncScene* scene) override;
  bool IsValidAsyncScene(AsyncScene const* scene) const override;
  void BindThisThread() override;
  void UnbindThisThread() override;
  bool IsSingleThreaded() const override;
  void SetIsSingleThreaded(bool isSingleThreaded) override;
  int GetNumThreads() const override;
  DebugServer& GetDebugServer() override;
  DebugServer const& GetDebugServer() const override;

  // For internal use only:
  void SetNumWorkerThreads(int numThreads); // must be called before Initialize()
  TaskScheduler& GetTaskScheduler();
  static ShapePtr CreateShapeFromModelData(ModelData&& model, Error& error);
  static ShapePtr
  CreateShapeFromModelData(ModelData&& model, ExperimentalModelData&& experimental, Error& error);
  ShapeHandle RegisterShape(ConstShapePtr shape, Error& error);
  ConstShapePtr GetShapeSharedPtr(ShapeHandle shape) const;

  // The C API needs a way to disable auto-cleanup of shape handles. Currently there is no public
  // option for this in languages that support auto-cleanup.
  void EnableAutomaticHandleCleanup(bool enable) {
    std::lock_guard lock(_mutex);
    _enableAutomaticHandleCleanup = enable;
  }

 private:
  AsyncScene* CreateAsyncSceneImpl(std::string_view name, bool startPaused, Error& error);

  struct FileCacheEntry {
    // Pointer to the loaded shape, or null if the shape is still loading or if it failed.
    // This shared_ptr may also be in the _shapes map.
    ConstShapePtr shape;

    // If a file failed to load, then presumably it will continue to fail. Store the result here.
    Error error;

    // It is possible for a FileCacheEntry to exist with a null shape. This means that another
    // thread has started loading the file but hasn't finished. In that case, wait for the semaphore
    // outside of the _mutex lock, then try again.
    TaskSemaphore semaphore;
  };
  using FileCacheEntryPtr = std::shared_ptr<FileCacheEntry>;

  // If the shape is already in the cache, then return a ShapeHandle immediately.
  // Else, use the provided function to perform the load and update the cache.
  ShapeHandle LoadShapeWithFileCache(
      std::string fileCacheKey,
      std::function<ShapePtr(Error&)> const& doLoad,
      Error& error);

  Handle::ValueType GenerateNewHandle();

  // We store a std::shared_ptr<WeakRef> and share it with ShapeHandle objects, when automatic
  // cleanup is enabled. It is used like a weak pointer.
  struct WeakRef : NoCopy {
    explicit WeakRef(ContextImpl* c) : context(c) {}
    std::mutex mutex;
    ContextImpl* context = {};
  };

  // Implement the automatic cleanup feature for a ShapeHandle.
  struct ShapeHandleAutoCleanup : ShapeHandle::AutoCleanup, NoCopy {
    using WeakRef = ContextImpl::WeakRef;
    std::atomic<ShapeHandle::ValueType> value = {};
    std::shared_ptr<WeakRef> const ref;

    explicit ShapeHandleAutoCleanup(ShapeHandle::ValueType v, std::shared_ptr<WeakRef> const& r)
        : value(v), ref(r) {}

    ~ShapeHandleAutoCleanup() override {
      auto v = value.load();
      if (v && ref) {
        std::lock_guard lock(ref->mutex);
        if (ref->context) {
          // If we reach this point, then we know that the context has not yet been destroyed, and
          // it can't be destroyed until we release the mutex.
          ref->context->ReleaseShape(ShapeHandle{v});
        }
      }
    }
  };

  mutable std::recursive_mutex _mutex;
  bool _shouldShutdownProfiler = false;
  bool _hasPreShutDown = false;
  bool _hasShutDown = false;
  bool _enableAutomaticHandleCleanup = true;
  std::atomic<bool> _isFileCacheEnabled = false;
  int _numWorkerThreadsRequested; // default computed in constructor
  std::atomic<Handle::ValueType> _nextHandle{1};
  std::unordered_map<ShapeHandle::ValueType, ConstShapePtr> _shapes;
  std::vector<std::unique_ptr<SceneImpl>> _scenes;
  std::vector<std::unique_ptr<IKSolverImpl>> _ikSolvers;
#if MOCHI_USE_OSC
  std::vector<std::unique_ptr<OperationalSpaceControllerImpl>> _OSCs;
#endif
  std::vector<std::unique_ptr<NewtonEulerTermsImpl>> _newtonEulerTerms;
  std::vector<std::unique_ptr<AsyncSceneImpl>> _asyncScenes;
  std::unique_ptr<TaskScheduler> _scheduler;
  std::shared_ptr<WeakRef> _weakRefToThis;
  std::unique_ptr<dbg::DebugServerInternal> _debugServer;

  // See GetFileCacheKey() for the format of the key.
  std::unordered_map<std::string, FileCacheEntryPtr> _fileCache;
};

inline Handle::ValueType ContextImpl::GenerateNewHandle() {
  return ++_nextHandle;
}

inline TaskScheduler& ContextImpl::GetTaskScheduler() {
  return *_scheduler;
}

} // namespace mochi
