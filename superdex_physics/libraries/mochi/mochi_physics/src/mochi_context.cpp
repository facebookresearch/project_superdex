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

#include "mochi_context.h"

#include "mochi_hdf5.h"
#include "mochi_rod.h"
#include "mochi_scene.h"

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/geometry/grid_sdf.h>
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/geometry/obb.h>
#include <mochi_core/geometry/plane.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/hdf5_utils.h>
#include <mochi_core/utils/json_utils.h> // TODO: Remove once articulated shape loading has been updated
#include <mochi_core/utils/mesh_embedding.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_physics/dbg/debug_server_internal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mochi::experimental;

namespace mochi {

static int GetDefaultNumWorkerThreads() {
  // By default, Mochi will try to use the majority of the physical CPU cores, but it will leave a
  // couple unused for other things like rendering and input. If the CPU is hyperthreaded, then it
  // may have a larger number of logical processors, but we don't use that many threads by default
  // because doing so does not generally improve overall performance. In fact, if we use too many
  // threads, it may force the operating system to perform more context switches, which can decrease
  // average performance and cause performance spikes.
  //
  // Example: A Windows computer with AMD Threadripper Pro has 32 physical cores and 64 logical.
  //          By default, Mochi will use (32 - 2) = 30 worker threads. Use of 60 worker threads on
  //          this computer has been empirically demonstrated to run 10-15% slower on average, for
  //          many Mochi samples.
  //
  //  To use a different number of worker threads, call ContextImpl::SetNumWorkerThreads()
  //  before ContextImpl::Initialize, or simply pass the thread count to
  //  mochi::CreateContextImpl.
  //
  int numCores = TaskScheduler::GetNumSupportedPhysicalProcessors();
  return Max(1, numCores - 2);
}

ContextImpl::ContextImpl() : _numWorkerThreadsRequested(GetDefaultNumWorkerThreads()) {
  _weakRefToThis = std::make_shared<WeakRef>(this);
}

// Defined here (not defaulted in the header) so the unique_ptr<DebugServerInternal>
// destructor is instantiated in this TU, where DebugServerInternal is complete.
ContextImpl::~ContextImpl() = default;

void ContextImpl::SetNumWorkerThreads(int numThreads) {
  std::scoped_lock lock(_mutex);
  if (_scheduler == nullptr) {
    _numWorkerThreadsRequested =
        Clamp(numThreads, 0, TaskScheduler::GetNumSupportedLogicalProcessors());
  } else {
    MOCHI_LOG_WARNING(
        "mochi::Context::SetNumWorkerThreads can only be called before mochi::Context::Initialize.");
  }
}

void ContextImpl::Initialize() {
  std::scoped_lock lock(_mutex);

  if (!ProfilerIsInitialized()) {
    ProfilerInitialize();
    _shouldShutdownProfiler = true;
  }

  _scheduler = std::make_unique<TaskScheduler>(_numWorkerThreadsRequested);
  _debugServer = dbg::CreateDebugServer(this);
}

void ContextImpl::PreShutDown() {
  // Stop the server, disconnect all clients, and unpause all scenes.
  if (_debugServer) {
    _debugServer->Stop();
  }

  // Invalidate _weakRefToThis to ShapeHandle automatic cleanup after this point.
  {
    MOCHI_ASSERT(_weakRefToThis != nullptr);
    std::scoped_lock lock(_weakRefToThis->mutex);
    _weakRefToThis->context = nullptr;
  }

  // If we have any AsyncScenes, then move them to a temporary vector, so that they can be destroyed
  // after we release the mutex lock. This prevents a deadlock because the destruction of an
  // AsyncScene includes a call to mochi::Context::DestroyScene (destroying the synchronous Scene).
  // That call will happen on the simulation loop thread, and it will also need to lock our mutex.
  std::vector<std::unique_ptr<AsyncSceneImpl>> asyncScenesToDestroy;
  {
    std::scoped_lock lock(_mutex);
    if (_hasPreShutDown) {
      return;
    }
    asyncScenesToDestroy = std::move(_asyncScenes);
    _hasPreShutDown = true;
  }
  asyncScenesToDestroy.clear(); // Destroy them

  // Now we can destroy the rest of the user-created objects.
  // This must happen before we can start shutting down shared services like TaskScheduler.
  std::scoped_lock lock(_mutex);
#if MOCHI_USE_OSC
  _OSCs.clear();
#endif
  _newtonEulerTerms.clear();
  _ikSolvers.clear();
  _scenes.clear();
  _shapes.clear();

  // All scenes have been destroyed, so it is now safe to destroy the debug server.
  _debugServer.reset();
}

void ContextImpl::ShutDown() {
  // PreShutDown always has to be completed before ShutDown. Do it now if necessary.
  PreShutDown();

  // Check _hasShutDown while holding the mutex lock
  {
    std::scoped_lock lock(_mutex);
    if (_hasShutDown) {
      return;
    }
    _hasShutDown = true;
  }

  // Shutting down the TaskScheduler may have to wait for pending tasks to complete. We do this
  // while we are NOT holding the mutex lock, just in case it is needed by one of those tasks.
  _scheduler.reset();

  // Finalize Petsc if we were the ones to initialize it
  std::scoped_lock lock(_mutex);

  if (_shouldShutdownProfiler) {
    ProfilerShutdown();
  }
}

static void ValidateShapeTransform(Real3 const& scale, TransformRT const& rt, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !IsFinite(scale) || !IsFinite(rt), error, "Invalid shape scale. Non-finite values detected.");
  MOCHI_ERROR_IF(
      NearZero(scale[0]) || NearZero(scale[1]) || NearZero(scale[2]),
      error,
      "Invalid shape scale. Scale must be non-zero on all 3 axes.");
}

// Generates the default frame axes of a valid polyline model that has none. Loads call this before
// baking a transform, so that the bake transforms default axes like authored ones.
static void GenerateMissingPolylineFrameAxes(ModelData& model) {
  if (!model.mesh || model.mesh->nodesPerElement != 2 || model.elementFrameAxes) {
    return;
  }
  DynamicArray<Real3> const axes = GenerateDiscreteBishopFrame(
      Unflatten<Real3 const>(model.mesh->coordinates), IsPolylineClosedLoop(*model.mesh));
  Span<real const> const flatAxes = Flatten(MakeConstSpan(axes));
  model.elementFrameAxes.emplace(flatAxes.begin(), flatAxes.end());
}

// The path component of a file cache key. Normalizing it lets different spellings of one file
// (mixed separators, "a/../b") share a cache entry, and lets ClearFileFromCache evict a file it
// was given under a different spelling than the load used. Case is deliberately not folded: that
// would be wrong on a case-sensitive filesystem.
static std::string NormalizeFileCachePath(std::string_view path) {
  return std::filesystem::path(path).lexically_normal().generic_string();
}

// Format a key for lookup in the file cache.
static std::string
GetFileCacheKey(std::string_view path, Real3 const& scale, TransformRT const& transform) {
  return Format(
      "%s:%s:%s:%s",
      NormalizeFileCachePath(path).c_str(),
      SReflect::ToJsonString(scale).c_str(),
      SReflect::ToJsonString(transform.GetRotation()).c_str(),
      SReflect::ToJsonString(transform.GetTranslation()).c_str());
}

ShapeHandle ContextImpl::LoadShapeWithFileCache(
    std::string fileCacheKey,
    std::function<ShapePtr(Error&)> const& doLoad,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // If the file cache is enabled, then find or insert a cache entry.
  FileCacheEntryPtr cacheEntry;
  bool isNewCacheEntry = false;
  if (_isFileCacheEnabled) {
    {
      std::scoped_lock lock(_mutex);
      if (_isFileCacheEnabled) {
        auto insertPair =
            _fileCache.insert(std::make_pair(std::move(fileCacheKey), FileCacheEntryPtr{}));
        isNewCacheEntry = insertPair.second;
        if (isNewCacheEntry) {
          // We just inserted a key. Create the FileCacheEntry and increment the semaphore.
          // We will decrement it outside of the mutex lock, after we finish loading.
          cacheEntry = insertPair.first->second = std::make_shared<FileCacheEntry>();
          cacheEntry->semaphore.Add(1);
        } else {
          // This key was already in the map.
          cacheEntry = insertPair.first->second;
          MOCHI_ASSERT(cacheEntry != nullptr);
        }
      }
    }

    if (cacheEntry && !isNewCacheEntry) {
      // A cache entry was found. Wait for the semaphore in case the entry was added recently by
      // another thread which is still loading it.
      cacheEntry->semaphore.Wait();

      if (cacheEntry->error.IsOK()) {
        // The cache entry has a shape that was loaded successfully.
        // Register a new ShapeHandle to track the caller's use of it.
        MOCHI_ASSERT_VERBOSE(
            cacheEntry->shape != nullptr, "Should be valid since there was no error");
        return RegisterShape(cacheEntry->shape, error);
      } else {
        // The cache entry refers to a file load attempt that failed in the past.
        // If it failed before, we assume that it will fail again, so let the same error bubble up.
        error = cacheEntry->error.Copy();
        return {};
      }
    }
  }

  // Do the work of actually loading the shape
  auto newShape = doLoad(error);

  // If we added a new entry to the cache, then store the results in that entry
  if (isNewCacheEntry) {
    if (error.IsOK()) {
      MOCHI_ASSERT_VERBOSE(newShape != nullptr, "Should be valid since there was no error");
      cacheEntry->shape = newShape;
    } else {
      cacheEntry->error = error.Copy();
    }

    // Notify other threads that may have been waiting for us to finish the load attempt.
    cacheEntry->semaphore.Done();
  }

  // Register a new ShapeHandle to track the caller's use of it.
  return RegisterShape(newShape, error);
}

ShapeHandle ContextImpl::LoadShapeFromFile(
    std::string_view filePath,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error) {
  ValidateShapeTransform(bakeScale, bakeTransform, error);
  MOCHI_ERROR_IF(filePath.empty(), error, "Invalid file path");
  MOCHI_ERROR_RETURN(error, {});
  ScopedSchedulerBinding schedulerBinding(GetTaskScheduler());

  auto doLoad = [&](Error& err) {
    // Load the file (any supported format)
    ModelData model = model::LoadFromFile(filePath, err);
    MOCHI_ERROR_RETURN(err, ShapePtr{});
    GenerateMissingPolylineFrameAxes(model);
    model::BakeTransform(model, bakeScale, bakeTransform, err);
    model::Validate(model, err); // Baking can invalidate a valid model
    MOCHI_ERROR_RETURN(err, ShapePtr{});

    // Load experimental feature if present.
    ExperimentalModelData experimentalData;
#if MOCHI_USE_HDF5
    if (model.experimentalDataDetected && filePath.ends_with(".h5")) {
      experimentalData =
          hdf5::LoadExperimentalModelDataFromFile(filePath, model, bakeScale, bakeTransform, err);
      MOCHI_ERROR_RETURN(err, ShapePtr{});
    }
#endif // MOCHI_USE_HDF5

    return CreateShapeFromModelData(std::move(model), std::move(experimentalData), err);
  };

  if (_isFileCacheEnabled) {
    return LoadShapeWithFileCache(
        GetFileCacheKey(filePath, bakeScale, bakeTransform), doLoad, error);
  } else {
    auto newShape = doLoad(error);
    return RegisterShape(newShape, error);
  }
}

// Experimental API
ShapeHandle experimental::CreateDeepFlowShape(
    Context* context,
    DeepModelParams const& params, // Parameters of a deep model
    Error& error) {
  MOCHI_ERROR_IF(!context, error, "Invalid context");
  MOCHI_ERROR_IF(
      !MOCHI_ENABLE_DEEP_FLOW_ACTORS,
      error,
      "Deep Flow shape creation is not supported in this build. To enable, define MOCHI_ENABLE_DEEP_FLOW_ACTORS=1");
  MOCHI_ERROR_RETURN(error, {});
  auto newShape = std::make_shared<DeepFlowShape>();
  newShape->flow = mochi::LoadDeepFlow(
      params.deepModelPath.c_str(),
      params.scale,
      Real3(params.shiftX, params.shiftY, params.shiftZ),
      params.numDof,
      error);
  MOCHI_ERROR_RETURN(error, {});

  return assert_cast<ContextImpl*>(context)->RegisterShape(newShape, error);
}

// Experimental API
ShapeHandle experimental::CreatePolylineShape(
    Context* context,
    Span<Real3 const> nodes,
    Span<Real3 const> elementFrameAxes,
    bool isClosedLoop,
    Error& error) {
  MOCHI_ERROR_IF(!context, error, "Invalid context");
  if (isClosedLoop) {
    MOCHI_ERROR_IF_NOT(
        isize(nodes) >= 3, error, "Must have at least 3 nodes for a closed-loop polyline.");
  } else {
    MOCHI_ERROR_IF_NOT(isize(nodes) >= 2, error, "Must have at least 2 nodes.");
  }
  MOCHI_ERROR_RETURN(error, {});

  ModelData model;
  model.mesh.emplace(MeshData{});
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = Flatten(nodes);
  model.elementFrameAxes = DynamicArray<real>{Flatten(elementFrameAxes)};
  // Encode polyline topology via connectivity. Closed-loop polylines need an explicit closing
  // segment; open polylines could leave it empty, but we emit it so the data is
  // self-describing all the way through the pipeline.
  model.mesh->connectivity = MakeSequentialPolylineConnectivity(isize(nodes), isClosedLoop);

  model::AutoCorrect(model, error);
  model::Validate(model, error);

  auto* contextImpl = assert_cast<ContextImpl*>(context);
  auto newShape = ContextImpl::CreateShapeFromModelData(std::move(model), error);
  return contextImpl->RegisterShape(newShape, error);
}

ShapeHandle ContextImpl::LoadShapeFromBytes(
    Span<char const> fileData,
    MeshFileType format,
    Real3 const& bakeScale,
    TransformRT const& bakeTransform,
    Error& error) {
  ValidateShapeTransform(bakeScale, bakeTransform, error);
  MOCHI_ERROR_IF(fileData.empty(), error, "No data");
  MOCHI_ERROR_RETURN(error, {});
  ScopedSchedulerBinding schedulerBinding(GetTaskScheduler());

  // Load the file from memory (any supported format)
  ModelData model = model::LoadFromBytes(fileData, format, error);
  MOCHI_ERROR_RETURN(error, {});
  GenerateMissingPolylineFrameAxes(model);
  model::BakeTransform(model, bakeScale, bakeTransform, error);
  model::Validate(model, error); // Baking can invalidate a valid model
  MOCHI_ERROR_RETURN(error, {});

  // Load experimental feature if present.
  ExperimentalModelData experimentalData;
#if MOCHI_USE_HDF5
  if (model.experimentalDataDetected && hdf5::LooksLikeHDF5(fileData)) {
    experimentalData =
        hdf5::LoadExperimentalModelDataFromBytes(fileData, model, bakeScale, bakeTransform, error);
    MOCHI_ERROR_RETURN(error, {});
  }
#endif // MOCHI_USE_HDF5

  auto newShape = CreateShapeFromModelData(std::move(model), std::move(experimentalData), error);

  // Allocate a new handle to track this request for the shape (if successful)
  return RegisterShape(newShape, error);
}

ShapeHandle ContextImpl::CreateTetMeshShape(
    Span<real const> coordinates,
    Span<int const> connectivity,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelDataView model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  model.mesh->coordinates = coordinates;
  model.mesh->connectivity = connectivity;
  return CreateModelShape(model, error);
}

#define MOCHI_ARRAY_CHECK_STRIDE(v, stride, error, descriptionStringLiteral)   \
  MOCHI_ERROR_IF((v).size() % (stride) != 0, error, descriptionStringLiteral); \
  MOCHI_ERROR_RETURN(error, {});

#define MOCHI_ARRAY_CHECK_LENGTH(v, length, error, descriptionStringLiteral) \
  MOCHI_ERROR_IF((v).size() != (length), error, descriptionStringLiteral);   \
  MOCHI_ERROR_RETURN(error, {});

static std::unordered_map<std::string, RomData>
MakeRomData(int numCoords, std::string const& name, Span<real const> basisIn, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (basisIn.empty()) {
    return {};
  }
  MOCHI_ERROR_IF(name.empty(), error, "Missing ROM model name");
  MOCHI_ARRAY_CHECK_STRIDE(
      basisIn, numCoords, error, "Size of linear rom basis must be multiple of coordinate size");
  std::unordered_map<std::string, RomData> romDataMap{};
  RowMatrix<real> basis =
      RowMatrixView<real const>(basisIn.data(), numCoords, basisIn.size() / numCoords);
  romDataMap[name] = LinearRomData{.needsRigidTransformLayer = false, .basis = std::move(basis)};
  return romDataMap;
}

ShapeHandle experimental::CreateModelShapeWithLinearRom(
    Context* context,
    ModelDataView const& model,
    std::string_view linearRomName,
    Span<real const> linearRomBasis,
    Error& error) {
  MOCHI_ERROR_IF(!context, error, "Invalid context");
  MOCHI_ERROR_IF(
      !MOCHI_ENABLE_ROM_ACTORS,
      error,
      "ROM shape creation is not supported in this build. To enable, define MOCHI_ENABLE_ROM_ACTORS=1");
  MOCHI_ERROR_IF(linearRomName.empty(), error, "ROM name cannot be empty");
  MOCHI_ERROR_IF(linearRomBasis.empty(), error, "ROM basis cannot be empty");
  MOCHI_ERROR_IF(
      !model.mesh || (model.mesh->nodesPerElement != 4),
      error,
      "Model must have a tetrahedral mesh.");
  MOCHI_ERROR_RETURN(error, {});

  auto modelData = ModelData{model}; // Copy
  model::AutoCorrect(modelData, error);
  model::Validate(modelData, error);
  MOCHI_ERROR_RETURN(error, {});

  ExperimentalModelData experimentalData;
  experimentalData.romData = MakeRomData(
      isize(model.mesh->coordinates), std::string{linearRomName}, linearRomBasis, error);
  MOCHI_ERROR_RETURN(error, {});

  auto* contextImpl = assert_cast<ContextImpl*>(context);
  auto shapePtr = contextImpl->CreateShapeFromModelData(
      std::move(modelData), std::move(experimentalData), error);
  return contextImpl->RegisterShape(shapePtr, error);
}

ShapeHandle ContextImpl::CreateMeshShape(MeshData const& mesh, Error& error) {
  return CreateMeshShape(MeshDataView{mesh}, error);
}

ShapeHandle ContextImpl::CreateMeshShape(MeshDataView const& mesh, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelDataView model;
  model.mesh = mesh;
  return CreateModelShape(model, error);
}

ShapeHandle ContextImpl::CreateModelShape(ModelData const& model, Error& error) {
  return CreateModelShape(ModelDataView{model}, error);
}

ShapeHandle ContextImpl::CreateModelShape(ModelDataView const& model, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto modelCopy = ModelData{model};
  model::AutoCorrect(modelCopy, error);
  model::Validate(modelCopy, error);
  MOCHI_ERROR_RETURN(error, {});
  auto shapePtr = CreateShapeFromModelData(std::move(modelCopy), error);
  return RegisterShape(shapePtr, error);
}

ShapeHandle ContextImpl::CreateTriMeshShape(
    Span<real const> coordinates,
    Span<int const> connectivity,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelDataView model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 3;
  model.mesh->coordinates = coordinates;
  model.mesh->connectivity = connectivity;
  return CreateModelShape(model, error);
}

ShapeHandle ContextImpl::CreateSphereShape(Real3 const& center, real radius, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelData model;
  model.sphere = Sphere{center, radius};
  return CreateModelShape(model, error);
}

ShapeHandle ContextImpl::CreatePlaneShape(Real3 const& normal, real distance, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  ModelData model;
  model.plane = Plane{normal, distance};
  return CreateModelShape(model, error);
}

void ContextImpl::ReleaseShape(ShapeHandle shape) {
  std::scoped_lock lock(_mutex);
  auto it = _shapes.find(shape.value);
  if (it != _shapes.end()) {
    _shapes.erase(it);

    // Prevent auto-cleanup since we already took care of it.
    if (auto* ref = assert_cast<ShapeHandleAutoCleanup*>(shape._cleanup.get())) {
      ref->value.store(0);
    }
  }
}

int ContextImpl::GetNumShapes() const {
  std::scoped_lock lock(_mutex);
  return isize(_shapes);
}

void ContextImpl::EnableFileCache(bool enable) {
  std::scoped_lock lock(_mutex);
  if (_isFileCacheEnabled && !enable) {
    ClearFileCache();
  }
  _isFileCacheEnabled = enable;
}

bool ContextImpl::IsFileCacheEnabled() const {
  std::scoped_lock lock(_mutex);
  return _isFileCacheEnabled;
}

void ContextImpl::ClearFileCache() {
  std::scoped_lock lock(_mutex);
  _fileCache.clear();
}

void ContextImpl::ClearFileFromCache(std::string_view filePath) {
  std::scoped_lock lock(_mutex);
  // Keys are "<normalized path>:<scale>:<rotation>:<translation>" (see GetFileCacheKey), so every
  // bake variant of this file is matched by the normalized path plus the separator.
  std::string keyPrefix = NormalizeFileCachePath(filePath);
  keyPrefix.push_back(':');
  std::erase_if(_fileCache, [&](auto const& pair) { return pair.first.starts_with(keyPrefix); });
}

Scene* ContextImpl::CreateScene(std::string_view name) {
  // Create a new scene
  auto newScene = std::make_unique<SceneImpl>(this, name, GenerateNewHandle());
  Scene* sceneRawPtr = newScene.get();

  std::scoped_lock lock(_mutex);

  // Add it to the collection
  _scenes.push_back(std::move(newScene));

  // Notify debug server
  _debugServer->OnAddScene(sceneRawPtr);

  return sceneRawPtr;
}

void ContextImpl::DestroyScene(Scene* scene) {
  std::unique_ptr<SceneImpl> sceneToDestroy;
  {
    std::scoped_lock lock(_mutex);
    auto it = std::ranges::find_if(_scenes, [scene](auto& ptr) { return ptr.get() == scene; });
    if (it == _scenes.end()) {
      return;
    }

    // Make sure it is not owned by an AsyncScene
    for (auto const& asyncScenePtr : _asyncScenes) {
      if (asyncScenePtr->GetSceneImplUnsafe() == scene) {
        MOCHI_LOG_WARNING(
            "Cannot destroy scene directly because it is owned by an AsyncScene. Call DestroyAsyncScene instead.");
        return;
      }
    }

    sceneToDestroy = std::move(*it);
    _debugServer->OnRemoveScene(sceneToDestroy.get());
    _scenes.erase(it);
  }

  // Destroy the scene after releasing the lock
  sceneToDestroy.reset();
}

experimental::IKSolver* ContextImpl::CreateIKSolver(Scene* scene, Error& error) {
  MOCHI_ERROR_IF(!IsValidScene(scene), error, "Scene does not belong to this context.");
  MOCHI_ERROR_RETURN(error, {})

  // Create a new IKSolver
  auto newIKSolver = std::make_unique<IKSolverImpl>(scene, error);
  MOCHI_ERROR_RETURN(error, {})
  experimental::IKSolver* IKSolverRawPtr = newIKSolver.get();

  // Add it to the collection
  std::scoped_lock lock(_mutex);
  _ikSolvers.push_back(std::move(newIKSolver));

  return IKSolverRawPtr;
}

void ContextImpl::DestroyIKSolver(experimental::IKSolver* solver) {
  std::unique_ptr<IKSolverImpl> IKSolverToDestroy;
  {
    std::scoped_lock lock(_mutex);
    auto it = std::ranges::find_if(_ikSolvers, [solver](auto& ptr) { return ptr.get() == solver; });
    if (it == _ikSolvers.end()) {
      return;
    }

    IKSolverToDestroy = std::move(*it);
    _ikSolvers.erase(it);
  }

  // Destroy the IKSolver after releasing the lock
  IKSolverToDestroy.reset();
}

bool ContextImpl::IsValidIKSolver(experimental::IKSolver const* solver) const {
  if (!solver) {
    return false;
  }
  std::scoped_lock lock(_mutex);
  auto it = std::ranges::find_if(_ikSolvers, [solver](auto& ptr) { return ptr.get() == solver; });
  return (it != _ikSolvers.end());
}

#if MOCHI_USE_OSC
OperationalSpaceController*
ContextImpl::CreateOperationalSpaceController(Actor* robot, int linkId, Error& error) {
  MOCHI_ERROR_RETURN(error, {})

  // Create a new OperationalSpaceController
  auto newOSC = std::make_unique<OperationalSpaceControllerImpl>(robot, linkId, error);
  MOCHI_ERROR_RETURN(error, {})
  OperationalSpaceController* OSCRawPtr = newOSC.get();

  // Add it to the collection
  std::scoped_lock lock(_mutex);
  _OSCs.push_back(std::move(newOSC));

  return OSCRawPtr;
}

void ContextImpl::DestroyOperationalSpaceController(OperationalSpaceController* controller) {
  std::unique_ptr<OperationalSpaceControllerImpl> OSCToDestroy;
  {
    std::scoped_lock lock(_mutex);
    auto it =
        std::ranges::find_if(_OSCs, [controller](auto& ptr) { return ptr.get() == controller; });
    if (it == _OSCs.end()) {
      return;
    }

    OSCToDestroy = std::move(*it);
    _OSCs.erase(it);
  }

  // Destroy the IKSolver after releasing the lock
  OSCToDestroy.reset();
}

bool ContextImpl::IsValidOperationalSpaceController(
    OperationalSpaceController const* controller) const {
  if (!controller) {
    return false;
  }
  std::scoped_lock lock(_mutex);
  auto it =
      std::ranges::find_if(_OSCs, [controller](auto& ptr) { return ptr.get() == controller; });
  return (it != _OSCs.end());
}
#endif

NewtonEulerTerms* ContextImpl::CreateNewtonEulerTerms(Actor* robot, Error& error) {
  MOCHI_ERROR_RETURN(error, {})

  auto newObj = std::make_unique<NewtonEulerTermsImpl>(robot, error);
  MOCHI_ERROR_RETURN(error, {})
  NewtonEulerTerms* rawPtr = newObj.get();

  std::scoped_lock lock(_mutex);
  _newtonEulerTerms.push_back(std::move(newObj));

  return rawPtr;
}

void ContextImpl::DestroyNewtonEulerTerms(NewtonEulerTerms* newtonEulerTerms) {
  std::unique_ptr<NewtonEulerTermsImpl> toDestroy;
  {
    std::scoped_lock lock(_mutex);
    auto it = std::ranges::find_if(
        _newtonEulerTerms, [newtonEulerTerms](auto& ptr) { return ptr.get() == newtonEulerTerms; });
    if (it == _newtonEulerTerms.end()) {
      return;
    }

    toDestroy = std::move(*it);
    _newtonEulerTerms.erase(it);
  }

  // Destroy after releasing the lock
  toDestroy.reset();
}

bool ContextImpl::IsValidNewtonEulerTerms(NewtonEulerTerms const* newtonEulerTerms) const {
  if (!newtonEulerTerms) {
    return false;
  }
  std::scoped_lock lock(_mutex);
  auto it = std::ranges::find_if(
      _newtonEulerTerms, [newtonEulerTerms](auto& ptr) { return ptr.get() == newtonEulerTerms; });
  return (it != _newtonEulerTerms.end());
}

AsyncScene*
ContextImpl::CreateAsyncSceneImpl(std::string_view name, bool startPaused, Error& error) {
  MOCHI_ERROR_IF(
      TaskScheduler::IsCurrentThreadAWorker(),
      error,
      "AsyncScene cannot be created from a Mochi worker thread. "
      "Create the AsyncScene from an application or control thread instead.");
  MOCHI_ERROR_IF(
      _scheduler->GetNumOtherThreads() == 0,
      error,
      "AsyncScene creation requires at least one Mochi worker thread and single-threaded mode disabled. "
      "Create the Context with at least one worker thread and ensure single-threaded mode is disabled before calling CreateAsyncScene or CreateAsyncScenePaused.");
  MOCHI_ERROR_RETURN(error, nullptr);

  auto newAsyncScene = std::make_unique<AsyncSceneImpl>(this, name, startPaused);
  AsyncScene* newAsyncSceneRawPtr = newAsyncScene.get();

  std::scoped_lock lock(_mutex);
  _asyncScenes.push_back(std::move(newAsyncScene));

  return newAsyncSceneRawPtr;
}

AsyncScene* ContextImpl::CreateAsyncScene(std::string_view name, Error& error) {
  return CreateAsyncSceneImpl(name, /*startPaused*/ false, error);
}

AsyncScene* ContextImpl::CreateAsyncScenePaused(std::string_view name, Error& error) {
  return CreateAsyncSceneImpl(name, /*startPaused*/ true, error);
}

void ContextImpl::DestroyAsyncScene(AsyncScene* scene) {
  std::unique_ptr<AsyncSceneImpl> victim;
  {
    std::scoped_lock lock(_mutex);
    auto it = std::ranges::find_if(_asyncScenes, [scene](auto& ptr) { return ptr.get() == scene; });
    if (it == _asyncScenes.end()) {
      return;
    }

    // Remove it from the list
    victim = std::move(*it);
    _asyncScenes.erase(it);
  }

  // Destroy it outside the lock
  victim.reset();
}

void ContextImpl::BindThisThread() {
  _scheduler->BindThisThread();
}

void ContextImpl::UnbindThisThread() {
  _scheduler->UnbindThisThread();
}

bool ContextImpl::IsSingleThreaded() const {
  return (_scheduler->GetNumThreads() == 0);
}

void ContextImpl::SetIsSingleThreaded(bool isSingleThreaded) {
  _scheduler->SetGlobalSingleThreadedMode(isSingleThreaded);
}

int ContextImpl::GetNumThreads() const {
  return _scheduler->GetNumThreads();
}

// Static API Method
bool Context::IsLogChannelEnabled(LogChannel channel) {
  return mochi::IsLogChannelEnabled(channel);
}

// Static API Method
void Context::EnableLogChannelInternal(LogChannel channel, bool enable) {
  mochi::EnableLogChannel(channel, enable);
}

// Static API Method
LogFn Context::GetLogCallback() {
  return mochi::GetLogCallback();
}

// Static API Method
void Context::SetLogCallbackInternal(LogFn callback) {
  mochi::SetLogCallback(callback);
}

// Static API Method
OnAssertFn Context::GetAssertionFailureCallback() {
  return mochi::GetAssertionFailureCallback();
}

namespace {

struct AuxiliaryMeshData {
  std::unique_ptr<TriangularMesh> mesh;
  std::unique_ptr<LinearMeshEmbedding> embedding;
};

AuxiliaryMeshData CreateAuxiliaryMeshData(std::optional<MeshData> const& data) {
  if (!data) {
    return {};
  }
  auto mesh = std::make_unique<TriangularMesh>(
      Unflatten<Real3 const>(MakeConstSpan(data->coordinates)),
      Unflatten<Int3 const>(MakeConstSpan(data->connectivity)));
  if (!data->skinning) {
    return {std::move(mesh), {}};
  }
  auto embedding = std::make_unique<LinearMeshEmbedding>(
      data->skinning->weightsPerNode,
      MakeConstSpan(data->skinning->indices),
      MakeConstSpan(data->skinning->weights));
  return {std::move(mesh), std::move(embedding)};
}

struct RodSurfaceData {
  std::shared_ptr<TriangularMesh const> mesh;
  std::shared_ptr<RodSurfaceEmbeddingData const> embedding;
};

RodSurfaceData CreateRodSurfaceData(
    std::optional<MeshData>& data,
    Span<Real3 const> rodNodes,
    Span<Real3 const> elementFrameAxes,
    bool isClosedLoop) {
  if (!data) {
    return {};
  }
  auto mesh = std::make_shared<TriangularMesh const>(
      DynamicArray<Real3>{Unflatten<Real3>(data->coordinates)},
      DynamicArray<Int3>{Unflatten<Int3>(data->connectivity)});
  if (!data->skinning) {
    return {std::move(mesh), {}};
  }
  auto embedding = std::make_shared<RodSurfaceEmbeddingData const>(ComputeRodSurfaceEmbedding(
      rodNodes, elementFrameAxes, *mesh, std::move(*data->skinning), isClosedLoop));
  return {std::move(mesh), std::move(embedding)};
}

} // namespace

// Static API Method
void Context::SetAssertionFailureCallbackInternal(OnAssertFn callback) {
  mochi::SetAssertionFailureCallback(callback);
}

ShapePtr ContextImpl::CreateShapeFromModelData(
    ModelData&& model,
    ExperimentalModelData&& experimental,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Implicit Box
  if (model.box) {
    auto obb = Obb{TransformRT{model.box->rotation, model.box->center}, model.box->halfExtents};
    return std::make_shared<ImplicitRigidShape>(obb);
  }

  // Implicit Plane
  if (model.plane) {
    return std::make_shared<ImplicitRigidShape>(*model.plane);
  }

  // Implicit Sphere:
  if (model.sphere) {
    return std::make_shared<ImplicitRigidShape>(*model.sphere);
  }

  // Else it must be a mesh (triangular or tetrahedral)
  MOCHI_ERROR_IF(!model.mesh.has_value(), error, "Model does not contain a supported shape type.");
  MOCHI_ERROR_RETURN(error, {});

  std::unique_ptr<SkinningData> skinning;
  if (model.mesh->skinning) {
    skinning = std::make_unique<SkinningData>(std::move(*model.mesh->skinning));
  }

  std::unique_ptr<ConstrainedNodesData> constrainedNodesData;
  if (model.constrainedNodes) {
    constrainedNodesData =
        std::make_unique<ConstrainedNodesData>(std::move(*model.constrainedNodes));
  }

  std::shared_ptr<GridSdf> gridSdf;
  if (model.sdf) {
    gridSdf = std::make_shared<GridSdf>(GridSdf::Create(std::move(*model.sdf)));
    model.sdf = std::nullopt;
  }

  // Build the blending map.
  std::shared_ptr<BlendingDataMap> shapeBlending;
  if (model.blending) {
    auto& meshBlending = *model.blending;
    shapeBlending = std::make_shared<BlendingDataMap>();
    for (auto& data : meshBlending) {
      shapeBlending->perSourceShapeData[data.sourceShape] = {
          std::move(data.indices), std::move(data.weights)};
    }
    shapeBlending->CopyMapToVectors();
  }

  if (model.mesh->nodesPerElement == 4) {
    // TODO: Move data instead of copying
    auto visualSurface = CreateAuxiliaryMeshData(model.visualMesh);
    auto contactSurface = CreateAuxiliaryMeshData(model.contactSkinMesh);
    auto tetMesh = std::make_unique<TetrahedralMesh>(
        Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates)),
        Unflatten<Int4 const>(MakeConstSpan(model.mesh->connectivity)));

    std::unique_ptr<PerElementSoftMaterialData> material;
    if (model.material) {
      material = std::make_unique<PerElementSoftMaterialData>(std::move(*model.material));
    }

    return std::make_shared<TetrahedralMeshShape>(
        std::move(tetMesh),
        std::move(skinning),
        std::move(constrainedNodesData),
        std::move(shapeBlending),
        std::move(visualSurface.mesh),
        std::move(visualSurface.embedding),
        std::move(contactSurface.mesh),
        std::move(contactSurface.embedding),
        std::move(gridSdf),
        std::move(experimental.romData),
        std::move(experimental.sampleMeshes),
        std::move(experimental.bshs),
        std::move(material));
  } else if (model.mesh->nodesPerElement == 3) {
    // TODO: Move data instead of copying
    auto visualSurface = CreateAuxiliaryMeshData(model.visualMesh);
    auto contactSurface = CreateAuxiliaryMeshData(model.contactSkinMesh);
    auto triMesh = std::make_unique<TriangularMesh>(
        Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates)),
        Unflatten<Int3 const>(MakeConstSpan(model.mesh->connectivity)));

    return std::make_shared<TriangularMeshShape>(
        std::move(triMesh),
        std::move(skinning),
        std::move(constrainedNodesData),
        std::move(shapeBlending),
        std::move(visualSurface.mesh),
        std::move(visualSurface.embedding),
        std::move(contactSurface.mesh),
        std::move(contactSurface.embedding),
        std::move(gridSdf));
  } else if (model.mesh->nodesPerElement == 2) {
    // TODO: Move data instead of copying
    auto nodes = DynamicArray<Real3>{Unflatten<Real3>(model.mesh->coordinates)};
    bool const isClosedLoop = IsPolylineClosedLoop(*model.mesh);
    GenerateMissingPolylineFrameAxes(model);
    auto frameAxes = DynamicArray<Real3>{Unflatten<Real3>(*model.elementFrameAxes)};

    auto visualSurface = CreateRodSurfaceData(
        model.visualMesh, MakeConstSpan(nodes), MakeConstSpan(frameAxes), isClosedLoop);
    auto contactSurface = CreateRodSurfaceData(
        model.contactSkinMesh, MakeConstSpan(nodes), MakeConstSpan(frameAxes), isClosedLoop);

    return std::make_shared<PolylineShape>(
        std::move(nodes),
        std::move(frameAxes),
        std::move(visualSurface.mesh),
        std::move(visualSurface.embedding),
        std::move(contactSurface.mesh),
        std::move(contactSurface.embedding),
        isClosedLoop);
  } else {
    MOCHI_ERROR_SET(
        error,
        "Mesh element size must be 2 for polyline, 3 for triangle, or 4 for tetrahedral mesh.");
    return {};
  }
}

ShapePtr ContextImpl::CreateShapeFromModelData(ModelData&& model, Error& error) {
  return CreateShapeFromModelData(std::move(model), ExperimentalModelData{}, error);
}

ShapeHandle ContextImpl::RegisterShape(ConstShapePtr newShape, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ASSERT(newShape != nullptr);

  // Allocate a unique handle (thread safe)
  ShapeHandle newHandle{GenerateNewHandle()};

  // Add it to the collection
  std::scoped_lock lock(_mutex);
  auto& shapeRef = _shapes[newHandle.value];
  MOCHI_ASSERT(shapeRef == nullptr, "ShapeHandle not unique!");
  shapeRef = std::move(newShape);

  // Enable auto-cleanup
  if (_enableAutomaticHandleCleanup) {
    newHandle._cleanup = std::make_shared<ShapeHandleAutoCleanup>(newHandle.value, _weakRefToThis);
  }

  return newHandle;
}

ConstShapePtr ContextImpl::GetShapeSharedPtr(ShapeHandle shapeHandle) const {
  std::scoped_lock lock(_mutex);
  auto it = _shapes.find(shapeHandle.value);
  return (it == _shapes.end()) ? ConstShapePtr{} : it->second;
}

template <typename MeshPtr>
static MeshDataView MakeMeshDataView(MeshPtr const& mesh) {
  MeshDataView view{};
  view.nodesPerElement = mesh->GetNumNodesPerElement();
  view.coordinates = Flatten(mesh->GetNodeCoordinates());
  view.connectivity = mesh->GetFlatConnectivity();
  return view;
}

static std::optional<SkinningDataView> MakeSkinningDataView(LinearMeshEmbedding const* embedding) {
  if (!embedding) {
    return std::nullopt;
  }
  SkinningDataView view;
  view.weightsPerNode = static_cast<int>(embedding->GetNumSkinningWeightsPerEntry());
  view.indices = embedding->GetIndices();
  view.weights = embedding->GetWeights();
  return view;
}

static std::optional<SkinningDataView> MakeSkinningDataView(
    RodSurfaceEmbeddingData const* embedding) {
  if (!embedding) {
    return std::nullopt;
  }
  SkinningDataView view;
  view.weightsPerNode = embedding->weightsPerNode;
  view.indices = embedding->elementIndices;
  view.weights = embedding->weights;
  return view;
}

static MeshDataView MakeAuxiliaryMeshDataView(
    TriangularMesh const* mesh,
    std::optional<SkinningDataView> const& skinning) {
  if (!mesh) {
    return {};
  }
  auto view = MakeMeshDataView(mesh);
  view.skinning = skinning;
  return view;
}

MeshDataView ContextImpl::GetShapeMesh(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  ConstShapePtr shapePtr = GetShapeSharedPtr(shape);
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot get shape mesh. Invalid shape handle.");
  MOCHI_ERROR_RETURN(error, {});

  if (auto const* tetmesh = dynamic_cast<TetrahedralMeshShape const*>(shapePtr.get())) {
    auto view = MakeMeshDataView(tetmesh->GetMesh());
    if (auto const& skinning = tetmesh->GetMeshSkinning()) {
      view.skinning = SkinningDataView{*skinning};
    }
    return view;
  }

  if (auto const* trimesh = dynamic_cast<TriangularMeshShape const*>(shapePtr.get())) {
    auto view = MakeMeshDataView(trimesh->GetMesh());
    if (auto const& skinning = trimesh->GetMeshSkinning()) {
      view.skinning = SkinningDataView{*skinning};
    }
    return view;
  }

  if (auto const* polyline = dynamic_cast<PolylineShape const*>(shapePtr.get())) {
    MeshDataView view{};
    view.nodesPerElement = 2;
    view.coordinates = Flatten(MakeConstSpan(polyline->GetNodes()));
    // Polyline connectivity carries the closed-loop information: 2*numNodes entries
    // for a closed loop, 2*(numNodes-1) for an open polyline.
    view.connectivity = polyline->GetFlatConnectivity();
    return view;
  }

  // Shape does not have a mesh.
  return {};
}

MeshDataView ContextImpl::GetShapeSurfaceMesh(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  ConstShapePtr shapePtr = GetShapeSharedPtr(shape);
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot get shape surface mesh. Invalid shape handle.");
  MOCHI_ERROR_RETURN(error, {});

  return shapePtr->GetSurfaceMeshData();
}

MeshDataView ContextImpl::GetShapeContactSkinMesh(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  ConstShapePtr shapePtr = GetShapeSharedPtr(shape);
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot get shape contact skin. Invalid shape handle.");
  MOCHI_ERROR_RETURN(error, {});

  TriangularMesh const* contactSkinPtr = nullptr;
  std::optional<SkinningDataView> skinning;

  if (auto const* tetmesh = dynamic_cast<TetrahedralMeshShape const*>(shapePtr.get())) {
    contactSkinPtr = tetmesh->GetContactSkin().get();
    skinning = MakeSkinningDataView(tetmesh->GetContactSkinEmbedding().get());
  } else if (auto const* trimesh = dynamic_cast<TriangularMeshShape const*>(shapePtr.get())) {
    contactSkinPtr = trimesh->GetContactSkin().get();
    skinning = MakeSkinningDataView(trimesh->GetContactSkinEmbedding().get());
  } else if (auto const* polyline = dynamic_cast<PolylineShape const*>(shapePtr.get())) {
    contactSkinPtr = polyline->GetContactSkin().get();
    skinning = MakeSkinningDataView(polyline->GetRodContactSkinEmbedding().get());
  }

  // Model validation rejects unreferenced contact-skin nodes.
  MOCHI_ASSERT_VERBOSE(
      !contactSkinPtr || contactSkinPtr->GetNumActiveNodes() == contactSkinPtr->GetNumNodes());
  return MakeAuxiliaryMeshDataView(contactSkinPtr, skinning);
}

MeshDataView ContextImpl::GetShapeVisualMesh(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  ConstShapePtr shapePtr = GetShapeSharedPtr(shape);
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot get shape visual mesh. Invalid shape handle.");
  MOCHI_ERROR_RETURN(error, {});

  TriangularMesh const* visualMeshPtr = nullptr;
  std::optional<SkinningDataView> skinning;

  if (auto const* tetmesh = dynamic_cast<TetrahedralMeshShape const*>(shapePtr.get())) {
    visualMeshPtr = tetmesh->GetVisualMesh().get();
    skinning = MakeSkinningDataView(
        dynamic_cast<LinearMeshEmbedding const*>(tetmesh->GetVisualEmbedding().get()));
  } else if (auto const* trimesh = dynamic_cast<TriangularMeshShape const*>(shapePtr.get())) {
    visualMeshPtr = trimesh->GetVisualMesh().get();
    skinning = MakeSkinningDataView(
        dynamic_cast<LinearMeshEmbedding const*>(trimesh->GetVisualEmbedding().get()));
  } else if (auto const* polyline = dynamic_cast<PolylineShape const*>(shapePtr.get())) {
    visualMeshPtr = polyline->GetVisualMesh().get();
    skinning = MakeSkinningDataView(polyline->GetRodVisualEmbedding().get());
  }

  return MakeAuxiliaryMeshDataView(visualMeshPtr, skinning);
}

Aabb ContextImpl::GetShapeAabb(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  ConstShapePtr shapePtr = GetShapeSharedPtr(shape);
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Not a valid shape handle");
  MOCHI_ERROR_RETURN(error, {});
  return GetAabb(shapePtr->GetBoundingVolume(error)); // Not all shapes have a bounding volume
}

Span<real const> ContextImpl::GetShapeElementFrameAxes(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  std::shared_ptr<PolylineShape const> shapePtr =
      std::dynamic_pointer_cast<PolylineShape const>(GetShapeSharedPtr(shape));
  MOCHI_ERROR_IF_NOT(
      shapePtr, error, "Cannot get shape element frame axes. Invalid polyline shape handle.");
  MOCHI_ERROR_RETURN(error, {});

  return Flatten(MakeConstSpan(shapePtr->GetElementFrameAxes()));
}

ArticulatedShapeInfo ContextImpl::GetArticulatedShapeInfo(ShapeHandle shape, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});

  // Look up the shape
  std::shared_ptr<ArticulatedBodyShape const> shapePtr =
      std::dynamic_pointer_cast<ArticulatedBodyShape const>(GetShapeSharedPtr(shape));
  MOCHI_ERROR_IF_NOT(
      shapePtr, error, "Cannot get articulated body description. Invalid shape parameter.");
  MOCHI_ERROR_RETURN(error, {});

  // Return articulated body description if the shape is an articulated body shape
  return shapePtr->GetArticulatedShapeInfo();
}

Scene* ContextImpl::GetScene(SceneHandle handle) {
  if (!handle.IsValid()) {
    return nullptr;
  }
  std::scoped_lock lock(_mutex);
  auto it =
      std::ranges::find_if(_scenes, [handle](auto& ptr) { return ptr->GetHandle() == handle; });
  if (it != _scenes.end()) {
    return (*it).get();
  }
  return nullptr;
}

Scene const* ContextImpl::GetScene(SceneHandle handle) const {
  // Shared implementation with non-const overload
  return const_cast<ContextImpl*>(this)->GetScene(handle);
}

bool ContextImpl::IsValidScene(Scene const* scene) const {
  if (!scene) {
    return false;
  }
  std::scoped_lock lock(_mutex);
  auto it = std::ranges::find_if(_scenes, [scene](auto& ptr) { return ptr.get() == scene; });
  return (it != _scenes.end());
}

bool ContextImpl::IsValidAsyncScene(AsyncScene const* scene) const {
  if (!scene) {
    return false;
  }
  std::scoped_lock lock(_mutex);
  auto it = std::ranges::find_if(_asyncScenes, [scene](auto& ptr) { return ptr.get() == scene; });
  return (it != _asyncScenes.end());
}

DebugServer& ContextImpl::GetDebugServer() {
  return *_debugServer;
}

DebugServer const& ContextImpl::GetDebugServer() const {
  return *_debugServer;
}

Context* CreateContext(int numWorkerThreads) {
  auto* mochiPhysics = new ContextImpl;
  if (numWorkerThreads >= 0) {
    mochiPhysics->SetNumWorkerThreads(numWorkerThreads);
  }
  mochiPhysics->Initialize();
  return mochiPhysics;
}

void DestroyContext(Context* mochiPhysicsInstance) {
  if (mochiPhysicsInstance) {
    auto* impl = assert_cast<ContextImpl*>(mochiPhysicsInstance);
    impl->PreShutDown();
    impl->ShutDown();
    delete impl;
  }
}

} // namespace mochi
