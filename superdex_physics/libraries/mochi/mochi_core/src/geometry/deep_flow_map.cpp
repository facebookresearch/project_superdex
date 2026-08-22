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

#include <mochi_core/ai/mlp.h>
#include <mochi_core/geometry/deep_flow_map.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array_utils.h>

// The current implementation of DeepFlowMap requires libtorch
#if MOCHI_USE_TORCH

MOCHI_WARNING_PUSH()
MOCHI_WARNING_IGNORE_MSVC(4067 4244 4251 4267 4275 4324 4458 4522 4624 4702 4805 4996)
#include <torch/script.h>
#include <torch/torch.h>
MOCHI_WARNING_POP()

#if MOCHI_PLATFORM_WINDOWS
#include <Windows.h>
#endif

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

// Must come after torch includes because the compiler complains about ambiguous references to
// c10::Allocator vs mochi::Allocator
using namespace mochi;

namespace mochi {

// TODO: The current implementation assumes the Torch module is an MLP with ELU activations,
// except the last layer that has no activation.
static ai::Mlp<real> ToMochiMlp(torch::jit::script::Module const& module) {
  DynamicArray<Matrix<real>> weights;
  DynamicArray<ColumnVector<real>> biases;
  for (auto const& param : module.named_parameters()) {
    auto tensor = param.value.to(torch::kCPU);
    if (param.name.find("weight") != std::string::npos) {
      weights.emplace_back(
          RowMatrixView<real const>(tensor.data_ptr<real>(), tensor.size(0), tensor.size(1)));
    } else if (param.name.find("bias") != std::string::npos) {
      biases.emplace_back(ColumnVectorView<real const>(tensor.data_ptr<real>(), tensor.size(0)));
    }
  }

  int const numLayers = isize(weights);
  MOCHI_ASSERT(numLayers == biases.size(), "Inconsistent weight and bias sizes.");
  DynamicArray<ai::MlpLayer<real>> mlpLayers;
  mlpLayers.reserve(numLayers);
  for (int iLayer = 0; iLayer < numLayers; ++iLayer) {
    if (iLayer < (numLayers - 1)) {
      mlpLayers.emplace_back(
          std::move(weights[iLayer]), std::move(biases[iLayer]), ai::ELUActivation<real>());
    } else {
      mlpLayers.emplace_back(
          std::move(weights[iLayer]), std::move(biases[iLayer]), ai::IdentityActivation<real>());
    }
  }

  return ai::Mlp<real>(std::move(mlpLayers));
}

class TorchDeepFlow final : public DeepFlow {
 public:
  TorchDeepFlow(
      torch::jit::script::Module&& module,
      torch::Device device,
      int numDofs,
      real scale,
      Real3 shift,
      bool computeGradient = true,
      int maxPoints = kMochiSamplesPrealloc)
      : DeepFlow(numDofs, scale, shift, computeGradient),
        _maxPoints(maxPoints),
        _module(std::move(module)),
        _device(device) {
    _module.to(device);

    _objFromLocalShiftTensor =
        torch::tensor({objFromLocalShift[0], objFromLocalShift[1], objFromLocalShift[2]})
            .to(device);
    _objFromLocalScaleTensor = torch::tensor(objFromLocalScale).to(device);

    _gradOutputTemplate[0] = _gradOutputTemplate[0].to(_device);
    _gradOutputTemplate[1] = _gradOutputTemplate[1].to(_device);
    _gradOutputTemplate[2] = _gradOutputTemplate[2].to(_device);

    if (_device == torch::DeviceType::CUDA) {
      // Libtorch suffers a GPU memory leak when a module is queried with tensors of increasing
      // size. Every time a larger tensor is passed, the necessary memory is allocated, but previous
      // memory is not freed. As a preemptive measure, we run a very large query when the module is
      // loaded, to ensure that a sufficiently large cache is allocated.
      // TODO: Is this still true of CUDA12?
      DynamicArray<real> result;
      DynamicArray<Real3> pointsLocal(_maxPoints);
      DynamicArray<Real3> pointsWorld(_maxPoints);
      DynamicArray<int> inds(_maxPoints);
      DynamicArray<real> dofs(numDofs);
      RunQuery(std::move(pointsLocal), std::move(pointsWorld), std::move(inds), dofs, result);
    }
  }

  void RunQueries(Span<MapQueryPtr> queries, DynamicArray<real>& outResult) override;

 private:
  // For small queries, it is more convenient to obtain all three components of the gradient
  // together, by replicating the query points three times. For large queries, it is more convenient
  // to loop through the three components of the gradient. The threshold query size was determined
  // by running 'PerformanceTest' in deep_flow_map_test.cpp.
  static constexpr int kNumPointsThreshold = 5000;

  // Size of expected largest query, to preallocate GPU memory and avoid memory leaks
  int const _maxPoints;

  at::Tensor _objFromLocalScaleTensor; // Tensor form of objFromLocalScale
  at::Tensor _objFromLocalShiftTensor; // Tensor form of objFromLocalShift

  // Torch module
  torch::jit::script::Module _module;
  torch::Device _device = torch::kCPU;

  // Template directions for gradients
  DynamicArray<at::Tensor> _gradOutputTemplate = {
      torch::tensor({1.0, 0.0, 0.0}),
      torch::tensor({0.0, 1.0, 0.0}),
      torch::tensor({0.0, 0.0, 1.0})};
};

void TorchDeepFlow::RunQueries(Span<MapQueryPtr> queries, DynamicArray<real>& outResult) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_PROFILE_DESCRIPTION_F(
      "TorchDeepFlow::RunQueries with %d queries and %d points.\n",
      isize(queries),
      [](Span<MapQueryPtr const> queries) {
        int numPoints = 0;
        for (auto const& q : queries) {
          numPoints += isize(q->_pointsLocal);
        }
        return numPoints;
      }(queries));
  if (queries.empty()) {
    return;
  }

  // Concatenate input data for all queries
  DynamicArray<at::Tensor> inputs;
  inputs.reserve(queries.size());
  int numPoints = 0;
  for (int i = 0; i < queries.size(); i++) {
    auto const& query = queries[i];
    auto const& points = query->_pointsLocal;
    auto const& dofs = query->_dofs;
    int const numPointsThis = isize(points);

    // Torch needs a non-const pointer even though it doesn't modify the data. :-(
    real* pointsData = const_cast<real*>(Flatten(MakeSpan(points)).data());
    at::Tensor x =
        torch::from_blob(pointsData, {numPointsThis, 3}, torch::requires_grad(computeGradient));

    // Concatenate the deformation descriptor to the input
    real* dofsData = const_cast<real*>(dofs.data());
    at::Tensor dofsTensor =
        torch::from_blob(dofsData, {1, numDofs}, torch::requires_grad(computeGradient));
    at::Tensor descriptor = torch::tile(dofsTensor, {numPointsThis, 1});
    inputs.push_back(torch::cat({x, descriptor}, -1));

    numPoints += numPointsThis;
  }
  if (numPoints > _maxPoints) {
    MOCHI_LOG_WARNING(
        "Querying deep flow with %d points, larger than the maximum %d points of the preallocated GPU memory. This will likely turn into a GPU memory leak.",
        numPoints,
        _maxPoints);
  }
  at::Tensor input = torch::cat(inputs, 0).to(_device);

  // If the number of points is small, compute all gradients together
  if (numPoints < kNumPointsThreshold) {
    input = torch::cat({input, input, input}, 0);
  }

  // Module::forward is a non-const method because it could modify state in the generic case (e.g.
  // when we are training the network). During network inference, however, it should be immutable
  // and it should be safe to call from multiple threads.
  // TODO: However, the computational graph should be retained for backward passes, and this
  // is likely not thread-safe. If that's the case, we should wrap the forward and backward queries
  // in a mutex (ideally at GPU level).
  at::Tensor mapTensor = _module.forward({input}).toTensor();

  // Transform the mapped points from local to object coords.
  mapTensor = _objFromLocalScaleTensor * mapTensor + _objFromLocalShiftTensor;

  at::Tensor gpuData;
  if (computeGradient) {
    // Compute gradients with respect to full input (points and deformation code).
    if (numPoints < kNumPointsThreshold) {
      // Do it together for all coordinates of the gradient.
      at::Tensor gradOutput = torch::cat(
          {torch::tile(_gradOutputTemplate[0], {numPoints, 1}),
           torch::tile(_gradOutputTemplate[1], {numPoints, 1}),
           torch::tile(_gradOutputTemplate[2], {numPoints, 1})},
          0);
      at::Tensor gradTensor = torch::autograd::grad({mapTensor}, {input}, {gradOutput}, false)[0];

      // Select and pack the output data
      mapTensor = mapTensor.view({3, numPoints, 3}).select(0, 0);
      gradTensor = gradTensor.view({3, numPoints, gradSize});
      gpuData = torch::cat(
          {mapTensor, gradTensor.select(0, 0), gradTensor.select(0, 1), gradTensor.select(0, 2)},
          1);
    } else {
      // Do it separately for each coordinate of the gradient.
      std::array<at::Tensor, 3> gradTensor;
      for (int i = 0; i < 3; i++) {
        at::Tensor gradOutput = torch::tile(_gradOutputTemplate[i], {numPoints, 1});
        gradTensor[i] = torch::autograd::grad({mapTensor}, {input}, {gradOutput}, i != 2)[0];
      }

      // Pack the output data
      gpuData = torch::cat({mapTensor, gradTensor[0], gradTensor[1], gradTensor[2]}, 1);
    }
  } else {
    // Pack the output data without gradient
    gpuData = mapTensor;
  }

  // Copy the result to the CPU
  at::Tensor cpuData = gpuData.to(torch::kCPU);
  real* cpuDataPtr = cpuData.data_ptr<real>();
  outResult.assign(cpuDataPtr, cpuDataPtr + (numPoints * dataSize));

  // Organize result views per query
  int offset = 0;
  for (auto& query : queries) {
    int const numPointsThis = isize(query->_pointsLocal);
    query->_result = Span(&outResult[offset], numPointsThis * dataSize);
    offset += numPointsThis * dataSize;
  }
}

class MochiDeepFlow final : public DeepFlow {
 public:
  MochiDeepFlow(
      torch::jit::script::Module const& module,
      int numDofs,
      real scale,
      Real3 shift,
      bool computeGradient = true)
      : DeepFlow(numDofs, scale, shift, computeGradient) {
    _network = std::make_unique<ai::Mlp<real>>(ToMochiMlp(module));
  }

  void RunQueries(Span<MapQueryPtr> queries, DynamicArray<real>& outResult) override;

 private:
  std::unique_ptr<ai::Mlp<real>> _network = nullptr;
};

void MochiDeepFlow::RunQueries(Span<MapQueryPtr> queries, DynamicArray<real>& outResult) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_PROFILE_DESCRIPTION_F("MochiDeepFlow::RunQueries with %d queries.\n", isize(queries));
  if (queries.empty()) {
    return;
  }

  int numPoints = 0;
  for (auto const& query : queries) {
    numPoints += isize(query->_pointsLocal);
  }

  // Assemble input data for all queries. Use col-major storage to improve neural network
  // performance.
  Matrix<real> input(3 + numDofs, numPoints);
  Matrix<real> output(3, numPoints);
  Matrix<real> dOutput_dInput(computeGradient ? 3 * numPoints : 0, gradSize);
  int colOffset = 0;
  for (int i = 0; i < isize(queries); i++) {
    auto const& query = queries[i];
    int const numQueryPoints = isize(query->_pointsLocal);
    real const* pointsData = Flatten(MakeSpan(query->_pointsLocal)).data();
    input.template Block<3, krylov::kDynamic>(0, colOffset, 3, numQueryPoints) =
        RowMatrixView<real const>(pointsData, numQueryPoints, 3).Transpose();
    for (int j = 0; j < numQueryPoints; ++j) {
      input.template Block<krylov::kDynamic, 1>(3, colOffset + j, numDofs, 1) =
          RowVectorView<real const>(query->_dofs.data(), numDofs).Transpose();
    }
    colOffset += numQueryPoints;
  }

  if (computeGradient) {
    _network->ForwardAndJacobian(input, output, dOutput_dInput);
  } else {
    _network->Forward(input, output);
  }

  for (int j = 0; j < numPoints; ++j) {
    output(0, j) = objFromLocalScale * output(0, j) + objFromLocalShift[0];
    output(1, j) = objFromLocalScale * output(1, j) + objFromLocalShift[1];
    output(2, j) = objFromLocalScale * output(2, j) + objFromLocalShift[2];
  }

  if (computeGradient) {
    dOutput_dInput *= objFromLocalScale;
  }

  // Store result into the output vector.
  outResult.resize_noinit(dataSize * numPoints);
  MatrixView<real> fullResult(outResult.data(), dataSize, numPoints);
  fullResult.template TopRows<3>(3) = output;
  if (computeGradient) {
    for (int iPoint = 0; iPoint < numPoints; ++iPoint) {
      for (int iDim = 0; iDim < 3; ++iDim) {
        fullResult.template Block<krylov::kDynamic, 1>(3 + iDim * gradSize, iPoint, gradSize, 1) =
            dOutput_dInput.Row(iPoint * 3 + iDim).Transpose();
      }
    }
  }

  // Store per-query results.
  size_t offset = 0;
  for (auto& query : queries) {
    auto const queryResultSize = query->_pointsLocal.size() * dataSize;
    query->_result = Span(&outResult[offset], queryResultSize);
    offset += queryResultSize;
  }
}

void DeepFlowMap::UpdateMap(Span<real const> dofs) {
  if (dofs.size() != _numDoFs) {
    MOCHI_LOG_WARNING("DeepFlowMap cannot set deformation. Incorrect number of DOF values.");
    return;
  }

  for (int i = 0; i < _numDoFs; i++) {
    _deformationDescriptor[i] = dofs[i] / _scale;
  }
}

void DeepFlowMap::MapPoints(
    Span<Real3 const> originalPoints,
    Span<int const> originalInds,
    BvhTree<Aabb> const* /*pointBvh*/,
    DynamicArray<Real3>& outMappedPoints,
    DynamicArray<int>& outInds,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) const {
  MOCHI_PROFILE_SCOPE();

  // If no points are passed, return
  if (originalPoints.empty()) {
    return;
  }

  // Transform the input to fill the unit sphere that was used to train the network
  DynamicArray<Real3> points;
  points.resize_noinit(originalPoints.size());
  real totalScale = _scale * flow->objFromLocalScale;
  Real3 totalShift = _scale * flow->objFromLocalShift;
  std::transform(originalPoints.begin(), originalPoints.end(), points.begin(), [&](auto& point) {
    return (point - totalShift) / totalScale;
  });

  DynamicArray<int> inds(originalInds.begin(), originalInds.end());
  DynamicArray<Real3> originalPointsCopy(originalPoints.begin(), originalPoints.end());
  outInds = inds;
  DynamicArray<real> result;
  flow->RunQuery(
      std::move(points),
      std::move(originalPointsCopy),
      std::move(inds),
      _deformationDescriptor,
      result);
  TransformResult(result, outMappedPoints, outMapJac, outDofsJac);
}

void DeepFlowMap::TransformResult(
    Span<real const> result,
    DynamicArray<Real3>& outPoints,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) const {
  // Transform the data
  int const numPoints = isize(result) / flow->dataSize;

  // There are 3 relevant reference frames:
  // - world: where the (possibly scaled) object lives
  // - obj: the object's default reference frame
  // - local: where the network lives
  // world = scaleObj * obj; obj = 1/scaleObj * world
  // obj = scaleFlow * local + shiftFlow; local = 1/scaleFlow * (obj - shiftFlow)
  //
  // The network infers ref_obj = flow(def_local, dofs_obj). Therefore, Autograd produces
  // the gradients dref_obj/ddef_local and dref_obj/ddofs_obj.
  //
  // The result points are:
  // ref_world = scaleObj * ref_obj
  //
  // The result Jacobians are:
  // dref_world/ddef_world^T = 1/scaleFlow * dref_obj/ddef_local^T
  // ddef_world/ddofs_world = - inv(dref_world/ddef_world) * dref_obj/ddofs_obj
  MOCHI_ASSERT(
      ColliderJacDofs::kMaxDoFs >= _numDoFs, "ColliderJacDofs::kMaxDoFs is not sufficiently large");
  outPoints.resize_noinit(numPoints);
  if (outMapJac) {
    outMapJac->resize_noinit(numPoints);
  }
  if (outDofsJac) {
    outDofsJac->resize_noinit(numPoints);
  }
  // For each query point, 'result' includes the following data, in this order: [ref, Dref_x/Ddef,
  // Dref_x/Ddofs, Dref_y/Ddef, Dref_y/Ddofs, Dref_z/Ddef, Dref_z/Ddofs]. This is of size dataSize,
  // hence 'offi' strides over the data of each query point. gradSize allows striding over
  // components of the gradient.
  auto transformFunc = [&, this](int i) {
    int offi = i * flow->dataSize;
    outPoints[i] = _scale * Real3(result[offi], result[offi + 1], result[offi + 2]);

    if (outMapJac || outDofsJac) {
      MOCHI_ASSERT(flow->computeGradient);
      VMatrix3x3r dref_ddef;
      dref_ddef[0] = Load<3, Vec4r>(&result[offi + 3]);
      dref_ddef[1] = Load<3, Vec4r>(&result[offi + 3 + flow->gradSize]);
      dref_ddef[2] = Load<3, Vec4r>(&result[offi + 3 + 2 * flow->gradSize]);
      dref_ddef = dref_ddef / flow->objFromLocalScale;
      if (outMapJac) {
        (*outMapJac)[i] = dref_ddef;
      }
      if (outDofsJac) {
        VMatrix3x3r inv_dref_ddefT = Invert3x3(Transpose3x3(dref_ddef));
        for (int j = 0, offij = offi + 6; j < _numDoFs; j++, offij++) {
          Vec4r neg_dref_ddof(
              -result[offij], -result[offij + flow->gradSize], -result[offij + 2 * flow->gradSize]);
          (*outDofsJac)[i].jac[j] = DotVecMat3x3(neg_dref_ddof, inv_dref_ddefT);
        }
        std::iota((*outDofsJac)[i].inds.begin(), (*outDofsJac)[i].inds.begin() + _numDoFs, 0);
      }
    }
  };
  ParallelForN("Deep-flow result transformation", numPoints, 5000, transformFunc);
}

} // namespace mochi

#endif // MOCHI_USE_TORCH

std::shared_ptr<mochi::DeepFlow> mochi::LoadDeepFlow(
    [[maybe_unused]] char const* torchFilePath,
    [[maybe_unused]] real scale,
    [[maybe_unused]] Real3 shift,
    [[maybe_unused]] int numDofs,
    [[maybe_unused]] NeuralComputeType computeType,
    [[maybe_unused]] int preallocMemSize,
    Error& error,
    [[maybe_unused]] bool computeGradient) {
  MOCHI_ERROR_RETURN(error, {});
#if MOCHI_USE_TORCH
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(torchFilePath);
  } catch (c10::Error const&) {
    MOCHI_ERROR_SET(error, "Failed to load Deep flow map from torch module file");
    return {};
  }
  if (computeType == NeuralComputeType::TorchCpu || computeType == NeuralComputeType::TorchGpu) {
    torch::Device device = torch::kCPU;
    if (computeType == NeuralComputeType::TorchGpu) {
      if (torch::cuda::is_available()) {
        device = torch::kCUDA;
      } else {
        MOCHI_LOG_WARNING("CUDA is NOT available for Flow Map evaluations. Using CPU instead.");
      }
    }
    return std::make_shared<TorchDeepFlow>(
        std::move(module), device, numDofs, scale, shift, computeGradient, preallocMemSize);
  } else {
    static_assert(
        static_cast<int>(NeuralComputeType::Count) == 3,
        "Please update this if statement if other neural compute types are added");
    MOCHI_ASSERT(computeType == NeuralComputeType::MochiCpu, "Unexpected deep flow compute type.");
    return std::make_shared<MochiDeepFlow>(module, numDofs, scale, shift, computeGradient);
  }
#else
  MOCHI_ERROR_SET(
      error,
      "DeepFlow requires libtorch. Compile Mochi with MOCHI_USE_TORCH=1 to enable this feature.");
  return {};
#endif
}

std::unique_ptr<mochi::DeepFlowMap> mochi::CreateDeepFlowMap(
    [[maybe_unused]] std::shared_ptr<DeepFlow> flow,
    [[maybe_unused]] real scaleDofs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
#if MOCHI_USE_TORCH
  return std::make_unique<DeepFlowMap>(flow, scaleDofs);
#else
  MOCHI_ERROR_SET(
      error,
      "DeepFlowMap requires libtorch. Compile Mochi with MOCHI_USE_TORCH=1 to enable this feature.");
  return {};
#endif
}
