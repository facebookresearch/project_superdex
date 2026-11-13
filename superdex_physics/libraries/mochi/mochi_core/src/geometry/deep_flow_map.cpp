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
#include <mochi_core/utils/group_rw.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

namespace mochi {

static std::optional<ai::MlpLayer<real>> LoadMlpLayer(GroupReader& reader, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  DynamicArray<real> weightData;
  size_t weightDims[2] = {};
  reader.ReadDataSet("weight", weightData, weightDims, error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      weightDims[0] == 0 || weightDims[1] == 0 ||
          weightDims[0] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
          weightDims[1] > static_cast<size_t>(std::numeric_limits<int>::max()),
      error,
      "MLP layer weight dimensions must be positive and fit in an int.");
  MOCHI_ERROR_RETURN(error, {});

  DynamicArray<real> biasData;
  reader.ReadDataSet("bias", biasData, error);
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      biasData.size() != weightDims[0],
      error,
      "MLP layer bias size must match the weight output dimension.");
  MOCHI_ERROR_RETURN(error, {});

  Matrix<real> weights{RowMatrixView<real const>(
      weightData.data(), static_cast<int>(weightDims[0]), static_cast<int>(weightDims[1]))};
  ColumnVector<real> biases{ColumnVectorView<real const>(biasData.data(), isize(biasData))};

  MOCHI_ERROR_IF_NOT(
      reader.HasGroup("activation"), error, "MLP layer activation group is required.");
  MOCHI_ERROR_RETURN(error, {});

  auto activationGroup = reader.EnterGroup("activation", error);
  std::string kind;
  reader.ReadAttribute("kind", kind, error);
  MOCHI_ERROR_RETURN(error, {});
  if (kind == "elu") {
    real alpha = 0_r;
    reader.ReadAttribute("alpha", alpha, error);
    MOCHI_ERROR_RETURN(error, {});
    return ai::MlpLayer<real>{
        std::move(weights), std::move(biases), ai::ELUActivation<real>{alpha}};
  }
  MOCHI_ERROR_IF(kind != "identity", error, "Only ELU and identity activation are supported.");
  MOCHI_ERROR_RETURN(error, {});

  return ai::MlpLayer<real>{std::move(weights), std::move(biases), ai::IdentityActivation<real>{}};
}

static std::optional<ai::Mlp<real>>
LoadMlp(GroupReader& reader, int expectedInputDim, int expectedOutputDim, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  std::map<int, ai::MlpLayer<real>> layers;
  auto const groupNames = reader.GetGroupNames(error);
  MOCHI_ERROR_RETURN(error, {});
  for (auto const& groupName : groupNames) {
    auto layerGroup = reader.EnterGroup(groupName, error);
    std::string name;
    reader.ReadAttribute("name", name, error);
    MOCHI_ERROR_IF(name != "Linear", error, "Only linear MLP layers are supported.");
    MOCHI_ERROR_RETURN(error, {});

    int index = -1;
    reader.ReadAttribute("index", index, error);
    MOCHI_ERROR_IF(index < 0, error, "Invalid MLP layer index.");
    MOCHI_ERROR_RETURN(error, {});

    auto layer = LoadMlpLayer(reader, error);
    MOCHI_ERROR_RETURN(error, {});
    bool const inserted = layers.emplace(index, std::move(*layer)).second;
    MOCHI_ERROR_IF_NOT(inserted, error, "MLP layer indices must be unique.");
    MOCHI_ERROR_RETURN(error, {});
  }
  MOCHI_ERROR_IF(layers.empty(), error, "No MLP layers found.");
  MOCHI_ERROR_RETURN(error, {});

  DynamicArray<ai::MlpLayer<real>> layerArray;
  layerArray.reserve(layers.size());
  int expectedIndex = 0;
  int previousOutputDim = 0;
  for (auto& indexedLayer : layers) {
    auto const& layer = indexedLayer.second;
    MOCHI_ERROR_IF(
        indexedLayer.first != expectedIndex,
        error,
        "MLP layer indices must be contiguous and start at zero.");
    MOCHI_ERROR_IF(
        expectedIndex > 0 && layer.InputDim() != previousOutputDim,
        error,
        "Adjacent MLP layer dimensions are incompatible.");
    MOCHI_ERROR_RETURN(error, {});

    previousOutputDim = layer.OutputDim();
    layerArray.emplace_back(std::move(indexedLayer.second));
    ++expectedIndex;
  }
  MOCHI_ERROR_IF(
      layerArray.front().InputDim() != expectedInputDim,
      error,
      "Deep Flow MLP input dimension must equal 3 + numDofs.");
  MOCHI_ERROR_IF(
      layerArray.back().OutputDim() != expectedOutputDim,
      error,
      "Deep Flow MLP output dimension must equal 3.");
  MOCHI_ERROR_RETURN(error, {});
  return ai::Mlp<real>{std::move(layerArray)};
}

namespace {

class MochiDeepFlow final : public DeepFlow {
 public:
  MochiDeepFlow(
      ai::Mlp<real>&& network,
      int numDofs,
      real scale,
      Real3 shift,
      bool computeGradient = true)
      : DeepFlow(numDofs, scale, shift, computeGradient),
        _network(std::make_unique<ai::Mlp<real>>(std::move(network))) {}

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

} // namespace

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

  // If no points are passed, return empty outputs
  if (originalPoints.empty()) {
    outMappedPoints.clear();
    outInds.clear();
    if (outMapJac) {
      outMapJac->clear();
    }
    if (outDofsJac) {
      outDofsJac->clear();
    }
    return;
  }

  // Transform the input to fill the unit sphere that was used to train the network
  DynamicArray<Real3> points;
  points.resize_noinit(originalPoints.size());
  real totalScale = _scale * flow->objFromLocalScale;
  Real3 totalShift = _scale * flow->objFromLocalShift;
  std::ranges::transform(originalPoints, points.begin(), [&](auto& point) {
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

std::shared_ptr<mochi::DeepFlow> mochi::LoadDeepFlow(
    char const* h5FilePath,
    real scale,
    Real3 shift,
    int numDofs,
    Error& error,
    bool computeGradient) {
  MOCHI_ERROR_IF(h5FilePath == nullptr, error, "Deep Flow model path must not be null.");
  MOCHI_ERROR_IF(
      numDofs < 0 || numDofs > ColliderJacDofs::kMaxDoFs,
      error,
      "Deep Flow numDofs is outside the supported range.");
  MOCHI_ERROR_RETURN(error, {});
  auto reader = CreateGroupReaderHDF5(h5FilePath, error);
  MOCHI_ERROR_IF(!reader, error, "Failed to create a Deep Flow HDF5 reader.");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF_NOT(
      reader->HasGroup("deep_flow"), error, "Deep Flow model must contain a 'deep_flow' group.");
  MOCHI_ERROR_RETURN(error, {});

  auto deepFlowGroup = reader->EnterGroup("deep_flow", error);
  auto network = LoadMlp(*reader, 3 + numDofs, 3, error);
  MOCHI_ERROR_RETURN(error, {});
  return std::make_shared<MochiDeepFlow>(
      std::move(*network), numDofs, scale, shift, computeGradient);
}

std::unique_ptr<mochi::DeepFlowMap>
mochi::CreateDeepFlowMap(std::shared_ptr<DeepFlow> flow, real scaleDofs, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  return std::make_unique<DeepFlowMap>(flow, scaleDofs);
}
