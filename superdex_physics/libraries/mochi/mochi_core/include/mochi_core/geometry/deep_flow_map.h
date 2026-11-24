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

#include <mochi_core/geometry/base_map.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>

#include <memory>
#include <utility>

namespace mochi {

/*
  Data handled by mapped querys. The mapping from deformed position to reference position is
  defined as ref = map(def, DoFs).
*/
struct MapQueryData {
  // Points and indices are owned by the query
  DynamicArray<Real3> _pointsLocal;
  DynamicArray<Real3> _pointsWorld;
  DynamicArray<int> _inds;
  // Dofs are owned by the map
  Span<real const> _dofs;
  // The result is just a view of the full result data
  Span<real const> _result;

  MapQueryData() = default;

  // The constructor takes ownership of the points and inds data, under the assumption that they are
  // created just for the query.
  MapQueryData(
      DynamicArray<Real3>&& pointsLocal,
      DynamicArray<Real3>&& pointsWorld,
      DynamicArray<int>&& inds,
      Span<real const> dofs)
      : _pointsLocal(std::move(pointsLocal)),
        _pointsWorld(std::move(pointsWorld)),
        _inds(std::move(inds)),
        _dofs(dofs) {}
};

using MapQueryPtr = std::shared_ptr<MapQueryData>;

/*
Base class to store the actual deep flow, possibly shared by multiple maps
The flow model ref = gamma(def, dofs) has sizes 3 (ref), 3 (def) and numDofs (dofs). Each model
gradient computation produces a vector of size gradSize = sizeof(Dref_i/Ddef) + sizeof(Dref_i/Ddofs)
= 3 + numDofs. The full data of a model query is of size dataSize = sizeof(ref) + sizeof(Dref/Ddef)
+ sizeof(Dref/Ddofs) = 3 + 3 * gradSize
*/
class DeepFlow {
 public:
  DeepFlow(int numDofs_, real scale_, Real3 shift_, bool computeGradient_)
      : numDofs(numDofs_),
        objFromLocalScale(scale_),
        objFromLocalShift(shift_),
        gradSize(computeGradient_ ? 3 + numDofs_ : 0),
        dataSize(3 + 3 * gradSize),
        computeGradient(computeGradient_) {}

  virtual ~DeepFlow() = default;

  void ClearQueries() {
    _result.clear();
    _queries.clear();
  }

  void RunQueries() {
    RunQueries(_queries, _result);
  }

  // RunQuery first creates a query and then runs it. It takes ownership of the data in points and
  // inds, under the assumption that this is temporary data needed only for the query.
  void RunQuery(
      DynamicArray<Real3>&& pointsLocal,
      DynamicArray<Real3>&& pointsWorld,
      DynamicArray<int>&& inds,
      Span<real const> dofs,
      DynamicArray<real>& outResult) {
    MapQueryPtr query = std::make_shared<MapQueryData>(
        std::move(pointsLocal), std::move(pointsWorld), std::move(inds), dofs);
    RunQueries(Span(&query, 1), outResult);
  }

  // Register a new query
  void RegisterNewQuery(MapQueryPtr query) {
    _queries.push_back(query);
  }

  int const numDofs; // Size of the deformation code.
  real const objFromLocalScale; // Scale from the flow's local frame to the object's frame.
  Real3 const objFromLocalShift; // Shift from the flow's local frame to the object's frame.
  int const gradSize; // Size of the gradient terms (3 + numDofs).
  int const dataSize; // Size of data per query point (3 + 3 * gradSize).
  bool const computeGradient; // Whether to compute the gradient of the outputs w.r.t. the inputs.

 protected:
  // Run a set of queries, described by the info in queries[].points and queries[].dofs, and write
  // the result to outResult. Also write views of the result to queries[].result
  virtual void RunQueries(Span<MapQueryPtr> queries, DynamicArray<real>& outResult) = 0;

  DynamicArray<MapQueryPtr> _queries; // Per-query data.
  DynamicArray<real> _result; // Scratch memory for the result of all queries.
};

/*
Class that implements a map from deformed position to reference position using a neural model. The
flow map is formally defined as ref = f(def, dofs). The result of the map must store Jacobians
dref/ddefT and ddef/dofs. The neural model returns gradients dref/ddef and dref/ddofs. To obtain
ddef/ddofs we consider a stationary reference position and apply the implicit function theorem to
the mapping, which yields dref/ddofs + dref/ddef * ddef/ddofs = 0 --> ddef/ddofs = - inv(dref/ddef)
* dref/ddofs.
*/
class DeepFlowMap : public BaseMap {
 public:
  std::shared_ptr<DeepFlow> flow = nullptr;

  DeepFlowMap(std::shared_ptr<DeepFlow> flowIn, real scale)
      : flow(flowIn), _scale(scale), _deformationDescriptor(flow->numDofs) {
    _numDoFs = flow->numDofs;
  }

  void MapPoints(
      Span<Real3 const> points,
      Span<int const> inds,
      DynamicArray<Real3>& outMappedPoints,
      DynamicArray<int>& outInds,
      DynamicArray<VMatrix3x3r>* outMapJac,
      DynamicArray<ColliderJacDofs>* outDofsJac) const override;

  void UpdateMap(Span<real const> dofs) override;

 private:
  void TransformResult(
      Span<real const> result,
      DynamicArray<Real3>& outPoints,
      DynamicArray<VMatrix3x3r>* outMapJac,
      DynamicArray<ColliderJacDofs>* outDofsJac) const;

  // The scale of the object from its default reference frame to the world. This is used for scaling
  // both the query points and the deformation descriptor.
  real const _scale;

  // Deformation
  DynamicArray<real> _deformationDescriptor;
};

// Create a DeepFlowMap given the DeepFlow implementation.
//
// Arguments:
//    deepFlow: the actual implementation of the deep flow
//    scaleDofs: Scale from world (simulation) space to the space used for training
//
std::unique_ptr<DeepFlowMap>
CreateDeepFlowMap(std::shared_ptr<DeepFlow> flow, real scaleDofs, Error& error);

/**
 * @brief Loads a @ref DeepFlow from a Mochi HDF5 model file.
 *
 * @param[in] h5FilePath Path to the HDF5 file containing the learned network.
 * @param[in] scale Scale from the network's local frame to the object's frame.
 * @param[in] shift Shift from the network's local frame to the object's frame.
 * @param[in] numDofs Size of the deformation descriptor, or zero if not deformable.
 * @param[out] error Error status. Check Error::IsOK() for success.
 * @param[in] computeGradient Whether to compute output gradients.
 * @return The loaded deep flow, or null on failure.
 *
 * @warning Requires `MOCHI_USE_HDF5=1`.
 *
 * @todo Points sampled outside the training sphere may produce incorrect results.
 */
std::shared_ptr<DeepFlow> LoadDeepFlow(
    char const* h5FilePath,
    real scale,
    Real3 shift,
    int numDofs,
    Error& error,
    bool computeGradient = true);

} // namespace mochi
