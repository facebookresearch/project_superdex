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

#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/elements/tetrahedral/finite_element_trace.h>
#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/blending.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/overload_visitor.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <variant>
#include <vector>

/**************************************************************************************************
 * All the classes in this file are used for the computation of contact Jacobians dy/dq, where y are
 * positions of contact points, and q is the state. y(q) is defined as a composition of
 * differentiable maps, and then dy/dq can be obtained by multiplying Jacobians of the various
 * maps. The differentiable maps should be volatile objects created on-the-fly for the evaluation of
 * Jacobians; they are not intended to persists as components, because they use pointers to memory
 * for efficiency and ease of implementation.
 */

namespace mochi::dmap {

/**************************************************************************************************
 * Virtual class that defines the interface of a differentiable map y = g(q, x), where 'y' are
 * output points, 'q' is the state, and 'x' are optional input points. If present, 'x' are the
 * result of some other differentiable map, x = h(q).
 */
class DMapImpl {
 public:
  // Helper structure to store remapped indices. For output indices 'indsY', it stores information
  // about the corresponding input indices 'indsX'.
  struct Remapping {
    // Span to the indices of the input. It points to 'indsXData' if remapping is necessary,
    // otherwise it points to the indices of the output 'indsY'.
    Span<int const> indsX;
    // Vector to store input indices. Filled only if remapping is necessary.
    std::vector<int> indsXData;
    // Map from output indices to input indices. Filled only if remapping is necessary. Each DMap
    // must know how to utilize this map.
    std::vector<int> map;
  };

  virtual ~DMapImpl() = default;

  // Given a set of output points defined by indices 'indsY', compute their Jacobian wrt the state
  // slice of this map. The default implementation does nothing, for maps that don't own a state
  // slice.
  virtual void WriteJacobianSlice(Span<int const> /* indsY */, Span<ContactJac> /* outJacsY */)
      const {}

  // Given a set of output points defined by indices 'indsY', find the indices of necessary input
  // points. The result is stored in 'outRemap'. The default implementation assumes that no
  // remapping is necessary, and makes 'indsX' = 'indsY'.
  virtual void RemapIndices(Span<int const> indsY, Remapping& outRemap) const {
    outRemap.indsX = indsY;
  }

  // Given a set of output points defined by indices 'indsY', propagate the Jacobian of input points
  // 'jacX', resulting in 'outJacY'. The implementation may use the result of index remapping
  // 'remap'. The default implementation does nothing, for maps that don't take input points 'x'.
  virtual void PropagateJacobianSlice(
      Remapping const& /* remap */,
      Span<int const> /* indsY */,
      ContactJac const& /* jacX */,
      ContactJac& /* outJacY */) const {}
};

/**************************************************************************************************
 * Recursive differentiable map conditioned by state. It describes a function y = g(q, h(q)), with
 * 'q' the state, 'y' a span of points, and 'h' another optional differentiable map. The state 'q'
 * may be split into slices {qi}, and then each differentiable map implementation must know its
 * governing slice 'qi'. Jacobians are split according to state slices. This class takes the
 * implementation of 'g', 'h' and all subsequent differentiable maps as variadic template arguments.
 * All those differentiable maps must be derived from DMapImpl.
 */

// Forward declaration
template <typename Impl, typename... Impls>
class DMap;

// Specialization with one template argument (non-recursive implementation)
template <typename Impl>
class DMap<Impl> {
  static_assert(std::is_base_of_v<DMapImpl, Impl>);

 public:
  DMap(Impl const* g) : _g(g) {}

  void GetJac(Span<int const> indsY, Span<ContactJac> outJacsY) const {
    // Write dg/dq, the Jacobian of this slice.
    _g->WriteJacobianSlice(indsY, outJacsY);
  }

 private:
  Impl const* _g = nullptr;
};

// Specialization with two or more template arguments (recursive implementation)
template <typename Impl1, typename Impl2, typename... Impls>
class DMap<Impl1, Impl2, Impls...> {
  static_assert(std::is_base_of_v<DMapImpl, Impl1>);

 public:
  template <typename... Args>
  DMap(Impl1 const* g, Args... h) : _g(g), _h(h...) {}

  void GetJac(Span<int const> indsY, Span<ContactJac> outJacsY) const {
    // Write dg/dq, the Jacobian of this slice.
    _g->WriteJacobianSlice(indsY, outJacsY);

    // Obtain output indices for h() by remapping indsY
    DMapImpl::Remapping remap;
    _g->RemapIndices(indsY, remap);

    // Get dh/dq
    MOCHI_FILO_STACK_ALLOCATOR(tempAlloc, sizeof(ContactJac) * 128); // Probably more than enough
    DynamicArray<ContactJac> dhdq(outJacsY.size(), &tempAlloc);
    _h.GetJac(remap.indsX, dhdq);

    // Propagate each slice of dh/dq
    for (int slice = 0; slice < dhdq.size(); slice++) {
      if (dhdq[slice].nContacts > 0) {
        _g->PropagateJacobianSlice(remap, indsY, dhdq[slice], outJacsY[slice]);
      }
    }
  }

 private:
  Impl1 const* _g = nullptr;
  DMap<Impl2, Impls...> _h;
};

/**************************************************************************************************
 * Differentiable map for a constant rigid transform. It is initialized with the transform. This
 * class is used for the root transform of soft actors.
 */
class DMapRTConst final : public DMapImpl {
 public:
  DMapRTConst(TransformRT const& transform) : _transform(transform) {}

  void PropagateJacobianSlice(
      Remapping const& remap,
      Span<int const> indsY,
      ContactJac const& jacX,
      ContactJac& outJacY) const override {
    MOCHI_ASSERT(remap.indsX.size() == indsY.size(), "Sizes of indices don't match");

    // Initialize the output Jacobian
    outJacY.Resize(
        jacX.hasSharedDoFs, jacX.hasSharedJacs, jacX.nDoFsInternal, jacX.nDoFsState, indsY.size());

    // Propagate partial derivatives wrt input. Consider the case where all jacX are the same.
    auto rodrig3x3 = Rodrigues(_transform.GetRotation().VToRotationVector());
    for (int i = 0; i < (outJacY.hasSharedJacs ? 1 : outJacY.nContacts); i++) {
      outJacY.Jac(i) = AsMatrixView(rodrig3x3) * jacX.Jac(i);
    }

    // Set DoF indices. Consider the case where all DoFs are the same.
    for (int i = 0; i < (outJacY.hasSharedDoFs ? 1 : outJacY.nContacts); i++) {
      std::copy(jacX.Inds(i).begin(), jacX.Inds(i).end(), outJacY.Inds(i).begin());
    }
  }

 private:
  /* Data members */
  TransformRT const& _transform;
};

/**************************************************************************************************
 * Differentiable map for a rigid transform for which the output points are known. It is initialized
 * with the state slice, the state, the DoF offset, and the local position of the center of mass.
 * Optionally, it receives an extra Jacobian and relative DoF indices. The method SetData() must be
 * called before computing the Jacobian, to pass the output points (expressed wrt the root
 * transform). This class is used for collider rigid and articulated-rigid actors, and it maps
 * world-space query points to the rigid collider's local frame.
 */
class DMapRTOutput final : public DMapImpl {
 public:
  DMapRTOutput(
      int slice,
      TransformRT const& state,
      int offset,
      Vec4r com,
      MatrixView<real const> jacAuxPersistentView = {}, // The owner must outlive the Jacobians.
      Span<int const> dofsAux = {})
      : _slice(slice),
        _state(state),
        _offset(offset),
        _com(com),
        _jacAux(jacAuxPersistentView),
        _dofsAux(dofsAux) {
    MOCHI_ASSERT(_jacAux.empty() == _dofsAux.empty(), "Inconsistent optional arguments");
  }

  void SetData(ContactDetectionResult const* query) {
    _query = query;
  }

  void WriteJacobianSlice(Span<int const> /* indsY */, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(_query, "Data is not set");
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    auto& outJacY = outJacsY[_slice];

    int stateSize = _dofsAux.empty() ? RigidSize::kDAll : isize(_dofsAux);
    outJacY.Resize(true, false, RigidSize::kDAll, stateSize, isize(_query->posColliding));

    // Compute partial derivatives wrt state
    VMatrix3x3r rotT = ToVMatrix3x3Transpose(_state.GetRotation());
    Matrix<real, 3, 3> jacTrans = AsMatrixView(-rotT);
    for (int i = 0; i < outJacY.nContacts; i++) {
      outJacY.Jac(i).template LeftCols<3>(3) = jacTrans;
      auto radiusVecLocal = ToSimd(_query->posColliding[i]) - _com;
      outJacY.Jac(i).template MiddleCols<3>(3, 3) =
          AsMatrixView(lie::DMultRotTVecDRot(rotT, radiusVecLocal));
    }

    // Possibly add aux Jacobian
    if (!_jacAux.empty()) {
      outJacY.SetJacAuxView(_jacAux);
    }

    // Set DoF indices
    if (_dofsAux.empty()) {
      std::iota(outJacY.Inds(0).begin(), outJacY.Inds(0).end(), _offset);
    } else {
      for (int i = 0; i < _dofsAux.size(); ++i) {
        outJacY.Inds(0)[i] = _offset + _dofsAux[i];
      }
    }
  }

 private:
  int const _slice; // State slice
  TransformRT const& _state; // Ref to the state
  int const _offset; // Offset for the DoFs
  Vec4r const _com; // Local position of the center of mass
  MatrixView<real const> _jacAux{}; // Jacobian of rigid state wrt actual state (optional)
  Span<int const> _dofsAux{}; // DoF indices of actual state (optional)
  ContactDetectionResult const* _query =
      nullptr; // Contact detection result including contact points
};

/**************************************************************************************************
 * Differentiable map for a rigid transform for which the input points are known. It is initialized
 * with the state slice, the state, the root transform, the DoF offset, the input set, and Jacobians
 * mapping world positions to the collider's local frame. Optionally, it receives an extra Jacobian
 * and relative DoF indices. For convenience, the input is expressed wrt the root transform, which
 * is not the same as the state. A translation is applied to remap the input wrt the state. This
 * class is used for colliding rigid and articulated-rigid actors.
 */
class DMapRTInput final : public DMapImpl {
 public:
  DMapRTInput(
      int slice,
      TransformRT const& state,
      TransformRT const& transform,
      int offset,
      Span<Real3 const> input,
      Span<VMatrix3x3r const> toColliderJacs,
      MatrixView<real const> jacAuxPersistentView = {}, // The owner must outlive the Jacobians.
      Span<int const> dofsAux = {})
      : _slice(slice),
        _state(state),
        _transform(transform),
        _offset(offset),
        _input(input),
        _toColliderJacs(toColliderJacs),
        _jacAux(jacAuxPersistentView),
        _dofsAux(dofsAux) {
    MOCHI_ASSERT(_jacAux.empty() == _dofsAux.empty(), "Inconsistent optional arguments");
  }

  void WriteJacobianSlice(Span<int const> indsY, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    MOCHI_ASSERT(
        _toColliderJacs.size() == 1 || _toColliderJacs.size() == indsY.size(),
        "Inconsistent to-collider Jacobians");
    if (indsY.empty()) {
      return;
    }
    auto& outJacY = outJacsY[_slice];

    int stateSize = _dofsAux.empty() ? RigidSize::kDAll : isize(_dofsAux);
    outJacY.Resize(true, false, RigidSize::kDAll, stateSize, isize(indsY));

    // Compute partial derivatives wrt state
    auto rotT = ToVMatrix3x3Transpose(_transform.GetRotation());
    auto delta = _transform.VGetTranslation() - _state.VGetTranslation();
    for (int i = 0, l = 0, dl = (_toColliderJacs.size() == 1 ? 0 : 1); i < outJacY.nContacts;
         i++, l += dl) {
      outJacY.Jac(i).template LeftCols<3>(3) = AsMatrixView(_toColliderJacs[l]);
      auto radiusVec = DotVecMat3x3(ToSimd(_input[indsY[i]]), rotT) + delta;
      outJacY.Jac(i).template MiddleCols<3>(3, 3) =
          AsMatrixView(Dot3x3(_toColliderJacs[l], lie::DMultRotVecDRot(radiusVec)));
    }

    // Possibly add aux Jacobian
    if (!_jacAux.empty()) {
      outJacY.SetJacAuxView(_jacAux);
    }

    // Set DoF indices
    if (_dofsAux.empty()) {
      std::iota(outJacY.Inds(0).begin(), outJacY.Inds(0).end(), _offset);
    } else {
      for (int i = 0; i < _dofsAux.size(); ++i) {
        outJacY.Inds(0)[i] = _offset + _dofsAux[i];
      }
    }
  }

 private:
  int const _slice;
  TransformRT const& _state;
  TransformRT const& _transform;
  int const _offset;
  Span<Real3 const> _input{};
  Span<VMatrix3x3r const> _toColliderJacs;
  MatrixView<real const> _jacAux{};
  Span<int const> _dofsAux{};
};

/**************************************************************************************************
 * Differentiable map used in sync rigid contact. It does not store per-contact Jacobians, only the
 * DoFs and the optional shared Jacobian.
 */
class DMapSyncRigid final : public DMapImpl {
 public:
  DMapSyncRigid(
      int slice,
      int offset,
      MatrixView<real const> jacAuxPersistentView = {}, // The owner must outlive the Jacobians.
      Span<int const> dofsAux = {})
      : _slice(slice), _offset(offset), _jacAux(jacAuxPersistentView), _dofsAux(dofsAux) {
    MOCHI_ASSERT(_jacAux.empty() == _dofsAux.empty(), "Inconsistent optional arguments");
  }

  void WriteJacobianSlice(Span<int const> indsY, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    auto& outJacY = outJacsY[_slice];

    int stateSize = _dofsAux.empty() ? RigidSize::kDAll : isize(_dofsAux);
    outJacY.Resize(true, true, RigidSize::kDAll, stateSize, isize(indsY));

    // Possibly add aux Jacobian
    if (!_jacAux.empty()) {
      outJacY.SetJacAuxView(_jacAux);
    }

    // Set DoF indices
    if (_dofsAux.empty()) {
      std::iota(outJacY.Inds(0).begin(), outJacY.Inds(0).end(), _offset);
    } else {
      for (int i = 0; i < _dofsAux.size(); ++i) {
        outJacY.Inds(0)[i] = _offset + _dofsAux[i];
      }
    }
  }

 private:
  int const _slice; // State slice
  int const _offset; // Offset for the DoFs
  MatrixView<real const> _jacAux{}; // Jacobian of rigid state wrt actual state (optional)
  Span<int const> _dofsAux{}; // DoF indices of actual state (optional)
};

/**************************************************************************************************
 * Differentiable map for a ROM. It is initialized with the state slice, the ROM Jacobian and the
 * DoF offset. This class is used for colliding ROMs.
 */
class DMapRom final : public DMapImpl {
  using dense_t = RowMatrix<real>;

 public:
  // Keeping this as a variant because we may want to support sparse matrices here too
  using VariantJacobian = std::variant<dense_t>;

  DMapRom(int slice, VariantJacobian const& jac, int offset)
      : _slice(slice), _jac(jac), _offset(offset) {}

  void WriteJacobianSlice(Span<int const> indsY, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    if (indsY.empty()) {
      return;
    }
    auto& outJacY = outJacsY[_slice];

    int ndofs = std::visit([](auto const& mat) { return mat.Cols(); }, _jac);
    outJacY.Resize(true, false, ndofs, ndofs, indsY.size());

    // Compute derivatives
    std::visit(
        OverloadVisitor{
            [&](dense_t const& dense) {
              for (int i = 0; i < outJacY.nContacts; i++) {
                outJacY.Jac(i) = dense.template MiddleRows<3>(3 * indsY[i], 3);
              }
            },
        },
        _jac);

    // Set DoF indices
    std::iota(outJacY.Inds(0).begin(), outJacY.Inds(0).end(), _offset);
  }

 private:
  int const _slice;
  VariantJacobian const& _jac;
  int const _offset;
};

/**************************************************************************************************
 * Differentiable map for a surface quadrature. It is initialized with the discretization info and
 * Jacobians mapping world positions to the collider's local frame. This class is used as the final
 * map for all colliding deforming actors.
 */
template <typename ElementT>
class DMapQuad final : public DMapImpl {
 public:
  DMapQuad(Span<ElementT const> femElements, Span<VMatrix3x3r const> toColliderJacs)
      : _femElements(femElements), _toColliderJacs(toColliderJacs) {}

  void RemapIndices(Span<int const> indsY, Remapping& outRemap) const override {
    MOCHI_ASSERT_VERBOSE(outRemap.map.empty() && outRemap.indsXData.empty());

    // Compute all input indices, with repetition.
    outRemap.indsXData.reserve(indsY.size() * ElementT::kNumNodes);
    int maxIndX = -1;
    for (int indY : indsY) {
      for (int indX : GetNodeInfo(_femElements, indY)) {
        outRemap.indsXData.push_back(indX);
        maxIndX = Max(maxIndX, indX);
      }
    }

    // Remove duplicate input indices and compute the map.
    MOCHI_FILO_STACK_ALLOCATOR(allocator, 8192 * sizeof(int)); // Up to 8192 boundary nodes
    DynamicArray<int> mapHelper(maxIndX + 1, -1, &allocator);
    outRemap.map.reserve(outRemap.indsXData.size());
    int numUniqueIndsX = 0;
    for (int indX : outRemap.indsXData) {
      if (mapHelper[indX] == -1) {
        mapHelper[indX] = numUniqueIndsX;
        outRemap.map.push_back(numUniqueIndsX);
        outRemap.indsXData[numUniqueIndsX] = indX;
        numUniqueIndsX++;
      } else {
        outRemap.map.push_back(mapHelper[indX]);
      }
    }
    outRemap.indsXData.resize(numUniqueIndsX);
    outRemap.indsX = MakeConstSpan(outRemap.indsXData);
  }

  void PropagateJacobianSlice(
      Remapping const& remap,
      Span<int const> indsY,
      ContactJac const& jacX,
      ContactJac& outJacY) const override {
    MOCHI_ASSERT(
        _toColliderJacs.size() == 1 || _toColliderJacs.size() == indsY.size(),
        "Inconsistent to-collider Jacobians");
    auto const& indsX = remap.map;
    MOCHI_ASSERT(
        indsX.size() == ElementT::kNumNodes * indsY.size(), "Sizes of indices don't match");

    // Initialize the output Jacobian
    int nDoFMultiplier = jacX.hasSharedDoFs ? 1 : ElementT::kNumNodes;
    outJacY.Resize(
        jacX.hasSharedDoFs,
        false,
        nDoFMultiplier * jacX.nDoFsInternal,
        nDoFMultiplier * jacX.nDoFsState,
        indsY.size());

    // Propagate partial derivatives wrt input. If the DoFs are shared, add the weighted input of
    // all ElementT::kNumNodes points. If the DoFs are not shared, write the weighted input of
    // each point to different columns. If both the incoming Jacobians and toColliderJacs are
    // shared, compute their product only once.
    int dstDelta = outJacY.hasSharedDoFs ? 0 : jacX.nDoFsInternal;
    if (_toColliderJacs.size() == 1 && jacX.hasSharedJacs) {
      Matrix<real, 3> jacProduct = AsMatrixView(_toColliderJacs[0]) * jacX.Jac(0);
      outJacY.SetZero();
      for (int i = 0; i < outJacY.nContacts; i++) {
        NdArray<real, ElementT::kNumNodes> const weights = GetWeightInfo(_femElements, indsY[i]);
        for (int k = 0, dst = 0; k < ElementT::kNumNodes; k++, dst += dstDelta) {
          outJacY.Jac(i).MiddleCols(dst, jacX.nDoFsInternal) += weights[k] * jacProduct;
        }
      }
    } else {
      outJacY.SetZero();
      for (int i = 0, j = 0, l = 0, dl = (_toColliderJacs.size() == 1 ? 0 : 1);
           i < outJacY.nContacts;
           i++, l += dl) {
        NdArray<real, ElementT::kNumNodes> const weights = GetWeightInfo(_femElements, indsY[i]);
        for (int k = 0, dst = 0; k < ElementT::kNumNodes; k++, j++, dst += dstDelta) {
          outJacY.Jac(i).MiddleCols(dst, jacX.nDoFsInternal) +=
              weights[k] * AsMatrixView(_toColliderJacs[l]) * jacX.Jac(indsX[j]);
        }
      }
    }

    // Set DoF indices. If the DoFs are shared, simply copy them over. If the DoFs are not shared,
    // append the DoFs of each point.
    if (outJacY.hasSharedDoFs) {
      std::copy(jacX.Inds(0).begin(), jacX.Inds(0).end(), outJacY.Inds(0).begin());
    } else {
      for (int i = 0, j = 0; i < outJacY.nContacts; i++) {
        auto* dst = outJacY.Inds(i).begin();
        for (int k = 0; k < ElementT::kNumNodes; k++, j++) {
          auto const& srcInds = jacX.Inds(indsX[j]);
          std::copy(srcInds.begin(), srcInds.end(), dst);
          dst += srcInds.size();
        }
      }
    }
  }

 private:
  Span<ElementT const> _femElements{};
  Span<VMatrix3x3r const> _toColliderJacs;
};

/**************************************************************************************************
 * Differentiable map for a skinning transformation. It is templatized whether it can take an input
 * or not (i.e. recursive or non-recursive). It is initialized with the state slice, the skinning
 * Jacobian, the articulated DoFs of the state slice and the DoF offset (plus the skinning data and
 * the bone rotations if it has an input). This class is used for articulated bodies and skinned
 * soft actors.
 */
template <bool InputT>
class DMapSkinning final : public DMapImpl {
 public:
  // Constructor if it has no input
  template <bool InputQ = InputT, typename = std::enable_if_t<!InputQ>>
  DMapSkinning(int slice, RowMatrixView<real const> jac, Span<int const> dofs, int offset)
      : _slice(slice), _jac(jac), _dofs(dofs), _offset(offset) {}

  // Constructor if it has an input
  template <bool InputQ = InputT, typename = std::enable_if_t<InputQ>>
  DMapSkinning(
      int slice,
      RowMatrixView<real const> jac,
      Span<int const> dofs,
      int offset,
      SkinningData const& skinning,
      Span<VMatrix3x3r const> boneRotations)
      : _slice(slice),
        _jac(jac),
        _dofs(dofs),
        _offset(offset),
        _skinning(skinning),
        _boneRotations(boneRotations) {}

  void WriteJacobianSlice(Span<int const> indsY, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    if (indsY.empty()) {
      return;
    }
    auto& outJacY = outJacsY[_slice];

    outJacY.Resize(true, false, _dofs.size(), _dofs.size(), indsY.size());

    // Compute derivatives
    for (int i = 0; i < outJacY.nContacts; i++) {
      auto jacRows = _jac.template MiddleRows<3>(3 * indsY[i], 3);
      for (int j = 0; j < _dofs.size(); j++) {
        outJacY.Jac(i).Col(j) = jacRows.Col(_dofs[j]);
      }
    }

    // Set DoF indices
    auto outInds = outJacY.Inds(0);
    for (int i = 0; i < outInds.size(); i++) {
      outInds[i] = _dofs[i] + _offset;
    }
  }

  void PropagateJacobianSlice(
      Remapping const& remap,
      Span<int const> indsY,
      ContactJac const& jacX,
      ContactJac& outJacY) const override {
    // Needed only if it takes an input
    if constexpr (InputT) {
      MOCHI_ASSERT(remap.indsX.size() == indsY.size(), "Sizes of indices don't match");

      // Initialize the output Jacobian
      outJacY.Resize(jacX.hasSharedDoFs, false, jacX.nDoFsInternal, jacX.nDoFsState, indsY.size());

      // Propagate partial derivatives wrt input
      for (int i = 0; i < outJacY.nContacts; i++) {
        VMatrix3x3r weightedRotations{};
        for (int j = 0, k = _skinning.weightsPerNode * indsY[i]; j < _skinning.weightsPerNode;
             j++, k++) {
          weightedRotations += _skinning.weights[k] * _boneRotations[_skinning.indices[k]];
        }
        auto weightedRotationsMat = AsMatrixView(weightedRotations);
        outJacY.Jac(i) = weightedRotationsMat * jacX.Jac(i);
      }

      // Set DoF indices. Consider the case where all DoFs are the same.
      for (int i = 0; i < (outJacY.hasSharedDoFs ? 1 : outJacY.nContacts); i++) {
        std::copy(jacX.Inds(i).begin(), jacX.Inds(i).end(), outJacY.Inds(i).begin());
      }
    }
  }

 private:
  struct Nothing {};
  int const _slice;
  RowMatrixView<real const> _jac;
  Span<int const> _dofs{};
  int const _offset;
  std::conditional_t<InputT, SkinningData const&, Nothing> _skinning;
  std::conditional_t<InputT, Span<VMatrix3x3r const>, Nothing> _boneRotations{};
};

using DMapSkinNoInput = DMapSkinning<false>;
using DMapSkinInput = DMapSkinning<true>;

/**************************************************************************************************
 * Differentiable map for sparse skinning. Maps skin mesh node positions to actor DoFs where
 * each node may depend on a different subset of DoFs.
 *
 * The Jacobian is stored as a @ref SparseMatrixView of size numVisNodes x numDofs, where each
 * non-zero entry is a @ref Real3 holding the (x, y, z) Jacobian contribution for that
 * (node, DoF) pair.
 *
 * This is used for transmitting contact forces from skin mesh samples back to actor DoFs
 * during contact resolution, when per-node DoF sets vary (e.g., rods, where each node
 * depends on DoFs from contributing element(s)).
 */
class DMapSparseSkinning final : public DMapImpl {
 public:
  DMapSparseSkinning(int slice, int offset, SparseMatrixView<Real3 const> skinningJacobian)
      : _slice(slice), _offset(offset), _skinningJacobian(skinningJacobian) {}

  void WriteJacobianSlice(Span<int const> indsY, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    if (indsY.empty()) {
      return;
    }
    auto& outJacY = outJacsY[_slice];

    // Find the maximum number of DoFs per visual node to size the ContactJac
    int maxDofsPerNode = 0;
    for (int const indY : indsY) {
      int const nnz = isize(_skinningJacobian.Indices(indY));
      maxDofsPerNode = Max(maxDofsPerNode, nnz);
    }

    // Each contact has its own set of DoF indices (not shared)
    outJacY.Resize(false, false, maxDofsPerNode, maxDofsPerNode, isize(indsY));

    // Fill Jacobian values and indices for each contact
    for (int i = 0; i < isize(indsY); ++i) {
      int const indY = indsY[i];
      int const nnz = isize(_skinningJacobian.Indices(indY));

      auto jacMat = outJacY.Jac(i);
      auto inds = outJacY.Inds(i);

      // Fill in the Jacobian values (3 rows x nnz columns)
      for (int j = 0; j < nnz; ++j) {
        for (int r = 0; r < 3; ++r) {
          jacMat(r, j) = _skinningJacobian.Values(indY)[j][r];
        }
      }

      // Zero out unused columns
      if (nnz < maxDofsPerNode) {
        jacMat.RightCols(maxDofsPerNode - nnz).SetZero();
      }

      // Set DoF indices (with offset applied)
      auto colIndices = _skinningJacobian.Indices(indY);
      for (int j = 0; j < nnz; ++j) {
        inds[j] = colIndices[j] + _offset;
      }
      // Set unused indices to first valid index (zero Jacobian columns won't contribute)
      for (int j = nnz; j < maxDofsPerNode; ++j) {
        inds[j] = nnz > 0 ? inds[0] : 0;
      }
    }
  }

 private:
  int const _slice;
  int const _offset;
  SparseMatrixView<Real3 const> const _skinningJacobian;
};

/**************************************************************************************************
 * Differentiable map for a deformable actor. It is initialized with the state slice and the DoF
 * offset, and templated on the number of fields, the first three of which are assumed to be
 * displacements.
 */
template <int kNumFields>
class DMapDeformable final : public DMapImpl {
  static_assert(kNumFields >= 3, "At least 3 fields are required");

 public:
  DMapDeformable(int slice, int offset) : _slice(slice), _offset(offset) {}

  void WriteJacobianSlice(Span<int const> indsY, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    if (indsY.empty()) {
      return;
    }
    auto& outJacY = outJacsY[_slice];

    outJacY.Resize(false, true, 3, 3, indsY.size());

    // Compute derivatives
    outJacY.Jac(0).SetIdentity();

    // Set DoF indices
    for (int i = 0; i < outJacY.nContacts; i++) {
      std::iota(outJacY.Inds(i).begin(), outJacY.Inds(i).end(), _offset + kNumFields * indsY[i]);
    }
  }

 private:
  int const _slice;
  int const _offset;
};

// Alias for deformable actors with only displacement DoFs
using DMapSoft = DMapDeformable<3>;

/**************************************************************************************************
 * Differentiable map for blended displacements. It is initialized with the blending info. This
 * class is used for soft skinned actors.
 */
class DMapBlending final : public DMapImpl {
 public:
  DMapBlending(BlendingDataSourceMesh const& blending) : _blending(blending) {}

  void RemapIndices(Span<int const> indsY, Remapping& outRemap) const override {
    MOCHI_ASSERT_VERBOSE(outRemap.map.empty() && outRemap.indsXData.empty());

    // Reserve output indices
    outRemap.map.reserve(indsY.size());
    outRemap.indsXData.reserve(indsY.size());

    // Fill the indices
    for (int indY : indsY) {
      int indX = _blending.mappingTargetToSource[indY];
      if (indX != -1) {
        outRemap.map.emplace_back(isize(outRemap.indsXData));
        outRemap.indsXData.emplace_back(indX);
      } else {
        outRemap.map.emplace_back(-1);
      }
    }

    // Assign remapped indices
    outRemap.indsX = MakeConstSpan(outRemap.indsXData);
  }

  void PropagateJacobianSlice(
      Remapping const& remap,
      Span<int const> indsY,
      ContactJac const& jacX,
      ContactJac& outJacY) const override {
    auto const& indsX = remap.map;
    MOCHI_ASSERT(indsX.size() == indsY.size(), "Sizes of indices don't match");

    // Initialize the output Jacobian
    outJacY.Resize(jacX.hasSharedDoFs, false, jacX.nDoFsInternal, jacX.nDoFsState, indsY.size());

    // Propagate partial derivatives wrt input, considering if the point is blended or not.
    for (int i = 0, j = 0; i < outJacY.nContacts; i++) {
      if (indsX[i] == -1) {
        outJacY.Jac(i).SetZero();
      } else {
        outJacY.Jac(i) = _blending.weightsSource[remap.indsX[j++]] * jacX.Jac(indsX[i]);
      }
    }

    // Set DoF indices. If the DoFs are shared, simply copy them over. If the DoFs are not shared,
    // copy the indices considering if the point is blended or not. If it is not, pick DoFs for an
    // arbitrary blended point (note that the Jacobian is zero in that case).
    if (outJacY.hasSharedDoFs) {
      std::copy(jacX.Inds(0).begin(), jacX.Inds(0).end(), outJacY.Inds(0).begin());
    } else {
      for (int i = 0; i < outJacY.nContacts; i++) {
        int indX = indsX[i] != -1 ? indsX[i] : 0;
        std::copy(jacX.Inds(indX).begin(), jacX.Inds(indX).end(), outJacY.Inds(i).begin());
      }
    }
  }

 private:
  BlendingDataSourceMesh const& _blending;
};

/**************************************************************************************************
 * Differentiable inverse map. It is initialized with the state slice, the DoF offset and a flag
 * indicating if DoFs are shared. The method SetData() must be called before computing the Jacobian,
 * to pass the output points (which carry the Jacobian of the inverse map). This class is used for
 * collider soft actors, and it maps world-space query points to the soft collider's local frame.
 */
class DMapInverse final : public DMapImpl {
 public:
  DMapInverse(int slice, int offset, bool sharedDofs)
      : _slice(slice), _offset(offset), _sharedDofs(sharedDofs) {}

  void SetData(ContactDetectionResult const* query) {
    _query = query;
  }

  void WriteJacobianSlice(Span<int const> /* indsY */, Span<ContactJac> outJacsY) const override {
    MOCHI_ASSERT(_query, "Data is not set");
    MOCHI_ASSERT(outJacsY.size() > _slice, "Insufficient Jacobian slices");
    auto& outJacY = outJacsY[_slice];

    outJacY.Resize(_sharedDofs, false, _query->ndofs, _query->ndofs, isize(_query->posColliding));

    // Compute Jacobians dref_ddofs = - dref_ddef * ddef_ddofs
    for (int i = 0, l = 0, dl = (_query->jacColliderFromWorld.size() == 1 ? 0 : 1);
         i < outJacY.nContacts;
         i++, l += dl) {
      for (int j = 0; j < outJacY.nDoFsInternal; j++) {
        outJacY.Jac(i).Col(j) = AsColumnVectorView<3>(
            DotMatVec3x3(_query->jacColliderFromWorld[l], -_query->jacWorldFromDofs[i].jac[j]));
      }
    }

    // Copy DoF indices
    if (_sharedDofs) {
      std::iota(outJacY.Inds(0).begin(), outJacY.Inds(0).end(), _offset);
    } else {
      for (int i = 0; i < outJacY.nContacts; i++) {
        for (int j = 0; j < outJacY.nDoFsInternal; j++) {
          outJacY.Inds(i)[j] = _offset + _query->jacWorldFromDofs[i].inds[j];
        }
      }
    }
  }

 private:
  int const _slice;
  int const _offset;
  bool _sharedDofs;
  ContactDetectionResult const* _query = nullptr;
};

} // namespace mochi::dmap
