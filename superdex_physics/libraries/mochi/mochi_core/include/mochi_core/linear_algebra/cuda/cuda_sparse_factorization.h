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

#include <mochi_core/mochi_config.h>

#if MOCHI_USE_CUDA

#include <mochi_core/linear_algebra/cuda/cuda_bsr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_csr_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>

#include <cusolverSp.h>

#if MOCHI_USE_CUDSS
#include <cudss.h>
#endif

namespace mochi::details::cuda {

template <typename Scalar, krylov::Direction kBlockDir>
void ConvertMatrixDeviceBsrToDeviceCsr(
    int blockSize,
    int n,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    Scalar const* d_blockValues,
    int* d_rowPtr,
    int* d_colIdx,
    Scalar* d_values);

template <typename Scalar, krylov::Direction kBlockDir>
void ConvertMatrixDeviceBsrToHostCsr(
    int blockSize,
    int n,
    int nnz,
    int const* d_blockPtr,
    int const* d_blockColIdx,
    Scalar const* d_blockValues,
    int* h_rowPtr,
    int* h_colIdx,
    Scalar* h_values) {
  //
  CudaVector<int> d_rowPtr(n + 1);
  CudaVector<int> d_colIdx(nnz);
  CudaVector<Scalar> d_values(nnz);
  //
  ConvertMatrixDeviceBsrToDeviceCsr<Scalar, kBlockDir>(
      blockSize,
      n,
      d_blockPtr,
      d_blockColIdx,
      d_blockValues,
      d_rowPtr.data(),
      d_colIdx.data(),
      d_values.data());
  //
  CudaMemCopy(h_rowPtr, d_rowPtr.data(), (n + 1) * sizeof(int));
  CudaMemCopy(h_colIdx, d_colIdx.data(), nnz * sizeof(int));
  CudaMemCopy(h_values, d_values.data(), nnz * sizeof(Scalar));
}

/// @brief Convert graph data from BSR format to CSR format
///
/// @param[in] blockSize Block size
/// @param[in] n Matrix dimension (assumed to be square)
/// @param[in] h_blockRowPtr Array for pointing block row starts (host)
/// @param[in] h_blockColIdx Array for block column indices (host)
/// @param[out] d_rowPtr Array for pointing row starts (device)
/// @param[out] d_colIdx Array for column indices (device)
void ConvertGraphHostBsrToDeviceCsr(
    int blockSize,
    int n,
    int const* h_blockRowPtr,
    int const* h_blockColIdx,
    int* d_rowPtr,
    int* d_colIdx);

/// @brief Convert graph data from BSR format to CSR format
///
/// @param[in] blockSize Block size
/// @param[in] n Matrix dimension (assumed to be square)
/// @param[in] h_blockRowPtr Array for pointing block row starts (host)
/// @param[in] h_blockColIdx Array for block column indices (host)
/// @param[out] h_rowPtr Array for pointing row starts (host)
/// @param[out] h_colIdx Array for column indices (host)
void ConvertGraphHostBsrToHostCsr(
    int blockSize,
    int n,
    int const* h_blockRowPtr,
    int const* h_blockColIdx,
    int* h_rowPtr,
    int* h_colIdx);

} // namespace mochi::details::cuda

namespace mochi::krylov {

#if !MOCHI_USE_CUDSS
/// @brief Class to solve A x = b with the LU factorization with partial pivoting.
///
/// @note NVIDIA will deprecate the function in a future release.
template <typename Scalar>
struct CudaSparseLU {
  using NonConstScalar = std::remove_const_t<Scalar>;

  template <typename AScalar, int kBlockSize, typename AIdx, krylov::Direction kBlockStorage>
  explicit CudaSparseLU(CudaBsrMatrix<AScalar, kBlockSize, AIdx, AIdx, kBlockStorage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    // Build the CSR data from the Cuda-BSR storage
    _ownsHRowPtr = true;
    _h_rowPtr = new int[_n + 1];
    //
    _ownsHColIdx = true;
    _h_colIdx = new int[_nnz];
    //
    _ownsHValues = true;
    _h_values = new NonConstScalar[_nnz];
    //
    mochi::details::cuda::ConvertMatrixDeviceBsrToHostCsr<NonConstScalar, kBlockStorage>(
        kBlockSize,
        _n,
        _nnz,
        A.Pointers().data(),
        A.Indices().data(),
        A.Values().data(),
        _h_rowPtr,
        _h_colIdx,
        _h_values);
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  template <
      typename AScalar,
      int kBlockSize,
      typename AIdx,
      template <typename, typename...> typename Storage>
  explicit CudaSparseLU(BlockSparseMatrix<AScalar, kBlockSize, AIdx, AIdx, Storage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    // Build the CSR data from the Mochi-BSR storage
    _ownsHRowPtr = true;
    _h_rowPtr = new int[_n + 1];
    //
    _ownsHColIdx = true;
    _h_colIdx = new int[_nnz];
    //
    _ownsHValues = false;
    _h_values = const_cast<NonConstScalar*>(A.Values().data());
    //
    mochi::details::cuda::ConvertGraphHostBsrToHostCsr(
        kBlockSize, _n, A.Pointers().data(), A.Indices().data(), _h_rowPtr, _h_colIdx);
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  template <typename AScalar, typename AIdx>
  explicit CudaSparseLU(CudaCsrMatrix<AScalar, AIdx, AIdx> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    _h_rowPtr = new int[_n + 1];
    MOCHI_CUDA_CHECK(
        cudaMemcpy(_h_rowPtr, A.Pointers().Data(), (_n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    _ownsHRowPtr = true;
    _h_colIdx = new int[_nnz];
    MOCHI_CUDA_CHECK(
        cudaMemcpy(_h_colIdx, A.Indices().Data(), _nnz * sizeof(int), cudaMemcpyDeviceToHost));
    _ownsHColIdx = true;
    _h_values = new NonConstScalar[_nnz];
    MOCHI_CUDA_CHECK(cudaMemcpy(
        _h_values, A.Values().Data(), _nnz * sizeof(NonConstScalar), cudaMemcpyDeviceToHost));
    _ownsHValues = true;
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  template <typename AScalar, typename AIdx, template <typename, typename...> typename Storage>
  explicit CudaSparseLU(SparseMatrix<AScalar, AIdx, AIdx, Storage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    //
    _h_rowPtr = const_cast<int*>(A.Pointers().data());
    _ownsHRowPtr = false;
    //
    _h_colIdx = const_cast<int*>(A.Indices().data());
    _ownsHColIdx = false;
    //
    _h_values = const_cast<NonConstScalar*>(A.Values().data());
    _ownsHValues = false;
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  ~CudaSparseLU() {
    if (_ownsHRowPtr) {
      delete[] _h_rowPtr;
    }
    if (_ownsHColIdx) {
      delete[] _h_colIdx;
    }
    if (_ownsHValues) {
      delete[] _h_values;
    }
  }

  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const {
    using NonConstScalarX = typename Input::NonConstScalar;
    static_assert(Input::kIsColMajor, "Input must be column major");
    static_assert(std::decay_t<Output>::kIsColMajor, "Output must be column major");
    using ScalarY = typename std::decay_t<Output>::Scalar;
    static_assert(
        std::is_same_v<NonConstScalar, NonConstScalarX> && std::is_same_v<NonConstScalar, ScalarY>,
        "Inconsistent scalar types");
    MOCHI_ASSERT_VERBOSE(
        (x.Rows() == y.Rows()) && (x.Cols() == y.Cols()), "Inconsistent matrix sizes.");
    int singularity = -1;
    cusparseMatDescr_t descrA = nullptr;
    cusparseCreateMatDescr(&descrA);
    // Host path for the matrix data
    MOCHI_ASSERT_VERBOSE(
        ((_h_rowPtr) && (_h_colIdx) && (_h_values)), "Matrix data must be on the device");
    static_assert(
        IsCudaMatrix<Input> && IsCudaMatrix<Output>,
        "Combination of Input and Output Types not currently supported.");
    // Execute the factorization and the solve in one call
    if constexpr (std::is_same_v<NonConstScalar, float>) {
      for (int j = 0; j < x.Cols(); ++j) {
        ColumnVector<NonConstScalar> h_x(x.Col(j));
        ColumnVector<NonConstScalar> h_y(y.Col(j));
        MOCHI_CUSOLVER_CHECK(cusolverSpScsrlsvluHost(
            _solverSpHandle,
            _n,
            _nnz,
            descrA,
            _h_values,
            _h_rowPtr,
            _h_colIdx,
            h_x.Data(),
            _pivotTol,
            _reOrder,
            h_y.Data(),
            &singularity));
        y.Col(j) = h_y;
      }
    } else {
      static_assert(std::is_same_v<NonConstScalar, double>);
      for (int j = 0; j < x.Cols(); ++j) {
        ColumnVector<NonConstScalar> h_x(x.Col(j));
        ColumnVector<NonConstScalar> h_y(y.Col(j));
        MOCHI_CUSOLVER_CHECK(cusolverSpDcsrlsvluHost(
            _solverSpHandle,
            _n,
            _nnz,
            descrA,
            _h_values,
            _h_rowPtr,
            _h_colIdx,
            h_x.Data(),
            _pivotTol,
            _reOrder,
            h_y.Data(),
            &singularity));
        y.Col(j) = h_y;
      }
    }
    if (singularity >= 0) {
      MOCHI_LOG_ERROR("Matrix appears to be singular");
    }
    cusparseDestroyMatDescr(descrA);
  }

  void Solve(CudaVectorView<Scalar const> x, CudaVectorView<NonConstScalar> Px) const {
    operator()(x, Px);
  }

 protected:
  int _n;
  int _nnz;
  /// @brief Flag for re-ordering algorithm
  /// 1 = symrcm
  /// 2 = symamd
  /// 3 = csrmetisnd (default)
  int _reOrder = 3;
  cusolverSpHandle_t _solverSpHandle;
  /// @brief Pointer of row beginning and ending in the column index array
  /// @note This array is on the host.
  /// @note It is allocated when the input matrix is block-sparse or on the device.
  int* _h_rowPtr = nullptr;
  bool _ownsHRowPtr = false;
  /// @brief Array of column indices
  /// @note This array is on the host.
  /// @note It is allocated when the input matrix is block-sparse or on the device.
  int* _h_colIdx = nullptr;
  bool _ownsHColIdx = false;
  /// @brief Array of values
  /// @note This array is on the host.
  /// @note It is allocated when the input matrix is block-sparse or on the device.
  NonConstScalar* _h_values = nullptr;
  bool _ownsHValues = false;
  /// @brief Absolute tolerance to detect a zero-pivot
  NonConstScalar _pivotTol = 0;
  /// @brief Component for absolute tolerance to detect a zero-pivot
  static NonConstScalar constexpr kUnscaledPivotTol =
      std::numeric_limits<NonConstScalar>::epsilon();
};

/// @brief Class to solve A x = b with the Cholesky factorization.
///
/// @note NVIDIA will deprecate the function in a future release.
template <typename Scalar>
struct CudaSparseCholesky {
  using NonConstScalar = std::remove_const_t<Scalar>;

  template <typename AScalar, int kBlockSize, typename AIdx, krylov::Direction kBlockStorage>
  explicit CudaSparseCholesky(
      CudaBsrMatrix<AScalar, kBlockSize, AIdx, AIdx, kBlockStorage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    // Build the CSR data from the Cuda-BSR storage
    _ownsDRowPtr = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_rowPtr, sizeof(int) * (_n + 1)));
    _ownsDColIdx = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_colIdx, sizeof(int) * _nnz));
    _ownsDValues = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_values, sizeof(NonConstScalar) * _nnz));
    //
    mochi::details::cuda::ConvertMatrixDeviceBsrToDeviceCsr<NonConstScalar, kBlockStorage>(
        kBlockSize,
        _n,
        A.Pointers().data(),
        A.Indices().data(),
        A.Values().data(),
        _d_rowPtr,
        _d_colIdx,
        _d_values);
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  template <
      typename AScalar,
      int kBlockSize,
      typename AIdx,
      template <typename, typename...> typename Storage>
  explicit CudaSparseCholesky(
      BlockSparseMatrix<AScalar, kBlockSize, AIdx, AIdx, Storage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    // Build the CSR data from the Mochi-BSR storage
    _ownsDRowPtr = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_rowPtr, sizeof(int) * (_n + 1)));
    _ownsDColIdx = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_colIdx, sizeof(int) * _nnz));
    //
    _ownsDValues = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_values, sizeof(NonConstScalar) * _nnz));
    MOCHI_CUDA_CHECK(cudaMemcpy(
        _d_values, A.Values().data(), _nnz * sizeof(NonConstScalar), cudaMemcpyHostToDevice));
    //
    mochi::details::cuda::ConvertGraphHostBsrToDeviceCsr(
        kBlockSize, _n, A.Pointers().data(), A.Indices().data(), _d_rowPtr, _d_colIdx);
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  template <typename AScalar, typename AIdx>
  explicit CudaSparseCholesky(CudaCsrMatrix<AScalar, AIdx, AIdx> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    _d_rowPtr = const_cast<int*>(A.Pointers().data());
    _ownsDRowPtr = false;
    _d_colIdx = const_cast<int*>(A.Indices().data());
    _ownsDColIdx = false;
    _d_values = const_cast<NonConstScalar*>(A.Values().data());
    _ownsDValues = false;
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  template <typename AScalar, typename AIdx, template <typename, typename...> typename Storage>
  explicit CudaSparseCholesky(SparseMatrix<AScalar, AIdx, AIdx, Storage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _solverSpHandle = reinterpret_cast<cusolverSpHandle_t>(mochi::details::GetCuSolverSpHandle());
    _n = A.Rows();
    _nnz = isize(A.Values());
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_rowPtr, (_n + 1) * sizeof(int)));
    MOCHI_CUDA_CHECK(
        cudaMemcpy(_d_rowPtr, A.Pointers().data(), (_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    _ownsDRowPtr = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_colIdx, _nnz * sizeof(int)));
    MOCHI_CUDA_CHECK(
        cudaMemcpy(_d_colIdx, A.Indices().data(), _nnz * sizeof(int), cudaMemcpyHostToDevice));
    _ownsDColIdx = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_values, _nnz * sizeof(NonConstScalar)));
    MOCHI_CUDA_CHECK(cudaMemcpy(
        _d_values, A.Values().data(), _nnz * sizeof(NonConstScalar), cudaMemcpyHostToDevice));
    _ownsDValues = true;
    // Compute the Frobenius norm of A to scale the pivot tolerance
    _pivotTol = A.Norm() * kUnscaledPivotTol / std::sqrt(Scalar(_nnz));
  }

  ~CudaSparseCholesky() {
    if (_ownsDRowPtr) {
      MOCHI_CUDA_CHECK(cudaFree(_d_rowPtr));
    }
    if (_ownsDColIdx) {
      MOCHI_CUDA_CHECK(cudaFree(_d_colIdx));
    }
    if (_ownsDValues) {
      MOCHI_CUDA_CHECK(cudaFree(_d_values));
    }
  }

  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const {
    using NonConstScalarX = typename Input::NonConstScalar;
    static_assert(Input::kIsColMajor, "Input must be column major");
    static_assert(std::decay_t<Output>::kIsColMajor, "Output must be column major");
    using ScalarY = typename std::decay_t<Output>::Scalar;
    static_assert(
        std::is_same_v<NonConstScalar, NonConstScalarX> && std::is_same_v<NonConstScalar, ScalarY>,
        "Inconsistent scalar types");
    MOCHI_ASSERT_VERBOSE(
        (x.Rows() == y.Rows()) && (x.Cols() == y.Cols()), "Inconsistent matrix sizes.");
    int singularity = -1;
    cusparseMatDescr_t descrA = nullptr;
    cusparseCreateMatDescr(&descrA);
    // GPU path for the matrix data
    MOCHI_ASSERT_VERBOSE(
        ((_d_rowPtr) && (_d_colIdx) && (_d_values)), "Matrix data must be on the device");
    static_assert(
        IsCudaMatrix<Input> && IsCudaMatrix<Output>,
        "Combination of Input and Output Types not currently supported.");
    // Execute the factorization and the solve in one call
    if constexpr (std::is_same_v<NonConstScalar, float>) {
      for (int j = 0; j < x.Cols(); ++j) {
        MOCHI_CUSOLVER_CHECK(cusolverSpScsrlsvchol(
            _solverSpHandle,
            _n,
            _nnz,
            descrA,
            _d_values,
            _d_rowPtr,
            _d_colIdx,
            x.Data() + j * x.LeadDim(),
            _pivotTol,
            _reOrder,
            y.Data() + j * y.LeadDim(),
            &singularity));
      }
    } else {
      static_assert(std::is_same_v<NonConstScalar, double>);
      for (int j = 0; j < x.Cols(); ++j) {
        MOCHI_CUSOLVER_CHECK(cusolverSpDcsrlsvchol(
            _solverSpHandle,
            _n,
            _nnz,
            descrA,
            _d_values,
            _d_rowPtr,
            _d_colIdx,
            x.Data() + j * x.LeadDim(),
            _pivotTol,
            _reOrder,
            y.Data() + j * y.LeadDim(),
            &singularity));
      }
    }
    if (singularity >= 0) {
      MOCHI_LOG_ERROR("Matrix appears to be singular");
    }
    cusparseDestroyMatDescr(descrA);
  }

  void Solve(CudaVectorView<Scalar const> x, CudaVectorView<NonConstScalar> Px) const {
    operator()(x, Px);
  }

 protected:
  int _n;
  int _nnz;
  /// @brief Flag for re-ordering algorithm
  /// 1 = symrcm
  /// 2 = symamd
  /// 3 = csrmetisnd (default)
  int _reOrder = 3;
  cusolverSpHandle_t _solverSpHandle;
  /// @brief Pointer of row beginning and ending in the column index array
  /// @note This array is on the device.
  /// @note It is allocated when the input matrix is block-sparse or on the host.
  int* _d_rowPtr = nullptr;
  bool _ownsDRowPtr = false;
  /// @brief Array of column indices
  /// @note This array is on the device.
  /// @note It is allocated when the input matrix is block-sparse or on the host.
  int* _d_colIdx = nullptr;
  bool _ownsDColIdx = false;
  /// @brief Array of values
  /// @note This array is on the device.
  /// @note It is allocated when the input matrix is block-sparse or on the host.
  NonConstScalar* _d_values = nullptr;
  bool _ownsDValues = false;
  /// @brief Absolute tolerance to detect a zero-pivot
  NonConstScalar _pivotTol = 0;
  /// @brief Component for absolute tolerance to detect a zero-pivot
  static NonConstScalar constexpr kUnscaledPivotTol =
      std::numeric_limits<NonConstScalar>::epsilon();
};
#else

enum class Factorization : char {
  Cholesky = 0,
  LDLt = 1,
  LU = 2,
};

/// @brief Class to call CUDSS-based sparse matrix factorization
template <typename Scalar, Factorization kFactType = Factorization::Cholesky>
struct CudaSparseMatrixFactorization {
  using NonConstScalar = std::remove_const_t<Scalar>;
  static cudaDataType_t constexpr kValueType =
      (std::is_same_v<Scalar, float>) ? CUDA_R_32F : CUDA_R_64F;

  // Documentation from https://docs.nvidia.com/cuda/cudss/types.html#cudssmatrixtype-t-label
  // Symmetric positive-definite matrix Cholesky factorization will be computed
  // with optional local pivoting
  static cudssMatrixType_t constexpr kMatrixType = (kFactType == Factorization::Cholesky)
      ? CUDSS_MTYPE_SPD
      : ((kFactType == Factorization::LDLt) ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_GENERAL);
  static cudssMatrixViewType_t constexpr kMatrixView = CUDSS_MVIEW_FULL;
  static cudssIndexBase_t constexpr kBase = CUDSS_BASE_ZERO;

  template <typename AScalar, int kBlockSize, typename AIdx, krylov::Direction kBlockStorage>
  explicit CudaSparseMatrixFactorization(
      CudaBsrMatrix<AScalar, kBlockSize, AIdx, AIdx, kBlockStorage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _n = A.Rows();
    _nnz = int64_t(A.Values().size());
    // Build the CSR data from the Cuda-BSR storage
    _ownsDRowPtr = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_rowPtr, sizeof(int) * (_n + 1)));
    _ownsDColIdx = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_colIdx, sizeof(int) * _nnz));
    _ownsDValues = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_values, sizeof(NonConstScalar) * _nnz));
    //
    mochi::details::cuda::ConvertMatrixDeviceBsrToDeviceCsr<NonConstScalar, kBlockStorage>(
        kBlockSize,
        _n,
        A.Pointers().data(),
        A.Indices().data(),
        A.Values().data(),
        _d_rowPtr,
        _d_colIdx,
        _d_values);
    //
    Factorize();
  }

  template <
      typename AScalar,
      int kBlockSize,
      typename AIdx,
      template <typename, typename...> typename Storage>
  explicit CudaSparseMatrixFactorization(
      BlockSparseMatrix<AScalar, kBlockSize, AIdx, AIdx, Storage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _n = A.Rows();
    _nnz = static_cast<int64_t>(A.Values().size());
    // Build the CSR data from the Mochi-BSR storage
    _ownsDRowPtr = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_rowPtr, sizeof(int) * (_n + 1)));
    _ownsDColIdx = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_colIdx, sizeof(int) * _nnz));
    //
    mochi::details::cuda::ConvertGraphHostBsrToDeviceCsr(
        kBlockSize, _n, A.Pointers().data(), A.Indices().data(), _d_rowPtr, _d_colIdx);
    //
    _ownsDValues = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_values, sizeof(NonConstScalar) * _nnz));
    MOCHI_CUDA_CHECK(cudaMemcpy(
        _d_values, A.Values().data(), _nnz * sizeof(NonConstScalar), cudaMemcpyHostToDevice));
    //
    Factorize();
  }

  template <typename AScalar, typename AIdx>
  explicit CudaSparseMatrixFactorization(CudaCsrMatrix<AScalar, AIdx, AIdx> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _nnz = static_cast<int64_t>(A.Values().size());
    _n = A.Rows();
    //
    _d_rowPtr = const_cast<int*>(A.Pointers().Data());
    _ownsDRowPtr = false;
    _d_colIdx = const_cast<int*>(A.Indices().Data());
    _ownsDColIdx = false;
    _d_values = const_cast<NonConstScalar*>(A.Values().Data());
    _ownsDValues = false;
    //
    Factorize();
  }

  template <typename AScalar, typename AIdx, template <typename, typename...> typename Storage>
  explicit CudaSparseMatrixFactorization(SparseMatrix<AScalar, AIdx, AIdx, Storage> const& A) {
    static_assert(std::is_same_v<AIdx const, int const>);
    static_assert(std::is_same_v<AScalar const, Scalar const>);
    _nnz = static_cast<int64_t>(A.Values().size());
    _n = A.Rows();
    // Build the CSR data on the device
    _ownsDRowPtr = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_rowPtr, sizeof(int) * (_n + 1)));
    MOCHI_CUDA_CHECK(
        cudaMemcpy(_d_rowPtr, A.Pointers().data(), (_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    _ownsDColIdx = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_colIdx, sizeof(int) * _nnz));
    MOCHI_CUDA_CHECK(
        cudaMemcpy(_d_colIdx, A.Indices().data(), _nnz * sizeof(int), cudaMemcpyHostToDevice));
    _ownsDValues = true;
    MOCHI_CUDA_CHECK(cudaMalloc((void**)&_d_values, sizeof(NonConstScalar) * _nnz));
    MOCHI_CUDA_CHECK(cudaMemcpy(
        _d_values, A.Values().data(), _nnz * sizeof(NonConstScalar), cudaMemcpyHostToDevice));
    //
    Factorize();
  }

  ~CudaSparseMatrixFactorization() {
    if (_ownsDRowPtr) {
      MOCHI_CUDA_CHECK(cudaFree(_d_rowPtr));
    }
    if (_ownsDColIdx) {
      MOCHI_CUDA_CHECK(cudaFree(_d_colIdx));
    }
    if (_ownsDValues) {
      MOCHI_CUDA_CHECK(cudaFree(_d_values));
    }
    MOCHI_CUDSS_CHECK(cudssMatrixDestroy(_A));
    MOCHI_CUDSS_CHECK(cudssDataDestroy(_dssHandle, _solverData));
    MOCHI_CUDSS_CHECK(cudssConfigDestroy(_solverConfig));
  }

  template <typename Input, typename Output>
  void operator()(Input const& x, Output&& y) const;

  void Solve(CudaVectorView<Scalar const> x, CudaVectorView<NonConstScalar> Px) const {
    operator()(x, Px);
  }

 protected:
  void Factorize() {
    _dssHandle = reinterpret_cast<cudssHandle_t>(mochi::details::GetCuDSSHandle());
    MOCHI_CUDSS_CHECK(cudssConfigCreate(&_solverConfig));
    MOCHI_CUDSS_CHECK(cudssDataCreate(_dssHandle, &_solverData));
    MOCHI_CUDSS_CHECK(cudssMatrixCreateCsr(
        &_A,
        _n,
        _n,
        _nnz,
        (void*)_d_rowPtr,
        NULL,
        (void*)_d_colIdx,
        (void*)_d_values,
        CUDA_R_32I,
        kValueType,
        kMatrixType,
        kMatrixView,
        kBase));
    //
    CudaVector<NonConstScalar> xtmp(_n), ytmp(_n);
    cudssMatrix_t d_xtmp, d_ytmp;
    MOCHI_CUDSS_CHECK(cudssMatrixCreateDn(
        &d_xtmp, _n, 1, _n, (void*)xtmp.data(), kValueType, CUDSS_LAYOUT_COL_MAJOR));
    MOCHI_CUDSS_CHECK(cudssMatrixCreateDn(
        &d_ytmp, _n, 1, _n, (void*)ytmp.data(), kValueType, CUDSS_LAYOUT_COL_MAJOR));
    //
    // The phases must always happen in the following order:
    // CUDSS_PHASE_REORDERING -> CUDSS_PHASE_SYMBOLIC_FACTORIZATION
    // -> CUDSS_PHASE_FACTORIZATION -> (optional) CUDSS_PHASE_REFACTORIZATION -> CUDSS_PHASE_SOLVE.
    // The optional refactorization is usually skipped before the first solve.
    // Re-using the analysis results is supported.
    // Users can change matrix values and only need to run the (re-)factorization and solve phases.
    //
    // Symbolic factorization
    MOCHI_CUDSS_CHECK(cudssExecute(
        _dssHandle, CUDSS_PHASE_ANALYSIS, _solverConfig, _solverData, _A, d_ytmp, d_xtmp));
    // Factorization
    MOCHI_CUDSS_CHECK(cudssExecute(
        _dssHandle, CUDSS_PHASE_FACTORIZATION, _solverConfig, _solverData, _A, d_ytmp, d_xtmp));
    MOCHI_CUDSS_CHECK(cudssMatrixDestroy(d_xtmp));
    MOCHI_CUDSS_CHECK(cudssMatrixDestroy(d_ytmp));
  }

  cudssHandle_t _dssHandle;
  cudssConfig_t _solverConfig;
  cudssData_t _solverData;
  cudssMatrix_t _A;
  int _n;
  int64_t _nnz;
  /// @brief Pointer of row beginning and ending in the column index array
  /// @note This array is on the device.
  /// @note It is allocated when the input matrix is block-sparse or on the host.
  int* _d_rowPtr = nullptr;
  bool _ownsDRowPtr = false;
  /// @brief Array of column indices
  /// @note This array is on the device.
  /// @note It is allocated when the input matrix is block-sparse or on the host.
  int* _d_colIdx = nullptr;
  bool _ownsDColIdx = false;
  /// @brief Array of values
  /// @note This array is on the device.
  /// @note It is allocated when the input matrix is block-sparse or on the host.
  NonConstScalar* _d_values = nullptr;
  bool _ownsDValues = false;
};

template <typename Scalar>
using CudaSparseCholesky = CudaSparseMatrixFactorization<Scalar, Factorization::Cholesky>;

template <typename Scalar>
using CudaSparseLDLt = CudaSparseMatrixFactorization<Scalar, Factorization::LDLt>;

template <typename Scalar>
using CudaSparseLU = CudaSparseMatrixFactorization<Scalar, Factorization::LU>;

template <typename Scalar, Factorization factType>
template <typename Input, typename Output>
void CudaSparseMatrixFactorization<Scalar, factType>::operator()(Input const& x, Output&& y) const {
  using NonConstScalarX = typename Input::NonConstScalar;
  static_assert(Input::kIsColMajor, "Input must be column major");
  static_assert(std::decay_t<Output>::kIsColMajor, "Output must be column major");
  using ScalarY = typename std::decay_t<Output>::Scalar;
  static_assert(
      std::is_same_v<NonConstScalar, NonConstScalarX> && std::is_same_v<NonConstScalar, ScalarY>,
      "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(
      (x.Rows() == y.Rows()) && (x.Cols() == y.Cols()), "Inconsistent matrix sizes.");
  static_assert(
      IsCudaMatrix<Input> && IsCudaMatrix<Output>,
      "Combination of Input and Output Types not currently supported.");
  int64_t nrows = int64_t(x.Rows()), nrhs = int64_t(x.Cols());
  int ldx = x.LeadDim(), ldy = y.LeadDim();
  cudssMatrix_t d_x, d_y;
  auto* xPtr = const_cast<Scalar*>(x.Data());
  MOCHI_CUDSS_CHECK(
      cudssMatrixCreateDn(&d_x, nrows, nrhs, ldx, (void*)xPtr, kValueType, CUDSS_LAYOUT_COL_MAJOR));
  MOCHI_CUDSS_CHECK(cudssMatrixCreateDn(
      &d_y, nrows, nrhs, ldy, (void*)y.Data(), kValueType, CUDSS_LAYOUT_COL_MAJOR));
  // Solving
  MOCHI_CUDSS_CHECK(
      cudssExecute(_dssHandle, CUDSS_PHASE_SOLVE, _solverConfig, _solverData, _A, d_y, d_x));
  MOCHI_CUDSS_CHECK(cudssMatrixDestroy(d_x));
  MOCHI_CUDSS_CHECK(cudssMatrixDestroy(d_y));
}
#endif

} // namespace mochi::krylov

#endif
