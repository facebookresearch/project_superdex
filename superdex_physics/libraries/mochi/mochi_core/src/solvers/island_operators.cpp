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

#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

// Like ToSparseMatrix, but with the added assumption that the matrix is symmetrical.
template <typename MatrixT>
static auto ToSymmetricalSparseMatrix(MatrixT const& mat) {
  if constexpr (IsMatrix<MatrixT>) {
    static_assert(MatrixT::kIsColMajor, "Expected Matrix<Scalar> which is col-major");
    // Transpose to get a row-major view before converting to SparseMatrix format.
    return ToSparseMatrix(Transpose(mat));
  } else {
    return ToSparseMatrix(mat);
  }
}

// Add two SparseMatrix which may have different sparsity patterns
// NOTE: The output indices are computed via AppendSum to allow for arbitrary col offsets, which is
// more expensive than Append. If needed for performance, this function could be templated by
// kColOffset = {kDynamic, 0} and use Append if kColOffset = 0.
template <typename T>
static SparseMatrix<T> AddMixedSparsity(
    std::tuple<int, int, SparseMatrixView<T const>> const& aInfo,
    std::tuple<int, int, SparseMatrixView<T const>> const& bInfo) {
  MOCHI_PROFILE_SCOPE();
  auto const& [aRowOffset, aColOffset, a] = aInfo;
  auto const& [bRowOffset, bColOffset, b] = bInfo;
  int const minRowOffset = Min(aRowOffset, bRowOffset);
  int const minRows = Min(aRowOffset + a.Rows(), bRowOffset + b.Rows());
  int const maxRows = Max(aRowOffset + a.Rows(), bRowOffset + b.Rows());
  int const maxCols = Max(aColOffset + a.Cols(), bColOffset + b.Cols());
  int const numNonZeroWorstCase = a.NumNonZeros() + b.NumNonZeros();

  DynamicArray<int> outPointers;
  DynamicArray<int> outIndices;
  DynamicArray<T> outValues;
  outPointers.reserve(maxRows + 1);
  outIndices.reserve(numNonZeroWorstCase); // conservative
  outValues.reserve(numNonZeroWorstCase); // conservative

  // Performance note: Parallelization doesn't seem to improve performance.
  outPointers.resize(minRowOffset + 1, 0);
  int r = minRowOffset;
  for (; r < minRows; ++r) {
    auto aIndices = (r >= aRowOffset) ? a.Indices(r - aRowOffset) : Span<int const>{};
    auto bIndices = (r >= bRowOffset) ? b.Indices(r - bRowOffset) : Span<int const>{};
    auto aValues = (r >= aRowOffset) ? a.Values(r - aRowOffset) : Span<T const>{};
    auto bValues = (r >= bRowOffset) ? b.Values(r - bRowOffset) : Span<T const>{};
    MOCHI_ASSERT_VERBOSE(isize(aIndices) == isize(aValues));
    MOCHI_ASSERT_VERBOSE(isize(bIndices) == isize(bValues));
    int ai = 0; // index in aIndices and aValues
    int bi = 0; // index in bIndices and bValues
    for (;;) {
      if (ai == isize(aValues)) {
        // Append any remaining values from matrix b
        AppendSum(outIndices, bIndices.subspan(bi), bColOffset);
        Append(outValues, bValues.subspan(bi));
        break;
      } else if (bi == isize(bValues)) {
        // Append any remaining values from matrix a
        AppendSum(outIndices, aIndices.subspan(ai), aColOffset);
        Append(outValues, aValues.subspan(ai));
        break;
      } else {
        auto aIdx = aIndices[ai] + aColOffset;
        auto bIdx = bIndices[bi] + bColOffset;
        if (aIdx < bIdx) {
          // The next value (in sorted order) comes from matrix a
          outIndices.push_back(aIdx);
          outValues.push_back(aValues[ai]);
          ++ai;
        } else if (aIdx > bIdx) {
          // The next value (in sorted order) comes from matrix b
          outIndices.push_back(bIdx);
          outValues.push_back(bValues[bi]);
          ++bi;
        } else {
          // Both matrices have the same DOF. Output the sum.
          outIndices.push_back(aIdx);
          outValues.push_back(aValues[ai] + bValues[bi]);
          ++ai;
          ++bi;
        }
      }
    }
    outPointers.push_back(isize(outValues));
  }

  // If one matrix had more rows, then append those extra rows.
  if (minRows < maxRows) {
    bool const isBTaller = (aRowOffset + a.Rows() < bRowOffset + b.Rows());
    SparseMatrixView<T const> tallerMat = isBTaller ? b : a;
    int const tallerRowOffset = isBTaller ? bRowOffset : aRowOffset;
    int const tallerColOffset = isBTaller ? bColOffset : aColOffset;
    for (; r < tallerRowOffset; ++r) {
      outPointers.push_back(isize(outValues));
    }
    for (; r < maxRows; ++r) {
      MOCHI_ASSERT_VERBOSE(isize(tallerMat.Indices()) == isize(tallerMat.Values()));
      AppendSum(outIndices, tallerMat.Indices(r - tallerRowOffset), tallerColOffset);
      Append(outValues, tallerMat.Values(r - tallerRowOffset));
      outPointers.push_back(isize(outValues));
    }
  }

  return SparseMatrix<T>{
      maxCols, std::move(outPointers), std::move(outIndices), std::move(outValues)};
}

// Add two BlockSparseMatrix which may have different sparsity patterns.
// NOTE: The output indices are computed via AppendSum to allow for arbitrary block col offsets,
// which is more expensive than Append. If needed for performance, this function could be templated
// by kColOffset = {kDynamic, 0} and use Append if kColOffset = 0.
template <typename T, int kBlockSize>
static BlockSparseMatrix<T, kBlockSize> AddMixedBlockSparsity(
    std::tuple<int, int, BlockSparseMatrixView<T const, kBlockSize>> const& aInfo,
    std::tuple<int, int, BlockSparseMatrixView<T const, kBlockSize>> const& bInfo) {
  MOCHI_PROFILE_SCOPE();
  auto const& [aRowOffset, aColOffset, A] = aInfo;
  auto const& [bRowOffset, bColOffset, B] = bInfo;
  MOCHI_ASSERT_VERBOSE(
      aRowOffset % kBlockSize == 0 && bRowOffset % kBlockSize == 0, "Invalid row offsets.");
  MOCHI_ASSERT_VERBOSE(
      aColOffset % kBlockSize == 0 && bColOffset % kBlockSize == 0, "Invalid col offsets.");

  int const aBrOffset = aRowOffset / kBlockSize;
  int const bBrOffset = bRowOffset / kBlockSize;
  int const aBcOffset = aColOffset / kBlockSize;
  int const bBcOffset = bColOffset / kBlockSize;
  int const minBrOffset = Min(aBrOffset, bBrOffset);
  int const minBlockRows = Min(aBrOffset + A.BlockRows(), bBrOffset + B.BlockRows());
  int const maxBlockRows = Max(aBrOffset + A.BlockRows(), bBrOffset + B.BlockRows());
  int const maxBlockCols = Max(aBcOffset + A.BlockCols(), bBcOffset + B.BlockCols());
  int const nnzBlocksConservative = A.NumNonZeroBlocks() + B.NumNonZeroBlocks();

  DynamicArray<int> outPointers;
  DynamicArray<int> outIndices;
  DynamicArray<T> outValues;
  outPointers.reserve(maxBlockRows + 1);
  outIndices.reserve(nnzBlocksConservative);
  outValues.reserve(nnzBlocksConservative * kBlockSize * kBlockSize);

  [[maybe_unused]] size_t const initialCapacity[] = {
      outPointers.capacity(), outIndices.capacity(), outValues.capacity()};

  // Temporary vectors for the values in the rows [1, ..., kBlockSize - 1] within a block row.
  std::array<DynamicArray<T>, kBlockSize - 1> tmpValues;
  for (auto& vals : tmpValues) {
    vals.reserve(A.MaxNnzPerRow() + B.MaxNnzPerRow()); // Conservative
  }

  // Process one block row at a time. The output sparsity pattern will be the union of the sparsity
  // patterns of A and B. Any blocks that exist in both will be added together. Performance note:
  // Parallelization doesn't seem to improve performance.
  outPointers.resize(minBrOffset + 1, 0);
  int br = minBrOffset;
  for (; br < minBlockRows; ++br) {
    auto const aIndices = (br >= aBrOffset) ? A.Indices(br - aBrOffset) : Span<int const>{};
    auto const bIndices = (br >= bBrOffset) ? B.Indices(br - bBrOffset) : Span<int const>{};

    // Fail fast if the block row is empty.
    if (aIndices.empty() && bIndices.empty()) {
      outPointers.push_back(isize(outIndices));
      continue;
    }

    auto const aValues = (br >= aBrOffset)
        ? A.Values(br - aBrOffset)
        : BlockRowView<T const, kBlockSize, int const>(nullptr, 0, 0);
    auto const bValues = (br >= bBrOffset)
        ? B.Values(br - bBrOffset)
        : BlockRowView<T const, kBlockSize, int const>(nullptr, 0, 0);
    int ai = 0; // Block index in aIndices and aValues
    int bi = 0; // Block index in bIndices and bValues
    for (auto& vals : tmpValues) {
      vals.clear();
    }
    while ((ai < isize(aIndices)) || (bi < isize(bIndices))) {
      if ((ai < isize(aIndices)) && (bi < isize(bIndices))) {
        auto aIdx = aBcOffset + aIndices[ai];
        auto bIdx = bBcOffset + bIndices[bi];
        if (aIdx < bIdx) {
          // Next block(s) are from A.
          auto const it = std::lower_bound(&aIndices[ai], aIndices.end(), bIdx - aBcOffset);
          auto const numBlocks = static_cast<int>(it - &aIndices[ai]);
          AppendSum(outIndices, Span(&aIndices[ai], numBlocks), aBcOffset);
          Append(outValues, Span(&aValues(0, ai * kBlockSize), kBlockSize * numBlocks));
          for (int k = 1; k < kBlockSize; ++k) {
            Append(tmpValues[k - 1], Span(&aValues(k, ai * kBlockSize), kBlockSize * numBlocks));
          }
          ai += numBlocks;
        } else if (aIdx > bIdx) {
          // Next block(s) are from B.
          auto const it = std::lower_bound(&bIndices[bi], bIndices.end(), aIdx - bBcOffset);
          auto const numBlocks = static_cast<int>(it - &bIndices[bi]);
          AppendSum(outIndices, Span(&bIndices[bi], numBlocks), bBcOffset);
          Append(outValues, Span(&bValues(0, bi * kBlockSize), kBlockSize * numBlocks));
          for (int k = 1; k < kBlockSize; ++k) {
            Append(tmpValues[k - 1], Span(&bValues(k, bi * kBlockSize), kBlockSize * numBlocks));
          }
          bi += numBlocks;
        } else {
          // Next block is from both.
          MOCHI_ASSERT_VERBOSE(aIdx == bIdx);
          outIndices.push_back(aIdx);
          AppendSum(
              outValues,
              Span(&aValues(0, ai * kBlockSize), kBlockSize),
              Span(&bValues(0, bi * kBlockSize), kBlockSize));
          for (int k = 1; k < kBlockSize; ++k) {
            AppendSum(
                tmpValues[k - 1],
                Span(&aValues(k, ai * kBlockSize), kBlockSize),
                Span(&bValues(k, bi * kBlockSize), kBlockSize));
          }
          ++ai;
          ++bi;
        }
      } else if (ai == isize(aIndices)) {
        // Append remaining block(s) from B.
        AppendSum(outIndices, bIndices.subspan(bi), bBcOffset);
        int const numCols = kBlockSize * (isize(bIndices) - bi);
        Append(outValues, Span(&bValues(0, bi * kBlockSize), numCols));
        for (int k = 1; k < kBlockSize; ++k) {
          Append(tmpValues[k - 1], Span(&bValues(k, bi * kBlockSize), numCols));
        }
        break;
      } else {
        // Append remaining block(s) from A.
        MOCHI_ASSERT_VERBOSE(bi == isize(bIndices));
        AppendSum(outIndices, aIndices.subspan(ai), aBcOffset);
        int const numCols = kBlockSize * (isize(aIndices) - ai);
        Append(outValues, Span(&aValues(0, ai * kBlockSize), numCols));
        for (int k = 1; k < kBlockSize; ++k) {
          Append(tmpValues[k - 1], Span(&aValues(k, ai * kBlockSize), numCols));
        }
        break;
      }
    }
    for (auto const& vals : tmpValues) {
      Append(outValues, vals);
    }
    outPointers.push_back(isize(outIndices));
  }

  // Confirm that we reserved enough memory to avoid re-allocation.
  MOCHI_ASSERT_VERBOSE(outPointers.capacity() == initialCapacity[0]);
  MOCHI_ASSERT_VERBOSE(outIndices.capacity() == initialCapacity[1]);
  MOCHI_ASSERT_VERBOSE(outValues.capacity() == initialCapacity[2]);

  // If one matrix has more block rows than the other, append the remaining block rows.
  if (minBlockRows < maxBlockRows) {
    bool const isBTaller = (aBrOffset + A.BlockRows() < bBrOffset + B.BlockRows());
    BlockSparseMatrixView<T const, kBlockSize> tallerMat = isBTaller ? B : A;
    int const tallerBrOffset = isBTaller ? bBrOffset : aBrOffset;
    int const tallerBcOffset = isBTaller ? bBcOffset : aBcOffset;
    auto const tallerPointers = tallerMat.Pointers();
    for (; br < tallerBrOffset; ++br) {
      outPointers.push_back(isize(outIndices));
    }
    AppendSum(
        outPointers,
        tallerPointers.subspan(1 + br - tallerBrOffset),
        outPointers.back() - tallerPointers[br - tallerBrOffset]);
    AppendSum(
        outIndices,
        tallerMat.Indices().subspan(tallerPointers[br - tallerBrOffset]),
        tallerBcOffset);
    Append(
        outValues,
        tallerMat.Values().subspan(kBlockSize * kBlockSize * tallerPointers[br - tallerBrOffset]));
  }

  return BlockSparseMatrix<T, kBlockSize>{
      maxBlockCols, std::move(outPointers), std::move(outIndices), std::move(outValues)};
}

// Append the rows of an actor's matrix to the global matrix. If the contact matrix overlaps any of
// these global rows, then add the contact values. The resulting sparsity pattern will be the union
// of tha actor's sparsity and the contact sparsity (which may include off-diagonal terms).
template <typename T>
static void AppendActorRowsToGlobalSparseMatrix(
    int actorOffset,
    SparseMatrixView<T const> actorMat,
    SparseMatrixView<T const> interactionMat,
    DynamicArray<int>& outPointers,
    DynamicArray<int>& outIndices,
    DynamicArray<T>& outValues) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(isize(outPointers) == actorOffset + 1);

  // Determine if the contact matrix has any rows that overlap the actor matrix
  bool hasContact = false;
  if (interactionMat.Rows() > actorOffset) {
    int contactRowBegin = actorOffset;
    int contactRowEnd = Min(interactionMat.Rows(), actorOffset + actorMat.Rows());
    auto contactRowOffsets = interactionMat.Pointers();
    if (contactRowOffsets[contactRowEnd] > contactRowOffsets[contactRowBegin]) {
      hasContact = true;
    }
  }

  // If there is no contact, then just append the actor rows to the global sparse matrix.
  if (!hasContact) {
    MOCHI_ASSERT(
        !outPointers.empty(),
        "This array should not be empty because the calling function adds the first value (always zero).");
    AppendSum(outPointers, actorMat.Pointers().subspan(1), outPointers.back());
    AppendSum(outIndices, actorMat.Indices(), actorOffset);
    Append(outValues, actorMat.Values());
    return;
  }

  // Output one row at a time. The output sparsity pattern will be the union of the actorMatrix
  // sparsity and the interactionMatrix sparsity. Any values that exist in both will be added
  // together. Performance note: Parallelization doesn't seem to improve performance.
  for (int r = 0; r < actorMat.Rows(); ++r) {
    int gr = r + actorOffset; // global row
    auto actorIndices = actorMat.Indices(r);
    auto actorValues = actorMat.Values(r);
    auto contactIndices =
        (gr < interactionMat.Rows()) ? interactionMat.Indices(gr) : Span<int const>{};
    auto contactValues = (gr < interactionMat.Rows()) ? interactionMat.Values(gr) : Span<T const>{};
    MOCHI_ASSERT_VERBOSE(isize(actorIndices) == isize(actorValues));
    MOCHI_ASSERT_VERBOSE(isize(contactIndices) == isize(contactValues));
    if (contactIndices.empty()) {
      // No contact data on this row
      AppendSum(outIndices, actorIndices, actorOffset);
      Append(outValues, actorValues);
    } else if (actorIndices.empty()) {
      // No actor data on this row (does this happen?)
      Append(outIndices, contactIndices);
      Append(outValues, contactValues);
    } else {
      int ai = 0; // index in actorIndices and actorValues
      int ci = 0; // index in contactIndices and contactValues
      for (;;) {
        if (ci == isize(contactValues)) {
          // No more contact values. Append any remaining actor values.
          AppendSum(outIndices, actorIndices.subspan(ai), actorOffset);
          Append(outValues, actorValues.subspan(ai));
          break;
        } else if (ai == isize(actorValues)) {
          // No more actor values. Append any remaining contact values.
          Append(outIndices, contactIndices.subspan(ci));
          Append(outValues, contactValues.subspan(ci));
          break;
        } else {
          auto actorColIdx = actorIndices[ai] + actorOffset;
          auto contactColIdx = contactIndices[ci];
          if (actorColIdx < contactColIdx) {
            // The next value (in sorted order) comes from actorMat
            outIndices.push_back(actorColIdx);
            outValues.push_back(actorValues[ai]);
            ++ai;
          } else if (actorColIdx > contactColIdx) {
            // The next value (in sorted order) comes form interactionMat, but there are still
            // values left from actorMat. This is typically because the column indices are smaller
            // than the actor offset. A large number of the next values may come from contact.
            // Process all of them at once.
            auto const it =
                std::lower_bound(&contactIndices[ci], contactIndices.end(), actorColIdx);
            auto const numCols = static_cast<int>(it - &contactIndices[ci]);
            MOCHI_ASSERT_VERBOSE(ci + numCols <= isize(contactValues));
            Append(outIndices, contactIndices.subspan(ci, numCols));
            Append(outValues, contactValues.subspan(ci, numCols));
            ci += numCols;
          } else {
            // Both matrices have the same DOF. Output the sum.
            outIndices.push_back(actorColIdx);
            outValues.push_back(actorValues[ai] + contactValues[ci]);
            ++ai;
            ++ci;
          }
        }
      }
    }
    outPointers.push_back(isize(outValues));
  }
}

template <int kBlockSize, typename T>
static void AppendActorToGlobalBlockSparseMatrix(
    int actorBlockRowOffset,
    BlockSparseMatrixView<T const, kBlockSize> A,
    BlockSparseMatrixView<T const, kBlockSize> C,
    DynamicArray<int>& outPointers,
    DynamicArray<int>& outIndices,
    DynamicArray<T>& outValues) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(isize(outPointers) == actorBlockRowOffset + 1);

  // Determine if the contact matrix has any rows that overlap the actor matrix.
  bool hasContact = false;
  int const contactBlockRowEnd = Min(C.BlockRows(), actorBlockRowOffset + A.BlockRows());
  auto const cPointers = C.Pointers();
  if (C.BlockRows() > actorBlockRowOffset) {
    int const contactBlockRowBegin = actorBlockRowOffset;
    if (cPointers[contactBlockRowEnd] > cPointers[contactBlockRowBegin]) {
      hasContact = true;
    }
  }

  // If there is no contact, append the actor matrix to the global matrix and return.
  if (!hasContact) {
    MOCHI_ASSERT(
        !outPointers.empty(),
        "Pointers should not be empty. The calling function must have added the first value (always zero).");
    AppendSum(outPointers, A.Pointers().subspan(1), outPointers.back());
    AppendSum(outIndices, A.Indices(), actorBlockRowOffset);
    Append(outValues, A.Values());
    return;
  }

  // Temporary vectors for the values in the rows [1, ..., kBlockSize - 1] within a block row.
  std::array<std::vector<T>, kBlockSize - 1> tmpValues;
  for (auto& vals : tmpValues) {
    vals.reserve(A.MaxNnzPerRow() + C.MaxNnzPerRow()); // Conservative
  }

  // Process one block row at a time. The output sparsity pattern will be the union of the actor
  // matrix sparsity and the contact matrix sparsity. Any blocks that exist in both will be added
  // together. Performance note: Parallelization doesn't seem to improve performance.
  for (int br = 0; br < A.BlockRows(); ++br) {
    int const gbr = br + actorBlockRowOffset;
    auto const aIndices = A.Indices(br);
    auto const aValues = A.Values(br);
    auto const cIndices = (gbr < C.BlockRows()) ? C.Indices(gbr) : Span<int const>{};

    if (cIndices.empty()) {
      // Performance note: No performance improvement by appending at once all block rows of the
      // actor matrix until the next non-empty block row of the contact matrix. Reference
      // implementation: P878695616
      AppendSum(outIndices, aIndices, actorBlockRowOffset);
      int const numNzCols = kBlockSize * isize(aIndices);
      MOCHI_ASSERT_VERBOSE(aValues.LeadDim() == numNzCols);
      Append(outValues, Span(&aValues(0, 0), kBlockSize * numNzCols));
      outPointers.push_back(isize(outIndices));
      continue;
    }

    auto const cValues = C.Values(gbr);
    for (auto& vals : tmpValues) {
      vals.clear();
    }
    if (aIndices.empty()) {
      // No actor data on this block row.
      Append(outIndices, cIndices);
      int const numNzCols = kBlockSize * isize(cIndices);
      MOCHI_ASSERT_VERBOSE(cValues.LeadDim() == numNzCols);
      Append(outValues, Span(&cValues(0, 0), kBlockSize * numNzCols));
    } else {
      int ai = 0; // Block index in aIndices and aValues.
      int ci = 0; // Block index in cIndices and cValues.
      while ((ai < isize(aIndices)) || (ci < isize(cIndices))) {
        if ((ci < isize(cIndices)) && (ai < isize(aIndices))) {
          int const aIdx = aIndices[ai] + actorBlockRowOffset;
          int const cIdx = cIndices[ci];
          if (aIdx < cIdx) {
            // The next block is from the actor matrix.
            outIndices.push_back(aIdx);
            Append(outValues, Span(&aValues(0, ai * kBlockSize), kBlockSize));
            for (int k = 1; k < kBlockSize; ++k) {
              Append(tmpValues[k - 1], Span(&aValues(k, ai * kBlockSize), kBlockSize));
            }
            ++ai;
          } else if (aIdx > cIdx) {
            // The next block(s) are from the contact matrix, but there are still blocks left from
            // the actor matrix. This is typically because the column indices are smaller than the
            // actor offset. A large number of the next blocks may come from contact. Process all of
            // them at once.
            auto const it = std::lower_bound(&cIndices[ci], cIndices.end(), aIdx);
            auto const numBlocks = static_cast<int>(it - &cIndices[ci]);
            Append(outIndices, Span(&cIndices[ci], numBlocks));
            Append(outValues, Span(&cValues(0, ci * kBlockSize), kBlockSize * numBlocks));
            for (int k = 1; k < kBlockSize; ++k) {
              Append(tmpValues[k - 1], Span(&cValues(k, ci * kBlockSize), kBlockSize * numBlocks));
            }
            ci += numBlocks;
          } else {
            // The next block is from both.
            MOCHI_ASSERT_VERBOSE(aIdx == cIdx);
            outIndices.push_back(aIdx);
            AppendSum(
                outValues,
                Span(&aValues(0, ai * kBlockSize), kBlockSize),
                Span(&cValues(0, ci * kBlockSize), kBlockSize));
            for (int k = 1; k < kBlockSize; ++k) {
              AppendSum(
                  tmpValues[k - 1],
                  Span(&aValues(k, ai * kBlockSize), kBlockSize),
                  Span(&cValues(k, ci * kBlockSize), kBlockSize));
            }
            ++ai;
            ++ci;
          }
        } else if (ci == isize(cIndices)) {
          // Append remaining actor block(s).
          AppendSum(outIndices, aIndices.subspan(ai), actorBlockRowOffset);
          int const numCols = kBlockSize * (isize(aIndices) - ai);
          Append(outValues, Span(&aValues(0, ai * kBlockSize), numCols));
          for (int k = 1; k < kBlockSize; ++k) {
            Append(tmpValues[k - 1], Span(&aValues(k, ai * kBlockSize), numCols));
          }
          break;
        } else {
          // Append remaining contact block(s).
          MOCHI_ASSERT_VERBOSE(ai == isize(aIndices));
          Append(outIndices, cIndices.subspan(ci));
          int const numCols = kBlockSize * (isize(cIndices) - ci);
          Append(outValues, Span(&cValues(0, ci * kBlockSize), numCols));
          for (int k = 1; k < kBlockSize; ++k) {
            Append(tmpValues[k - 1], Span(&cValues(k, ci * kBlockSize), numCols));
          }
          break;
        }
      }
    }
    for (auto const& vals : tmpValues) {
      Append(outValues, vals);
    }
    outPointers.push_back(isize(outIndices));
  }
}

template <typename T>
SparseMatrix<T> IslandOperators<T>::FullSparseMatrix() const {
  MOCHI_PROFILE_SCOPE();

  // Dimensions of the global matrix
  int const globalNumRows = Rows();
  int const globalNumCols = globalNumRows;

  // The current implementation only supports interaction matrices in SparseMatrix format.
  // Other matrix formats will be converted to SparseMatrix (not optimal) for now.
  std::vector<SparseMatrix<T>> tempMatrixStorage;
  std::vector<std::tuple<int, int, SparseMatrixView<T const>>> interactionMatrixViews;
  for (auto const& [rOffset, cOffset, anyMat, _] : this->_interactionMatrices) {
    std::visit(
        [&, rOffset_ = rOffset, cOffset_ = cOffset](auto const& mat) {
          if constexpr (IsSparseMatrix<decltype(mat)>) {
            interactionMatrixViews.emplace_back(
                rOffset_, cOffset_, AsConstView(mat)); // View of existing data
          } else {
            tempMatrixStorage.emplace_back(ToSparseMatrix(mat)); // Convert format
            interactionMatrixViews.emplace_back(
                rOffset_,
                cOffset_,
                AsConstView(tempMatrixStorage.back())); // View of converted format
          }
        },
        anyMat);
  }

  // If there are multiple interaction matrices, then condense them down to one.
  SparseMatrix<T> combinedInteractionMatrix{}; // Temporary owner.
  SparseMatrixView<T const> combinedInteractionMatrixView; // As const view.
  int iInter = 0;
  if (isize(interactionMatrixViews) > 1) {
    combinedInteractionMatrix.Reset(
        AddMixedSparsity<T>(interactionMatrixViews[0], interactionMatrixViews[1]));
    iInter += 2;
  }
  for (; iInter < isize(interactionMatrixViews); ++iInter) {
    combinedInteractionMatrix.Reset(
        AddMixedSparsity<T>(
            {0, 0, AsConstView(combinedInteractionMatrix)}, interactionMatrixViews[iInter]));
  }
  combinedInteractionMatrixView.Reset(combinedInteractionMatrix);

  // Get a conservative count for the total number of non-zero values
  int conservativeNnz = combinedInteractionMatrixView.NumNonZeros();
  for (auto const& [actorOffset, anyMat] : this->_actorMatrices) {
    conservativeNnz += GetNumValues(anyMat);
  }

  // Allocate storage for the global sparse matrix. May be more than we need (optimizing for speed).
  DynamicArray<int> pointers, indices;
  DynamicArray<T> values;
  pointers.reserve(globalNumRows + 1);
  pointers.push_back(0);
  indices.reserve(conservativeNnz);
  values.reserve(conservativeNnz);
  [[maybe_unused]] size_t const initialCapacity[] = {
      pointers.capacity(), indices.capacity(), values.capacity()};

  // Append rows to the global sparse matrix, one actor at a time.
  for (auto const& [actorOffset, anyMat] : this->_actorMatrices) {
    std::visit(
        [&, actorOffset_ = actorOffset](auto const& mat) {
          AppendActorRowsToGlobalSparseMatrix(
              actorOffset_,
              ToSymmetricalSparseMatrix(mat),
              combinedInteractionMatrixView,
              pointers,
              indices,
              values);
        },
        anyMat);
  }

  // Confirm that we reserved enough memory to avoid re-allocation
  MOCHI_ASSERT_VERBOSE(pointers.capacity() == initialCapacity[0]);
  MOCHI_ASSERT_VERBOSE(indices.capacity() == initialCapacity[1]);
  MOCHI_ASSERT_VERBOSE(values.capacity() == initialCapacity[2]);

  // Move the data arrays into a new SparseMatrix
  return SparseMatrix<T>{globalNumCols, std::move(pointers), std::move(indices), std::move(values)};
}

template <typename T>
template <int kBlockSize>
BlockSparseMatrix<T, kBlockSize> IslandOperators<T>::FullBlockSparseMatrix() const {
  MOCHI_PROFILE_SCOPE();

  // Dimensions of the global matrix.
  int const numRows = Rows();
  int const numCols = numRows;
  if ((numRows % kBlockSize != 0) || !IsBlockable<kBlockSize>()) {
    // Return empty matrix if not blockable.
    return {};
  }
  int const numBlockRows = numRows / kBlockSize;
  int const numBlockCols = numCols / kBlockSize;

  // Condense interaction matrices into a single one.
  std::vector<std::tuple<int, int, BlockSparseMatrixView<T const, kBlockSize>>>
      interactionMatrixViews; // As block sparse views
  std::vector<BlockViewStructure<kBlockSize, int, int>> blockedStructures;
  std::vector<BlockSparseMatrix<T, kBlockSize>> denseAsBlockSparse;
  interactionMatrixViews.reserve(this->_interactionMatrices.size());
  blockedStructures.reserve(this->_interactionMatrices.size()); // Conservative
  denseAsBlockSparse.reserve(this->_interactionMatrices.size()); // Conservative
  for (auto const& [rOffset, cOffset, anyMat, _] : this->_interactionMatrices) {
    static_assert(
        std::variant_size_v<decltype(anyMat)> == 4,
        "Please update the code below if the interaction matrix types change");
    auto* bsp = std::get_if<BlockSparseMatrixView<T const, kBlockSize>>(&anyMat);
    auto* sp = std::get_if<SparseMatrixView<T const>>(&anyMat);
    auto* dense = std::get_if<MatrixView<T const>>(&anyMat);
    if (bsp) {
      interactionMatrixViews.emplace_back(rOffset, cOffset, *bsp);
    } else if (sp) {
      blockedStructures.push_back(BlockedStructure<kBlockSize>(*sp));
      interactionMatrixViews.emplace_back(
          rOffset, cOffset, blockedStructures.back().ConstView(sp->Values().data()));
    } else if (dense) {
      // TODO(@pabfer): Create BlockedStructure overload for Matrix.
      denseAsBlockSparse.push_back(ToBlockSparseMatrix<kBlockSize>(*dense));
      interactionMatrixViews.emplace_back(rOffset, cOffset, denseAsBlockSparse.back());
    } else {
      MOCHI_ASSERT(false, "Unexpected matrix type."); // Must not be reached if blockable
      return {};
    }
  }

  BlockSparseMatrix<T, kBlockSize> condensedInteractionMatrix; // Temporary owner.
  BlockSparseMatrixView<T const, kBlockSize> condensedInteractionMatrixView; // As block sparse view
  int iInter = 0;
  if (isize(interactionMatrixViews) > 1) {
    condensedInteractionMatrix =
        AddMixedBlockSparsity<T, kBlockSize>(interactionMatrixViews[0], interactionMatrixViews[1]);
    iInter += 2;
  }
  for (; iInter < isize(interactionMatrixViews); ++iInter) {
    condensedInteractionMatrix.Reset(
        AddMixedBlockSparsity<T, kBlockSize>(
            {0, 0, AsConstView(condensedInteractionMatrix)}, interactionMatrixViews[iInter]));
  }
  condensedInteractionMatrixView.Reset(condensedInteractionMatrix);

  // Allocate storage for the global BlockSparseMatrix. It may be more than necessary (optimizing
  // for speed).
  int nnzConservative = condensedInteractionMatrixView.NumNonZeros();
  for (auto const& [actorOffset, anyMat] : this->_actorMatrices) {
    nnzConservative += GetNumValues(anyMat);
  }
  MOCHI_ASSERT_VERBOSE(nnzConservative % (kBlockSize * kBlockSize) == 0);
  int const nnzBlocksConservative = nnzConservative / (kBlockSize * kBlockSize);
  DynamicArray<int> pointers, indices;
  DynamicArray<T> values;
  pointers.reserve(numBlockRows + 1);
  indices.reserve(nnzBlocksConservative);
  values.reserve(nnzConservative);

  [[maybe_unused]] size_t const initialCapacity[] = {
      pointers.capacity(), indices.capacity(), values.capacity()};

  // Append rows to the global block sparse matrix, one actor at a time.
  pointers.push_back(0);
  for (auto const& [actorOffset, anyMat] : this->_actorMatrices) {
    static_assert(
        std::variant_size_v<decltype(anyMat)> == 4,
        "Please update the code below if the actor matrix types change");
    MOCHI_ASSERT_VERBOSE(actorOffset % kBlockSize == 0);
    int const actorBlockRowOffset = actorOffset / kBlockSize;
    auto* bsp = std::get_if<BlockSparseMatrixView<T const, kBlockSize>>(&anyMat);
    auto* sp = std::get_if<SparseMatrixView<T const>>(&anyMat);
    auto* dense = std::get_if<MatrixView<T const>>(&anyMat);
    if (bsp) {
      AppendActorToGlobalBlockSparseMatrix<kBlockSize>(
          actorBlockRowOffset, *bsp, condensedInteractionMatrixView, pointers, indices, values);
    } else if (sp) {
      BlockViewStructure<kBlockSize> blockedStructure = BlockedStructure<kBlockSize>(*sp);
      AppendActorToGlobalBlockSparseMatrix<kBlockSize>(
          actorBlockRowOffset,
          blockedStructure.ConstView(sp->Values().data()),
          condensedInteractionMatrixView,
          pointers,
          indices,
          values);
    } else if (dense) {
      // TODO(@pabfer): Create BlockedStructure overload for Matrix.
      AppendActorToGlobalBlockSparseMatrix<kBlockSize>(
          actorBlockRowOffset,
          ToBlockSparseMatrix<kBlockSize>(*dense),
          condensedInteractionMatrixView,
          pointers,
          indices,
          values);
    } else {
      MOCHI_ASSERT(false, "Unexpected matrix type."); // Must not be reached if blockable
      return {};
    }
  }

  // Confirm that we reserved enough memory to avoid re-allocation.
  MOCHI_ASSERT_VERBOSE(pointers.capacity() == initialCapacity[0]);
  MOCHI_ASSERT_VERBOSE(indices.capacity() == initialCapacity[1]);
  MOCHI_ASSERT_VERBOSE(values.capacity() == initialCapacity[2]);

  // Move the data into a new BlockSparseMatrix.
  return BlockSparseMatrix<T, kBlockSize>{
      numBlockCols, std::move(pointers), std::move(indices), std::move(values)};
}

template <typename T>
template <int kBlockSize>
bool IslandOperators<T>::IsBlockable() const {
  // Check interaction matrices first to fail fast.
  for (auto const& [rOffset, cOffset, anyMat, _] : this->_interactionMatrices) {
    if ((rOffset % kBlockSize != 0) || (cOffset % kBlockSize != 0)) {
      return false;
    }
    static_assert(
        std::variant_size_v<decltype(anyMat)> == 4,
        "Please update the code below if the interaction matrix types change");
    auto* bsp = std::get_if<BlockSparseMatrixView<T const, kBlockSize>>(&anyMat);
    auto* sp = std::get_if<SparseMatrixView<T const>>(&anyMat);
    auto* dense = std::get_if<MatrixView<T const>>(&anyMat);
    if (bsp || (sp && mochi::IsBlockable<kBlockSize>(*sp)) ||
        (dense && mochi::IsBlockable<kBlockSize>(*dense))) {
      continue;
    } else {
      return false;
    }
  }

  // Check actor matrices.
  for (auto const& [actorOffset, anyMat] : this->_actorMatrices) {
    MOCHI_ASSERT_VERBOSE(actorOffset % kBlockSize == 0); // Must be true if we have gotten this far.
    static_assert(
        std::variant_size_v<decltype(anyMat)> == 4,
        "Please update the code below if the actor matrix types change");
    auto* bsp = std::get_if<BlockSparseMatrixView<T const, kBlockSize>>(&anyMat);
    auto* sp = std::get_if<SparseMatrixView<T const>>(&anyMat);
    auto* dense = std::get_if<MatrixView<T const>>(&anyMat);
    if (bsp || (sp && mochi::IsBlockable<kBlockSize>(*sp)) ||
        (dense && mochi::IsBlockable<kBlockSize>(*dense))) {
      continue;
    } else {
      return false;
    }
  }

  return true;
}

template <typename T>
AnyMatrix<T> IslandOperators<T>::CondenseFullMatrix() const {
  static_assert(
      std::variant_size_v<AnyMatrix<T>> == 4,
      "Please update the code below if AnyMatrix is updated");
  if (IsBlockable<3>()) {
    return FullBlockSparseMatrix<3>();
  } else if (IsBlockable<4>()) {
    return FullBlockSparseMatrix<4>();
  } else {
    auto sp = FullSparseMatrix();
    // TODO(@natepayne): The implementation of the branches below is sub-optimal:
    // ToBlockSparseMatrix calls again 'IsBlockable' and copies the values (which could be moved
    // here).
    if (mochi::IsBlockable<3>(sp)) {
      return ToBlockSparseMatrix<3>(sp);
    } else if (mochi::IsBlockable<4>(sp)) {
      return ToBlockSparseMatrix<4>(sp);
    } else {
      return sp;
    }
  }
}

// Explicit instantiations
template struct IslandOperators<float>;
template struct IslandOperators<double>;

} // namespace mochi
