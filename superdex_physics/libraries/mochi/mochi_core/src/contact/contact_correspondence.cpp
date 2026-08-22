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

#include <mochi_core/contact/contact_correspondence.h>

#include <limits>

using namespace mochi;

void ContactCorrespondence::Clear() {
  // Check roll-over of the epoch counter.
  if (_epoch == std::numeric_limits<int>::max()) {
    _epoch = 0;
    std::fill(_stamp.begin(), _stamp.end(), 0);
  }

  // Update epoch.
  ++_epoch;

  // Clear the missing samples.
  _missing.clear();
}

Span<ContactCorrespondence::Pair const> ContactCorrespondence::GetMissingSamplesImpl(
    Span<int const> src,
    Span<int const> dst) {
  Clear();

  // Registration phase: update the stamp for dst samples.
  for (int const sampleIdx : dst) {
    MOCHI_ASSERT_VERBOSE(sampleIdx >= 0 && sampleIdx < isize(_stamp), "Invalid sample index");
    _stamp[sampleIdx] = _epoch;
  }

  // Matching phase: find missing samples in src.
  for (int contactIdx = 0; contactIdx < isize(src); ++contactIdx) {
    int const sampleIdx = src[contactIdx];
    MOCHI_ASSERT_VERBOSE(sampleIdx >= 0 && sampleIdx < isize(_stamp), "Invalid sample index");
    if (_stamp[sampleIdx] != _epoch) {
      _missing.emplace_back(sampleIdx, contactIdx);
    }
  }

  return _missing;
}

Span<ContactCorrespondence::Pair const> ContactCorrespondence::GetMissingSamplesImpl(
    Span<int const> srcSamples,
    Span<int const> srcFeatures,
    Span<int const> dstSamples,
    Span<int const> dstFeatures) {
  MOCHI_ASSERT_VERBOSE(srcSamples.size() == srcFeatures.size(), "Mismatched src sizes");
  MOCHI_ASSERT_VERBOSE(dstSamples.size() == dstFeatures.size(), "Mismatched dst sizes");

  Clear();

  // If needed, allocate memory for arrays used by colliders with feature indices.
  _startIdx.resize_noinit(_numSamples);
  _count.resize_noinit(_numSamples);

  // Registration phase: record start index and count for each sample in dst.
  for (int i = 0; i < isize(dstSamples); ++i) {
    int const sample = dstSamples[i];
    MOCHI_ASSERT_VERBOSE(sample >= 0 && sample < isize(_stamp), "Invalid sample index");
    if (_stamp[sample] != _epoch) {
      _stamp[sample] = _epoch;
      _startIdx[sample] = i;
      _count[sample] = 1;
    } else {
      MOCHI_ASSERT_VERBOSE(
          i == _startIdx[sample] + _count[sample],
          "Same sample must appear consecutively in dst arrays");
      ++_count[sample];
    }
  }

  // Matching phase: find missing (sample, feature) pairs.
  for (int contactIdx = 0; contactIdx < isize(srcSamples); ++contactIdx) {
    int const sample = srcSamples[contactIdx];
    int const feature = srcFeatures[contactIdx];
    MOCHI_ASSERT_VERBOSE(sample >= 0 && sample < isize(_stamp), "Invalid sample index");

    if (_stamp[sample] != _epoch) {
      // Sample not in dst.
      _missing.emplace_back(sample, contactIdx);
    } else {
      // Sample in dst — linear scan for matching feature.
      int const start = _startIdx[sample];
      int const end = start + _count[sample];
      bool found = false;
      for (int j = start; j < end; ++j) {
        if (dstFeatures[j] == feature) {
          found = true;
          break;
        }
      }
      if (!found) {
        _missing.emplace_back(sample, contactIdx);
      }
    }
  }

  return _missing;
}
