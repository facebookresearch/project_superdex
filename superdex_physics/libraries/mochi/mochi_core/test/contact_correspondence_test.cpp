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
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

using namespace mochi;

TEST(ContactCorrespondence, NoFeaturesEmptySrcAndDst) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src;
  DynamicArray<int> dst;

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, NoFeaturesEmptySrc) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src;
  DynamicArray<int> dst = {2, 0, 1};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, NoFeaturesEmptyDst) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src = {2, 0, 1};
  DynamicArray<int> dst;

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  // All src samples are missing since dst is empty.
  DynamicArray<ContactCorrespondence::Pair> const expected = {{2, 0}, {0, 1}, {1, 2}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, NoFeaturesAllSamplesMatch) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src = {3, 1, 0, 2};
  DynamicArray<int> dst = {2, 0, 3, 1};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, NoFeaturesAllSamplesMissing) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src = {1, 2, 0};
  DynamicArray<int> dst = {7, 5, 6};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  // All src samples are missing.
  DynamicArray<ContactCorrespondence::Pair> const expected = {{1, 0}, {2, 1}, {0, 2}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, NoFeaturesSomeSamplesMissing) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src = {4, 0, 6, 2};
  DynamicArray<int> dst = {3, 2, 5, 1, 4};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  // Samples 0 and 6 are missing
  DynamicArray<ContactCorrespondence::Pair> const expected = {{0, 1}, {6, 2}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, NoFeaturesDstSupersetOfSrc) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src = {4, 2};
  DynamicArray<int> dst = {6, 2, 0, 4, 1, 5, 3};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, NoFeaturesDuplicateSamplesInSrc) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> src = {3, 1, 1, 3};
  DynamicArray<int> dst = {4, 0, 2};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(src, {}, dst, {});

  // All src samples (1 and 3) are missing, with their respective contact indices.
  DynamicArray<ContactCorrespondence::Pair> const expected = {{3, 0}, {1, 1}, {1, 2}, {3, 3}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, NoFeaturesManyEpochs) {
  int constexpr kMaxCount = 1000;
  ContactCorrespondence correspondence(kMaxCount);
  DynamicArray<int> src = {2, 0, 1, 3};
  DynamicArray<int> dst = {1, 2};

  // Call GetMissingSamples many times to exercise epoch handling.
  for (int epoch = 1; epoch < kMaxCount - 3; ++epoch) {
    for (int& s : src) {
      ++s;
    }
    for (int& d : dst) {
      ++d;
    }
    Span<ContactCorrespondence::Pair const> missing =
        correspondence.GetMissingSamples(src, {}, dst, {});
    DynamicArray<ContactCorrespondence::Pair> const expected = {{epoch, 1}, {epoch + 3, 3}};
    EXPECT_EQ(missing, MakeConstSpan(expected));
  }
}

TEST(ContactCorrespondence, WithFeaturesEmptySrcAndDst) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples;
  DynamicArray<int> srcFeatures;
  DynamicArray<int> dstSamples;
  DynamicArray<int> dstFeatures;

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, WithFeaturesEmptySrc) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples;
  DynamicArray<int> srcFeatures;
  DynamicArray<int> dstSamples = {2, 0, 1};
  DynamicArray<int> dstFeatures = {5, 3, 7};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, WithFeaturesEmptyDst) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples = {2, 0, 1};
  DynamicArray<int> srcFeatures = {5, 3, 7};
  DynamicArray<int> dstSamples;
  DynamicArray<int> dstFeatures;

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  // All src (sample, feature) pairs are missing.
  DynamicArray<ContactCorrespondence::Pair> const expected = {{2, 0}, {0, 1}, {1, 2}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, WithFeaturesAllPairsMatch) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples = {3, 1, 0};
  DynamicArray<int> srcFeatures = {5, 2, 8};
  DynamicArray<int> dstSamples = {0, 1, 3};
  DynamicArray<int> dstFeatures = {8, 2, 5};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, WithFeaturesAllPairsMissing) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples = {1, 2, 0};
  DynamicArray<int> srcFeatures = {5, 3, 7};
  DynamicArray<int> dstSamples = {7, 5, 6};
  DynamicArray<int> dstFeatures = {1, 2, 3};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  // All src (sample, feature) pairs are missing (different samples).
  DynamicArray<ContactCorrespondence::Pair> const expected = {{1, 0}, {2, 1}, {0, 2}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, WithFeaturesSampleMatchesButFeatureDiffers) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples = {2, 2, 3};
  DynamicArray<int> srcFeatures = {5, 6, 7};
  DynamicArray<int> dstSamples = {2, 3};
  DynamicArray<int> dstFeatures = {5, 8}; // Feature 7 missing for sample 3

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  // Sample 2 with feature 6 is missing, sample 3 with feature 7 is missing.
  DynamicArray<ContactCorrespondence::Pair> const expected = {{2, 1}, {3, 2}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, WithFeaturesMultipleFeaturesPerSample) {
  ContactCorrespondence correspondence(10);
  // Sample 2 has features 5, 6, 7 (consecutive in src)
  DynamicArray<int> srcSamples = {2, 2, 2, 3};
  DynamicArray<int> srcFeatures = {5, 6, 7, 9};
  // Sample 2 has features 5, 7 in dst (consecutive)
  DynamicArray<int> dstSamples = {2, 2, 3};
  DynamicArray<int> dstFeatures = {5, 7, 9};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  // Sample 2 with feature 6 is missing (at contact index 1).
  DynamicArray<ContactCorrespondence::Pair> const expected = {{2, 1}};
  EXPECT_EQ(missing, MakeConstSpan(expected));
}

TEST(ContactCorrespondence, WithFeaturesDstSupersetOfSrc) {
  ContactCorrespondence correspondence(10);
  DynamicArray<int> srcSamples = {4, 2};
  DynamicArray<int> srcFeatures = {1, 3};
  DynamicArray<int> dstSamples = {2, 2, 4, 4, 4};
  DynamicArray<int> dstFeatures = {3, 5, 0, 1, 2};

  Span<ContactCorrespondence::Pair const> missing =
      correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);

  EXPECT_TRUE(missing.empty());
}

TEST(ContactCorrespondence, WithFeaturesManyEpochs) {
  int constexpr kMaxCount = 1000;
  ContactCorrespondence correspondence(kMaxCount);
  DynamicArray<int> srcSamples = {2, 2, 0, 3};
  DynamicArray<int> srcFeatures = {5, 6, 7, 8};
  DynamicArray<int> dstSamples = {2, 2};
  DynamicArray<int> dstFeatures = {5, 6};

  // Call GetMissingSamples many times to exercise epoch handling.
  for (int epoch = 1; epoch < kMaxCount - 3; ++epoch) {
    for (int& s : srcSamples) {
      ++s;
    }
    for (int& d : dstSamples) {
      ++d;
    }
    Span<ContactCorrespondence::Pair const> missing =
        correspondence.GetMissingSamples(srcSamples, srcFeatures, dstSamples, dstFeatures);
    // Samples 0+epoch and 3+epoch are missing (at contact indices 2 and 3).
    DynamicArray<ContactCorrespondence::Pair> const expected = {{epoch, 2}, {epoch + 3, 3}};
    EXPECT_EQ(missing, MakeConstSpan(expected));
  }
}
