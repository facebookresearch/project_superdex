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

#include <mochi_renderer/buffer.h>
#include <mochi_renderer/render_space.h>
#include <mochi_renderer/types.h>

#include <mochi_core/utils/coordinate_space_converter.h>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

using namespace mochi_renderer;

// --- RenderResult tests ---

TEST(RenderResultTest, DefaultIsInvalid) {
  RenderResult result;
  EXPECT_FALSE(result.IsValid());
  EXPECT_TRUE(result.pixels.empty());
  EXPECT_EQ(result.width, 0);
  EXPECT_EQ(result.height, 0);
}

TEST(RenderResultTest, PopulatedIsValid) {
  RenderResult result;
  result.width = 64;
  result.height = 48;
  result.channels = 4;
  result.format = ImageFormat::RGBA8;
  result.pixels.resize(64 * 48 * 4, 0);
  EXPECT_TRUE(result.IsValid());
  EXPECT_EQ(result.pixels.size(), 64u * 48u * 4u);
}

TEST(RenderResultTest, ZeroDimensionIsInvalid) {
  RenderResult result;
  result.pixels.resize(100, 0);
  result.width = 0;
  result.height = 10;
  EXPECT_FALSE(result.IsValid());
}

// --- SceneViewSettings tests ---

TEST(SceneViewSettingsTest, DefaultValues) {
  SceneViewSettings settings;
  EXPECT_TRUE(settings.msaaEnabled);
  EXPECT_EQ(settings.msaaSampleCount, 4);
  EXPECT_TRUE(settings.shadowsEnabled);
  EXPECT_TRUE(settings.ssaoEnabled);
  EXPECT_TRUE(settings.postProcessingEnabled);
  EXPECT_TRUE(settings.bloomEnabled);
  EXPECT_FLOAT_EQ(settings.bloomStrength, 0.05f);
  EXPECT_TRUE(settings.vignetteEnabled);
}

TEST(SceneViewSettingsTest, CustomValues) {
  SceneViewSettings settings;
  settings.msaaEnabled = false;
  settings.msaaSampleCount = 8;
  settings.bloomStrength = 0.1f;
  EXPECT_FALSE(settings.msaaEnabled);
  EXPECT_EQ(settings.msaaSampleCount, 8);
  EXPECT_FLOAT_EQ(settings.bloomStrength, 0.1f);
}

// --- PipelineMode tests ---

TEST(PipelineModeTest, EnumValues) {
  PipelineMode sync = PipelineMode::Synchronized;
  PipelineMode delay = PipelineMode::OneFrameDelay;
  PipelineMode perf = PipelineMode::MaxPerformance;
  EXPECT_NE(sync, delay);
  EXPECT_NE(delay, perf);
  EXPECT_NE(sync, perf);
}

// --- ProducerConsumerBuffer tests ---

TEST(ProducerConsumerBufferTest, InitialConsumeReturnsFalse) {
  ProducerConsumerBuffer<int> buffer;
  EXPECT_FALSE(buffer.Consume());
}

TEST(ProducerConsumerBufferTest, ProduceThenConsume) {
  ProducerConsumerBuffer<int> buffer;
  buffer.GetProducerData() = 42;
  buffer.Produce();
  EXPECT_TRUE(buffer.Consume());
  EXPECT_EQ(buffer.GetConsumerData(), 42);
}

TEST(ProducerConsumerBufferTest, ConsumeOnlyOnce) {
  ProducerConsumerBuffer<int> buffer;
  buffer.GetProducerData() = 10;
  buffer.Produce();
  EXPECT_TRUE(buffer.Consume());
  EXPECT_EQ(buffer.GetConsumerData(), 10);
  // Second consume without new produce returns false
  EXPECT_FALSE(buffer.Consume());
  // But data is still readable
  EXPECT_EQ(buffer.GetConsumerData(), 10);
}

TEST(ProducerConsumerBufferTest, LatestValueWins) {
  ProducerConsumerBuffer<int> buffer;
  buffer.GetProducerData() = 1;
  buffer.Produce();
  buffer.GetProducerData() = 2;
  buffer.Produce();
  buffer.GetProducerData() = 3;
  buffer.Produce();
  // Consumer should get the latest produced value
  EXPECT_TRUE(buffer.Consume());
  // The value should be one of the produced values (triple buffer
  // means the latest swapped-in value is available)
  int val = buffer.GetConsumerData();
  EXPECT_TRUE(val >= 1 && val <= 3);
}

TEST(ProducerConsumerBufferTest, ThreadSafety) {
  ProducerConsumerBuffer<int> buffer;
  constexpr int kIterations = 10000;

  std::atomic<bool> producerReady{false};
  std::atomic<bool> consumerReady{false};
  std::atomic<bool> producerDone{false};

  std::thread producer([&]() {
    producerReady.store(true, std::memory_order_release);
    while (!consumerReady.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kIterations; ++i) {
      buffer.GetProducerData() = i;
      buffer.Produce();
    }
    producerDone.store(true, std::memory_order_release);
  });

  int lastConsumed = -1;
  int consumeCount = 0;
  std::thread consumer([&]() {
    consumerReady.store(true, std::memory_order_release);
    while (!producerReady.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (true) {
      if (buffer.Consume()) {
        int val = buffer.GetConsumerData();
        EXPECT_GE(val, lastConsumed);
        lastConsumed = val;
        ++consumeCount;
      } else if (producerDone.load(std::memory_order_acquire)) {
        break;
      }
    }
  });

  producer.join();
  consumer.join();
  EXPECT_GT(consumeCount, 0);
}

TEST(ProducerConsumerBufferTest, StructType) {
  struct Payload {
    float x = 0.0f;
    float y = 0.0f;
    std::string name;
  };

  ProducerConsumerBuffer<Payload> buffer;
  auto& data = buffer.GetProducerData();
  data.x = 1.5f;
  data.y = 2.5f;
  data.name = "test";
  buffer.Produce();

  EXPECT_TRUE(buffer.Consume());
  auto& consumed = buffer.GetConsumerData();
  EXPECT_FLOAT_EQ(consumed.x, 1.5f);
  EXPECT_FLOAT_EQ(consumed.y, 2.5f);
  EXPECT_EQ(consumed.name, "test");
}
