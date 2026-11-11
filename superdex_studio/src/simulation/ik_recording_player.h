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

#include <superdex_robotics/superdex_robotics.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace superdex::studio {

struct IKRecordingSample {
  uint64_t frame = 0;
  uint64_t recordingOrder = 0; // Reader-local tie-break, not a payload field.
  int64_t time = 0;
  bool tracked =
      false; // Legacy availability bit; simulated snapshots do not encode tracking confidence.
  bool hasPose = false; // True only for a complete, validated native physics snapshot.
  mochi::DynamicArray<mochi::real> inputBotConfiguration;
  mochi::TransformRT recordedWorldFrameFromInputBaseFrame;
};

/* @brief Immutable native physics-scene articulation clip (simulated, not raw tracking).
 * Loading is off-thread; sampling performs no file I/O.
 */
class IKRecordingClip {
 public:
  // Native Open/ReadScalars are eager and non-cancellable; cancellation is checked around them
  // and while constructing the immutable clip.
  static std::shared_ptr<IKRecordingClip const> Load(
      std::string const& path,
      superdex::robotics::BotPrefab const& inputBot,
      std::string& diagnostic,
      std::atomic_bool const* cancelled = nullptr);

  IKRecordingSample const* SampleAt(int64_t time) const;
  mochi::TransformRT InputBotRoot(
      IKRecordingSample const& sample,
      mochi::TransformRT const& authoredWorldFrameFromInputBaseFrame,
      bool fixedBase) const;
  int64_t GetFirstTime() const {
    return _first;
  }
  int64_t GetLastTime() const {
    return _last;
  }
  double GetDurationSeconds() const;
  std::vector<int64_t> const& GetTimes() const {
    return _times;
  }
  std::string const& GetCompatibility() const {
    return _compatibility;
  }

 private:
  int64_t _first = 0;
  int64_t _last = 0;
  mochi::TransformRT _firstRecordedWorldFrameFromInputBaseFrame;
  std::vector<IKRecordingSample> _samples;
  std::vector<int64_t> _times;
  std::string _compatibility;
};

/* @brief 1x active steady-clock cursor; pause/deactivation exclude idle wall time. */
class IKRecordingClock {
 public:
  using Clock = std::chrono::steady_clock;
  void Configure(IKRecordingClip const& clip);
  void Configure(int64_t first, int64_t last, std::vector<int64_t> times);
  void Play(Clock::time_point now = Clock::now());
  void Pause(Clock::time_point now = Clock::now());
  void Stop();
  void Step(Clock::time_point now = Clock::now());
  int64_t Tick(Clock::time_point now = Clock::now());
  int64_t GetCursor() const {
    return _cursor;
  }
  bool IsPlaying() const {
    return _playing;
  }

 private:
  int64_t _first = 0;
  int64_t _last = 0;
  int64_t _cursor = 0;
  std::vector<int64_t> _times;
  Clock::time_point _lastTick{};
  bool _playing = false;
};

} // namespace superdex::studio
