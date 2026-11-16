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

#include <superdex_physics.h>
#include <superdex_robotics/core/component_base.h>
#include <superdex_robotics/sensors/sensor_format.h>

#include <mochi_core/utils/transform_rt.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace superdex::robotics {

class RoboticsContext;
class SensorRecordingSink;

/* @brief Simulation or hardware time supplied after one acquisition step. */
struct StepTime {
  double t = 0.0;
  double dt = 0.0;
  int64_t step = 0;
};

/* @brief Borrowed view of the latest completed sensor output.
 * The payload remains valid until this sensor publishes again, resets, is initialized again, or is
 * destroyed. SensorGraph execution and borrowed reads are externally serialized. */
struct SensorSample {
  int64_t captureStep = -1;
  double captureTime = 0.0;
  void const* data = nullptr;
  int64_t elementCount = 0;

  [[nodiscard]] MOCHI_API Span<float const> Floats() const;
};

/* @brief Result of advancing one sensor for one graph step.
 * Meaningful only when the Step error argument remains OK; a non-OK error is always failure. */
enum class SensorStepResult {
  NoUpdate,
  Published,
};

/* @brief Resources available while a graph initializes a sensor. */
struct SensorScope {
  Scene* scene = nullptr;
  RoboticsContext* robotics = nullptr;
};

/* @brief Base class for synchronous source and derived sensor nodes. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API SensorBase : public ComponentBase {
 public:
  /* @brief Bind fixed graph inputs and prepare immutable output storage. */
  void Init(Span<SensorBase const* const> inputs, SensorScope const& scope, superdex::Error& error);

  /* @brief Clear the published sample, then reset sensor-specific episode state. */
  void Reset() final;

  /* @brief Advance this sensor after one physics or hardware step. The returned state
   * is meaningful only while @p error is OK. */
  [[nodiscard]] virtual SensorStepResult Step(StepTime const& now, superdex::Error& error) = 0;

  [[nodiscard]] SensorFormat const& GetFormat() const {
    return _format;
  }

  /* @brief Return the latest sample without triggering acquisition or allocation. */
  [[nodiscard]] SensorSample Latest() const {
    return _latest;
  }

  /* @brief Copy the latest flat payload for owning API and language-binding boundaries. */
  [[nodiscard]] DynamicArray<real> CopyLatestValues() const;

  /* @brief Query optional sensor-specific metadata by a stable library-owned key.
   * The bytes are an opaque borrowed view of one live producer-owned object, not a wire format.
   * Interpret them only through TryGetSensorMetadata() while this initialization remains live; do
   * not copy, persist or deserialize the raw bytes. */
  [[nodiscard]] virtual Span<std::byte const> TryGetMetadata(std::string_view key) const;

  /* @brief Write constant, sensor-specific recording fields. */
  virtual void ConfigureRecording(SensorRecordingSink& sink, superdex::Error& error) const;

  /* @brief Write the latest completed sample and any sensor-specific per-sample fields. */
  virtual void Record(SensorRecordingSink& sink, superdex::Error& error) const;

  /* @brief The sensor's pose relative to its parent frame: the associated link actor for attached
   * sensors, or the scene root for unattached sensors. Uses the aFromB convention. */
  [[nodiscard]] TransformRT const& GetParentFromSensor() const {
    return _parentFromSensor;
  }

  /* @brief Set the parent-from-sensor transform using the aFromB convention. */
  void SetParentFromSensor(TransformRT const& parentFromSensor) {
    _parentFromSensor = parentFromSensor;
  }

  /* @brief Return the sensor pose in world space.
   * Attached actors are already world-resolved. For an unattached sensor, parentFromSensor is the
   * scene-root-to-sensor transform and the scene-root-to-world transform is currently identity. */
  [[nodiscard]] TransformRT GetWorldTransform() const;

 protected:
  /* @brief Construct an attached sensor. Sensors that require an actor validate it themselves. */
  SensorBase(Actor* actor, superdex::Error& error);

  /* @brief Construct an unattached node such as a fusion sensor. */
  explicit SensorBase(superdex::Error& error);

  /* @brief Resolve sensor-specific resources and return the fixed output format. */
  [[nodiscard]] virtual SensorFormat InitSensor(
      SensorScope const& scope,
      superdex::Error& error) = 0;

  /* @brief Reset sensor-specific state. Stateless sensors implement an explicit no-op. */
  virtual void ResetSensor() = 0;

  [[nodiscard]] Span<SensorBase const* const> Inputs() const {
    return _inputs;
  }

  /* @brief Complete the sensor's write stroke by copying one full sample into owned storage. */
  bool PublishSample(
      StepTime const& now,
      Span<float const> sample,
      superdex::Error& error,
      SensorSample const* source = nullptr);

  /* Sensor pose relative to its parent frame; see GetParentFromSensor(). */
  TransformRT _parentFromSensor;

  ~SensorBase() override;

 private:
  void PrepareOutput(SensorFormat format, superdex::Error& error);
  void ClearOutput();

  SensorFormat _format;
  DynamicArray<SensorBase const*> _inputs;
  DynamicArray<float> _output;
  SensorSample _latest;
};

template <typename T>
struct SensorMetadataKey;

/* @brief Query optional typed sensor metadata. The returned object remains producer-owned and is
 * valid only while the sensor's current initialization remains live. */
template <typename T>
[[nodiscard]] T const* TryGetSensorMetadata(SensorBase const& sensor) {
  Span<std::byte const> const bytes = sensor.TryGetMetadata(SensorMetadataKey<T>::Key);
  if (bytes.empty() || bytes.size() != sizeof(T)) {
    return nullptr;
  }
  auto const address = reinterpret_cast<uintptr_t>(bytes.data());
  if (address % alignof(T) != 0) {
    return nullptr;
  }
  // TryGetMetadata returns the byte view of an already-live T owned by the producer. This recovers
  // that original pointer; it does not materialize T from copied object representation, so metadata
  // may contain owning strings and other non-trivially-copyable members.
  return reinterpret_cast<T const*>(bytes.data());
}

} // namespace superdex::robotics
