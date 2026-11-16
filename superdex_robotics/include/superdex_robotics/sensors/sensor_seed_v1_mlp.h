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

#include <mochi_core/utils/span.h>
#include <superdex_physics.h>
#include <superdex_robotics/sensors/sensor_base.h>

#include <memory>

namespace mochi {
struct ContactPoint;
} // namespace mochi

namespace mochi::ai {
template <typename T>
class Mlp;
} // namespace mochi::ai

namespace superdex::robotics {

/* @brief SEED tactile sensor parameters. */
struct SensorSeedV1MLPParams {
  /* Path to the JSON weights file (forcenet_weights.json). */
  DynamicString weightsPath;

  /* Offset from link origin to hemisphere center in link-local frame [m]. */
  Real3 hemisphereCenterOffset = {};

  /* When true, output = raw contact force (skip MLP inference). */
  bool bypassModel = false;

  /* @brief Load sensor parameters from a JSON file.
   * @param path Path to a .superdex_sensor JSON file.
   * @param error Error status.
   * @return Loaded parameters. */

  /* @brief Save sensor parameters to a JSON file.
   * @param path Destination file path.
   * @param error Error status. */

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorSeedV1MLPParams)
  MOCHI_FIELD(weightsPath);
  MOCHI_FIELD(hemisphereCenterOffset) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(bypassModel);
  MOCHI_STRUCT_END()
};

/* @brief SEED tactile sensor result. */
struct SensorSeedV1MLPResult {
  Real3 inputForce = {};
  Real3 inputNormal = {};
  Real3 outputForce = {};
  Real3 contactCentroid = {};
  Real3 contactNormal = {};
  int numContacts = 0;
};

/* [Experimental] SEED tactile sensor using Mochi-native MLP inference.
 *
 * Functionally identical to the internal Torch implementation but uses mochi::ai::Mlp instead of
 * libtorch. Loads weights from a JSON file exported by sensor_models/seed/export.py
 * (--export_weights_json flag). Suitable for platforms without libtorch. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API SensorSeedV1MLP : public SensorBase {
 public:
  /* @brief Registration type name for this sensor (see RoboticsContext::RegisterSensorType).
   * @return This sensor's registration type name. */
  static constexpr std::string_view TypeName() {
    return "SENSOR_SEED_V1_MLP";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  using Params = SensorSeedV1MLPParams;
  using Result = SensorSeedV1MLPResult;

  /* @brief Construct from @p paramArgs -- either a params file path or an inline JSON string (empty
   * uses defaults). When loaded from a file, resolves a relative @c weightsPath against the params
   * file's directory; loads the MLP model. This is the factory RoboticsContext::RegisterSensor
   * uses. */
  SensorSeedV1MLP(Actor* linkActor, std::string_view paramArgs, superdex::Error& error);

  /* @brief Construct from a pre-loaded params struct (programmatic / non-filesystem construction).
   * @c weightsPath is used as-is (assumed already resolved). */
  SensorSeedV1MLP(Actor* linkActor, Params const& params, superdex::Error& error);

  MOCHI_DECLARE_NO_COPY_NO_MOVE(SensorSeedV1MLP)

  ~SensorSeedV1MLP() override;

  /* @brief Compute sensor signal.
   * @param error Error status.
   * @return Sensor result. */
  Result ComputeSignal(superdex::Error& error);

  /* @brief Compute the sensor signal from externally provided contacts (for testing).
   * @param contacts Contact points to process.
   * @param error Error status.
   * @return Computed sensor result. */
  Result ComputeSignal(Span<ContactPoint const> contacts, superdex::Error& error);

  /* @brief Get sensor parameters.
   * @return Reference to parameters. */
  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  /* @brief Set sensor parameters.
   * @param params New parameters.
   * @param error Error status. */
  void SetParams(Params const& params, superdex::Error& error);

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override {
    /* Params and the loaded MLP are configuration; each sample reads contacts afresh. */
  }

  /* Register the contact query and apply @p params (loading the MLP model). Shared by both ctors.
   */
  void Initialize(Params const& params, superdex::Error& error);
  Result ProcessContacts(Span<ContactPoint const> contacts, superdex::Error& error);

  Params _params;
  std::unique_ptr<ai::Mlp<float>> _mlp;
  QueryHandle _contactQueryHandle = {};
  bool _modelLoaded = false;
};

} // namespace superdex::robotics
