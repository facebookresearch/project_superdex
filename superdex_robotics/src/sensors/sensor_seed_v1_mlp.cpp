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

#include <mochi_core/ai/mlp.h>
#include <superdex_robotics/sensors/sensor_seed_v1_mlp.h>

#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/json_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <optional>

#include <array>
#include <fstream>

using namespace mochi;
using namespace superdex::robotics;

namespace {

/* Load a Mochi ai::Mlp<float> from a JSON weights file exported by
 * sensor_models/seed/export.py --export_weights_json. Format:
 *   { "architecture": {...}, "layers": [ {"weight": [[...]], "bias": [...]}, ... ] }
 * Weight matrices are row-major (out_features x in_features). */
std::optional<ai::Mlp<float>> LoadMlpFromJson(std::string_view path, Error& error) {
  auto json = ParseJsonFromFile(path, error);
  MOCHI_ERROR_RETURN(error, std::nullopt);

  if (!json.is<picojson::object>()) {
    MOCHI_ERROR_SET(error, "Weights JSON must be an object.");
    return std::nullopt;
  }
  auto const& root = json.get<picojson::object>();

  auto layersIt = root.find("layers");
  if (layersIt == root.end() || !layersIt->second.is<picojson::array>()) {
    MOCHI_ERROR_SET(error, "Weights JSON missing 'layers' array.");
    return std::nullopt;
  }
  auto const& layersArr = layersIt->second.get<picojson::array>();
  int const numLayers = isize(layersArr);
  if (numLayers < 1) {
    MOCHI_ERROR_SET(error, "Weights JSON has no layers.");
    return std::nullopt;
  }

  DynamicArray<ai::MlpLayer<float>> mlpLayers;
  mlpLayers.reserve(numLayers);

  for (int iLayer = 0; iLayer < numLayers; ++iLayer) {
    if (!layersArr[iLayer].is<picojson::object>()) {
      MOCHI_ERROR_SET(error, "Layer entry must be an object.");
      return std::nullopt;
    }
    auto const& layerObj = layersArr[iLayer].get<picojson::object>();

    // Parse weight matrix (row-major: out_features x in_features)
    auto wIt = layerObj.find("weight");
    if (wIt == layerObj.end() || !wIt->second.is<picojson::array>()) {
      MOCHI_ERROR_SET(error, "Layer missing 'weight' array.");
      return {};
    }
    auto const& wRows = wIt->second.get<picojson::array>();
    int const outDim = isize(wRows);
    if (outDim == 0 || !wRows[0].is<picojson::array>()) {
      MOCHI_ERROR_SET(error, "Weight rows must be non-empty arrays.");
      return {};
    }
    int const inDim = isize(wRows[0].get<picojson::array>());

    constexpr int kMaxDim = 1 << 16;
    MOCHI_ERROR_IF(outDim > kMaxDim || inDim > kMaxDim, error, "Layer dimensions exceed limit.");
    MOCHI_ERROR_RETURN(error, {});

    Matrix<float> W(outDim, inDim);
    for (int r = 0; r < outDim; ++r) {
      if (!wRows[r].is<picojson::array>()) {
        MOCHI_ERROR_SET(error, "Weight row is not an array.");
        return {};
      }
      auto const& row = wRows[r].get<picojson::array>();
      if (isize(row) != inDim) {
        MOCHI_ERROR_SET(error, "Weight row has wrong number of elements.");
        return {};
      }
      for (int c = 0; c < inDim; ++c) {
        if (!row[c].is<double>()) {
          MOCHI_ERROR_SET(error, "Weight value is not a number.");
          return {};
        }
        W(r, c) = static_cast<float>(row[c].get<double>());
      }
    }

    // Parse bias vector
    auto bIt = layerObj.find("bias");
    if (bIt == layerObj.end() || !bIt->second.is<picojson::array>()) {
      MOCHI_ERROR_SET(error, "Layer missing 'bias' array.");
      return {};
    }
    auto const& bArr = bIt->second.get<picojson::array>();
    if (isize(bArr) != outDim) {
      MOCHI_ERROR_SET(error, "Bias array size does not match output dimension.");
      return {};
    }
    ColumnVector<float> b(outDim);
    for (int i = 0; i < outDim; ++i) {
      if (!bArr[i].is<double>()) {
        MOCHI_ERROR_SET(error, "Bias value is not a number.");
        return {};
      }
      b(i) = static_cast<float>(bArr[i].get<double>());
    }

    // All hidden layers use ReLU, final layer uses Identity
    if (iLayer < numLayers - 1) {
      mlpLayers.emplace_back(std::move(W), std::move(b), ai::ReLUActivation<float>{});
    } else {
      mlpLayers.emplace_back(std::move(W), std::move(b), ai::IdentityActivation<float>{});
    }
  }

  return ai::Mlp<float>(std::move(mlpLayers));
}

} // namespace

SensorSeedV1MLP::SensorSeedV1MLP(
    Actor* linkActor,
    SensorSeedV1MLP::Params const& params,
    Error& error)
    : SensorBase(linkActor, error) {
  Initialize(params, error);
}

SensorSeedV1MLP::SensorSeedV1MLP(Actor* linkActor, std::string_view paramArgs, Error& error)
    : SensorBase(linkActor, error) {
  auto params = LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  MOCHI_ERROR_RETURN(error);
  // Paths in params files use the same resolution as other bot-file references.
  if (!paramArgs.empty() && !IsInlineJson(paramArgs)) {
    MakePathAbsolute(params.weightsPath, paramArgs, kBotPathMaxParentDepth, error);
    MOCHI_ERROR_RETURN(error);
  }
  Initialize(params, error);
}

SensorSeedV1MLP::~SensorSeedV1MLP() {
  Actor* const actor = GetActor();
  if (actor && _contactQueryHandle.value != 0) {
    actor->CancelQuery(_contactQueryHandle);
  }
}

SensorFormat SensorSeedV1MLP::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "SensorSeedV1MLP does not accept inputs");
  MOCHI_ERROR_RETURN(error, {});
  return {
      .kind = DynamicString{kSensorFormatKindForce3D},
      .shape = {3},
      .axes = {MakeForceComponentAxis()},
  };
}

SensorStepResult SensorSeedV1MLP::Step(StepTime const& now, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.empty(), error, "SensorSeedV1MLP does not accept inputs");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  if (now.dt <= 0.0) {
    return SensorStepResult::NoUpdate;
  }
  Result const signal = ComputeSignal(error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  std::array<float, 3> const values = {
      static_cast<float>(signal.outputForce[0]),
      static_cast<float>(signal.outputForce[1]),
      static_cast<float>(signal.outputForce[2]),
  };
  PublishSample(now, Span<float const>{values}, error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  return SensorStepResult::Published;
}

void SensorSeedV1MLP::Initialize(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  // SEED harvests contact points from a live articulated body, so it requires an actor (SensorBase
  // itself no longer enforces this, to allow actor-less sensors like fusion sensors).
  Actor* const actor = GetActor();
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorSeedV1MLP requires a non-null link actor");
    return;
  }
  _contactQueryHandle = actor->RegisterQuery(QueryType::ContactPoints, error);
  SetParams(params, error);
}

void SensorSeedV1MLP::SetParams(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  bool const pathChanged = !_modelLoaded || params.weightsPath != _params.weightsPath;

  if (pathChanged) {
    if (!params.weightsPath.empty()) {
      auto mlp = LoadMlpFromJson(params.weightsPath, error);
      if (!error.IsOK()) {
        return;
      }
      if (!mlp.has_value()) {
        MOCHI_ERROR_SET(error, "ForceNet MLP loader returned no model");
        return;
      }
      MOCHI_ERROR_IF(
          mlp->InputDim() != 6 || mlp->OutputDim() != 3,
          error,
          "ForceNet MLP must have input_dim=6 and output_dim=3.");
      if (!error.IsOK()) {
        return;
      }
      _mlp = std::make_unique<ai::Mlp<float>>(std::move(*mlp));
      _modelLoaded = true;
    } else {
      _mlp.reset();
      _modelLoaded = false;
    }
  }
  _params = params;
}

SensorSeedV1MLP::Result SensorSeedV1MLP::ComputeSignal(Error& error) {
  Result result;
  Actor* const actor = GetActor();
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorSeedV1MLP: actor is null.");
    return result;
  }

  auto const contacts = actor->GetContactPointsWorld(error);
  MOCHI_ERROR_RETURN(error, result);

  return ProcessContacts(contacts, error);
}

SensorSeedV1MLP::Result SensorSeedV1MLP::ComputeSignal(
    Span<ContactPoint const> contacts,
    Error& error) {
  return ProcessContacts(contacts, error);
}

SensorSeedV1MLP::Result SensorSeedV1MLP::ProcessContacts(
    Span<ContactPoint const> contacts,
    Error& error) {
  Result result;
  result.numContacts = isize(contacts);
  if (result.numContacts == 0) {
    return result;
  }

  // Aggregate contact forces in world frame, compute force-weighted centroid
  Real3 contactForceWorld = {};
  real contactForceMag = 0_r;
  Real3 contactPosWorld = {};

  Actor* const actor = GetActor();
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorSeedV1MLP: actor is null.");
    return result;
  }
  ActorHandle const myHandle = actor->GetHandle();
  for (int i = 0; i < isize(contacts); ++i) {
    auto const& c = contacts[i];
    Real3 force;
    Real3 pos;
    if (c.actorA == myHandle) {
      force = c.force;
      pos = c.posA;
    } else {
      force = -c.force;
      pos = c.posB;
    }
    contactForceWorld += force;
    real const f = Norm(force);
    contactForceMag += f;
    contactPosWorld += f * pos;
  }

  if (contactForceMag <= 0_r) {
    return result;
  }
  contactPosWorld = contactPosWorld * (1_r / contactForceMag);

  // Transform force and centroid to sensor-local frame
  TransformRT const worldFromLink = actor->GetRootTransform();
  Real3 const forceLocal = worldFromLink.TransformDirectionInverse(contactForceWorld);
  Real3 const centroidLocal = worldFromLink.TransformPointInverse(contactPosWorld);

  // Compute normal: direction from hemisphere center to contact centroid
  Real3 const toContact = centroidLocal - _params.hemisphereCenterOffset;
  real const normalLen = Norm(toContact);

  Real3 normalLocal = {};
  if (normalLen > 0_r) {
    normalLocal = toContact * (1_r / normalLen);
  }

  result.contactCentroid = centroidLocal;
  result.contactNormal = normalLocal;
  result.inputForce = forceLocal;
  result.inputNormal = normalLocal;

  if (_params.bypassModel || !_modelLoaded || !_mlp) {
    result.outputForce = forceLocal;
    return result;
  }

  // Evaluate ForceNet via Mochi MLP: input = [force_local(3), normal_local(3)]
  ColumnVector<float, 6> input;
  input(0) = static_cast<float>(forceLocal[0]);
  input(1) = static_cast<float>(forceLocal[1]);
  input(2) = static_cast<float>(forceLocal[2]);
  input(3) = static_cast<float>(result.inputNormal[0]);
  input(4) = static_cast<float>(result.inputNormal[1]);
  input(5) = static_cast<float>(result.inputNormal[2]);

  ColumnVector<float, 3> output;
  _mlp->Forward(input, output);

  result.outputForce = {
      static_cast<real>(output(0)), static_cast<real>(output(1)), static_cast<real>(output(2))};

  return result;
}
