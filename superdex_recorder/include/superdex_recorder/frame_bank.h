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
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

namespace superdex::recorder {

class EpisodeReader;

inline constexpr std::string_view kFrameBankExtension = ".superdex_framebank";

/* The recorded controller fields a bank holds: BASIC_JSC_PD's and the lab joint controller's
 * position target, the OSCs' target pose, and the lab joint controller's closure target. */
inline constexpr std::array<std::string_view, 4> kBankTargetFields{
    "joint_position_target",
    "target_translation_xyz",
    "target_quaternion_xyzw",
    "closure_target",
};

/* @brief A recorded actor's state at each frame of an episode; a static actor has one row, its
 * pose at the first frame. Per frame, as EpisodeRestorer sets them: xyzw rotations and xyz
 * translations, and one value per DOF, except a rod's displacements, which are xyz per node. */
struct BankActor {
  mochi::DynamicString path{}; // "data/physics/actors/cube"
  mochi::DynamicString runtimeName{}; // binding names; spawnName is empty for none
  mochi::DynamicString spawnName{};
  mochi::DynamicString type{}; // "rigid", "articulated", "soft" or "rod", as recorded
  bool isStatic = false;
  mochi::DynamicArray<mochi::real> rootRotations{}, rootTranslations{};
  mochi::DynamicArray<mochi::real> comTranslations{}, linearVelocities{},
      angularVelocities{}; // rigid
  mochi::DynamicArray<mochi::real> jointPositions{}, jointVelocities{}; // articulated
  mochi::DynamicArray<mochi::real> controllerTargets{}, jointEfforts{}; // optional, articulated
  mochi::DynamicArray<mochi::real> displacements{}, nodeVelocities{}; // soft and rod
  mochi::DynamicArray<mochi::real> elementFrameAxes{}; // rod: xyz per element
};

/* @brief One recorded target field of a controller, held at each frame from its latest sample. */
struct BankTarget {
  mochi::DynamicString name{}; // one of kBankTargetFields
  int32_t width = 0;
  mochi::DynamicArray<mochi::real> values{}; // width per frame; NaN before the first sample
};

struct BankController {
  int32_t robotActor = -1; // index into BankEpisode::actors
  mochi::DynamicString id{}; // recorded controller ID
  mochi::DynamicArray<BankTarget> targets{};
};

struct BankProperty {
  mochi::DynamicString key{}, value{};
};

/* @brief What EpisodeRestorer restores from one recorded episode. */
struct BankEpisode {
  mochi::DynamicString source{}; // the recording, relative to the packed recordings' directory
  mochi::DynamicArray<BankProperty> properties{};
  mochi::DynamicArray<int64_t> frames{}; // recorded frame numbers
  mochi::DynamicArray<double> simulationTimesSec{}, stepDtsSec{};
  mochi::DynamicArray<BankActor> actors{}; // sorted by path
  mochi::DynamicArray<BankController> controllers{};
};

struct FrameBank {
  mochi::DynamicArray<BankEpisode> episodes{};
};

/* @brief Reads what EpisodeRestorer restores from a recorded episode, and each recorded robot
 * controller's kBankTargetFields. Efforts and targets hold from their latest sample. */
BankEpisode ReadBankEpisode(EpisodeReader const& recording, mochi::Error& error);

/* @brief Reads recorded episodes into one bank, in the order given. */
FrameBank PackFrameBank(mochi::Span<std::filesystem::path const> recordings, mochi::Error& error);

/* @brief Writes a bank file: magic, a hash of the bank structs' reflected layout, then the bank in
 * Mochi reflection's binary format. The same bank always writes the same bytes. */
void SaveFrameBank(FrameBank const& bank, std::filesystem::path const& path, mochi::Error& error);

/* @brief Reads a bank file. Refuses a file written with other bank structs or another Mochi
 * precision: re-pack it from its recordings. */
std::shared_ptr<FrameBank const> LoadFrameBank(
    std::filesystem::path const& path,
    mochi::Error& error);

} // namespace superdex::recorder
