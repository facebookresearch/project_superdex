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

#include <superdex_recorder/frame_bank.h>

#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/formatted_error.h>

#include <mochi_core/utils/reflection.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

MOCHI_STRUCT_BEGIN_EX(superdex::recorder::BankActor)
MOCHI_FIELD(path)
MOCHI_FIELD(runtimeName)
MOCHI_FIELD(spawnName)
MOCHI_FIELD(type)
MOCHI_FIELD(isStatic)
MOCHI_FIELD(rootRotations)
MOCHI_FIELD(rootTranslations)
MOCHI_FIELD(comTranslations)
MOCHI_FIELD(linearVelocities)
MOCHI_FIELD(angularVelocities)
MOCHI_FIELD(jointPositions)
MOCHI_FIELD(jointVelocities)
MOCHI_FIELD(controllerTargets)
MOCHI_FIELD(jointEfforts)
MOCHI_FIELD(displacements)
MOCHI_FIELD(nodeVelocities)
MOCHI_FIELD(elementFrameAxes)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::recorder::BankTarget)
MOCHI_FIELD(name)
MOCHI_FIELD(width)
MOCHI_FIELD(values)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::recorder::BankController)
MOCHI_FIELD(robotActor)
MOCHI_FIELD(id)
MOCHI_FIELD(targets)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::recorder::BankProperty)
MOCHI_FIELD(key)
MOCHI_FIELD(value)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::recorder::BankEpisode)
MOCHI_FIELD(source)
MOCHI_FIELD(properties)
MOCHI_FIELD(frames)
MOCHI_FIELD(simulationTimesSec)
MOCHI_FIELD(stepDtsSec)
MOCHI_FIELD(actors)
MOCHI_FIELD(controllers)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::recorder::FrameBank)
MOCHI_FIELD(episodes)
MOCHI_STRUCT_END_EX()

namespace superdex::recorder {
namespace {

// "SDXBANK" in the first seven bytes of a bank file; the eighth is the bytes per real it holds.
constexpr uint64_t kMagicMask = 0x00FF'FFFF'FFFF'FFFF;
constexpr uint64_t kMagic = 0x004B'4E41'4258'4453 | (uint64_t{sizeof(mochi::real)} << 56);

/* A hash of the bank structs' reflected names, types, sizes and offsets. */
uint64_t GetSchema() {
  static uint64_t const schema = [] {
    std::string const json = SReflect::GetTypeInfo<FrameBank>().TypeInfoToJson(false);
    return SReflect::CalcHash64(json.data(), json.size());
  }();
  return schema;
}

std::filesystem::path GetAbsolutePath(std::filesystem::path const& path) {
  std::error_code ignored;
  return std::filesystem::absolute(path, ignored).lexically_normal();
}

/* The deepest directory that holds every recording. */
std::filesystem::path GetCommonDirectory(mochi::Span<std::filesystem::path const> recordings) {
  std::filesystem::path common = GetAbsolutePath(recordings[0]).parent_path();
  for (std::filesystem::path const& recording : recordings) {
    std::filesystem::path const directory = GetAbsolutePath(recording).parent_path();
    std::filesystem::path shared;
    for (auto a = common.begin(), b = directory.begin();
         a != common.end() && b != directory.end() && *a == *b;
         ++a, ++b) {
      shared /= *a;
    }
    common = shared;
  }
  return common;
}

/* A recording's path relative to `directory` in UTF-8 with '/' separators, or its absolute path
 * when it has no relative one, such as on another drive. */
mochi::DynamicString GetSource(
    std::filesystem::path const& recording,
    std::filesystem::path const& directory) {
  std::filesystem::path const path = GetAbsolutePath(recording);
  std::filesystem::path const relative = path.lexically_relative(directory);
  std::u8string const text = (relative.empty() ? path : relative).generic_u8string();
  return {text.begin(), text.end()};
}

} // namespace

FrameBank PackFrameBank(mochi::Span<std::filesystem::path const> recordings, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(recordings.empty(), error, "No recordings to pack.");
  MOCHI_ERROR_RETURN(error, {});
  std::filesystem::path const directory = GetCommonDirectory(recordings);
  FrameBank bank;
  for (std::filesystem::path const& path : recordings) {
    mochi::Error readError;
    std::unique_ptr<EpisodeReader> const recording = EpisodeReader::Open(path, readError);
    BankEpisode episode =
        recording == nullptr ? BankEpisode{} : ReadBankEpisode(*recording, readError);
    if (!readError.IsOK()) {
      SUPERDEX_ERROR_SET(
          error, "Failed to read '%s': %s", path.string().c_str(), readError.GetDescription());
      return {};
    }
    episode.source = GetSource(path, directory);
    bank.episodes.push_back(std::move(episode));
  }
  return bank;
}

void SaveFrameBank(FrameBank const& bank, std::filesystem::path const& path, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  SReflect::VecStreamWriter bytes;
  bool const isSerialized = SReflect::ToBytes(kMagic, bytes) &&
      SReflect::ToBytes(GetSchema(), bytes) && SReflect::ToBytes(bank, bytes);
  std::ofstream file{path, std::ios::binary | std::ios::trunc};
  file.write(
      reinterpret_cast<char const*>(bytes.GetBytes().data()),
      static_cast<std::streamsize>(bytes.GetNumBytesWritten()));
  file.close();
  if (!isSerialized || !file) {
    SUPERDEX_ERROR_SET(error, "Failed to write bank file '%s'.", path.string().c_str());
  }
}

std::shared_ptr<FrameBank const> LoadFrameBank(
    std::filesystem::path const& path,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  std::ifstream file{path, std::ios::binary | std::ios::ate};
  std::vector<uint8_t> bytes(file ? static_cast<size_t>(file.tellg()) : 0);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    SUPERDEX_ERROR_SET(error, "Failed to read bank file '%s'.", path.string().c_str());
    return nullptr;
  }
  SReflect::SpanStreamReader reader{{bytes.data(), bytes.size()}};
  uint64_t magic = 0;
  uint64_t schema = 0;
  auto bank = std::make_shared<FrameBank>();
  std::string const name = path.string();
  if (!SReflect::FromBytes(reader, magic) || (magic & kMagicMask) != (kMagic & kMagicMask)) {
    SUPERDEX_ERROR_SET(error, "'%s' is not a frame bank file.", name.c_str());
  } else if (magic != kMagic) {
    SUPERDEX_ERROR_SET(
        error,
        "'%s' was packed with another Mochi precision; re-pack it with %s.",
        name.c_str(),
        sizeof(mochi::real) == 4 ? "fp32" : "fp64");
  } else if (!SReflect::FromBytes(reader, schema) || schema != GetSchema()) {
    SUPERDEX_ERROR_SET(
        error,
        "'%s' was packed by another build or an older format; re-pack it from its recordings.",
        name.c_str());
  } else if (
      !SReflect::FromBytes(reader, *bank) || reader.HasUnreadBytes() || bank->episodes.empty()) {
    SUPERDEX_ERROR_SET(error, "'%s' is truncated, corrupt or empty.", name.c_str());
  }
  MOCHI_ERROR_RETURN(error, nullptr);
  return bank;
}

} // namespace superdex::recorder
