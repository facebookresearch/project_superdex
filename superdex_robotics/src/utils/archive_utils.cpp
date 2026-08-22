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

#include <superdex_robotics/core/loader.h>
#if MOCHI_INTERNAL
#include <superdex_robotics/internal/bot_scene.h>
#endif
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/json_utils.h>
#include <mochi_core/utils/overload_visitor.h>

#include <mochi_physics/utils/mochi_prefab.h>

#include <miniz.h>

#include <chrono>
#include <ctime>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;

bool superdex::robotics::IsBotArchivePath(std::string_view path) {
  if (path.ends_with(kBotArchiveExtension)) {
    return true;
  }
#if MOCHI_INTERNAL
  if (path.ends_with(kBotArchiveExtensionLegacy)) {
    return true;
  }
#endif
  return false;
}

// One entry in the output archive. Either srcFile or data is the bytes source:
//   srcFile non-empty: read content from this file on disk via miniz's add_file.
//   srcFile empty:     write `data` via minz's add_mem (can be empty, as in the case of
//                      .superdex_root
struct ZipEntry {
  std::string archivePath;
  std::filesystem::path srcFile;
  std::string data;
};

static void ZipEntries(
    std::vector<ZipEntry> const& entries,
    std::filesystem::path const& dstZipFile,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Remove any pre-existing destination file so miniz starts from a clean slate.
  std::error_code ec;
  std::filesystem::remove(dstZipFile, ec);
  MOCHI_ERROR_IF(ec, error, "Failed to remove pre-existing destination file.");
  MOCHI_ERROR_RETURN(error);
  // Init zip.
  mz_zip_archive zip{};
  if (!mz_zip_writer_init_file(&zip, dstZipFile.string().c_str(), 0)) {
    MOCHI_ERROR_SET(error, "Failed to initialize zip writer.");
    return;
  }
  MOCHI_DEFER(mz_zip_writer_end(&zip));
  // Add entries to zip.
  for (auto const& entry : entries) {
    bool ok = false;
    if (!entry.srcFile.empty()) {
      auto const srcStr = entry.srcFile.string();
      ok = mz_zip_writer_add_file(
               &zip,
               entry.archivePath.c_str(),
               srcStr.c_str(),
               /*pComment=*/nullptr,
               /*comment_size=*/0,
               MZ_BEST_COMPRESSION) != 0;
    } else {
      ok = mz_zip_writer_add_mem(
               &zip,
               entry.archivePath.c_str(),
               entry.data.data(),
               entry.data.size(),
               MZ_BEST_COMPRESSION) != 0;
    }
    if (!ok) {
      MOCHI_LOG_ERROR("Failed to add %s to zip", entry.archivePath.c_str());
      MOCHI_ERROR_SET(error, "Failed to add entry to zip archive.");
      return;
    }
  }
  // Wrap things up.
  if (!mz_zip_writer_finalize_archive(&zip)) {
    MOCHI_ERROR_SET(error, "Failed to finalize zip archive.");
  }
}

// Convert a path to a form that bypasses the Windows legacy MAX_PATH (260-char)
// limit during file I/O. On Windows this returns an absolute, normalized path
// prefixed with the extended-length marker (\\?\); miniz's extraction and
// std::filesystem then operate without the length cap. A no-op elsewhere.
static std::filesystem::path MakeLongPath(std::filesystem::path const& p) {
#if MOCHI_PLATFORM_WINDOWS
  std::error_code ec;
  std::filesystem::path abs = std::filesystem::absolute(p, ec);
  if (ec) {
    abs = p;
  }
  std::wstring native = abs.lexically_normal().native();
  // Only plain drive paths (C:\...) need the prefix; leave existing \\?\ and
  // UNC (\\server\...) paths untouched.
  if (native.size() < 2 || native[0] != L'\\' || native[1] != L'\\') {
    native.insert(0, LR"(\\?\)");
  }
  return std::filesystem::path(native);
#else
  return p;
#endif
}

static void UnzipToDirectory(std::string_view srcZipFile, std::string_view dstDir, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!std::filesystem::exists(srcZipFile)) {
    MOCHI_ERROR_SET(error, "Source zip file does not exist.");
    return;
  }
  std::error_code ec;
  // Wipe any pre-existing destination directory so we extract into a clean state.
  if (std::filesystem::exists(dstDir)) {
    std::filesystem::remove_all(dstDir, ec);
    MOCHI_ERROR_IF(ec, error, "Failed to remove pre-existing destination directory.");
    MOCHI_ERROR_RETURN(error);
  }
  std::filesystem::create_directories(MakeLongPath(dstDir), ec);
  MOCHI_ERROR_IF(ec, error, "Failed to create destination directory.");
  MOCHI_ERROR_RETURN(error);

  mz_zip_archive zip{};
  if (!mz_zip_reader_init_file(&zip, std::string(srcZipFile).c_str(), 0)) {
    MOCHI_ERROR_SET(error, "Failed to open zip archive.");
    return;
  }
  MOCHI_DEFER(mz_zip_reader_end(&zip));

  auto const numEntries = mz_zip_reader_get_num_files(&zip);
  for (mz_uint i = 0; i < numEntries; ++i) {
    mz_zip_archive_file_stat stat{};
    if (!mz_zip_reader_file_stat(&zip, i, &stat)) {
      MOCHI_ERROR_SET(error, "Failed to stat zip entry.");
      return;
    }
    auto const entryPath = dstDir / std::filesystem::path(stat.m_filename);
    if (mz_zip_reader_is_file_a_directory(&zip, i)) {
      std::filesystem::create_directories(MakeLongPath(entryPath), ec);
      MOCHI_ERROR_IF(ec, error, "Failed to create directory entry.");
      MOCHI_ERROR_RETURN(error);
      continue;
    }
    auto const parent = entryPath.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(MakeLongPath(parent), ec);
      MOCHI_ERROR_IF(ec, error, "Failed to create parent directory.");
      MOCHI_ERROR_RETURN(error);
    }
    if (!mz_zip_reader_extract_to_file(&zip, i, MakeLongPath(entryPath).string().c_str(), 0)) {
      MOCHI_LOG_ERROR("Failed to extract %s", stat.m_filename);
      MOCHI_ERROR_SET(error, "Failed to extract zip entry.");
      return;
    }
  }
}

static DynamicString GetCurrentDateString() {
  auto const now = std::chrono::system_clock::now();
  auto const tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &tt);
#else
  gmtime_r(&tt, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
  return DynamicString{buf};
}

#ifdef MOCHI_BOTS_WITH_MECURIAL
static DynamicString GetSourceCommitHash() {
#ifdef _WIN32
  FILE* pipe = _popen("hg id -i 2>NUL", "r");
#else
  FILE* pipe = popen("hg id -i 2>/dev/null", "r");
#endif
  if (pipe == nullptr) {
    return DynamicString{};
  }
  std::string out;
  char buffer[128];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    out += buffer;
  }
#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  while (!out.empty() &&
         (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '+')) {
    out.pop_back();
  }
  return DynamicString{out.c_str(), out.size()};
}
#else
static DynamicString GetSourceCommitHash() {
  // TODO: Implement this for non-Meta internal builds assuming git.
  return DynamicString{};
}
#endif // MOCHI_BOTS_WITH_MECURIAL

/* Record an asset the source tree could not supply, and say so in the log. Both, deliberately: the
 * log serves whoever is watching the build, the record travels inside the archive for whoever picks
 * it up later. */
static void NoteMissingAsset(DynamicArray<DynamicString>& warnings, std::string const& message) {
  MOCHI_LOG_WARNING("ArchiveBot: %s", message.c_str());
  warnings.push_back(DynamicString(message));
}

// Collect files referenced from within a sensor's params (.superdex_sensor)
// file. Sensor params reference companion assets by paths relative to the
// params file's directory -- a learned sensor naming its weights file, say --
// and without these, a sensor declared in the archive cannot load. We walk the params JSON and
// collect any string that resolves to an existing file relative to the params directory. This is
// type-agnostic, so it covers every sensor type (including those whose params struct lives in a
// library the archiver does not link) without per-type knowledge here.
//
// Limitations, both acceptable in practice: (1) only params-dir-relative
// references are discovered (the convention sensor params use); a "//root"- or
// "@tag"-relative reference would be missed. (2) Any string that happens to
// resolve to an existing file is collected, so an incidental value could be
// over-collected — harmless (extra bytes only).
//
// A params file that cannot be parsed warns and contributes nothing rather than
// failing the archive. The archive's job is to mirror the assets it was built
// from, broken ones included: the file itself is still bundled by the caller, so
// loading from the archive fails exactly where loading from the source tree
// does, at the same parse. Failing here instead would make a repo that loads
// badly impossible to archive at all, and the two would stop agreeing.
static void CollectSensorParamsReferences(
    std::filesystem::path const& paramsFile,
    std::set<std::filesystem::path>& collectedFiles,
    DynamicArray<DynamicString>& warnings,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  Error parseError;
  auto const json = ParseJsonFromFile(paramsFile.generic_string(), parseError);
  if (!parseError.IsOK()) {
    NoteMissingAsset(
        warnings,
        "sensor params '" + paramsFile.generic_string() +
            "' cannot be parsed; archived as-is without its companion assets");
    return;
  }

  auto const baseDir = paramsFile.parent_path();
  std::vector<picojson::value const*> pending{&json};
  while (!pending.empty()) {
    picojson::value const* const value = pending.back();
    pending.pop_back();
    if (value->is<picojson::object>()) {
      for (auto const& entry : value->get<picojson::object>()) {
        pending.push_back(&entry.second);
      }
    } else if (value->is<picojson::array>()) {
      for (auto const& element : value->get<picojson::array>()) {
        pending.push_back(&element);
      }
    } else if (value->is<std::string>()) {
      // An empty string resolves to baseDir (a directory), which the
      // is_regular_file check below correctly rejects — no empty guard needed.
      auto const candidate = NormalizeBotPath(baseDir / value->get<std::string>());
      if (std::filesystem::is_regular_file(candidate)) {
        collectedFiles.insert(candidate);
      }
    }
  }
}

static void CollectLinkAssets(
    BotLinkPrefab const& link,
    std::set<std::filesystem::path>& collectedFiles,
    DynamicArray<DynamicString>& warnings,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  /* Geometry the build does not have is skipped with a warning, like sensor params below: the
   * archive mirrors the assets that exist, and a reference to a file that is missing from the
   * source tree is equally missing from the archive, so both fail the same way when something
   * actually needs it. Collecting it regardless would fail the copy and prevent archiving at all.
   */
  auto const collectIfPresent = [&](DynamicString const& file, char const* what) {
    if (file.empty()) {
      return;
    }
    auto const path = NormalizeBotPath(std::string(file));
    if (!std::filesystem::exists(path)) {
      NoteMissingAsset(
          warnings,
          std::string(what) + " '" + path.generic_string() + "' on link '" +
              std::string(link.name) + "' does not exist; archived without it");
      return;
    }
    collectedFiles.insert(path);
  };
  collectIfPresent(link.shapeFile, "collision shape");
  collectIfPresent(link.renderModelFile, "render model");
  for (auto const& sensor : link.sensors) {
    if (sensor.params.empty() ||
        IsInlineJson(std::string_view(sensor.params.c_str(), sensor.params.size()))) {
      // Inline JSON has no backing file to archive.
      continue;
    }
    auto const paramsPath = NormalizeBotPath(std::string(sensor.params));
    /* A params file the build does not have is skipped with a warning rather than failing the
     * archive. A sensor type can be absent from a build while the bots that declare it are still
     * present -- the open-source export strips the internal sensors and their params but keeps the
     * geometry -- and such a sensor is skipped at load time too, so there is nothing to bundle.
     *
     * A params file that exists but cannot be parsed is bundled as-is; see the note on
     * CollectSensorParamsReferences. Either way the archive is written and mirrors the source. */
    if (!std::filesystem::exists(paramsPath)) {
      NoteMissingAsset(
          warnings,
          "sensor params '" + paramsPath.generic_string() + "' on link '" + std::string(link.name) +
              "' do not exist; archived without them");
      continue;
    }
    collectedFiles.insert(paramsPath);
    CollectSensorParamsReferences(paramsPath, collectedFiles, warnings, error);
    MOCHI_ERROR_RETURN(error);
  }
}

static void CollectBotDependencies(
    std::string_view path,
    FileBotLoader const& loader,
    std::set<std::filesystem::path>& collectedFiles,
    DynamicArray<DynamicString>& warnings,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const canonical = NormalizeBotPath(std::string(path));
  auto [iter, inserted] = collectedFiles.insert(canonical);
  if (!inserted) {
    return;
  }
  auto const type = loader.GetBotFileType(path, error);
  if (type == BotFileType::ModBotPrefab) {
    auto modBotPrefab = loader.LoadModBotPrefab(path, error);
    CollectBotDependencies(modBotPrefab.base.c_str(), loader, collectedFiles, warnings, error);
    for (auto const& mod : modBotPrefab.modifications) {
      std::visit(
          OverloadVisitor{
              [&](AttachBot const& m) {
                CollectBotDependencies(m.path.c_str(), loader, collectedFiles, warnings, error);
              },
              [&](ReplaceLinkWithBot const& m) {
                CollectBotDependencies(m.path.c_str(), loader, collectedFiles, warnings, error);
              },
              [&](AttachLink const& m) {
                CollectLinkAssets(m.link, collectedFiles, warnings, error);
              },
              [&](ReplaceLink const& m) {
                CollectLinkAssets(m.link, collectedFiles, warnings, error);
              },
          },
          mod);
      MOCHI_ERROR_RETURN(error);
    }
  } else {
    auto botPrefab = loader.LoadBotPrefab(path, error);
    for (auto const& link : botPrefab.links) {
      CollectLinkAssets(link, collectedFiles, warnings, error);
      MOCHI_ERROR_RETURN(error);
    }
  }
}

// Compute the deepest common ancestor (by path components) of a non-empty
// set of absolute paths. Returns an empty path when no common ancestor
// exists (e.g. paths on different Windows drive letters).
//
// std::set orders std::filesystem::path lexicographically by component, so
// every element lies between *paths.begin() and *paths.rbegin() under that
// same ordering. Hence the component-wise longest common prefix of those
// two extremes is a common prefix of every element
static std::filesystem::path DeepestCommonAncestor(std::set<std::filesystem::path> const& paths) {
  if (paths.empty()) {
    return {};
  }
  auto const& first = *paths.begin();
  auto const& last = *paths.rbegin();
  std::filesystem::path result;
  auto itA = first.begin();
  auto itB = last.begin();
  for (; itA != first.end() && itB != last.end() && *itA == *itB; ++itA, ++itB) {
    result /= *itA;
  }
  return result;
}

static std::string JoinArchivePath(
    std::filesystem::path const& rootArchiveRel,
    std::string_view sub) {
  if (rootArchiveRel.empty() || rootArchiveRel == std::filesystem::path(".")) {
    return std::string(sub);
  }
  return rootArchiveRel.generic_string() + "/" + std::string(sub);
}

void superdex::robotics::ArchiveBot(ArchiveParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(!IsBotPath(params.src), error, "Source must be a .superdex_bot file.");
  MOCHI_ERROR_IF(
      !params.dst.ends_with(kBotArchiveExtension),
      error,
      "Destination must end with .superdex_bot_archive.");
  auto const srcCanonical = std::filesystem::weakly_canonical(std::string(params.src));
  MOCHI_ERROR_IF(!std::filesystem::exists(srcCanonical), error, "Source bot file does not exist.");
  auto const botsRoot = FindBotsRoot(srcCanonical);
  MOCHI_ERROR_IF(!botsRoot.has_value(), error, "No .superdex_root file found.");
  MOCHI_ERROR_RETURN(error);
  auto const dstPath = std::filesystem::path(std::string(params.dst));
  // Ensure the parent of the destination archive exists.
  EnsureDirectoriesCreated(dstPath, error);
  MOCHI_ERROR_RETURN(error);
  // Collect all transitive dependencies from the source bot.
  FileBotLoader loader;
  std::set<std::filesystem::path> collectedFiles;
  DynamicArray<DynamicString> warnings;
  CollectBotDependencies(srcCanonical.generic_string(), loader, collectedFiles, warnings, error);
  MOCHI_ERROR_RETURN(error);
  // Attribute every collected file to its owning .superdex_root and accumulate
  // the set of distinct roots involved in the archive.
  std::map<std::filesystem::path, std::filesystem::path> fileToRoot;
  std::set<std::filesystem::path> roots;
  for (auto const& file : collectedFiles) {
    auto rootOpt = FindBotsRoot(file);
    MOCHI_ERROR_IF(
        !rootOpt.has_value(), error, "Collected dependency has no .superdex_root ancestor.");
    MOCHI_ERROR_RETURN(error);
    fileToRoot.emplace(file, *rootOpt);
    roots.insert(*rootOpt);
  }
  // The archive's top folder is the deepest common ancestor of all involved
  // roots. Each root is laid out inside it at its on-disk relative offset.
  auto const archiveTop = DeepestCommonAncestor(roots);
  if (archiveTop.empty()) {
    MOCHI_LOG_ERROR(
        "Cannot archive '%s': the %zu referenced .superdex_root directories share no common ancestor:",
        srcCanonical.generic_string().c_str(),
        roots.size());
    for (auto const& root : roots) {
      MOCHI_LOG_ERROR("  - %s", root.generic_string().c_str());
    }
  }
  MOCHI_ERROR_IF(
      archiveTop.empty(),
      error,
      "Cannot archive: referenced .superdex_root directories share no common ancestor.");
  MOCHI_ERROR_RETURN(error);
  // Map each root to its archive-relative directory.
  std::map<std::filesystem::path, std::filesystem::path> rootToArchiveRel;
  for (auto const& root : roots) {
    rootToArchiveRel.emplace(root, std::filesystem::relative(root, archiveTop));
  }
  // Zip entries and record of seen paths.
  std::vector<ZipEntry> entries;
  std::set<std::string> seenArchivePaths;
  // Helper to add a file entry not seen yet.
  auto const AddFileEntry = [&](std::string archivePath, std::filesystem::path srcFile) {
    if (!seenArchivePaths.insert(archivePath).second) {
      return;
    }
    entries.push_back({std::move(archivePath), std::move(srcFile), {}});
  };
  // Helper to add a data entry not seen yet.
  auto const AddDataEntry = [&](std::string archivePath, std::string data) {
    if (!seenArchivePaths.insert(archivePath).second) {
      return;
    }
    entries.push_back({std::move(archivePath), {}, std::move(data)});
  };
  // For each archived root, write a (possibly rewritten) .superdex_root.
  // Tags whose target is not in the archived root set are dropped; surviving
  // tags are rewritten as the relative path from this root's archive directory
  // to the target root's archive directory.
  for (auto const& root : roots) {
    auto const& thisArchiveRel = rootToArchiveRel.at(root);
    auto const parsed = ParseRootFile(*FindRootMarker(root), error);
    MOCHI_ERROR_RETURN(error);
    std::unordered_map<std::string, std::string> rewrittenTags;
    for (auto const& [tagName, tagTarget] : parsed.tags) {
      auto it = rootToArchiveRel.find(tagTarget);
      if (it == rootToArchiveRel.end()) {
        /* The tag points at a root outside this archive, so it cannot be rewritten and is dropped.
         * Recorded rather than dropped silently: a reference of the form "@tag/..." resolves in the
         * source tree and simply will not in the archive, which is a gap of the same kind as a
         * missing file and just as invisible at extraction time. */
        NoteMissingAsset(
            warnings,
            "tag '" + std::string(tagName) + "' targets '" + tagTarget.generic_string() +
                "', a root outside this archive; dropped, so '@" + std::string(tagName) +
                "/...' references will not resolve from the archive");
        continue;
      }
      auto rel = std::filesystem::relative(it->second, thisArchiveRel);
      rewrittenTags.emplace(tagName, rel.empty() ? std::string(".") : rel.generic_string());
    }
    // ParseRootFile accepts `{}` (and an empty file) as a valid empty tag table.
    auto rootJson = SReflect::ToJsonString(rewrittenTags, true);
    AddDataEntry(JoinArchivePath(thisArchiveRel, kRootMarkerFile), std::move(rootJson));
  }
  // Add every collected file at its root-relative position inside the archive.
  for (auto const& [file, root] : fileToRoot) {
    auto const rel = file.lexically_relative(root).generic_string();
    AddFileEntry(JoinArchivePath(rootToArchiveRel.at(root), rel), file);
  }
  // Compute the archive-relative path of the source .mochi_bot file for the
  // metadata target. No "//" prefix: it is just an archive-root-relative path
  // that GetExtractedBotArchiveTarget joins with the extracted directory.
  auto const& srcRoot = fileToRoot.at(srcCanonical);
  auto const srcArchivePath = JoinArchivePath(
      rootToArchiveRel.at(srcRoot),
      std::filesystem::relative(srcCanonical, srcRoot).generic_string());
  // Add archive metadata.
  BotArchiveMetadata metadata;
  metadata.date = GetCurrentDateString();
  metadata.botHash = HashBotFile(srcCanonical.generic_string(), error);
  metadata.commitHash = GetSourceCommitHash();
  metadata.target = DynamicString{srcArchivePath};
  metadata.comment = params.comment;
  /* Carried into the archive so the gaps are discoverable later, not only by whoever watched the
   * build. An archive mirrors its source tree, and this is the account of what that tree lacked. */
  metadata.warnings = warnings;
  auto const metadataJson = SReflect::ToJsonString(metadata, true);
  AddDataEntry(std::string(kArchiveMetadataFile), metadataJson);
  // Zip up all entries.
  ZipEntries(entries, dstPath, error);
}

// Shared implementation for content-hash-keyed archive extraction with atomic rename.
static DynamicString ExtractArchiveToCacheImpl(
    std::string_view archiveFile,
    std::string_view cacheSubdir,
    std::string_view metadataFile,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto const canonical = std::filesystem::weakly_canonical(archiveFile);
  if (!std::filesystem::exists(canonical)) {
    MOCHI_ERROR_SET(error, "Archive file does not exist.");
    return {};
  }
  auto const archiveHash = HashGenericFile(canonical.generic_string(), error);
  MOCHI_ERROR_RETURN(error, {});
  auto const dirName = canonical.stem().string() + "_" + std::string(archiveHash);
  auto const cacheRoot = std::filesystem::temp_directory_path() / "superdex_robotics" / cacheSubdir;
  auto const dstDir = cacheRoot / dirName;
  // Cache hit: a complete extraction already exists for this exact content.
  // The metadata file is the publish sentinel — it is bundled into the archive
  // last and only appears at dstDir via the atomic rename below, so its presence
  // means the extraction is complete. The recovery logic in tryRename relies on
  // this: a dstDir without metadata is always safe to discard and re-publish.
  if (std::filesystem::exists(dstDir / metadataFile)) {
    return DynamicString(dstDir.generic_string());
  }
  // Cache miss. Extract into a unique sibling temp dir, then atomically rename.
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> dist;
  auto const ns =
      static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  auto const aslr = reinterpret_cast<uint64_t>(&ns);
  auto const tempDir = cacheRoot / (dirName + ".tmp." + std::to_string(dist(rd) ^ ns ^ aslr));
  MOCHI_DEFER({
    std::error_code cleanupEc;
    std::filesystem::remove_all(tempDir, cleanupEc);
  });
  UnzipToDirectory(canonical.generic_string(), tempDir.generic_string(), error);
  MOCHI_ERROR_RETURN(error, {});
  // Tracks the last failure to remove a stale destination, so an undeletable
  // cache directory (e.g. a file inside it held open by another process) is
  // reported as the cause rather than an opaque publish failure.
  std::error_code lastStaleEc;
  auto const tryRename = [&](std::error_code& ec) {
    constexpr int kMaxAttempts = 6;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
      if (attempt > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50 << (attempt - 1)));
      }
      std::filesystem::rename(tempDir, dstDir, ec);
      if (!ec) {
        return true;
      }
      // A peer process may have concurrently published a complete extraction
      // (metadata present means complete, per the sentinel invariant above);
      // accept it as a cache hit.
      if (std::filesystem::exists(dstDir / metadataFile)) {
        return true;
      }
      // Otherwise the destination is occupied by a stale or incomplete
      // extraction with no metadata file (e.g. left by an interrupted run or an
      // older in-place extractor). std::filesystem::rename cannot replace a
      // non-empty directory on Windows, so clear it before the next attempt.
      if (std::filesystem::exists(dstDir)) {
        std::filesystem::remove_all(dstDir, lastStaleEc);
      }
    }
    return false;
  };
  std::error_code ec;
  if (tryRename(ec)) {
    return DynamicString(dstDir.generic_string());
  }
  if (lastStaleEc) {
    MOCHI_ERROR_SET(
        error,
        "Failed to publish extracted archive cache: could not remove stale cache directory.");
    return {};
  }
  MOCHI_ERROR_SET(error, "Failed to publish extracted archive cache.");
  return {};
}

// Shared implementation for reading the target path from an extracted archive's metadata.
template <typename MetadataT>
static DynamicString GetExtractedArchiveTargetImpl(
    std::string_view extractedDir,
    std::string_view metadataFile,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::filesystem::path extractedDirPath(extractedDir);
  std::string const metadataPath = (extractedDirPath / metadataFile).string();
  MetadataT metadata;
  int numIssues = 0;
  bool const parsed = SReflect::LoadFromJsonFile(
      metadata, metadataPath.c_str(), SReflect::DeserializeFlags::Default, numIssues);
  MOCHI_ERROR_IF(!parsed, error, "Failed to load archive metadata file.");
  MOCHI_ERROR_RETURN(error, {});
  std::string const targetPath = (extractedDirPath / metadata.target.c_str()).generic_string();
  return DynamicString(targetPath);
}

DynamicString superdex::robotics::ExtractBotArchiveToCache(
    std::string_view archiveFile,
    Error& error) {
  return ExtractArchiveToCacheImpl(archiveFile, "bot_archive_cache", kArchiveMetadataFile, error);
}

[[nodiscard]] DynamicString superdex::robotics::GetExtractedBotArchiveTarget(
    std::string_view extractedDir,
    Error& error) {
  return GetExtractedArchiveTargetImpl<BotArchiveMetadata>(
      extractedDir, kArchiveMetadataFile, error);
}

BotArchiveMetadata superdex::robotics::ReadBotArchiveMetadata(
    std::string_view extractedDir,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  auto const metadataPath = (std::filesystem::path(extractedDir) / kArchiveMetadataFile).string();
  BotArchiveMetadata metadata;
  int numIssues = 0;
  bool const parsed = SReflect::LoadFromJsonFile(
      metadata, metadataPath.c_str(), SReflect::DeserializeFlags::Default, numIssues);
  MOCHI_ERROR_IF(!parsed, error, "Failed to load archive metadata file.");
  MOCHI_ERROR_RETURN(error, {});
  return metadata;
}

// ---------------------------------------------------------------------------
// Bot scene archives (.mochi_bot_scene_archive)
// ---------------------------------------------------------------------------

#if MOCHI_INTERNAL
static void CollectScenePrefabDependencies(
    std::filesystem::path const& scenePath,
    std::set<std::filesystem::path>& collectedFiles,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const canonical = std::filesystem::weakly_canonical(scenePath);
  auto [iter, inserted] = collectedFiles.insert(canonical);
  if (!inserted) {
    return; // Already visited (handles circular refs).
  }
  auto prefab = mochi::prefab::ShallowLoadFromFile(canonical.generic_string(), error);
  MOCHI_ERROR_RETURN(error);
  auto const parentDir = canonical.parent_path();
  // Helper: resolve a path relative to the scene file's directory and collect it.
  auto const CollectAsset = [&](DynamicString const& asset) {
    if (asset.empty()) {
      return;
    }
    auto const resolved = NormalizeBotPath(parentDir / asset.c_str());
    collectedFiles.insert(resolved);
  };
  // Rigid actors.
  for (auto const& actor : prefab.actors.rigid) {
    CollectAsset(actor.shapeFile);
    CollectAsset(actor.renderModelFile);
  }
  // Soft actors.
  for (auto const& actor : prefab.actors.soft) {
    CollectAsset(actor.shapeFile);
    CollectAsset(actor.renderModelFile);
    CollectAsset(actor.flowFile);
  }
  // Articulated actors.
  for (auto const& actor : prefab.actors.articulated) {
    for (auto const& link : actor.links) {
      CollectAsset(link.shapeFile);
      CollectAsset(link.renderModelFile);
    }
  }
  // Soft-skinned actors (skeleton links + soft bodies).
  for (auto const& actor : prefab.actors.softSkinned) {
    for (auto const& link : actor.skeletonParams.links) {
      CollectAsset(link.shapeFile);
      CollectAsset(link.renderModelFile);
    }
    for (auto const& soft : actor.softParams) {
      CollectAsset(soft.shapeFile);
      CollectAsset(soft.renderModelFile);
      CollectAsset(soft.flowFile);
    }
  }
  // Nested prefabs.
  for (auto const& nested : prefab.prefabs) {
    if (!nested.path.empty()) {
      auto const nestedPath = NormalizeBotPath(parentDir / nested.path.c_str());
      CollectScenePrefabDependencies(nestedPath, collectedFiles, error);
      MOCHI_ERROR_RETURN(error);
    }
  }
}

void superdex::robotics::ArchiveBotScene(ArchiveParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !std::string_view(params.src).ends_with(kBotSceneExtension),
      error,
      "Source must be a .mochi_bot_scene file.");
  MOCHI_ERROR_IF(
      !std::string_view(params.dst).ends_with(kSceneArchiveExtension),
      error,
      "Destination must end with .mochi_bot_scene_archive.");
  MOCHI_ERROR_RETURN(error);

  auto const sceneCanonical = std::filesystem::weakly_canonical(std::string(params.src));
  MOCHI_ERROR_IF(
      !std::filesystem::exists(sceneCanonical),
      error,
      "Source .mochi_bot_scene file does not exist.");
  MOCHI_ERROR_RETURN(error);

  // Load the scene prefab (paths resolved to absolute).
  auto const prefab = LoadBotScenePrefabFromFile(sceneCanonical.generic_string(), error);
  MOCHI_ERROR_RETURN(error);

  // Validate all bot paths are bot archives (reject raw .superdex_bot/.mochi_bot).
  for (auto const& bot : prefab.bots) {
    if (!IsBotArchivePath(bot.path)) {
      MOCHI_LOG_ERROR(
          "Bot '%s' path '%s' is not a bot archive. "
          "ArchiveBotScene requires all bots to be pre-archived.",
          bot.name.c_str(),
          bot.path.c_str());
      MOCHI_ERROR_SET(error, "All bots[].path entries must be .superdex_bot_archive files.");
      return;
    }
  }

  auto const dstPath = std::filesystem::path(std::string(params.dst));
  EnsureDirectoriesCreated(dstPath, error);
  MOCHI_ERROR_RETURN(error);

  // Collect all transitive dependencies.
  std::set<std::filesystem::path> collectedFiles;

  // 1. The .mochi_bot_scene file itself.
  collectedFiles.insert(sceneCanonical);

  // 2. Base scene and its transitive shape/prefab dependencies.
  {
    auto const baseScenePath = NormalizeBotPath(std::string(prefab.scene.baseScene));
    CollectScenePrefabDependencies(baseScenePath, collectedFiles, error);
    MOCHI_ERROR_RETURN(error);
  }

  // 3. Spawnable prefab dependencies.
  for (auto const& spawnPrefab : prefab.scene.spawnablePrefabs) {
    auto const spawnPath = NormalizeBotPath(std::string(spawnPrefab.path));
    auto const spawnPathString = spawnPath.generic_string();
    if (IsBotArchivePath(spawnPathString)) {
      collectedFiles.insert(spawnPath);
    } else {
      CollectScenePrefabDependencies(spawnPath, collectedFiles, error);
      MOCHI_ERROR_RETURN(error);
    }
  }

  // 4. Bot archive files (opaque blobs — we do not walk their dependency trees).
  for (auto const& bot : prefab.bots) {
    auto const botPath = NormalizeBotPath(std::string(bot.path));
    collectedFiles.insert(botPath);
  }

  // 5. Controller param files.
  for (auto const& bot : prefab.bots) {
    for (auto const& ctrl : bot.controllers) {
      if (ctrl.params.empty() ||
          IsInlineJson(std::string_view(ctrl.params.c_str(), ctrl.params.size()))) {
        // Inline JSON has no backing file to archive.
        continue;
      }
      auto const paramsPath = NormalizeBotPath(std::string(ctrl.params));
      if (std::filesystem::is_regular_file(paramsPath)) {
        collectedFiles.insert(paramsPath);
      }
    }
  }

  // Map files to roots and compute deepest common ancestor (same logic as ArchiveBot).
  std::map<std::filesystem::path, std::filesystem::path> fileToRoot;
  std::set<std::filesystem::path> roots;
  for (auto const& file : collectedFiles) {
    auto rootOpt = FindBotsRoot(file);
    MOCHI_ERROR_IF(
        !rootOpt.has_value(), error, "Collected dependency has no .superdex_root ancestor.");
    MOCHI_ERROR_RETURN(error);
    fileToRoot.emplace(file, *rootOpt);
    roots.insert(*rootOpt);
  }

  auto const archiveTop = DeepestCommonAncestor(roots);
  if (archiveTop.empty()) {
    MOCHI_LOG_ERROR(
        "Cannot archive '%s': the %zu referenced .superdex_root directories share no common ancestor.",
        sceneCanonical.generic_string().c_str(),
        roots.size());
  }
  MOCHI_ERROR_IF(
      archiveTop.empty(),
      error,
      "Cannot archive: referenced .superdex_root directories share no common ancestor.");
  MOCHI_ERROR_RETURN(error);

  // Map each root to its archive-relative directory.
  std::map<std::filesystem::path, std::filesystem::path> rootToArchiveRel;
  for (auto const& root : roots) {
    rootToArchiveRel.emplace(root, std::filesystem::relative(root, archiveTop));
  }

  // Build zip entries.
  std::vector<ZipEntry> entries;
  std::set<std::string> seenArchivePaths;

  auto const AddFileEntry = [&](std::string archPath, std::filesystem::path srcFile) {
    if (!seenArchivePaths.insert(archPath).second) {
      return;
    }
    entries.push_back({std::move(archPath), std::move(srcFile), {}});
  };
  auto const AddDataEntry = [&](std::string archPath, std::string data) {
    if (!seenArchivePaths.insert(archPath).second) {
      return;
    }
    entries.push_back({std::move(archPath), {}, std::move(data)});
  };

  // Rewrite .superdex_root tag tables (same logic as ArchiveBot lines 340-357).
  for (auto const& root : roots) {
    auto const& thisArchiveRel = rootToArchiveRel.at(root);
    auto const parsed = ParseRootFile(*FindRootMarker(root), error);
    MOCHI_ERROR_RETURN(error);
    std::unordered_map<std::string, std::string> rewrittenTags;
    for (auto const& [tagName, tagTarget] : parsed.tags) {
      auto it = rootToArchiveRel.find(tagTarget);
      if (it == rootToArchiveRel.end()) {
        continue; // Tag points to a root not present in the archive; drop.
      }
      auto rel = std::filesystem::relative(it->second, thisArchiveRel);
      rewrittenTags.emplace(tagName, rel.empty() ? std::string(".") : rel.generic_string());
    }
    auto rootJson = SReflect::ToJsonString(rewrittenTags, true);
    AddDataEntry(JoinArchivePath(thisArchiveRel, kRootMarkerFile), std::move(rootJson));
  }

  // Add every collected file at its root-relative position inside the archive.
  for (auto const& [file, root] : fileToRoot) {
    auto const rel = file.lexically_relative(root).generic_string();
    AddFileEntry(JoinArchivePath(rootToArchiveRel.at(root), rel), file);
  }

  // Compute the archive-relative path of the source .mochi_bot_scene for metadata.
  auto const& srcRoot = fileToRoot.at(sceneCanonical);
  auto const srcArchivePath = JoinArchivePath(
      rootToArchiveRel.at(srcRoot),
      std::filesystem::relative(sceneCanonical, srcRoot).generic_string());

  // Write metadata.
  BotSceneArchiveMetadata metadata;
  metadata.date = GetCurrentDateString();
  metadata.sceneHash = HashGenericFile(sceneCanonical.generic_string(), error);
  MOCHI_ERROR_RETURN(error);
  metadata.commitHash = GetSourceCommitHash();
  metadata.target = DynamicString{srcArchivePath};
  metadata.comment = params.comment;
  auto const metadataJson = SReflect::ToJsonString(metadata, true);
  AddDataEntry(std::string(kSceneArchiveMetadataFile), metadataJson);

  // Zip up all entries.
  ZipEntries(entries, dstPath, error);
}

DynamicString superdex::robotics::ExtractBotSceneArchiveToCache(
    std::string_view archiveFile,
    Error& error) {
  return ExtractArchiveToCacheImpl(
      archiveFile, "scene_archive_cache", kSceneArchiveMetadataFile, error);
}

DynamicString superdex::robotics::GetExtractedBotSceneArchiveTarget(
    std::string_view extractedDir,
    Error& error) {
  return GetExtractedArchiveTargetImpl<BotSceneArchiveMetadata>(
      extractedDir, kSceneArchiveMetadataFile, error);
}
#endif // MOCHI_INTERNAL
