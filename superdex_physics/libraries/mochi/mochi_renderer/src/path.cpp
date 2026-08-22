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

#include <mochi_renderer/path.h>

#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <cctype>
#include <system_error>
#include <utility>

namespace mochi {

/// Compute the storable (absolute, lexically-normal, generic) form of `p`.
/// Returns empty on empty input. If `std::filesystem::absolute` fails, logs the
// error and returns empty — `Path`'s invariant is/ "always absolute or empty", and
// returning the raw non-absolute input would silently violate that invariant.
static std::filesystem::path Normalize(std::filesystem::path const& p) {
  if (p.empty()) {
    return {};
  }
  std::error_code ec;
  auto abs = std::filesystem::absolute(p, ec);
  if (ec) {
    MOCHI_LOG_ERROR(
        "mochi::Path: std::filesystem::absolute failed for '%s' (%s); the path"
        " cannot be made absolute and will be treated as empty.",
        p.string().c_str(),
        ec.message().c_str());
    return {};
  }
  // lexically_normal()+generic_string() round-trip strips redundant `.`/`..`
  // and unifies separators.
  auto const normal = abs.lexically_normal();
  std::string generic = normal.generic_string();
  // Drop a trailing separator so the canonical form is unique. lexically_normal
  // can emit one (e.g. "foo/bar/.." -> "foo/") and dialog-provided paths may
  // include one; a trailing '/' breaks prefix/equality comparisons such as
  // IsDescendantOf and operator==. Preserve filesystem roots ("/", "C:/").
  std::string const root = normal.root_path().generic_string();
  if (generic.size() > root.size() && generic.back() == '/') {
    generic.pop_back();
  }
  return std::filesystem::path{generic};
}

static std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

Path::Path(char const* path) : Path(std::filesystem::path{path}) {}
Path::Path(std::string const& path) : Path(std::filesystem::path{path}) {}
Path::Path(std::string_view path) : Path(std::filesystem::path{path}) {}

Path::Path(std::filesystem::path const& path) : _normalizedPath(Normalize(path)) {
  _normalizedPathLowercase = ToLower(_normalizedPath.generic_string());
}

std::string Path::ToString() const {
  return _normalizedPath.generic_string();
}

std::string const& Path::AsLowercaseString() const {
  return _normalizedPathLowercase;
}

std::filesystem::path const& Path::AsFilesystemPath() const {
  return _normalizedPath;
}

bool Path::IsEmpty() const {
  return _normalizedPathLowercase.empty();
}

std::string Path::GetFilename() const {
  return _normalizedPath.filename().generic_string();
}

std::string Path::GetStem() const {
  return _normalizedPath.stem().generic_string();
}

std::string Path::GetExtension() const {
  return _normalizedPath.extension().generic_string();
}

std::string Path::GetExtensionLowercase() const {
  return ToLower(GetExtension());
}

Path Path::GetParentPath() const {
  return Path{_normalizedPath.parent_path()};
}

Path& Path::ReplaceExtension(std::string_view newExt) {
  _normalizedPath.replace_extension(std::filesystem::path{newExt});
  auto generic = _normalizedPath.generic_string();
  _normalizedPath = std::filesystem::path{generic};
  _normalizedPathLowercase = ToLower(std::move(generic));
  return *this;
}

Path Path::operator/(std::string_view subpath) const {
  if (subpath.empty()) {
    return *this;
  }
  std::filesystem::path const rhsFs{subpath};
  if (rhsFs.is_absolute()) {
    MOCHI_LOG_ERROR(
        "mochi::Path: cannot append absolute subpath '%s' to path '%s'.",
        rhsFs.string().c_str(),
        _normalizedPath.string().c_str());
    return *this;
  }
  return Path{_normalizedPath / rhsFs};
}

Path Path::operator/(std::string const& subpath) const {
  return *this / std::string_view{subpath};
}

Path Path::operator/(char const* subpath) const {
  return *this / std::string_view{subpath};
}

bool Path::IsDescendantOf(Path const& parent) const {
  if (parent._normalizedPathLowercase.empty() || _normalizedPathLowercase.empty()) {
    return false;
  }
  if (_normalizedPathLowercase.size() < parent._normalizedPathLowercase.size()) {
    return false;
  }
  if (_normalizedPathLowercase.compare(
          0, parent._normalizedPathLowercase.size(), parent._normalizedPathLowercase) != 0) {
    return false;
  }
  // Boundary: same length (equal) or next char is the separator.
  if (_normalizedPathLowercase.size() == parent._normalizedPathLowercase.size()) {
    return true;
  }
  return _normalizedPathLowercase[parent._normalizedPathLowercase.size()] == '/';
}

std::string Path::RelativeToParent(Path const& parent) const {
  if (parent._normalizedPathLowercase.empty() || _normalizedPathLowercase.empty()) {
    return {};
  }
  if (!IsDescendantOf(parent)) {
    return {};
  }
  // _normalizedPathLowercase starts with parent._normalizedPathLowercase.
  std::size_t off = parent._normalizedPathLowercase.size();
  if (off < _normalizedPathLowercase.size() && _normalizedPathLowercase[off] == '/') {
    ++off;
  }
  return _normalizedPath.generic_string().substr(off);
}

} // namespace mochi
