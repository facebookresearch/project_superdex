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

#include <cstddef>
#include <filesystem>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>

namespace mochi {

class Path;

// Value type for an absolute, lexically-normal filesystem path.
//
// Internally caches two forms side-by-side:
//   - storable:   absolute, lexically-normal, forward-slash separators, ORIGINAL case
//                 (used for display, serialization, and re-loading from disk).
//   - comparable: same path lowercased (used for hashing and comparison).
//
// All equality/ordering/hashing operates on the comparable form, so two `Path`s
// constructed from `C:\Foo\Bar.txt` and `c:/foo/bar.txt` compare equal — this
// makes `Path` a safe key for `std::map` / `std::unordered_map` on Windows.
//
// Both forms are computed once in the constructor; downstream comparisons and
// hashing become pure string operations (no `std::filesystem` calls).
class Path {
 public:
  /// Default — empty path.
  Path() = default;

  /// Implicit constructors from common path-bearing types. Empty input yields an
  /// empty Path without touching the filesystem; otherwise the input is
  /// canonicalized via `std::filesystem::absolute(...).lexically_normal()` and
  /// converted to generic (forward-slash) form.
  Path(char const* path);
  Path(std::string const& path);
  Path(std::string_view path);
  Path(std::filesystem::path const& path);

  /// Catch-all for other char-based string types (e.g., custom allocators).
  /// Disabled for `std::string` to avoid ambiguity with the explicit overload.
  template <
      typename Traits,
      typename Alloc,
      typename = std::enable_if_t<
          !std::is_same_v<Alloc, std::allocator<char>> ||
          !std::is_same_v<Traits, std::char_traits<char>>>>
  Path(std::basic_string<char, Traits, Alloc> const& s)
      : Path(std::string_view{s.data(), s.size()}) {}

  /// Path as a `std::string`. Original case is preserved.
  std::string ToString() const;

  // Path as a `std::string`. Lowercased for comparison.
  std::string const& AsLowercaseString() const;

  std::filesystem::path const& AsFilesystemPath() const;

  /// Returns true of the path is an empty string.
  bool IsEmpty() const;

  /// Filename component (e.g. `"baz.txt"` from `/foo/bar/baz.txt`).
  std::string GetFilename() const;

  /// Stem component (filename without its final extension).
  std::string GetStem() const;

  /// Extension component (e.g. `".txt"`, including the leading dot).
  /// Will not detect double extensions!
  std::string GetExtension() const;

  /// Extension component lowercased for case-insensitive comparisons.
  std::string GetExtensionLowercase() const;

  /// Parent path.
  Path GetParentPath() const;

  /// Replace file extension with a new one.
  Path& ReplaceExtension(std::string_view newExt);

  /// Append a sub-path. The right-hand side is treated as a relative segment
  /// (it is NOT re-absolutized like the Path constructor would do).
  Path operator/(std::string_view subpath) const;
  Path operator/(std::string const& subpath) const;
  Path operator/(char const* subpath) const;

  /// True iff `*this` is `parent` or a descendant directory/file of `parent`.
  /// Pure prefix check on the comparable form — no filesystem call.
  bool IsDescendantOf(Path const& parent) const;

  /// Path of `*this` relative to `parent`, computed lexically. Returns an empty
  /// string if `*this` is not a descendant of `parent`.
  ///
  /// A relative path is conceptually a fragment — it cannot satisfy the
  /// always-absolute invariant of `Path`, so it is returned as `std::string`
  /// (in forward-slash, original-case form). Re-join with `operator/`:
  ///     `auto rel = file.RelativeToParent(dir); auto joined = dir / rel;`
  std::string RelativeToParent(Path const& parent) const;

 private:
  std::filesystem::path
      _normalizedPath; ///< Absolute, lexically-normal, forward slash, *original case*.
  std::string
      _normalizedPathLowercase; ///< Absolute, lexically-normal, forward slash, *lowercased*.
};

inline bool operator==(Path const& a, Path const& b) {
  return a.AsLowercaseString() == b.AsLowercaseString();
}

inline bool operator!=(Path const& a, Path const& b) {
  return !(a == b);
}

inline bool operator<(Path const& a, Path const& b) {
  return a.AsLowercaseString() < b.AsLowercaseString();
}

inline bool operator<=(Path const& a, Path const& b) {
  return a.AsLowercaseString() <= b.AsLowercaseString();
}
inline bool operator>(Path const& a, Path const& b) {
  return a.AsLowercaseString() > b.AsLowercaseString();
}

inline bool operator>=(Path const& a, Path const& b) {
  return a.AsLowercaseString() >= b.AsLowercaseString();
}

inline std::ostream& operator<<(std::ostream& os, Path const& p) {
  return os << p.ToString();
}

} // namespace mochi

namespace std {
template <>
struct hash<mochi::Path> {
  size_t operator()(mochi::Path const& p) const noexcept {
    return std::hash<std::string>{}(p.AsLowercaseString());
  }
};
} // namespace std
