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

#include <mochi_renderer/path.h>

#include <functional>

namespace superdex::studio {

// Mix-in interface for entities that hold path references to files and want to participate in
// AssetReferenceManager tracking.
//
// Implementers can be assets (e.g., BotAsset) or session-level containers (e.g., ViewportTab).
class IAssetReferencer {
 public:
  virtual ~IAssetReferencer() = default;

  /// Get the display name of the referencer for asset deletion popups etc.
  virtual std::string const& GetReferencerName() const = 0;

  /// Enumerate file paths referenced by this referencer.
  /// Implementations should emit absolute paths.
  virtual void ForEachReferencedPath(
      std::function<void(mochi::Path const&)> const& callback) const = 0;

  /// Rewrite any internal references from `oldPath` to `newPath`.
  ///
  /// Implementers are responsible for any side effects (e.g., MarkDirty,
  /// undo-stack reset) that follow from the rewrite.
  ///
  /// @return true if any internal reference was actually rewritten. The manager
  ///   uses this to keep its forward/reverse indices in sync — when an
  ///   implementer returns false, the index entry for `asset` is NOT updated.
  virtual bool RewriteReferencedPath(mochi::Path const& oldPath, mochi::Path const& newPath) = 0;
};

} // namespace superdex::studio
