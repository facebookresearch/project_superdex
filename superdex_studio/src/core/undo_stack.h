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

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace superdex::studio {

/// Snapshot-based undo/redo stack.
///
/// Stores full JSON snapshots of document state. The caller provides callbacks
/// for serialization (snapshot) and deserialization (restore). The stack handles
/// push timing (mouse-button debounce), depth capping, fork-on-edit-after-undo,
/// and dirty-state tracking relative to the last save.
class UndoStack {
 public:
  using SnapshotFn = std::function<std::string()>;
  using SelectionFn = std::function<int()>;
  using RestoreFn = std::function<void(std::string const&, int selectionIndex)>;

  UndoStack() = default;

  /// Set up snapshot/restore callbacks and push a baseline snapshot.
  void Initialize(SnapshotFn snapshotFn, RestoreFn restoreFn, size_t maxDepth = 100);

  /// Optionally provide a callback that returns the current selection index.
  void SetSelectionFn(SelectionFn selectionFn);

  /// Call when an edit is made. Updates the internal timestamp.
  void MarkEdited();

  /// Call once per frame. Pushes a snapshot if there is a pending edit and the
  /// user is not currently interacting (e.g. dragging a slider).
  /// @param isInteracting true while the user is mid-interaction (mouse down)
  void MaybePushSnapshot(bool isInteracting);

  /// Push a snapshot immediately (for discrete operations like delete/add).
  void PushNow();

  void Undo();
  void Redo();
  bool CanUndo() const;
  bool CanRedo() const;

  /// Mark the current stack position as the saved state.
  void SetCurrentAsSaved();

  /// Returns true if the current position matches the last saved position.
  bool IsAtSavedState() const;

  /// Clear the stack and push a fresh baseline snapshot.
  void Reset();

  /// Returns true if Initialize() has been called.
  bool IsInitialized() const;

 private:
  void PushSnapshotInternal();

  struct Entry {
    std::string json;
    int selectionIndex = -1;
  };

  std::vector<Entry> _stack;
  int _currentIndex = -1;
  int _savedAtIndex = 0;
  size_t _maxDepth = 100;

  using Clock = std::chrono::steady_clock;
  Clock::time_point _timeLastEdited{};
  Clock::time_point _timeLastPushed{};

  SnapshotFn _snapshotFn;
  SelectionFn _selectionFn;
  RestoreFn _restoreFn;
  bool _initialized = false;
};

} // namespace superdex::studio
