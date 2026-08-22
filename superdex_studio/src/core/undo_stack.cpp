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

#include "core/undo_stack.h"

namespace superdex::studio {

void UndoStack::Initialize(SnapshotFn snapshotFn, RestoreFn restoreFn, size_t maxDepth) {
  _snapshotFn = std::move(snapshotFn);
  _restoreFn = std::move(restoreFn);
  _maxDepth = maxDepth;
  _initialized = true;
  Reset();
}

void UndoStack::SetSelectionFn(SelectionFn selectionFn) {
  _selectionFn = std::move(selectionFn);
}

void UndoStack::MarkEdited() {
  _timeLastEdited = Clock::now();
}

void UndoStack::MaybePushSnapshot(bool isInteracting) {
  if (!_initialized || isInteracting || _timeLastEdited <= _timeLastPushed) {
    return;
  }
  PushSnapshotInternal();
}

void UndoStack::PushNow() {
  if (!_initialized) {
    return;
  }
  PushSnapshotInternal();
}

void UndoStack::PushSnapshotInternal() {
  // Fork: if we've undone past the end, truncate the redo future
  if (_currentIndex < static_cast<int>(_stack.size()) - 1) {
    _stack.erase(_stack.begin() + _currentIndex + 1, _stack.end());
    if (_savedAtIndex > _currentIndex) {
      _savedAtIndex = -1;
    }
  }

  Entry entry;
  entry.json = _snapshotFn();
  entry.selectionIndex = _selectionFn ? _selectionFn() : -1;
  _stack.push_back(std::move(entry));
  _currentIndex = static_cast<int>(_stack.size()) - 1;
  _timeLastPushed = Clock::now();

  // Cap depth: remove oldest entries beyond the limit
  while (_stack.size() > _maxDepth) {
    _stack.erase(_stack.begin());
    --_currentIndex;
    if (_savedAtIndex > 0) {
      --_savedAtIndex;
    } else if (_savedAtIndex == 0) {
      _savedAtIndex = -1;
    }
  }
}

void UndoStack::Undo() {
  if (!CanUndo()) {
    return;
  }
  --_currentIndex;
  _restoreFn(_stack[_currentIndex].json, _stack[_currentIndex].selectionIndex);
  // Prevent the deserialization side-effects from triggering another push
  _timeLastPushed = Clock::now();
}

void UndoStack::Redo() {
  if (!CanRedo()) {
    return;
  }
  ++_currentIndex;
  _restoreFn(_stack[_currentIndex].json, _stack[_currentIndex].selectionIndex);
  _timeLastPushed = Clock::now();
}

bool UndoStack::CanUndo() const {
  return _initialized && _currentIndex > 0;
}

bool UndoStack::CanRedo() const {
  return _initialized && _currentIndex < static_cast<int>(_stack.size()) - 1;
}

void UndoStack::SetCurrentAsSaved() {
  _savedAtIndex = _currentIndex;
}

bool UndoStack::IsAtSavedState() const {
  return _savedAtIndex == _currentIndex;
}

void UndoStack::Reset() {
  _stack.clear();
  _savedAtIndex = 0;
  if (_snapshotFn) {
    Entry entry;
    entry.json = _snapshotFn();
    entry.selectionIndex = _selectionFn ? _selectionFn() : -1;
    _stack.push_back(std::move(entry));
  }
  _currentIndex = _stack.empty() ? -1 : 0;
  _timeLastPushed = Clock::now();
}

bool UndoStack::IsInitialized() const {
  return _initialized;
}

} // namespace superdex::studio
