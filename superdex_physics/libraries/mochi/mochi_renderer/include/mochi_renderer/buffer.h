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

#include <memory>
#include <mutex>
#include <utility>

namespace mochi_renderer {

// Manages the thread-safe hand-off of data from a single producer to a single consumer.
// Used to sync data between simulation thread and render thread.
template <typename T>
class ProducerConsumerBuffer {
 public:
  ProducerConsumerBuffer();

  // Producer thread writes data then calls Produce().
  T& GetProducerData();
  void Produce();

  // Consumer thread calls Consume(). If it returns true,
  // then there is new data the consumer can read.
  bool Consume();
  T& GetConsumerData();

 private:
  std::mutex _mutex;
  std::unique_ptr<T> _front;
  std::unique_ptr<T> _middle;
  std::unique_ptr<T> _back;
  bool _middleDirty = false;
};

template <typename T>
inline ProducerConsumerBuffer<T>::ProducerConsumerBuffer()
    : _front(std::make_unique<T>()), _middle(std::make_unique<T>()), _back(std::make_unique<T>()) {}

template <typename T>
inline T& ProducerConsumerBuffer<T>::GetProducerData() {
  return *_back;
}

template <typename T>
inline void ProducerConsumerBuffer<T>::Produce() {
  std::lock_guard lock(_mutex);
  std::swap(_middle, _back);
  _middleDirty = true;
}

template <typename T>
inline T& ProducerConsumerBuffer<T>::GetConsumerData() {
  return *_front;
}

template <typename T>
inline bool ProducerConsumerBuffer<T>::Consume() {
  std::lock_guard lock(_mutex);
  if (_middleDirty) {
    std::swap(_front, _middle);
    _middleDirty = false;
    return true;
  } else {
    return false;
  }
}

} // namespace mochi_renderer
