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

#include <cstdint>

namespace mochi {

struct CoordinateSpace;

class DebugServer {
 public:
  virtual void Start(uint16_t preferredPort = 7333) = 0;
  virtual void Stop() = 0;
  virtual bool HasStarted() const = 0;
  virtual bool HasConnection() const = 0;
  virtual uint16_t GetPort() const = 0;
  virtual void SetCoordinateSpace(CoordinateSpace const& space) = 0;

 protected:
  // DebugServer is owned by the Context. Do not delete it.
  ~DebugServer() = default;
};

} // namespace mochi
