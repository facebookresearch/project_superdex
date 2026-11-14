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

#include <superdex_robotics/controllers/controller_base.h>

namespace superdex::robotics {

int ControllerBase::GetInputSize() const {
  return 0;
}

ControllerInputKind ControllerBase::GetInputKind() const {
  return ControllerInputKind::None;
}

Span<int const> ControllerBase::GetControlledDofs() const {
  return {};
}

void ControllerBase::GetCurrentInput(Span<real> /*input*/, superdex::Error& error) {
  MOCHI_ERROR_SET(error, "This controller is not driven by floats");
}

void ControllerBase::ComputeOutputFromInput(
    Span<real const> /*input*/,
    real /*dt*/,
    Span<real> /*efforts*/,
    superdex::Error& error) {
  MOCHI_ERROR_SET(error, "This controller is not driven by floats");
}

} // namespace superdex::robotics
