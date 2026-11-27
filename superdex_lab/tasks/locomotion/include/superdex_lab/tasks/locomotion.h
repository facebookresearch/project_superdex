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

namespace superdex::lab {

class LabKinds;

/// Registers the velocity-tracking task family's components and terms (unitree_rl_mjlab's velocity
/// task): see locomotion_components.h and locomotion_terms.h.
void RegisterLocomotionKinds(LabKinds& kinds);

} // namespace superdex::lab
