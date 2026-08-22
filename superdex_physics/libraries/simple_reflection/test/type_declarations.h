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

#include <simple_reflection/simple_reflection.h>

#include <string>

namespace SReflectTest {

// A struct with an external reflection definition in this file
struct MyStructEx {
  int value = 0;
  bool operator==(const MyStructEx& rhs) const {
    return value == rhs.value;
  }
};

// A struct with an external reflection declaration in this file. Defined in the cpp.
struct MyDeclaredStructEx {
  int value = 0;
  bool operator==(const MyDeclaredStructEx& rhs) const {
    return value == rhs.value;
  }
};

} // namespace SReflectTest

// NOTE: External reflection declarations/definitions must go in the global namespace for now.

SR_BeginStructEx(SReflectTest::MyStructEx);
SR_Field(value);
SR_EndStructEx();

SR_DeclareStructEx(SReflectTest::MyDeclaredStructEx);
