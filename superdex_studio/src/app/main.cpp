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

#include <imguios/gpu_selector.h>

#include "app/app.h"

#include <mochi_core/utils/console.h>

int main(int, char**) {
  // This is a GUI-subsystem binary on Windows, so it starts with no console and would otherwise
  // discard everything it prints -- including the reason it is about to fail.
  mochi::AttachParentConsole();
  superdex::studio::SuperDexStudio app;
  app.Run();
  return 0;
}
