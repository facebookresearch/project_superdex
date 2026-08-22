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

/**
  Macros used to declare the MochiPhysics public API
*/
#pragma once

// DLL Import/Export Attributes
#ifndef MOCHI_DLL_EXPORT
#if defined(_MSC_VER)
#define MOCHI_DLL_EXPORT __declspec(dllexport)
#define MOCHI_DLL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define MOCHI_DLL_EXPORT __attribute__((visibility("default")))
#define MOCHI_DLL_IMPORT
#else
#define MOCHI_DLL_EXPORT
#define MOCHI_DLL_IMPORT
#pragma warning Unknown dynamic link import / export semantics.
#endif
#endif

// MOCHI_PHYSICS_EXPORTS should be defined to 1 when compiling the internals of MochiPhysics project
// (lib or dll). It should be 0 (or undefined) when including MochiPhysics headers from elsewhere.
#ifndef MOCHI_PHYSICS_EXPORTS
#define MOCHI_PHYSICS_EXPORTS 0
#endif

// By default, assume this library is being build as a DLL. If you want to build it as a static,
// then define MOCHI_PHYSICS_DYNAMICALLY_LINKED to 0 in your build system (CMake, Visual Studio,
// etc...)
#ifndef MOCHI_PHYSICS_DYNAMICALLY_LINKED
#define MOCHI_PHYSICS_DYNAMICALLY_LINKED 1
#endif
#if MOCHI_PHYSICS_DYNAMICALLY_LINKED && MOCHI_PHYSICS_EXPORTS
// Declare API symbols as MOCHI_DLL_EXPORT when building the CPP files in this project into a DLL.
// That is the only time MOCHI_PHYSICS_EXPORTS is 1.
#define MOCHI_API MOCHI_DLL_EXPORT
#elif MOCHI_PHYSICS_DYNAMICALLY_LINKED && !MOCHI_PHYSICS_EXPORTS
// Declare API symbols as MOCHI_DLL_IMPORT when this project is being build as a DLL, but the
// headers are being included elsewhere.
#define MOCHI_API MOCHI_DLL_IMPORT
#else
// Declare API symbols with no import/export semantics if this project is being built as a static
// library
#define MOCHI_API
#endif
