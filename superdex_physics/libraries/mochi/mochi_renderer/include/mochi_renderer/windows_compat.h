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

// Windows compatibility header for Filament.
//
// On Windows, system headers (included transitively by STL headers like
// <filesystem>, <functional>, etc.) define macros that conflict with
// identifiers used by Filament:
//
//   OPAQUE, TRANSPARENT  (wingdi.h)  — conflict with filament::BlendMode
//   near, far            (windef.h)  — conflict with parameter names
//   min, max             (windef.h)  — conflict with std::min/max
//   ERROR                (wingdi.h)  — conflict with log levels
//   PURE                 (objbase.h) — conflict with filament enum
//   DOMAIN               (math.h)   — conflict with filament enum
//
// This header is force-included (/FI) on MSVC before all source files,
// ensuring Windows.h is included first (setting its include guard) and
// then cleaning up the conflicting macros. Subsequent includes of
// Windows.h by STL headers become no-ops due to the include guard.

#pragma once

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>

// Clean up macros that conflict with Filament identifiers.
// This is equivalent to filament's <utils/unwindows.h>.
#ifdef OPAQUE
#undef OPAQUE
#endif

#ifdef TRANSPARENT
#undef TRANSPARENT
#endif

#ifdef near
#undef near
#endif

#ifdef far
#undef far
#endif

#ifdef NEAR
#undef NEAR
#endif

#ifdef FAR
#undef FAR
#endif

#ifdef ERROR
#undef ERROR
#endif

#ifdef PURE
#undef PURE
#endif

#ifdef DOMAIN
#undef DOMAIN
#endif

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#endif // _WIN32
