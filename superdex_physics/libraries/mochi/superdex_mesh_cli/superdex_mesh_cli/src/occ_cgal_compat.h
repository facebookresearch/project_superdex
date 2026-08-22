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

// Enforces correct include ordering between CGAL and OpenCascade (OCCT). OCCT defines
//   #define Handle(Class) opencascade::handle<Class>
// which collides with CGAL/Boost preprocessor usage of the identifier `Handle`. Any translation
// unit that needs both libraries must include its CGAL headers BEFORE this header (and before any
// OCCT header). This header includes no CGAL headers itself -- each consumer needs different CGAL
// modules -- it only checkpoints the ordering at compile time.

// CGAL sets CGAL_VERSION_NR via its config header, which any CGAL include pulls in. If this fires,
// move your CGAL includes above this header.
#if !defined(CGAL_VERSION_NR)
#error \
    "CGAL headers must be included before occ_cgal_compat.h to avoid the OCCT Handle macro conflict."
#endif
