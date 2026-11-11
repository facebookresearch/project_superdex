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

#include <mochi_core/mochi_init.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/span.h>

#include <mutex>

using namespace mochi;

#if !MOCHI_OPTIMIZED && MOCHI_WARN_IF_NOT_OPTIMIZED
#pragma message(                                                                                       \
    "\n\n******************************************************************************************\n" \
    "WARNING: Mochi is being compiled without optimizations. This will greatly reduce performance.\n"  \
    "\n"                                                                                               \
    "For single-configuration CMake generators (for example, Ninja or Unix Makefiles),\n"              \
    "  configure with \"-DCMAKE_BUILD_TYPE=Release\".\n"                                               \
    "\n"                                                                                               \
    "For multi-configuration CMake generators (for example, Ninja Multi-Config or Visual Studio),\n"   \
    "  pass \"--config Release\" to \"cmake --build\" or select Release in your IDE.\n"                \
    "\n"                                                                                               \
    "To silence this warning, configure CMake with \"-DMOCHI_WARN_IF_NOT_OPTIMIZED=OFF\"\n"            \
    "  or define the MOCHI_WARN_IF_NOT_OPTIMIZED preprocessor macro to 0.\n"                           \
    "******************************************************************************************\n\n")

#endif

#if MOCHI_ARCH_CPU && \
    ((!MOCHI_ARCH_X64_AVX2 && !MOCHI_ARCH_ARM_NEON) || (MOCHI_ARCH_X64 && !MOCHI_ARCH_X64_FMA))
#error \
    "Please enable SIMD + FMA in your compiler settings. FMA is important for both performance and rounding accuracy.\n" \
    "- For x86-64 with AVX2:\n" \
    "  * MSVC: Add '/arch:AVX2' compiler flag (implies FMA).\n" \
    "  * clang-cl: Add '/arch:AVX2 -mfma' compiler flags.\n" \
    "  * GCC/Clang: Add '-mavx2 -mfma' compiler flags.\n" \
    "- For x86-64 with AVX-512:\n" \
    "  * MSVC: Add '/arch:AVX512' compiler flag (implies FMA).\n" \
    "  * clang-cl: Add '/arch:AVX512 -mfma' compiler flags.\n" \
    "  * GCC/Clang: Add '-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mfma' compiler flags.\n" \
    "- For ARM64 platforms:\n" \
    "  * GCC/Clang: Add '-march=armv8-a+simd' compiler flag (implies FMA).\n" \
    "- For CUDA GPU:\n" \
    "  * NVCC: No flag required. SIMD instructions are not available on CUDA.\n" \
    "Other architectures and compilers are not supported."
#endif

// Global Initialization State
static std::mutex g_initMutex;
static bool g_initialized = false;

void mochi::Initialize() {
  std::scoped_lock lock(g_initMutex);
  if (g_initialized) {
    MOCHI_LOG_WARNING("Redundant call to mochi::Initialize");
  } else {
    g_initialized = true;

    ProfilerInitialize();
  }
}

bool mochi::IsInitialized() {
  std::scoped_lock lock(g_initMutex);
  return g_initialized;
}

void mochi::ShutDown() {
  std::scoped_lock lock(g_initMutex);
  if (g_initialized) {
    g_initialized = false;

    ProfilerShutdown();
  } else {
    MOCHI_LOG_WARNING("Redundant call to mochi::ShutDown");
  }
}

#if MOCHI_USE_EXTERN_TEMPLATE
// Explicit template instantiation can go in any cpp file.
// The corresponding "extern template" declarations reduce code bloat from out-of-line
// function definitions, especially in debug builds.
namespace mochi {
template class NdArray<int, 2>;
template class NdArray<int, 3>;
template class NdArray<int, 4>;
template class NdArray<real, 2>;
template class NdArray<real, 3>;
template class NdArray<real, 4>;
template class NdArray<real, 2, 2>;
template class NdArray<real, 2, 3>;
template class NdArray<real, 3, 2>;
template class NdArray<real, 3, 3>;
template class NdArray<real, 4, 3>;
template class Span<int>;
template class Span<NdArray<int, 2>>;
template class Span<NdArray<int, 3>>;
template class Span<NdArray<int, 4>>;
template class Span<real>;
template class Span<NdArray<real, 3>>;
template class Span<int, int>;
template class Span<NdArray<int, 2>, int>;
template class Span<NdArray<int, 3>, int>;
template class Span<NdArray<int, 4>, int>;
template class Span<real, int>;
template class Span<NdArray<real, 3>, int>;
} // namespace mochi
#endif // MOCHI_USE_EXTERN_TEMPLATE
