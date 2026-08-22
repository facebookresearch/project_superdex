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

#include "mesh_cli_client.h"

#include "mesh_cli_adapter.h"

#include <mochi_core/mochi_platform.h>
#include <mochi_mesh/mesh_cli_control.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if MOCHI_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#if MOCHI_PLATFORM_MACOS
#include <crt_externs.h> // _NSGetEnviron
#include <mach-o/dyld.h> // _NSGetExecutablePath
#define MOCHI_ENVIRON (*_NSGetEnviron())
#else
// NOLINTNEXTLINE(readability-redundant-declaration)
extern char** environ;
#define MOCHI_ENVIRON environ
#endif
#endif

using namespace mochi;

namespace {

// Serialize pipe setup and process creation across RunSubprocess calls. This closes the non-atomic
// pipe()+fcntl window between these calls and prevents their Windows helpers from inheriting one
// another's temporarily inheritable child handles.
// NOLINTNEXTLINE(facebook-thread-safety-analysis)
std::mutex g_spawnMutex;

// Protects the live-process registry so cancellation cannot race handle or pid release.
// NOLINTNEXTLINE(facebook-thread-safety-analysis)
std::mutex g_liveProcessMutex;

// Name of the deployed CLI executable.
#if MOCHI_PLATFORM_WINDOWS
constexpr char const* kCliExecutableName = "superdex_mesh_cli.exe";
#else
constexpr char const* kCliExecutableName = "superdex_mesh_cli";
#endif

// Sets @p error to a dynamically-built message. Error keeps the description pointer without
// copying, so the message is stashed in a thread-local that outlives this call.
void SetClientError(Error& error, std::string message) {
  static thread_local std::string storage;
  storage = std::move(message);
  error.SetFirstError(storage.c_str(), __FILE__, __LINE__);
}

#if MOCHI_PLATFORM_WINDOWS
std::vector<HANDLE> g_liveProcesses;
void RegisterLiveProcess(HANDLE process) {
  std::scoped_lock const lock(g_liveProcessMutex);
  g_liveProcesses.push_back(process);
}
void UnregisterLiveProcess(HANDLE process) {
  std::scoped_lock const lock(g_liveProcessMutex);
  g_liveProcesses.erase(
      std::remove(g_liveProcesses.begin(), g_liveProcesses.end(), process), g_liveProcesses.end());
}
#else
std::vector<pid_t> g_liveProcesses;
void RegisterLiveProcess(pid_t pid) {
  std::scoped_lock const lock(g_liveProcessMutex);
  g_liveProcesses.push_back(pid);
}
void UnregisterLiveProcess(pid_t pid) {
  std::scoped_lock const lock(g_liveProcessMutex);
  g_liveProcesses.erase(
      std::remove(g_liveProcesses.begin(), g_liveProcesses.end(), pid), g_liveProcesses.end());
}
#endif

#if MOCHI_PLATFORM_WINDOWS

std::string GetExecutablePath() {
  wchar_t buffer[MAX_PATH];
  DWORD const length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  if (length == 0 || length == MAX_PATH) {
    return {};
  }
  int const utf8Length = WideCharToMultiByte(
      CP_UTF8, 0, buffer, static_cast<int>(length), nullptr, 0, nullptr, nullptr);
  std::string utf8(static_cast<size_t>(utf8Length), '\0');
  WideCharToMultiByte(
      CP_UTF8, 0, buffer, static_cast<int>(length), utf8.data(), utf8Length, nullptr, nullptr);
  return utf8;
}

std::wstring Utf8ToWide(std::string const& utf8) {
  if (utf8.empty()) {
    return {};
  }
  int const length =
      MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
  std::wstring wide(static_cast<size_t>(length), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), length);
  return wide;
}

std::wstring QuoteWindowsArgument(std::wstring const& argument) {
  std::wstring quoted{L'"'};
  size_t backslashes = 0;
  for (wchar_t const character : argument) {
    if (character == L'\\') {
      ++backslashes;
    } else if (character == L'"') {
      quoted.append(backslashes * 2 + 1, L'\\');
      quoted.push_back(character);
      backslashes = 0;
    } else {
      quoted.append(backslashes, L'\\');
      quoted.push_back(character);
      backslashes = 0;
    }
  }
  quoted.append(backslashes * 2, L'\\');
  quoted.push_back(L'"');
  return quoted;
}

// Spawns the helper, writes the whole request to its stdin, then reads its stdout to EOF. The
// helper reads stdin to EOF before writing, so this one-shot write-then-read sequence cannot
// deadlock.
bool RunSubprocess(
    std::string const& exePath,
    std::string const& helperArgument,
    Span<char const> input,
    std::vector<char>& output,
    Error& error) {
  HANDLE childStdinRead = nullptr;
  HANDLE childStdinWrite = nullptr;
  HANDLE childStdoutRead = nullptr;
  HANDLE childStdoutWrite = nullptr;
  std::wstring const exeWide = Utf8ToWide(exePath);
  std::wstring commandLine = QuoteWindowsArgument(exeWide);
  if (!helperArgument.empty()) {
    commandLine += L" " + QuoteWindowsArgument(Utf8ToWide(helperArgument));
  }
  std::vector<wchar_t> commandLineBuffer(commandLine.begin(), commandLine.end());
  commandLineBuffer.push_back(L'\0');

  PROCESS_INFORMATION processInfo{};
  BOOL created = FALSE;
  {
    std::scoped_lock const spawnLock(g_spawnMutex);

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    if (!CreatePipe(&childStdinRead, &childStdinWrite, &sa, 0)) {
      MOCHI_ERROR_SET(error, "Failed to create stdin pipe for superdex_mesh_cli.");
      return false;
    }
    if (!CreatePipe(&childStdoutRead, &childStdoutWrite, &sa, 0)) {
      CloseHandle(childStdinRead);
      CloseHandle(childStdinWrite);
      MOCHI_ERROR_SET(error, "Failed to create stdout pipe for superdex_mesh_cli.");
      return false;
    }
    // The parent's pipe ends must not be inherited by the child.
    if (!SetHandleInformation(childStdinWrite, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(childStdoutRead, HANDLE_FLAG_INHERIT, 0)) {
      CloseHandle(childStdinRead);
      CloseHandle(childStdinWrite);
      CloseHandle(childStdoutRead);
      CloseHandle(childStdoutWrite);
      MOCHI_ERROR_SET(error, "Failed to configure pipes for superdex_mesh_cli.");
      return false;
    }

    STARTUPINFOW startupInfo{};
    startupInfo.cb = sizeof(startupInfo);
    startupInfo.dwFlags = STARTF_USESTDHANDLES;
    startupInfo.hStdInput = childStdinRead;
    startupInfo.hStdOutput = childStdoutWrite;
    startupInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    // CREATE_NO_WINDOW: the helper is a console-subsystem program, so without this a console window
    // flashes on screen for every mesh operation. Its stdio already goes to the pipes above.
    created = CreateProcessW(
        exeWide.c_str(),
        commandLineBuffer.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startupInfo,
        &processInfo);

    // Close the inheritable child ends before another request enters its spawn transaction.
    CloseHandle(childStdinRead);
    CloseHandle(childStdoutWrite);
  }

  if (!created) {
    CloseHandle(childStdinWrite);
    CloseHandle(childStdoutRead);
    MOCHI_ERROR_SET(error, "Failed to spawn superdex_mesh_cli.");
    return false;
  }
  // Make the child cancellable for the duration of this call (until just before its handle closes).
  RegisterLiveProcess(processInfo.hProcess);

  bool writeOk = true;
  size_t offset = 0;
  while (offset < input.size()) {
    DWORD const toWrite = static_cast<DWORD>(std::min<size_t>(input.size() - offset, 1u << 20));
    DWORD written = 0;
    if (!WriteFile(childStdinWrite, input.data() + offset, toWrite, &written, nullptr)) {
      writeOk = false;
      break;
    }
    offset += written;
  }
  CloseHandle(childStdinWrite);

  char buffer[4096];
  DWORD bytesRead = 0;
  while (ReadFile(childStdoutRead, buffer, sizeof(buffer), &bytesRead, nullptr) && bytesRead > 0) {
    output.insert(output.end(), buffer, buffer + bytesRead);
  }
  CloseHandle(childStdoutRead);

  WaitForSingleObject(processInfo.hProcess, INFINITE);
  DWORD exitCode = 0;
  bool const gotExitCode = GetExitCodeProcess(processInfo.hProcess, &exitCode) != 0;
  // Stop tracking the child before closing its handle so a concurrent cancel never terminates a
  // closed handle.
  UnregisterLiveProcess(processInfo.hProcess);
  CloseHandle(processInfo.hProcess);
  CloseHandle(processInfo.hThread);

  // The helper always exits 0 on a handled error (reporting detail in-band), so a non-zero exit
  // means it crashed before producing a response (which includes a cancel that terminated it).
  // Surface that instead of a generic decode failure.
  if (gotExitCode && exitCode != 0) {
    SetClientError(error, "superdex_mesh_cli exited with status " + std::to_string(exitCode) + ".");
    return false;
  }
  MOCHI_ERROR_IF(!writeOk, error, "Failed to write the request to superdex_mesh_cli.");
  MOCHI_ERROR_RETURN(error, false);
  return true;
}

#else // POSIX

std::string GetExecutablePath() {
#if MOCHI_PLATFORM_MACOS
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    return {};
  }
  buffer.resize(std::strlen(buffer.c_str()));
  return buffer;
#else
  char buffer[4096];
  ssize_t const length = readlink("/proc/self/exe", buffer, sizeof(buffer));
  if (length <= 0) {
    return {};
  }
  return {buffer, static_cast<size_t>(length)};
#endif
}

// Writes to a helper that has already exited would otherwise raise SIGPIPE and crash the process.
// The plan requires a missing/unspawnable helper to surface a clean Error, never a crash, so we
// install SIG_IGN once (writes then fail with EPIPE, which the write loop handles).
void EnsureSigPipeIgnored() {
  static bool const ignored = []() {
    std::signal(SIGPIPE, SIG_IGN);
    return true;
  }();
  (void)ignored;
}

bool SetCloseOnExec(int fd) {
  int flags = -1;
  do {
    flags = fcntl(fd, F_GETFD);
  } while (flags < 0 && errno == EINTR);
  if (flags < 0) {
    return false;
  }

  int result = -1;
  do {
    result = fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
  } while (result < 0 && errno == EINTR);
  return result == 0;
}

// The pipe()+fcntl fallback is race-free only because RunSubprocess holds g_spawnMutex across
// pipe creation and posix_spawn. Keep this helper inside that transaction.
bool CreateCloseOnExecPipe(int pipeFds[2]) {
#if MOCHI_PLATFORM_LINUX || MOCHI_PLATFORM_ANDROID
  if (pipe2(pipeFds, O_CLOEXEC) == 0) {
    return true;
  }
  // ENOSYS/EINVAL mean the atomic API is unavailable. Other errors describe the pipe request
  // itself, so retrying with a less-safe implementation would mask the real failure.
  if (errno != ENOSYS && errno != EINVAL) {
    return false;
  }
#endif

  if (pipe(pipeFds) != 0) {
    return false;
  }
  if (!SetCloseOnExec(pipeFds[0]) || !SetCloseOnExec(pipeFds[1])) {
    close(pipeFds[0]);
    close(pipeFds[1]);
    return false;
  }
  return true;
}

bool RunSubprocess(
    std::string const& exePath,
    std::string const& helperArgument,
    Span<char const> input,
    std::vector<char>& output,
    Error& error) {
  EnsureSigPipeIgnored();

  int stdinPipe[2]; // parent writes stdinPipe[1] -> child reads stdinPipe[0] as stdin
  int stdoutPipe[2]; // child writes stdoutPipe[1] as stdout -> parent reads stdoutPipe[0]
  pid_t pid = 0;
  int spawnResult = 0;
  {
    std::scoped_lock const spawnLock(g_spawnMutex);
    if (!CreateCloseOnExecPipe(stdinPipe)) {
      MOCHI_ERROR_SET(error, "Failed to create stdin pipe for superdex_mesh_cli.");
      return false;
    }
    if (!CreateCloseOnExecPipe(stdoutPipe)) {
      close(stdinPipe[0]);
      close(stdinPipe[1]);
      MOCHI_ERROR_SET(error, "Failed to create stdout pipe for superdex_mesh_cli.");
      return false;
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, stdinPipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, stdoutPipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, stdinPipe[1]);
    posix_spawn_file_actions_addclose(&actions, stdoutPipe[0]);
    posix_spawn_file_actions_addclose(&actions, stdinPipe[0]);
    posix_spawn_file_actions_addclose(&actions, stdoutPipe[1]);

    std::string mutableExePath = exePath;
    std::string mutableHelperArgument = helperArgument;
    char* argvWithArgument[] = {mutableExePath.data(), mutableHelperArgument.data(), nullptr};
    char* argvWithoutArgument[] = {mutableExePath.data(), nullptr};
    char** argv = helperArgument.empty() ? argvWithoutArgument : argvWithArgument;
    spawnResult = posix_spawn(&pid, exePath.c_str(), &actions, nullptr, argv, MOCHI_ENVIRON);
    posix_spawn_file_actions_destroy(&actions);
  }

  // The parent does not use the child's ends of the pipes.
  close(stdinPipe[0]);
  close(stdoutPipe[1]);

  if (spawnResult != 0) {
    close(stdinPipe[1]);
    close(stdoutPipe[0]);
    MOCHI_ERROR_SET(error, "Failed to spawn superdex_mesh_cli.");
    return false;
  }
  // Make the child cancellable for the duration of this call (until it is reaped below).
  RegisterLiveProcess(pid);

  bool writeOk = true;
  size_t offset = 0;
  while (offset < input.size()) {
    ssize_t const written = write(stdinPipe[1], input.data() + offset, input.size() - offset);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      writeOk = false;
      break;
    }
    offset += static_cast<size_t>(written);
  }
  close(stdinPipe[1]);

  char buffer[4096];
  while (true) {
    ssize_t const bytesRead = read(stdoutPipe[0], buffer, sizeof(buffer));
    if (bytesRead < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (bytesRead == 0) {
      break;
    }
    output.insert(output.end(), buffer, buffer + bytesRead);
  }
  close(stdoutPipe[0]);

  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  // Stop tracking the child once it is reaped so a concurrent cancel never signals a reused pid.
  UnregisterLiveProcess(pid);

  // The helper always exits 0 on a handled error (reporting detail in-band), so a non-zero exit or
  // a signal means it crashed before producing a response (which includes a cancel that killed it).
  // Surface that instead of a generic decode failure.
  if (WIFSIGNALED(status)) {
    SetClientError(
        error,
        "superdex_mesh_cli was terminated by signal " + std::to_string(WTERMSIG(status)) + ".");
    return false;
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
    SetClientError(
        error, "superdex_mesh_cli exited with status " + std::to_string(WEXITSTATUS(status)) + ".");
    return false;
  }
  MOCHI_ERROR_IF(!writeOk, error, "Failed to write the request to superdex_mesh_cli.");
  MOCHI_ERROR_RETURN(error, false);
  return true;
}

#endif

std::vector<char> InvokeMeshCliAtPath(
    std::string const& cliPath,
    std::string const& helperArgument,
    mochi::mesh::cli::GeometryOp op,
    Span<char const> requestPayload,
    Error& error) {
  std::vector<char> const request = mochi::mesh::cli::EncodeRequestFrame(op, requestPayload);
  std::vector<char> responseBytes;
  RunSubprocess(cliPath, helperArgument, request, responseBytes, error);
  MOCHI_ERROR_RETURN(error, {});

  uint32_t status = 0;
  std::vector<char> payload;
  MOCHI_ERROR_IF(
      !mochi::mesh::cli::DecodeResponseFrame(responseBytes, status, payload),
      error,
      "superdex_mesh_cli returned a malformed or empty response.");
  MOCHI_ERROR_RETURN(error, {});

  if (status != 0) {
    static thread_local std::string helperErrorMessage;
    helperErrorMessage.assign(payload.begin(), payload.end());
    if (helperErrorMessage.empty()) {
      helperErrorMessage = "superdex_mesh_cli reported an error processing the request.";
    }
    static_cast<Error&>(error).SetFirstError(helperErrorMessage.c_str(), __FILE__, __LINE__);
    return {};
  }

  return payload;
}

} // namespace

void mochi::mesh::CancelInFlightMeshCli() {
  std::scoped_lock const lock(g_liveProcessMutex);
#if MOCHI_PLATFORM_WINDOWS
  for (HANDLE const process : g_liveProcesses) {
    TerminateProcess(process, 1);
  }
#else
  for (pid_t const pid : g_liveProcesses) {
    kill(pid, SIGKILL);
  }
#endif
}

std::string mochi::mesh::FindMeshCliPath() {
  if (char const* const envPath = std::getenv("SUPERDEX_MESH_CLI_PATH")) {
    if (envPath[0] != '\0') {
      // Checked like the branches below rather than trusted. This is the one candidate that
      // names a path instead of finding one, so without this a stale, misspelled or relative
      // value comes back as a perfectly good helper -- and callers are promised an absolute
      // path to something that exists.
      //
      // An override that does not resolve ends the search rather than falling through to it.
      // The variable exists so a developer can point at a locally built helper, so quietly
      // substituting a different one is the worst answer available: the caller gets results
      // from a binary they did not choose, with nothing to say why.
      std::error_code canonicalEc;
      std::filesystem::path const resolved =
          std::filesystem::weakly_canonical(envPath, canonicalEc);
      std::error_code statEc;
      if (!canonicalEc && std::filesystem::is_regular_file(resolved, statEc)) {
        return resolved.string();
      }
      return {};
    }
  }

  std::string const exePath = GetExecutablePath();
  if (exePath.empty()) {
    return {};
  }
  std::filesystem::path const exeDir = std::filesystem::path(exePath).parent_path();

  std::error_code ec;
  std::filesystem::path const adjacent = exeDir / kCliExecutableName;
  if (std::filesystem::exists(adjacent, ec)) {
    return adjacent.string();
  }

  // The wheel layout: each distribution stores its native payload in `<package>/_native/`, so a
  // caller at `site-packages/superdex_studio/_native/` reaches the separately-shipped GPL helper
  // two levels up.
  std::filesystem::path const sibling =
      exeDir / ".." / ".." / "superdex_mesh_cli" / "_native" / kCliExecutableName;
  if (std::filesystem::exists(sibling, ec)) {
    std::error_code canonicalEc;
    std::filesystem::path const resolved = std::filesystem::weakly_canonical(sibling, canonicalEc);
    return canonicalEc ? sibling.string() : resolved.string();
  }

  return {};
}

std::string mochi::mesh::ResolveMeshCliPath() {
  return FindMeshCliPath();
}

std::vector<char>
mochi::mesh::InvokeMeshCli(cli::GeometryOp op, Span<char const> requestPayload, Error& error) {
  return InvokeMeshCli(std::string{}, op, requestPayload, error);
}

std::vector<char> mochi::mesh::InvokeMeshCli(
    std::string const& cliExtraArg,
    cli::GeometryOp op,
    Span<char const> requestPayload,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  std::string const cliPath = FindMeshCliPath();
  MOCHI_ERROR_IF(
      cliPath.empty(),
      error,
      "superdex_mesh_cli helper not found, so this mesh operation cannot run. Install the "
      "superdex-mesh-cli distribution, place the helper next to this executable, or point "
      "SUPERDEX_MESH_CLI_PATH at it.");
  MOCHI_ERROR_RETURN(error, {});

  return InvokeMeshCliAtPath(cliPath, cliExtraArg, op, requestPayload, error);
}

mochi::MeshData
mochi::mesh::InvokeMeshOp(cli::GeometryOp op, cli::PayloadWriter const& writer, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  std::vector<char> const response = InvokeMeshCli(op, writer.Bytes(), error);
  MOCHI_ERROR_RETURN(error, {});

  cli::PayloadReader reader(response);
  cli::MeshData cliMesh;
  MOCHI_ERROR_IF(
      !reader.ReadMeshData(cliMesh) || !reader.AtEnd(),
      error,
      "Malformed mesh response from superdex_mesh_cli.");
  MOCHI_ERROR_RETURN(error, {});

  return cli_adapter::FromCliMeshData(cliMesh, error);
}
