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

#include <mochi_core/mochi_platform.h>

// On Windows, <winsock2.h> must be included before any header that pulls in
// <Windows.h> (mochi_core/utils/log.h below does). Including it first defines the
// _WINSOCKAPI_ guard so Windows.h skips the legacy <winsock.h>, whose declarations
// conflict with <winsock2.h>. <ws2tcpip.h> depends on <winsock2.h> and must follow
// it; keep this group ordered (lowercase names also keep clang-format from sorting
// them the wrong way).
//
// <winsock2.h> transitively includes <Windows.h>, so define NOMINMAX first to stop it
// leaking the min/max function-like macros, which would otherwise clobber std::min /
// std::max in later-included mochi headers (e.g. dynamic_array.h).
#if MOCHI_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <mochi_core/net/client_socket.h>
#include <mochi_core/net/server_list.h>
#include <mochi_core/net/server_socket.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <marl/thread.h> // For marl::Thread::setName

#if !MOCHI_PLATFORM_WINDOWS
#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h> // getifaddrs / freeifaddrs (POSIX interface enumeration)
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace mochi::net {

// -------------------------------------------------------------------------------------------------
// Platform Socket Abstraction
// -------------------------------------------------------------------------------------------------

#if MOCHI_PLATFORM_WINDOWS

using SocketHandle = SOCKET;
static constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;

static void PlatformInit() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
    // Intentionally do not call WSACleanup: Winsock refcounts per process, so the single ref is
    // reclaimed at process exit; an atexit cleanup could run before static-lifetime socket
    // destructors and tear Winsock down underneath them.
  });
}

static void CloseSocket(SocketHandle s) {
  closesocket(s);
}

[[nodiscard]] static bool SetNonBlocking(SocketHandle s) {
  u_long mode = 1;
  return ioctlsocket(s, FIONBIO, &mode) == 0;
}

[[nodiscard]] static bool SetBlocking(SocketHandle s) {
  u_long mode = 0;
  return ioctlsocket(s, FIONBIO, &mode) == 0;
}

#else

using SocketHandle = int;
static constexpr SocketHandle kInvalidSocket = -1;

static void PlatformInit() {}

static void CloseSocket(SocketHandle s) {
  close(s);
}

[[nodiscard]] static bool SetNonBlocking(SocketHandle s) {
  int const flags = fcntl(s, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  return fcntl(s, F_SETFL, flags | O_NONBLOCK) == 0;
}

[[nodiscard]] static bool SetBlocking(SocketHandle s) {
  int const flags = fcntl(s, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  return fcntl(s, F_SETFL, flags & ~O_NONBLOCK) == 0;
}

#endif

// Flags passed to send(): MSG_NOSIGNAL (where available) suppresses SIGPIPE on a broken pipe.
#if defined(MSG_NOSIGNAL)
static constexpr int kSendFlags = MSG_NOSIGNAL;
#else
static constexpr int kSendFlags = 0;
#endif

#if defined(MSG_DONTWAIT)
static constexpr int kNonBlockingSendFlags = kSendFlags | MSG_DONTWAIT;
#else
static constexpr int kNonBlockingSendFlags = kSendFlags;
#endif

// Apple lacks MSG_NOSIGNAL; SO_NOSIGPIPE is the per-socket equivalent. No-op elsewhere.
static void SetNoSigPipe([[maybe_unused]] SocketHandle s) {
#if MOCHI_PLATFORM_MACOS && defined(SO_NOSIGPIPE)
  int const opt = 1;
  setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &opt, sizeof(opt));
#endif
}

// Set an integer-valued socket option, hiding the Windows requirement that the value be passed via
// a char pointer (POSIX accepts the int directly).
static void SetSockOptInt(SocketHandle s, int level, int optname, int value) {
#if MOCHI_PLATFORM_WINDOWS
  setsockopt(s, level, optname, reinterpret_cast<char const*>(&value), sizeof(value));
#else
  setsockopt(s, level, optname, &value, sizeof(value));
#endif
}

// Disable Nagle's algorithm so small request/response messages are sent immediately. Without this,
// Nagle (which coalesces small writes) interacting with the peer's delayed ACKs adds a ~40 ms
// latency to each tiny message round trip, unacceptable for an interactive debugger.
static void SetTcpNoDelay(SocketHandle s) {
  SetSockOptInt(s, IPPROTO_TCP, TCP_NODELAY, 1);
}

// Bound how long a blocking recv() on `s` may wait before failing with a timeout. A zero duration
// clears the timeout, restoring indefinite blocking. Used to drop a peer that completes the TCP
// connection but never sends the handshake, so it cannot park a recv thread (and the client slot it
// holds) forever. On a timeout recv() returns < 0, which RecvAll/RecvAndValidateHandshake already
// treat as a failed read.
static void SetRecvTimeout(SocketHandle s, std::chrono::milliseconds timeout) {
#if MOCHI_PLATFORM_WINDOWS
  DWORD const ms = static_cast<DWORD>(timeout.count());
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char const*>(&ms), sizeof(ms));
#else
  timeval tv{};
  tv.tv_sec = static_cast<decltype(tv.tv_sec)>(timeout.count() / 1000);
  tv.tv_usec = static_cast<decltype(tv.tv_usec)>((timeout.count() % 1000) * 1000);
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

// Windows has no MSG_DONTWAIT equivalent for send(), so the client send thread cannot use the
// wakeable non-blocking send loop there. Bound only the Windows fallback with a conservative
// timeout: long enough for ordinary debugger pauses, but not forever if a peer stops reading
// permanently.
static void SetSendTimeout(
    [[maybe_unused]] SocketHandle s,
    [[maybe_unused]] std::chrono::milliseconds timeout) {
#if MOCHI_PLATFORM_WINDOWS
  DWORD const ms = static_cast<DWORD>(timeout.count());
  setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<char const*>(&ms), sizeof(ms));
#endif
}

// Allow bind() to reuse a port left in TIME_WAIT by a recent close. Used together with
// SetSocketReusePort for the shared UDP discovery socket, where multiple servers on the same host
// deliberately bind the same port.
//
// Not for the TCP listen socket: on Windows SO_REUSEADDR lets two live sockets bind the same active
// port at once, which would defeat listen-port auto-increment. Use SetListenSocketExclusive there.
static void SetSocketReuseAddr(SocketHandle s) {
  SetSockOptInt(s, SOL_SOCKET, SO_REUSEADDR, 1);
}

// Allow multiple sockets to bind the same port and each receive a copy of inbound broadcast
// datagrams. The shared UDP discovery socket needs this: on Linux/Apple a second wildcard UDP bind
// to the same port fails with SO_REUSEADDR alone, so without SO_REUSEPORT only the first server on
// a host would be discoverable. No-op where SO_REUSEPORT is unavailable (e.g. Windows, where
// SO_REUSEADDR already permits the shared bind). Note: unicast datagrams to a SO_REUSEPORT group
// are load-balanced to a single socket, so a same-host probe must be a broadcast (see the loopback
// directed broadcast in ServerList::Refresh) to reach every local server rather than just one.
static void SetSocketReusePort([[maybe_unused]] SocketHandle s) {
#if !MOCHI_PLATFORM_WINDOWS && defined(SO_REUSEPORT)
  SetSockOptInt(s, SOL_SOCKET, SO_REUSEPORT, 1);
#endif
}

// Reserve a TCP listen port exclusively: bind() fails if another socket already holds the port, so
// the server auto-increments to the next free one. On POSIX, SO_REUSEADDR gives exactly this (it
// only rebinds a TIME_WAIT port, never an active one); on Windows that flag instead allows sharing
// an active port, so SO_EXCLUSIVEADDRUSE is needed for the same semantics. Closed listen sockets do
// not enter TIME_WAIT, so an immediate restart on the same port still works.
static void SetListenSocketExclusive(SocketHandle s) {
#if MOCHI_PLATFORM_WINDOWS
  SetSockOptInt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
  SetSockOptInt(s, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
}

// Allow sendto() with INADDR_BROADCAST (required for UDP discovery).
static void SetSocketBroadcast(SocketHandle s) {
  SetSockOptInt(s, SOL_SOCKET, SO_BROADCAST, 1);
}

// Half-close both directions to wake any thread blocked in recv()/send() on this socket. Does not
// release the file descriptor (a separate CloseSocket call does that).
static void ShutdownSocket(SocketHandle s) {
#if MOCHI_PLATFORM_WINDOWS
  shutdown(s, SD_BOTH);
#else
  shutdown(s, SHUT_RDWR);
#endif
}

// Block until `s` is readable or `wakeSocket` is signaled. Returns true if `s` is readable.
static bool WaitForSocket(SocketHandle s, SocketHandle wakeSocket) {
  fd_set readSet;
  FD_ZERO(&readSet);
  FD_SET(s, &readSet);
  FD_SET(wakeSocket, &readSet);

#if MOCHI_PLATFORM_WINDOWS
  select(0, &readSet, nullptr, nullptr, nullptr);
#else
  int const maxFd = (s > wakeSocket ? s : wakeSocket) + 1;
  select(maxFd, &readSet, nullptr, nullptr, nullptr);
#endif

  return FD_ISSET(s, &readSet) != 0;
}

static int LastSocketError() {
#if MOCHI_PLATFORM_WINDOWS
  return WSAGetLastError();
#else
  return errno;
#endif
}

static bool IsConnectInProgress(int error) {
#if MOCHI_PLATFORM_WINDOWS
  return error == WSAEINPROGRESS || error == WSAEWOULDBLOCK || error == WSAEALREADY;
#else
  return error == EINPROGRESS || error == EALREADY;
#endif
}

static int GetSocketError(SocketHandle s) {
  int error = 0;
#if MOCHI_PLATFORM_WINDOWS
  int optLen = sizeof(error);
  getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &optLen);
#else
  socklen_t optLen = sizeof(error);
  getsockopt(s, SOL_SOCKET, SO_ERROR, &error, &optLen);
#endif
  return error;
}

// Wait for a non-blocking connect() to finish or for Stop() to signal `wakeSocket`.
static bool WaitForConnect(SocketHandle s, SocketHandle wakeSocket) {
  fd_set writeSet;
  FD_ZERO(&writeSet);
  FD_SET(s, &writeSet);

  fd_set readSet;
  FD_ZERO(&readSet);
  FD_SET(wakeSocket, &readSet);

#if MOCHI_PLATFORM_WINDOWS
  // On Windows a failed connect() is reported only via the exception set, never as writable. Watch
  // exceptfds too, otherwise select() blocks forever on a refused/unreachable connect and the retry
  // path is never reached. A failed socket lands in exceptSet (not writeSet), so the writable check
  // below returns false and the caller retries.
  fd_set exceptSet;
  FD_ZERO(&exceptSet);
  FD_SET(s, &exceptSet);
  int const ready = select(0, &readSet, &writeSet, &exceptSet, nullptr);
#else
  int const maxFd = (s > wakeSocket ? s : wakeSocket) + 1;
  int const ready = select(maxFd, &readSet, &writeSet, nullptr, nullptr);
#endif
  return ready > 0 && FD_ISSET(s, &writeSet) != 0 && GetSocketError(s) == 0;
}

// Block until `s` is writable or `wakeSocket` is signaled. Returns true if `s` is writable.
static bool WaitForWritableSocket(SocketHandle s, SocketHandle wakeSocket) {
  fd_set writeSet;
  FD_ZERO(&writeSet);
  FD_SET(s, &writeSet);

  fd_set readSet;
  FD_ZERO(&readSet);
  FD_SET(wakeSocket, &readSet);

#if MOCHI_PLATFORM_WINDOWS
  int const ready = select(0, &readSet, &writeSet, nullptr, nullptr);
#else
  int const maxFd = (s > wakeSocket ? s : wakeSocket) + 1;
  int const ready = select(maxFd, &readSet, &writeSet, nullptr, nullptr);
#endif
  return ready > 0 && FD_ISSET(s, &writeSet) != 0;
}

// Sleep for up to `duration`, returning early if `wakeSocket` becomes readable (signaled). Lets a
// teardown that signals the wake pipe cut a connect-retry backoff short instead of waiting it out.
static void SleepInterruptible(SocketHandle wakeSocket, std::chrono::milliseconds duration) {
  fd_set readSet;
  FD_ZERO(&readSet);
  FD_SET(wakeSocket, &readSet);

  timeval tv{};
  tv.tv_sec = static_cast<decltype(tv.tv_sec)>(duration.count() / 1000);
  tv.tv_usec = static_cast<decltype(tv.tv_usec)>((duration.count() % 1000) * 1000);

#if MOCHI_PLATFORM_WINDOWS
  select(0, &readSet, nullptr, nullptr, &tv);
#else
  select(static_cast<int>(wakeSocket) + 1, &readSet, nullptr, nullptr, &tv);
#endif
}

// Create a connected pair of sockets for cross-thread wakeup signaling. Returns false (leaving both
// ends kInvalidSocket) if the pair could not be created; the owning WakePipe is then left invalid
// so callers can fail cleanly instead of running a loop whose select() could never be woken for
// teardown.
static bool CreateWakePipe(SocketHandle& readEnd, SocketHandle& writeEnd) {
  readEnd = kInvalidSocket;
  writeEnd = kInvalidSocket;
#if MOCHI_PLATFORM_WINDOWS
  // Windows lacks socketpair(); use a loopback TCP connection. Every step is checked and on failure
  // we close what we opened and bail. Because accept() is reached only after connect() has
  // succeeded, the connection is already queued, so the (blocking) accept() returns immediately and
  // can never deadlock waiting for a peer that will never arrive.
  SocketHandle const listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listener == kInvalidSocket) {
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(listener, 1) != 0) {
    CloseSocket(listener);
    return false;
  }

  auto addrLen = static_cast<int>(sizeof(addr));
  if (getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &addrLen) != 0) {
    CloseSocket(listener);
    return false;
  }

  SocketHandle const writer = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (writer == kInvalidSocket) {
    CloseSocket(listener);
    return false;
  }
  if (connect(writer, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    CloseSocket(writer);
    CloseSocket(listener);
    return false;
  }

  SocketHandle const reader = accept(listener, nullptr, nullptr);
  CloseSocket(listener);
  if (reader == kInvalidSocket) {
    CloseSocket(writer);
    return false;
  }
  readEnd = reader;
  writeEnd = writer;
#else
  // Initialize: a failed socketpair() leaves fds unspecified, and closing/sending on an
  // uninitialized descriptor could clobber an unrelated fd.
  int fds[2] = {kInvalidSocket, kInvalidSocket};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    return false;
  }
  readEnd = fds[0];
  writeEnd = fds[1];
#endif

  // We send on the wake pipe (a stream socket), so suppress SIGPIPE on it. On Apple this relies on
  // SO_NOSIGPIPE since there is no MSG_NOSIGNAL; elsewhere SetNoSigPipe is a no-op and the send
  // uses kSendFlags instead.
  SetNoSigPipe(readEnd);
  SetNoSigPipe(writeEnd);
  return true;
}

// Send exactly `size` bytes, looping over partial sends and retrying on EINTR.
static bool SendAll(SocketHandle s, void const* data, size_t size) {
  auto const* ptr = static_cast<char const*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    auto const sent = send(s, ptr, static_cast<int>(remaining), kSendFlags);
    if (sent <= 0) {
#if !MOCHI_PLATFORM_WINDOWS
      if (sent < 0 && errno == EINTR) {
        continue;
      }
#endif
      return false;
    }
    ptr += sent;
    remaining -= static_cast<size_t>(sent);
  }
  return true;
}

// Send exactly `size` bytes without allowing teardown to park forever behind a full send window.
static bool SendAllInterruptible(
    SocketHandle s,
    void const* data,
    size_t size,
    SocketHandle wakeSocket,
    std::atomic<bool> const& running) {
  auto const* ptr = static_cast<char const*>(data);
  size_t remaining = size;
  while (remaining > 0 && running.load()) {
    auto const sent = send(s, ptr, static_cast<int>(remaining), kNonBlockingSendFlags);
    if (sent > 0) {
      ptr += sent;
      remaining -= static_cast<size_t>(sent);
      continue;
    }
#if MOCHI_PLATFORM_WINDOWS
    int const error = LastSocketError();
    if (sent < 0 && (error == WSAEWOULDBLOCK || error == WSAEINTR)) {
      if (!WaitForWritableSocket(s, wakeSocket)) {
        return false;
      }
      continue;
    }
#else
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      if (!WaitForWritableSocket(s, wakeSocket)) {
        return false;
      }
      continue;
    }
#endif
    return false;
  }
  return remaining == 0;
}

// Receive exactly `size` bytes, looping over partial reads and retrying on EINTR. A return of 0
// from recv() means the peer performed an orderly shutdown, which we treat as a disconnect.
static bool RecvAll(SocketHandle s, void* data, size_t size) {
  auto* ptr = static_cast<char*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    auto const received = recv(s, ptr, static_cast<int>(remaining), 0);
    if (received <= 0) {
#if !MOCHI_PLATFORM_WINDOWS
      if (received < 0 && errno == EINTR) {
        continue;
      }
#endif
      return false;
    }
    ptr += received;
    remaining -= static_cast<size_t>(received);
  }
  return true;
}

// Maximum size [bytes] of a single framed TCP message. Adjustable. Guards against a corrupt or
// hostile length prefix triggering a huge allocation; an over-cap frame tears down the connection.
// Only applies to the real-socket path; the in-proc queue path carries no length prefix and is
// uncapped.
static constexpr uint32_t kMaxMessageSize = 64 * 1024 * 1024; // 64 MB

// Send a length-prefixed message: [4-byte network-order size][payload]. Refuses (and reports the
// connection as failed) if `size` exceeds kMaxMessageSize.
static bool SendFramed(SocketHandle s, void const* data, size_t size) {
  if (size > kMaxMessageSize) {
    MOCHI_LOG_ERROR_ONCE(
        "mochi::net: refusing to send oversized message (%zu bytes > %u byte cap)",
        size,
        kMaxMessageSize);
    return false;
  }
  uint32_t const len = htonl(static_cast<uint32_t>(size));
  if (!SendAll(s, &len, sizeof(len))) {
    return false;
  }
  return SendAll(s, data, size);
}

static bool SendFramedInterruptible(
    SocketHandle s,
    void const* data,
    size_t size,
    SocketHandle wakeSocket,
    std::atomic<bool> const& running) {
  if (size > kMaxMessageSize) {
    MOCHI_LOG_ERROR_ONCE(
        "mochi::net: refusing to send oversized message (%zu bytes > %u byte cap)",
        size,
        kMaxMessageSize);
    return false;
  }
  uint32_t const len = htonl(static_cast<uint32_t>(size));
  if (!SendAllInterruptible(s, &len, sizeof(len), wakeSocket, running)) {
    return false;
  }
  return SendAllInterruptible(s, data, size, wakeSocket, running);
}

// Receive a length-prefixed message: [4-byte network-order size][payload]. A decoded length over
// kMaxMessageSize tears down the connection (returns false) rather than allocating it.
static bool RecvFramed(SocketHandle s, DynamicArray<uint8_t>& out) {
  uint32_t netLen = 0;
  if (!RecvAll(s, &netLen, sizeof(netLen))) {
    return false;
  }
  uint32_t const len = ntohl(netLen);
  if (len > kMaxMessageSize) {
    MOCHI_LOG_ERROR_ONCE(
        "mochi::net: received oversized message length (%u bytes > %u byte cap); dropping "
        "connection",
        len,
        kMaxMessageSize);
    return false;
  }
  out.resize_noinit(len);
  if (len == 0) {
    return true;
  }
  return RecvAll(s, out.data(), len);
}

namespace {

// -------------------------------------------------------------------------------------------------
// RAII Socket Ownership
// -------------------------------------------------------------------------------------------------

// Move-only owner of exactly one socket fd. Closes the fd on destruction, so a connection's
// lifetime follows the lifetime of the Socket that holds it — there is exactly one Socket per fd
// and the close happens exactly once.
class Socket {
  MOCHI_DECLARE_NO_COPY(Socket);

 public:
  Socket() = default;
  explicit Socket(SocketHandle handle) : _handle(handle) {}

  Socket(Socket&& other) noexcept : _handle(other._handle) {
    other._handle = kInvalidSocket;
  }

  Socket& operator=(Socket&& other) noexcept {
    if (this != &other) {
      Close();
      _handle = other._handle;
      other._handle = kInvalidSocket;
    }
    return *this;
  }

  ~Socket() {
    Close();
  }

  [[nodiscard]] SocketHandle Get() const {
    return _handle;
  }

  [[nodiscard]] bool IsValid() const {
    return _handle != kInvalidSocket;
  }

  // Half-close to wake a thread blocked in recv()/select() on this fd, without releasing it. The fd
  // stays valid (so a concurrent select()/recv() never reads a closed/reused fd) until Close().
  void Shutdown() const {
    if (_handle != kInvalidSocket) {
      ShutdownSocket(_handle);
    }
  }

  void Close() {
    if (_handle != kInvalidSocket) {
      CloseSocket(_handle);
      _handle = kInvalidSocket;
    }
  }

 private:
  SocketHandle _handle{kInvalidSocket};
};

// A connected client: its owned Socket plus a per-client mutex that serializes framed sends on
// this one fd. Held via shared_ptr so a sender can copy the pointer under the server _state lock,
// release _state, and then perform the (potentially blocking) send without keeping the fd alive
// concern or the shared lock — the fd is not closed until the last shared_ptr drops.
struct ClientConnection {
  explicit ClientConnection(Socket s) : socket(std::move(s)) {}

  Socket socket; // RAII owner of the fd.
  std::mutex sendMutex; // Serializes SendFramed on this socket only.
};

// Move-only owner of a connected socket pair used to wake a blocked select()/WaitForSocket() from
// another thread. Signal() writes a byte to the read end, which the select() observes as readable.
class WakePipe {
  MOCHI_DECLARE_NO_COPY(WakePipe);

 public:
  WakePipe() {
    SocketHandle readEnd = kInvalidSocket;
    SocketHandle writeEnd = kInvalidSocket;
    if (CreateWakePipe(readEnd, writeEnd)) {
      _read = Socket(readEnd);
      _write = Socket(writeEnd);
    }
  }

  WakePipe(WakePipe&&) noexcept = default;
  WakePipe& operator=(WakePipe&&) noexcept = default;
  ~WakePipe() = default;

  // True only if both ends were created. When false the pipe cannot wake a blocked select(), so an
  // owner must not run any loop that relies on it for teardown.
  [[nodiscard]] bool IsValid() const {
    return _read.IsValid() && _write.IsValid();
  }

  [[nodiscard]] SocketHandle ReadEnd() const {
    return _read.Get();
  }

  // Signal the pipe so a select()/WaitForSocket() watching ReadEnd() wakes up.
  void Signal() const {
    char const byte = 0;
    send(_write.Get(), &byte, 1, kSendFlags);
  }

 private:
  Socket _read;
  Socket _write;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------------------------------

static constexpr auto kConnectRetryDelay = std::chrono::milliseconds(100);
static constexpr auto kWindowsSendTimeout = std::chrono::seconds(30);

// UDP discovery. The magic tag is sent on the wire in network byte order (htonl) so that discovery
// is endian-safe between hosts of different endianness.
static constexpr uint32_t kDiscoveryMagic = 0x4D4F4348; // 'M','O','C','H'
static constexpr size_t kDiscoveryBufferSize = 512;

// Directed broadcast for the loopback subnet (127.255.255.255, host order). Same-host servers share
// the discovery port via SO_REUSEADDR/SO_REUSEPORT; the kernel load-balances a loopback *unicast*
// probe to a single member of that group, but delivers a *broadcast* datagram to every member. So
// the same-host probe targets this address to reach all local servers, not just one.
//
// Note: this loopback directed broadcast is delivered on Linux but NOT on macOS/BSD, whose loopback
// interface is not broadcast-capable — hence the multicast group below, which is the portable way
// to fan a same-host probe out to every member of the SO_REUSEPORT group.
static constexpr uint32_t kLoopbackBroadcast = 0x7FFFFFFF;

// Administratively-scoped multicast group (239.0.0.0/8) used for same-host discovery over the
// loopback interface. Servers join this group on loopback (JoinDiscoveryMulticastGroup) and the
// client sends a probe to it over loopback (SetMulticastLoopbackEgress). Multicast is delivered to
// every member of the SO_REUSEPORT group on all platforms — including macOS, whose loopback
// interface is multicast-capable but not broadcast-capable, so kLoopbackBroadcast above never fans
// out there. Value is host order; convert with htonl() at the point of use.
static constexpr uint32_t kDiscoveryMulticastGroup = 0xEFFF4342; // 239.255.67.66

// Connection-time handshake. Immediately after the TCP connection is established (and before any
// framed messages), the client sends a fixed header, all fields in network byte order:
//   [magic(u32)][protocolVersion(u32)][userVersionHi(u32)][userVersionLo(u32)]
// The 64-bit user-defined version is split into two htonl halves for endian safety, consistent
// with the rest of the wire code. The server validates the header before reporting the client as
// Connected; a mismatch (foreign peer, incompatible socket protocol, or mismatched user-defined
// version) is dropped without firing a connect event. Bump kNetProtocolVersion manually if the
// socket-level wire conventions ever change; the user-defined version is opaque to this layer.
static constexpr uint32_t kHandshakeMagic = 0x4D434E54; // 'M','C','N','T' (Mochi Connect)
static constexpr uint32_t kNetProtocolVersion = 1;

// How long a freshly accepted peer has to deliver the connection handshake before its recv thread
// drops it. Bounds a peer that completes the TCP connection but never sends the handshake, so it
// cannot hold a client slot forever.
static constexpr auto kHandshakeTimeout = std::chrono::milliseconds(5000);

// Split a 64-bit value into network-order high and low 32-bit halves.
static void EncodeU64(uint64_t value, uint32_t& netHi, uint32_t& netLo) {
  netHi = htonl(static_cast<uint32_t>(value >> 32));
  netLo = htonl(static_cast<uint32_t>(value & 0xFFFFFFFFu));
}

// Reassemble a 64-bit value from network-order high and low 32-bit halves.
static uint64_t DecodeU64(uint32_t netHi, uint32_t netLo) {
  return (static_cast<uint64_t>(ntohl(netHi)) << 32) | static_cast<uint64_t>(ntohl(netLo));
}

static void CheckNotCurrentThread(std::thread const& thread, [[maybe_unused]] char const* message) {
  if (std::this_thread::get_id() == thread.get_id()) {
    MOCHI_ASSERT(false, "%s", message);
    std::terminate();
  }
}

// Send the connection handshake. Returns false if the socket write failed.
static bool SendHandshake(SocketHandle s, uint64_t userVersion) {
  uint32_t hi = 0;
  uint32_t lo = 0;
  EncodeU64(userVersion, hi, lo);
  uint32_t const header[4] = {htonl(kHandshakeMagic), htonl(kNetProtocolVersion), hi, lo};
  return SendAll(s, header, sizeof(header));
}

// Receive and validate the connection handshake. Returns true only if the peer sent the expected
// magic, a matching protocol version, and the expected user-defined version.
static bool RecvAndValidateHandshake(SocketHandle s, uint64_t expectedUserVersion) {
  uint32_t header[4] = {0, 0, 0, 0};
  if (!RecvAll(s, header, sizeof(header))) {
    return false;
  }
  return ntohl(header[0]) == kHandshakeMagic && ntohl(header[1]) == kNetProtocolVersion &&
      DecodeU64(header[2], header[3]) == expectedUserVersion;
}

// Join the same-host discovery multicast group on the loopback interface so a socket bound to the
// shared discovery port also receives multicast probes sent to that group on this host.
// Best-effort: on failure it silently does nothing (e.g. a loopback interface without multicast
// support, as on Linux, where the loopback directed broadcast covers same-host discovery instead).
// Used together with SO_REUSEPORT so every same-host server receives its own copy of each probe.
static void JoinDiscoveryMulticastGroup(SocketHandle s) {
  ip_mreq mreq{};
  mreq.imr_multiaddr.s_addr = htonl(kDiscoveryMulticastGroup);
  mreq.imr_interface.s_addr = htonl(INADDR_LOOPBACK);
#if MOCHI_PLATFORM_WINDOWS
  setsockopt(s, IPPROTO_IP, IP_ADD_MEMBERSHIP, reinterpret_cast<char const*>(&mreq), sizeof(mreq));
#else
  setsockopt(s, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
#endif
}

// Route a socket's outgoing multicast over the loopback interface and ensure it is looped back to
// local group members, so a same-host discovery probe reaches every server that joined the group on
// loopback. Best-effort. Only affects this socket's multicast sends; its broadcast and unicast
// sends are unaffected (they continue to route normally). IP_MULTICAST_LOOP takes a u_char on POSIX
// but a DWORD on Windows, so the two are set separately.
static void SetMulticastLoopbackEgress(SocketHandle s) {
  in_addr iface{};
  iface.s_addr = htonl(INADDR_LOOPBACK);
#if MOCHI_PLATFORM_WINDOWS
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_IF, reinterpret_cast<char const*>(&iface), sizeof(iface));
  DWORD const loop = 1;
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_LOOP, reinterpret_cast<char const*>(&loop), sizeof(loop));
#else
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_IF, &iface, sizeof(iface));
  unsigned char const loop = 1;
  setsockopt(s, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));
#endif
}

namespace {

// -------------------------------------------------------------------------------------------------
// Thread-Safe Payload Queue
// -------------------------------------------------------------------------------------------------

// FIFO queue of raw byte payloads. Used for the client's outbound send queue, so a Send issued
// while the connection is still pending is buffered until the send thread can flush it.
class PayloadQueue {
 public:
  void Push(DynamicArray<uint8_t> payload) {
    {
      std::lock_guard lock(_mutex);
      _queue.push_back(std::move(payload));
    }
    _cv.notify_one();
  }

  bool Pop(DynamicArray<uint8_t>& out) {
    std::lock_guard lock(_mutex);
    if (_queue.empty()) {
      return false;
    }
    out = std::move(_queue.front());
    _queue.pop_front();
    return true;
  }

  // Blocks until a payload is available or the queue is shut down. Returns true if the queue is
  // non-empty (caller should Pop), false if it was woken by Shutdown with nothing left to send.
  //
  // Notify-driven with no timeout: the only wakeups are Push (new payload) and Shutdown (teardown).
  // Every transition that must stop the consumer therefore has to call Shutdown(), or the consumer
  // parks here forever. The send thread upholds this on both of its exit paths.
  bool Wait() {
    std::unique_lock lock(_mutex);
    _cv.wait(lock, [this] { return !_queue.empty() || _shutdown; });
    return !_queue.empty();
  }

  void Shutdown() {
    {
      std::lock_guard lock(_mutex);
      _shutdown = true;
    }
    _cv.notify_all();
  }

  void Clear() {
    std::lock_guard lock(_mutex);
    _queue.clear();
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<DynamicArray<uint8_t>> _queue;
  bool _shutdown{false};
};

// -------------------------------------------------------------------------------------------------
// Callback Holders
// -------------------------------------------------------------------------------------------------

// Each holder wraps a user callback behind its own lock and is owned via shared_ptr by the public
// ClientSocket/ServerSocket Impl. The transport receives a shared_ptr copy, so:
//   - Set*Callback works before connecting, after connecting, and across transport replacement
//     (reconnect): the holder outlives every transport that references it.
//   - Delivery copies the callback out under the lock, releases the lock, then invokes it, so the
//     callback may freely call back into the socket (e.g. to send a reply) without a lock-order
//     inversion, and a delivery in flight can never touch a destroyed callback.

// Client receive callback: invoked for each message received from the server.
struct ClientReceiveHolder {
  using CallbackFn = std::function<void(void const* data, size_t size)>;
  Guarded<CallbackFn> callback;

  void Deliver(void const* data, size_t size) {
    auto cb = callback.Load();
    if (cb) {
      cb(data, size);
    }
  }
};

// Client status: mirrors the latest status into an atomic so GetStatus() is a lock-free load, then
// fires the user status callback.
struct StatusHolder {
  using CallbackFn = std::function<void(SocketStatus)>;

  std::atomic<SocketStatus> status{SocketStatus::None};
  Guarded<CallbackFn> callback;

  void Deliver(SocketStatus s) {
    status.store(s);
    auto cb = callback.Load();
    if (cb) {
      cb(s);
    }
  }
};

// Server receive callback: invoked for each message received from any client, tagged with sender.
struct ServerReceiveHolder {
  using CallbackFn = std::function<void(ClientId, void const* data, size_t size)>;
  Guarded<CallbackFn> callback;

  void Deliver(ClientId id, void const* data, size_t size) {
    auto cb = callback.Load();
    if (cb) {
      cb(id, data, size);
    }
  }
};

// Server client-event callback: invoked on client connect/disconnect.
struct ClientEventHolder {
  using CallbackFn = std::function<void(ClientId, ClientEvent)>;
  Guarded<CallbackFn> callback;

  void Deliver(ClientId id, ClientEvent event) {
    auto cb = callback.Load();
    if (cb) {
      cb(id, event);
    }
  }
};

// -------------------------------------------------------------------------------------------------
// Transport Interfaces
// -------------------------------------------------------------------------------------------------

class InProcServerTransport;

// A client's transport (real TCP or in-process). The public ClientSocket holds one behind a
// Guarded<shared_ptr> and forwards Send through it; replacing/destroying it (Disconnect, reconnect)
// drops the shared_ptr, whose destructor performs all teardown.
class ClientTransport {
 public:
  virtual ~ClientTransport() = default;

  virtual bool Send(void const* data, size_t size) = 0;

  // The server address and port this transport connects to. Empty string and 0 by default, which is
  // the correct answer for in-process transports; TCP overrides it with the real endpoint.
  virtual void GetAddress(std::string& outAddress, uint16_t& outPort) const {
    outAddress.clear();
    outPort = 0;
  }
};

// A server's transport (real TCP or in-process). The public ServerSocket holds one behind a
// Guarded<shared_ptr> and forwards through it; Stop()/destruction drops the shared_ptr, whose
// destructor performs all teardown.
class ServerTransport {
 public:
  virtual ~ServerTransport() = default;

  [[nodiscard]] virtual DynamicArray<ClientId> GetClients() const = 0;
  virtual bool SendTo(ClientId client, void const* data, size_t size) = 0;
  virtual void Broadcast(void const* data, size_t size) = 0;
  [[nodiscard]] virtual uint16_t GetPort() const = 0;
  virtual void SetLabel(std::string_view label) = 0;

  // Returns this when the transport is the in-process server, nullptr otherwise. Lets
  // ClientSocket::ConnectInProc reach the in-proc server through the abstract handle without a raw
  // downcast.
  [[nodiscard]] virtual InProcServerTransport* AsInProc() {
    return nullptr;
  }
};

// -------------------------------------------------------------------------------------------------
// TCP Server Transport
// -------------------------------------------------------------------------------------------------

class TcpServerTransport final : public ServerTransport {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(TcpServerTransport);

 public:
  TcpServerTransport(
      std::shared_ptr<ServerReceiveHolder> receiveHolder,
      std::shared_ptr<ClientEventHolder> eventHolder,
      uint64_t version,
      uint16_t discoveryPort)
      : _receiveHolder(std::move(receiveHolder)),
        _eventHolder(std::move(eventHolder)),
        _version(version),
        _discoveryPort(discoveryPort) {}

  ~TcpServerTransport() override {
    Stop();
  }

  // Bind, listen, and start the accept thread. Returns false (and hosts nothing) if no port could
  // be bound; the caller then leaves the server un-hosted.
  bool Start(uint16_t preferredPort, int maxClients, DynamicString label) {
    // The accept thread blocks in select() and is woken for teardown only by the wake pipe; without
    // it Stop() would hang joining that thread. Refuse to host if the pipe is unavailable. Log on
    // the info channel to avoid failing CI tests where a server is started but no client connects.
    if (!_wakePipe.IsValid()) {
      MOCHI_LOG("mochi::net: wake pipe unavailable; not hosting");
      return false;
    }
    SocketHandle const listen = CreateListenSocket(preferredPort, maxClients);
    if (listen == kInvalidSocket) {
      return false;
    }
    if (!SetNonBlocking(listen)) {
      CloseSocket(listen);
      MOCHI_LOG("mochi::net: failed to set server socket non-blocking; not hosting");
      return false;
    }
    _listenSocket = Socket(listen);

    uint16_t const port = BoundPort(listen);
    _state.Mutate([&](State& s) {
      s.maxClients = maxClients;
      s.label = std::move(label);
      s.port = port;
    });

    // Best-effort UDP discovery responder, serviced by the accept thread. Multiple servers
    // can share the configured discovery port via SO_REUSEADDR + SO_REUSEPORT; if the
    // bind fails we simply skip discovery for this server.
    if (_discoveryPort != 0) {
      _discoverySocket = CreateDiscoverySocket(_discoveryPort);
    }

    _running.store(true);
    _acceptThread = std::thread([this] {
      marl::Thread::setName("Mochi Net Accept");
      RunAcceptLoop();
    });
    return true;
  }

  [[nodiscard]] uint16_t GetPort() const override {
    return _state.Read(&State::port);
  }

  void SetLabel(std::string_view label) override {
    _state.Mutate([&](State& s) { s.label = DynamicString{label}; });
  }

  [[nodiscard]] DynamicArray<ClientId> GetClients() const override {
    return _state.Read(&State::clientIds);
  }

  bool SendTo(ClientId client, void const* data, size_t size) override {
    // Snapshot the target connection under _state, then send outside the lock. Holding only the
    // per-client sendMutex during the (potentially blocking) send keeps framing atomic without
    // freezing the rest of the server or deadlocking Stop().
    std::shared_ptr<ClientConnection> conn;
    _state.Read([&](State const& s) {
      for (size_t i = 0; i < s.clientIds.size(); ++i) {
        if (s.clientIds[i] == client) {
          conn = s.clients[i];
          break;
        }
      }
    });
    if (!conn) {
      return false;
    }
    std::lock_guard<std::mutex> const lock(conn->sendMutex);
    return SendFramed(conn->socket.Get(), data, size);
  }

  void Broadcast(void const* data, size_t size) override {
    // Snapshot all connections under _state, then send outside the lock (sequential, blocking per
    // client). A slow client only slows this Broadcast, not the whole server, and Stop() can still
    // interrupt an in-flight send via shutdown().
    DynamicArray<std::shared_ptr<ClientConnection>> conns;
    _state.Read([&](State const& s) { conns = s.clients; });
    for (auto const& conn : conns) {
      std::lock_guard<std::mutex> const lock(conn->sendMutex);
      SendFramed(conn->socket.Get(), data, size);
    }
  }

 private:
  struct State {
    DynamicString label;
    int maxClients{0};
    uint16_t port{0};
    // Parallel arrays: clients[i] is the connection for clientIds[i]. Each ClientConnection owns
    // its fd and a per-client sendMutex. Senders copy the shared_ptr under this Guarded lock,
    // release it, then send under the connection's own sendMutex — so a blocking send never holds
    // _state, and the fd outlives the unlocked send because the sender holds its own shared_ptr
    // ref.
    DynamicArray<std::shared_ptr<ClientConnection>> clients;
    DynamicArray<ClientId> clientIds;
  };

  // Create and bind the TCP listen socket, auto-incrementing the port if the preferred one is in
  // use. Returns kInvalidSocket on failure.
  //
  // All failures here log on the info channel only (never warning/error) to avoid failing CI tests
  // when a server is started but no clients connect.
  static SocketHandle CreateListenSocket(uint16_t preferredPort, int maxClients) {
    SocketHandle const s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == kInvalidSocket) {
      MOCHI_LOG("mochi::net: failed to create server socket; not hosting");
      return kInvalidSocket;
    }
    SetListenSocketExclusive(s);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;

    uint16_t port = preferredPort;
    constexpr int kMaxPortAttempts = 16;
    for (int attempt = 0; attempt < kMaxPortAttempts; ++attempt) {
      addr.sin_port = htons(port);
      if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
        if (listen(s, maxClients) == 0) {
          return s;
        }
        // Bound but could not listen; this is not port contention, so do not retry.
        CloseSocket(s);
        MOCHI_LOG("mochi::net: failed to listen on server socket; not hosting");
        return kInvalidSocket;
      }
      ++port;
    }
    CloseSocket(s);
    MOCHI_LOG(
        "mochi::net: failed to bind server socket after %d attempts; not hosting",
        kMaxPortAttempts);
    return kInvalidSocket;
  }

  // Read back the actual port a bound socket is listening on.
  static uint16_t BoundPort(SocketHandle s) {
    sockaddr_in addr{};
    auto addrLen = static_cast<socklen_t>(sizeof(addr));
    if (getsockname(s, reinterpret_cast<sockaddr*>(&addr), &addrLen) != 0) {
      return 0;
    }
    return ntohs(addr.sin_port);
  }

  // Create the best-effort UDP discovery responder socket, or an empty Socket if it cannot bind.
  static Socket CreateDiscoverySocket(uint16_t discoveryPort) {
    SocketHandle const udp = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp == kInvalidSocket) {
      return Socket{};
    }
    SetSocketReuseAddr(udp);
    SetSocketReusePort(udp);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(discoveryPort);
    if (bind(udp, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      CloseSocket(udp);
      return Socket{};
    }
    // Also receive same-host probes via multicast: on macOS the loopback is not broadcast-capable,
    // so the client's loopback directed broadcast never arrives; joining this group on loopback is
    // the portable path that reaches every SO_REUSEPORT member. Best-effort (skipped where loopback
    // multicast is unavailable, e.g. Linux, which uses the loopback broadcast instead).
    JoinDiscoveryMulticastGroup(udp);
    return Socket(udp);
  }

  ClientId AllocateClientId() {
    return _nextClientId.fetch_add(1);
  }

  void RunAcceptLoop() {
    SocketHandle const listenFd = _listenSocket.Get();
    SocketHandle const wakeFd = _wakePipe.ReadEnd();
    SocketHandle const discoveryFd = _discoverySocket.Get();

    while (_running.load()) {
      fd_set readSet;
      FD_ZERO(&readSet);
      FD_SET(listenFd, &readSet);
      FD_SET(wakeFd, &readSet);
      if (discoveryFd != kInvalidSocket) {
        FD_SET(discoveryFd, &readSet);
      }

#if MOCHI_PLATFORM_WINDOWS
      select(0, &readSet, nullptr, nullptr, nullptr);
#else
      SocketHandle maxFd = listenFd;
      if (wakeFd > maxFd) {
        maxFd = wakeFd;
      }
      if (discoveryFd > maxFd) {
        maxFd = discoveryFd;
      }
      select(maxFd + 1, &readSet, nullptr, nullptr, nullptr);
#endif

      if (!_running.load()) {
        break;
      }

      // Accept and drain pending clients before servicing discovery, so a burst of best-effort UDP
      // discovery probes can never starve client accepts.
      if (FD_ISSET(listenFd, &readSet)) {
        AcceptPendingClients();
      }

      // Discovery is best-effort and lower priority: service at most one datagram per wakeup.
      if (discoveryFd != kInvalidSocket && FD_ISSET(discoveryFd, &readSet)) {
        ServiceDiscovery(discoveryFd);
      }
    }
  }

  // Accept and register every currently-pending client, then return. The listen socket is
  // non-blocking, so accept() returns kInvalidSocket once the backlog is drained. Draining here
  // (rather than one accept per wakeup) keeps a burst of UDP discovery probes from delaying client
  // accepts. Runs on the accept thread.
  void AcceptPendingClients() {
    SocketHandle const listenFd = _listenSocket.Get();
    while (_running.load()) {
      sockaddr_in addr{};
      auto addrLen = static_cast<socklen_t>(sizeof(addr));
      SocketHandle const rawClient = accept(listenFd, reinterpret_cast<sockaddr*>(&addr), &addrLen);
      if (rawClient == kInvalidSocket) {
        break;
      }

      // Accepted sockets must be blocking: the recv loop relies on blocking reads woken by
      // shutdown() at teardown.
      if (!SetBlocking(rawClient)) {
        CloseSocket(rawClient);
        continue;
      }
      SetNoSigPipe(rawClient);
      SetTcpNoDelay(rawClient);
      Socket clientSocket(rawClient);

      ClientId const id = AllocateClientId();
      bool accepted = false;
      _state.Mutate([&](State& s) {
        if (isize(s.clientIds) >= s.maxClients) {
          return;
        }
        s.clients.push_back(std::make_shared<ClientConnection>(std::move(clientSocket)));
        s.clientIds.push_back(id);
        accepted = true;
      });
      if (!accepted) {
        // clientSocket still owns the fd (the move never ran); it closes on scope exit.
        continue;
      }

      // Reap finished recv threads before spawning the next one (see ReapFinishedClientThreads).
      ReapFinishedClientThreads();

      auto finished = std::make_shared<std::atomic<bool>>(false);
      _clientThreads.push_back(
          ClientThread{
              std::thread([this, rawClient, id, finished] {
                marl::Thread::setName("Mochi Net Recv %u", id);
                RunClientRecvLoop(rawClient, id);
                finished->store(true); // Signals the accept thread it may join this handle.
              }),
              std::move(finished),
          });
    }
  }

  // Reply to a discovery probe with [magic(u32, network order)][compact-JSON ServerInfo]. The JSON
  // carries label, port, live client count, capacity, and the user-defined version; address is left
  // empty and filled in by the receiver from the UDP source IP.
  void ServiceDiscovery(SocketHandle discoveryFd) {
    char buf[kDiscoveryBufferSize];
    sockaddr_in fromAddr{};
    auto fromLen = static_cast<socklen_t>(sizeof(fromAddr));

    auto const n = recvfrom(
        discoveryFd, buf, sizeof(buf), 0, reinterpret_cast<sockaddr*>(&fromAddr), &fromLen);
    if (n <= 0 || static_cast<size_t>(n) < sizeof(uint32_t)) {
      return;
    }

    uint32_t netMagic = 0;
    std::memcpy(&netMagic, buf, sizeof(netMagic));
    if (ntohl(netMagic) != kDiscoveryMagic) {
      return;
    }

    ServerInfo info;
    info.version = _version;
    _state.Read([&](State const& s) {
      info.label = s.label;
      info.port = s.port;
      info.numClients = static_cast<uint16_t>(s.clientIds.size());
      info.maxClients = static_cast<uint16_t>(s.maxClients);
    });

    std::string const json = SReflect::ToJsonString(info, /*pretty*/ false);

    uint32_t const replyMagic = htonl(kDiscoveryMagic);
    // Best-effort: if the framed reply would not fit, skip it (consistent with discovery being a
    // best-effort mechanism). The receiver caps reads at kDiscoveryBufferSize too.
    if (sizeof(replyMagic) + json.size() > kDiscoveryBufferSize) {
      return;
    }

    char reply[kDiscoveryBufferSize];
    std::memcpy(reply, &replyMagic, sizeof(replyMagic));
    std::memcpy(reply + sizeof(replyMagic), json.data(), json.size());
    size_t const offset = sizeof(replyMagic) + json.size();

    sendto(
        discoveryFd,
        reply,
        static_cast<int>(offset),
        0,
        reinterpret_cast<sockaddr*>(&fromAddr),
        fromLen);
  }

  void RunClientRecvLoop(SocketHandle s, ClientId id) {
    // Validate the connection handshake before treating the peer as connected. A foreign or
    // version-incompatible client is dropped here without ever firing ClientEvent::Connected.
    //
    // Bound the handshake read with a receive timeout so a peer that connects but never sends the
    // handshake cannot park this thread forever and hold a client slot. The timeout is cleared
    // after a successful handshake: the steady-state recv loop below legitimately blocks
    // indefinitely between messages.
    SetRecvTimeout(s, kHandshakeTimeout);
    if (!RecvAndValidateHandshake(s, _version)) {
      RemoveClient(id, /*fireDisconnected*/ false);
      return;
    }
    SetRecvTimeout(s, std::chrono::milliseconds(0));

    _eventHolder->Deliver(id, ClientEvent::Connected);

    while (_running.load()) {
      DynamicArray<uint8_t> data;
      if (!RecvFramed(s, data)) {
        break;
      }
      _receiveHolder->Deliver(id, data.data(), data.size());
    }

    RemoveClient(id, /*fireDisconnected*/ true);
  }

  // Remove a client's entry and optionally fire the disconnect event. Runs on the client's own recv
  // thread. Erasing the shared_ptr drops one ref; the fd closes when the last ref drops, which is
  // here unless a concurrent send still holds a ref, in which case the close waits for that send.
  // This thread's recv() has already returned, so no recv() is in flight when the fd closes.
  void RemoveClient(ClientId id, bool fireDisconnected) {
    bool found = false;
    _state.Mutate([&](State& s) {
      for (size_t i = 0; i < s.clientIds.size(); ++i) {
        if (s.clientIds[i] == id) {
          s.clients.erase(s.clients.begin() + static_cast<ptrdiff_t>(i));
          s.clientIds.erase(s.clientIds.begin() + static_cast<ptrdiff_t>(i));
          found = true;
          break;
        }
      }
    });
    if (fireDisconnected && found) {
      _eventHolder->Deliver(id, ClientEvent::Disconnected);
    }
  }

  // Join and erase recv threads that have signaled completion via their `finished` flag. Run on the
  // accept thread before each new client, so handles cannot accumulate across connect/disconnect
  // churn. Only finished threads are joined, so join() never blocks the accept loop.
  void ReapFinishedClientThreads() {
    for (size_t i = 0; i < _clientThreads.size();) {
      if (_clientThreads[i].finished->load()) {
        _clientThreads[i].thread.join();
        _clientThreads.erase(_clientThreads.begin() + static_cast<ptrdiff_t>(i));
      } else {
        ++i;
      }
    }
  }

  // Teardown order: stop accepting, then half-close sockets to wake blocked threads, then join,
  // then close fds via RAII. shutdown() (not close()) wakes recv()/select() while keeping each fd
  // valid until its owning thread has exited.
  void Stop() {
    _running.store(false);

    // Stop the accept thread first so it cannot add new clients or spawn new recv threads while we
    // tear down the existing ones.
    _wakePipe.Signal();
    if (_acceptThread.joinable()) {
      CheckNotCurrentThread(
          _acceptThread, "ServerSocket::Stop must not be called from the TCP accept thread.");
      _acceptThread.join();
    }

    // Wake every client recv thread by half-closing its socket, then join. Each recv thread removes
    // and closes its own socket on exit, so we only shutdown() here (we must not close). shutdown()
    // is non-blocking, so doing it under _state is safe — no blocking send holds _state anymore.
    _state.Read([](State const& s) {
      for (auto const& conn : s.clients) {
        conn->socket.Shutdown();
      }
    });
    for (auto& ct : _clientThreads) {
      if (ct.thread.joinable()) {
        CheckNotCurrentThread(
            ct.thread, "ServerSocket::Stop must not be called from a TCP server callback.");
        ct.thread.join();
      }
    }
    _clientThreads.clear();

    // After joining, every recv thread has removed and closed its own socket; clear any leftover
    // bookkeeping (normally empty) and reset the port: per GetPort()'s contract a stopped server is
    // no longer listening.
    _state.Mutate([](State& s) {
      s.clients.clear();
      s.clientIds.clear();
      s.port = 0;
    });

    _discoverySocket.Close();
    _listenSocket.Close();
  }

  std::shared_ptr<ServerReceiveHolder> _receiveHolder;
  std::shared_ptr<ClientEventHolder> _eventHolder;
  uint64_t _version{0};
  uint16_t _discoveryPort{0};

  Guarded<State> _state;
  std::atomic<uint32_t> _nextClientId{1};

  // Fds owned by the transport and read by the accept thread; set before the thread starts and
  // closed after it joins, so the accept thread never sees a closed/reused fd.
  Socket _listenSocket;
  Socket _discoverySocket;
  WakePipe _wakePipe;
  std::atomic<bool> _running{false};
  std::thread _acceptThread;

  // One handle per accepted client, owned solely by the accept thread until it is joined in Stop(),
  // so no lock is needed. The recv thread sets `finished` as its last action; the accept thread
  // joins+erases finished handles (ReapFinishedClientThreads) and Stop() joins the rest.
  struct ClientThread {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> finished;
  };
  DynamicArray<ClientThread> _clientThreads;
};

// -------------------------------------------------------------------------------------------------
// In-Proc Callback Dispatch (reentrancy-safe)
// -------------------------------------------------------------------------------------------------

// In-proc delivery is synchronous on the calling thread: Send/SendTo/Broadcast normally invoke the
// peer's callback inline. That makes sending from inside a callback unsafe, because the peer's
// callback would run nested inside the current one (and a reply-to-a-reply could recurse without
// bound). To make sending from callbacks safe, all in-proc callbacks on a thread are funneled
// through one pump:
//
//   * The outermost in-proc callback on the thread runs immediately (PumpInProc); when it returns,
//     the pump flushes a FIFO queue of any deliveries requested while it (or a drained delivery)
//     was running.
//   * A Send/SendTo/Broadcast issued while a callback is running does not deliver inline: it copies
//     its payload and enqueues the delivery (DeferInProc). Nested sends keep enqueuing and are
//     drained by the same loop, so callbacks never nest and ping-pong cannot overflow the stack.
//
// The state is thread_local because in-proc delivery never crosses threads, so no locks are needed
// and the fast path (no callback running) is a single bool test with no copy or allocation. Only
// the in-proc transport uses this; TCP delivers on its own threads and needs none of it.

thread_local bool tls_inProcCallback = false;
thread_local std::deque<std::function<void()>> tls_inProcQueue;

// True while an in-proc callback is running on the calling thread. Disconnect()/Stop() assert on
// this: tearing a transport down from inside its own callback would destroy an object whose method
// is still on the stack.
[[nodiscard]] bool InInProcCallback() {
  return tls_inProcCallback;
}

// Run `body` (which invokes one or more in-proc user callbacks) as the outermost dispatch on this
// thread, then flush, in FIFO order, any deliveries queued while it ran. Precondition: no in-proc
// callback is already running on this thread.
template <typename Body>
void PumpInProc(Body&& body) {
  MOCHI_ASSERT(!tls_inProcCallback);
  tls_inProcCallback = true;
  // On any exit path (normal or exception), drop any not-yet-run deliveries and clear the flag so
  // the next dispatch on this thread starts clean.
  MOCHI_DEFER({
    tls_inProcQueue.clear();
    tls_inProcCallback = false;
  });
  body();
  while (!tls_inProcQueue.empty()) {
    std::function<void()> job = std::move(tls_inProcQueue.front());
    tls_inProcQueue.pop_front();
    job();
  }
}

// Copy `data`/`size` into an owned buffer and enqueue `deliver` (invoked with that buffer) to run
// when the current in-proc callback returns. Precondition: an in-proc callback is running on this
// thread (the caller checks InInProcCallback()).
template <typename Deliver>
void DeferInProc(void const* data, size_t size, Deliver deliver) {
  DynamicArray<uint8_t> payload;
  payload.resize_noinit(size);
  if (size > 0) {
    std::memcpy(payload.data(), data, size);
  }
  tls_inProcQueue.push_back([payload = std::move(payload), deliver = std::move(deliver)]() {
    deliver(payload.data(), payload.size());
  });
}

// -------------------------------------------------------------------------------------------------
// In-Proc Transports
// -------------------------------------------------------------------------------------------------

// Forward-declared so InProcServerTransport can hold weak_ptrs to its connected clients.
class InProcClientTransport;

class InProcServerTransport final : public ServerTransport,
                                    public std::enable_shared_from_this<InProcServerTransport> {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(InProcServerTransport);

 public:
  InProcServerTransport(
      std::shared_ptr<ServerReceiveHolder> receiveHolder,
      std::shared_ptr<ClientEventHolder> eventHolder,
      int maxClients,
      DynamicString label,
      uint64_t version)
      : _receiveHolder(std::move(receiveHolder)),
        _eventHolder(std::move(eventHolder)),
        _version(version) {
    _state.Mutate([&](State& s) {
      s.maxClients = maxClients;
      s.label = std::move(label);
    });
  }

  ~InProcServerTransport() override {
    // Snapshot connected clients under the lock, release it, then transition each to Lost. Because
    // the server lock is not held while notifying, the server lock and a client's callback lock are
    // never held simultaneously (no lock-order inversion with client->server calls).
    DynamicArray<std::weak_ptr<InProcClientTransport>> clients;
    DynamicArray<ClientId> clientIds;
    _state.Mutate([&](State& s) {
      clients = std::move(s.clients);
      clientIds = std::move(s.clientIds);
      s.clients.clear();
      s.clientIds.clear();
    });
    // Fire the server-side disconnect for every still-connected client, matching
    // TcpServerTransport::Stop(), then transition each client to Lost. Run through the pump so a
    // send from any of these callbacks is queued rather than reentering user code.
    PumpInProc([&] {
      for (ClientId const id : clientIds) {
        _eventHolder->Deliver(id, ClientEvent::Disconnected);
      }
      NotifyServerStopped(clients);
    });
  }

  [[nodiscard]] InProcServerTransport* AsInProc() override {
    return this;
  }

  [[nodiscard]] uint16_t GetPort() const override {
    return 0;
  }

  void SetLabel(std::string_view label) override {
    _state.Mutate([&](State& s) { s.label = DynamicString(label); });
  }

  [[nodiscard]] DynamicArray<ClientId> GetClients() const override {
    return _state.Read(&State::clientIds);
  }

  // Register a connecting client. Returns the assigned ClientId, or 0 if rejected (version mismatch
  // or at capacity). The Connected event is fired separately by FireConnected so the caller can
  // publish the client transport first (see ClientSocket::ConnectInProc). Handshake parity with the
  // real-socket path.
  ClientId RegisterClient(
      std::weak_ptr<InProcClientTransport> client,
      uint32_t protocolVersion,
      uint64_t userVersion) {
    if (protocolVersion != kNetProtocolVersion || userVersion != _version) {
      return 0;
    }
    ClientId id = 0;
    _state.Mutate([&](State& s) {
      if (isize(s.clientIds) >= s.maxClients) {
        return;
      }
      id = AllocateClientId();
      s.clients.push_back(std::move(client));
      s.clientIds.push_back(id);
    });
    return id;
  }

  // Fire the Connected event for a client just registered via RegisterClient. Called inside a pump
  // (see ClientSocket::ConnectInProc) after the client transport is published, so the event
  // callback -- and any reply it triggers -- sees a fully-connected client. No-op if the client is
  // already gone (e.g. a concurrent Stop removed it).
  void FireConnected(ClientId id) {
    if (HasClient(id)) {
      _eventHolder->Deliver(id, ClientEvent::Connected);
    }
  }

  // Detach a client (on the client's Disconnect). Fires the disconnect event if the client was
  // still registered.
  void RemoveClient(ClientId id) {
    bool found = false;
    _state.Mutate([&](State& s) {
      for (size_t i = 0; i < s.clientIds.size(); ++i) {
        if (s.clientIds[i] == id) {
          s.clients.erase(s.clients.begin() + static_cast<ptrdiff_t>(i));
          s.clientIds.erase(s.clientIds.begin() + static_cast<ptrdiff_t>(i));
          found = true;
          break;
        }
      }
    });
    if (found) {
      PumpInProc([&] { _eventHolder->Deliver(id, ClientEvent::Disconnected); });
    }
  }

  [[nodiscard]] bool HasClient(ClientId id) const {
    return _state.Read([&](State const& s) { return Contains(s.clientIds, id); });
  }

  // Resolve client `id` to a live transport, or null if it is gone. Looks up under the server lock,
  // then locks the weak_ptr (see State::clients).
  [[nodiscard]] std::shared_ptr<InProcClientTransport> LockClient(ClientId id) const {
    return _state.Read([&](State const& s) -> std::shared_ptr<InProcClientTransport> {
      for (size_t i = 0; i < s.clientIds.size(); ++i) {
        if (s.clientIds[i] == id) {
          return s.clients[i].lock();
        }
      }
      return {};
    });
  }

  bool SendTo(ClientId client, void const* data, size_t size) override;
  void Broadcast(void const* data, size_t size) override;

  // Deliver a client->server message to the server's receive callback (invoked with no server lock
  // held, so the callback may reply without a lock-order inversion). Always called from inside the
  // pump (a synchronous send or a drained deferred send), so it never reenters user code.
  void DeliverFromClient(ClientId id, void const* data, size_t size) {
    _receiveHolder->Deliver(id, data, size);
  }

  // Re-resolve client `id` and deliver `data` to its receive callback, if still connected. Defined
  // out-of-line (touches InProcClientTransport). Used by deferred SendTo/Broadcast jobs, which
  // resolve their target when the pump runs them.
  void DeliverToClientById(ClientId id, void const* data, size_t size);

 private:
  struct State {
    DynamicString label;
    int maxClients{0};
    // Parallel arrays: clients[i] is the (weak) transport for clientIds[i]. weak_ptr::lock() is the
    // single "is this client still alive?" check at delivery time.
    DynamicArray<std::weak_ptr<InProcClientTransport>> clients;
    DynamicArray<ClientId> clientIds;
  };

  ClientId AllocateClientId() {
    return _nextClientId.fetch_add(1);
  }

  // Transition each still-alive client to Lost. Defined out-of-line because it touches the
  // (still incomplete here) InProcClientTransport.
  static void NotifyServerStopped(
      DynamicArray<std::weak_ptr<InProcClientTransport>> const& clients);

  std::shared_ptr<ServerReceiveHolder> _receiveHolder;
  std::shared_ptr<ClientEventHolder> _eventHolder;
  uint64_t _version{0};
  Guarded<State> _state;
  std::atomic<uint32_t> _nextClientId{1};
};

class InProcClientTransport final : public ClientTransport,
                                    public std::enable_shared_from_this<InProcClientTransport> {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(InProcClientTransport);

 public:
  InProcClientTransport(
      std::shared_ptr<ClientReceiveHolder> receiveHolder,
      std::shared_ptr<StatusHolder> statusHolder,
      uint64_t version)
      : _receiveHolder(std::move(receiveHolder)),
        _statusHolder(std::move(statusHolder)),
        _version(version) {}

  ~InProcClientTransport() override {
    // If still connected (transport replaced/destroyed without an explicit Disconnect), leave the
    // server so it stops referencing us and fires its disconnect event.
    LeaveServer();
  }

  // Connect to the given in-proc server. Returns the assigned ClientId on success, or 0 if
  // rejected. Registers with the server but does not fire the Connected event; the caller
  // (ClientSocket::ConnectInProc) publishes this transport and then drives the connect callbacks.
  // The caller holds a shared_ptr to the server transport for the duration, so the server cannot be
  // destroyed mid-connect.
  ClientId ConnectTo(InProcServerTransport& server) {
    ClientId const id = server.RegisterClient(shared_from_this(), kNetProtocolVersion, _version);
    if (id == 0) {
      return 0;
    }
    _clientId = id;
    _server.Store(server.weak_from_this());
    return id;
  }

  bool Send(void const* data, size_t size) override {
    // weak_ptr::lock() is the single peer-liveness check: if the server is gone, the lock fails and
    // Send reports failure with no chance of touching freed memory.
    auto server = _server.Read([](auto& weak) { return weak.lock(); });
    if (!server || _statusHolder->status.load() != SocketStatus::Connected) {
      return false;
    }
    if (InInProcCallback()) {
      // A callback is on the stack: defer so we do not synchronously invoke the server's receive
      // callback from underneath it. The deferred job re-resolves the server when the pump runs it.
      DeferInProc(data, size, [self = weak_from_this()](void const* d, size_t n) {
        if (auto client = self.lock()) {
          client->DeliverToServer(d, n);
        }
      });
    } else {
      PumpInProc([&] { server->DeliverFromClient(_clientId, data, size); });
    }
    return true;
  }

  // Re-resolve the server and deliver `data` to its receive callback, if still connected. Used by
  // deferred Send jobs, which resolve the server when the pump runs them.
  void DeliverToServer(void const* data, size_t size) {
    if (_statusHolder->status.load() != SocketStatus::Connected) {
      return;
    }
    if (auto server = _server.Read([](auto& weak) { return weak.lock(); })) {
      server->DeliverFromClient(_clientId, data, size);
    }
  }

  // Deliver a server->client message to this client's receive callback. Always called from inside
  // the pump (a synchronous send or a drained deferred send), so it never reenters user code.
  void DeliverToClient(void const* data, size_t size) {
    _receiveHolder->Deliver(data, size);
  }

  // Called by the server (with no server lock held) when it stops. Detaches and goes Lost.
  void OnServerStopped() {
    _server.Store({});
    _statusHolder->Deliver(SocketStatus::Lost);
  }

  // Detach from the server (on Disconnect/destruction). Whoever wins the exchange against
  // OnServerStopped tells the server we are leaving; weak_ptr::lock() guards against a dead server.
  void LeaveServer() {
    auto old = _server.Exchange({});
    if (auto server = old.lock()) {
      server->RemoveClient(_clientId);
    }
  }

 private:
  std::shared_ptr<ClientReceiveHolder> _receiveHolder;
  std::shared_ptr<StatusHolder> _statusHolder;
  uint64_t _version{0};
  // Assigned once in ConnectTo (from RegisterClient) and read-only thereafter. Nothing reads
  // it until status reaches Connected, so the brief window before it is set is harmless.
  ClientId _clientId{0};
  // Guarded so Disconnect (LeaveServer) and server teardown (OnServerStopped) can race safely.
  Guarded<std::weak_ptr<InProcServerTransport>> _server;
};

void InProcServerTransport::DeliverToClientById(ClientId id, void const* data, size_t size) {
  if (auto target = LockClient(id)) {
    target->DeliverToClient(data, size);
  }
}

bool InProcServerTransport::SendTo(ClientId client, void const* data, size_t size) {
  auto target = LockClient(client);
  if (!target) {
    return false;
  }
  if (InInProcCallback()) {
    // Defer so we do not synchronously reenter user code from underneath the running callback; the
    // deferred job re-resolves the client when the pump runs it. See InProcClientTransport::Send.
    DeferInProc(data, size, [self = weak_from_this(), client](void const* d, size_t n) {
      if (auto server = self.lock()) {
        server->DeliverToClientById(client, d, n);
      }
    });
  } else {
    PumpInProc([&] { target->DeliverToClient(data, size); });
  }
  return true;
}

void InProcServerTransport::Broadcast(void const* data, size_t size) {
  if (InInProcCallback()) {
    // Queue one delivery per currently-registered client; each re-resolves its target when the pump
    // runs it, matching the reentrancy treatment of SendTo.
    DynamicArray<ClientId> ids = _state.Read(&State::clientIds);
    for (ClientId const id : ids) {
      DeferInProc(data, size, [self = weak_from_this(), id](void const* d, size_t n) {
        if (auto server = self.lock()) {
          server->DeliverToClientById(id, d, n);
        }
      });
    }
    return;
  }
  // Snapshot all live clients under the lock, release it, then deliver inline under the pump.
  DynamicArray<std::shared_ptr<InProcClientTransport>> targets;
  _state.Read([&](State const& s) {
    for (auto const& weak : s.clients) {
      if (auto client = weak.lock()) {
        targets.push_back(std::move(client));
      }
    }
  });
  PumpInProc([&] {
    for (auto& target : targets) {
      target->DeliverToClient(data, size);
    }
  });
}

void InProcServerTransport::NotifyServerStopped(
    DynamicArray<std::weak_ptr<InProcClientTransport>> const& clients) {
  for (auto const& weak : clients) {
    if (auto client = weak.lock()) {
      client->OnServerStopped();
    }
  }
}

// -------------------------------------------------------------------------------------------------
// TCP Client Transport
// -------------------------------------------------------------------------------------------------

class TcpClientTransport final : public ClientTransport {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(TcpClientTransport);

 public:
  TcpClientTransport(
      std::shared_ptr<ClientReceiveHolder> receiveHolder,
      std::shared_ptr<StatusHolder> statusHolder,
      uint64_t version)
      : _receiveHolder(std::move(receiveHolder)),
        _statusHolder(std::move(statusHolder)),
        _version(version) {}

  ~TcpClientTransport() override {
    Stop();
  }

  void Start(std::string address, uint16_t port) {
    // Record the endpoint before any early return so GetAddress reflects the requested target even
    // if the connection never establishes. Written once here, before this transport is published to
    // ClientSocket::Impl::transport, so later reads under that guard need no extra synchronization.
    _address = address;
    _port = port;

    // During the connect-retry phase there is no connected socket to shutdown(), so the
    // connect/send threads are woken for teardown only by the wake pipe. Without it Stop() could
    // hang, so report the connection lost instead of starting. A client may warn on socket errors.
    if (!_wakePipe.IsValid()) {
      MOCHI_LOG_WARNING("mochi::net: wake pipe unavailable; cannot connect");
      _statusHolder->Deliver(SocketStatus::Lost);
      return;
    }
    _running.store(true);
    _thread = std::thread([this, address = std::move(address), port] {
      marl::Thread::setName("Mochi Net Client");
      RunConnectLoop(address, port);
    });
  }

  void GetAddress(std::string& outAddress, uint16_t& outPort) const override {
    outAddress = _address;
    outPort = _port;
  }

  bool Send(void const* data, size_t size) override {
    if (!_running.load() || _statusHolder->status.load() == SocketStatus::Lost) {
      return false;
    }
    DynamicArray<uint8_t> payload;
    payload.resize_noinit(size);
    if (size > 0) {
      std::memcpy(payload.data(), data, size);
    }
    _outbox.Push(std::move(payload));
    return true;
  }

 private:
  // Teardown: stop retrying/sending, wake any blocked recv()/send() by half-closing the connected
  // socket, then join. The connect-loop thread performs the single-owner close of the fd once both
  // it and its send thread are done.
  void Stop() {
    _running.store(false);
    _outbox.Shutdown();
    _wakePipe.Signal();
    _socket.Read([](Socket const& s) { s.Shutdown(); });
    if (_thread.joinable()) {
      CheckNotCurrentThread(
          _thread, "ClientSocket::Disconnect must not be called from a TCP client callback.");
      _thread.join();
    }
  }

  void RunConnectLoop(std::string const& address, uint16_t port) {
    _statusHolder->Deliver(SocketStatus::Pending);

    while (_running.load()) {
      SocketHandle const raw = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (raw == kInvalidSocket) {
        SleepInterruptible(_wakePipe.ReadEnd(), kConnectRetryDelay);
        continue;
      }
      Socket sock(raw);
      if (!SetNonBlocking(sock.Get())) {
        SleepInterruptible(_wakePipe.ReadEnd(), kConnectRetryDelay);
        continue;
      }

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      // IPv4 literal only (no DNS). An unparseable address can never succeed, so fail fast and stop
      // retrying instead of spinning in Pending forever.
      if (inet_pton(AF_INET, address.c_str(), &addr.sin_addr) != 1) {
        MOCHI_LOG_ERROR("mochi::net: invalid IPv4 address '%s'; not retrying", address.c_str());
        _outbox.Clear();
        _running.store(false);
        _statusHolder->Deliver(SocketStatus::Lost);
        return;
      }

      if (connect(sock.Get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        int const error = LastSocketError();
        if (!IsConnectInProgress(error) || !WaitForConnect(sock.Get(), _wakePipe.ReadEnd())) {
          if (_running.load()) {
            SleepInterruptible(_wakePipe.ReadEnd(), kConnectRetryDelay);
          }
          continue;
        }
      }

      // The connected socket is normally blocking: the recv loop relies on blocking reads woken by
      // shutdown() from Stop(). The send thread uses non-blocking send flags where supported; on
      // Windows it falls back to a conservative send timeout because MSG_DONTWAIT is unavailable.
      if (!SetBlocking(sock.Get())) {
        if (_running.load()) {
          SleepInterruptible(_wakePipe.ReadEnd(), kConnectRetryDelay);
        }
        continue;
      }
      SetNoSigPipe(sock.Get());
      SetTcpNoDelay(sock.Get());
      SetSendTimeout(sock.Get(), kWindowsSendTimeout);

      // Send the connection handshake before anything else. If the server rejects it (magic,
      // protocol, or user-version mismatch) it simply closes the socket, which the recv loop below
      // observes as a disconnect.
      if (!SendHandshake(sock.Get(), _version)) {
        SleepInterruptible(_wakePipe.ReadEnd(), kConnectRetryDelay);
        continue;
      }

      RunConnectedSession(std::move(sock));

      if (!_running.load()) {
        break;
      }

      // A dropped established connection is terminal for this client. App-level code can decide
      // whether to create/connect another client.
      _outbox.Clear();
      _running.store(false);
      break;
    }
  }

  // Run one connected session: hand the fd to _socket, spin up the send thread, and run the recv
  // loop on this thread until the connection drops or teardown is requested. Closes the fd once
  // both threads are finished with it (single-owner close).
  void RunConnectedSession(Socket sock) {
    SocketHandle const fd = sock.Get();
    _socket.Store(std::move(sock));
    _statusHolder->Deliver(SocketStatus::Connected);

    std::thread sendThread([this, fd] {
      marl::Thread::setName("Mochi Net Send");
      while (_running.load() && _statusHolder->status.load() == SocketStatus::Connected) {
        DynamicArray<uint8_t> payload;
        if (_outbox.Wait() && _outbox.Pop(payload)) {
          if (!SendFramedInterruptible(
                  fd, payload.data(), payload.size(), _wakePipe.ReadEnd(), _running)) {
            break;
          }
        }
      }
    });

    while (_running.load() && _statusHolder->status.load() == SocketStatus::Connected) {
      DynamicArray<uint8_t> data;
      if (!RecvFramed(fd, data)) {
        break;
      }
      _receiveHolder->Deliver(data.data(), data.size());
    }

    _statusHolder->Deliver(SocketStatus::Lost);

    // Wake the send thread (Shutdown also aborts a blocked send) and join it, then close the fd.
    // Both threads are done with the fd before it closes, so the close is single-owner and the fd
    // is never read after close.
    _outbox.Shutdown();
    _socket.Read([](Socket const& s) { s.Shutdown(); });
    if (sendThread.joinable()) {
      sendThread.join();
    }
    _socket.Store(Socket{});
  }

  std::shared_ptr<ClientReceiveHolder> _receiveHolder;
  std::shared_ptr<StatusHolder> _statusHolder;
  uint64_t _version{0};

  // Target server endpoint, set once in Start before publish and read-only thereafter (empty/0 if
  // never started).
  std::string _address;
  uint16_t _port{0};

  std::atomic<bool> _running{false};
  std::thread _thread;
  // The connected fd, owned here. Guarded so Stop() (any thread) can half-close it to wake the recv
  // loop while the connect-loop thread owns its full lifecycle.
  Guarded<Socket> _socket;
  WakePipe _wakePipe;
  PayloadQueue _outbox;
};

} // namespace

// -------------------------------------------------------------------------------------------------
// ServerSocket::Impl
// -------------------------------------------------------------------------------------------------

struct ServerSocket::Impl {
  // Callbacks live in shared_ptr-owned holders so they survive transport replacement and can be set
  // before/after hosting starts.
  std::shared_ptr<ServerReceiveHolder> receiveHolder{std::make_shared<ServerReceiveHolder>()};
  std::shared_ptr<ClientEventHolder> eventHolder{std::make_shared<ClientEventHolder>()};

  // The active transport (TCP or in-proc), or null when not hosting. Methods copy the shared_ptr
  // under the lock (Load), release the lock, then call through the copy. Start/Stop use Exchange so
  // the previous transport's destructor (which joins threads) runs outside the lock.
  Guarded<std::shared_ptr<ServerTransport>> transport;

  // User-defined connection version. Set via SetVersion before hosting; captured when a transport
  // is created.
  uint64_t version{0};

  // Discovery UDP port. 0 disables discovery for this server.
  uint16_t discoveryPort{0};
};

ServerSocket::ServerSocket() : _impl(std::make_unique<Impl>()) {}

ServerSocket::~ServerSocket() {
  Stop();
}

void ServerSocket::Start(uint16_t preferredPort, int maxClients, std::string_view label) {
  Stop();
  PlatformInit();
  auto transport = std::make_shared<TcpServerTransport>(
      _impl->receiveHolder, _impl->eventHolder, _impl->version, _impl->discoveryPort);
  if (transport->Start(preferredPort, maxClients, DynamicString(label.data(), label.size()))) {
    _impl->transport.Store(std::move(transport));
  }
}

void ServerSocket::StartInProc(int maxClients, std::string_view label) {
  Stop();
  _impl->transport.Store(
      std::make_shared<InProcServerTransport>(
          _impl->receiveHolder,
          _impl->eventHolder,
          maxClients,
          DynamicString(label.data(), label.size()),
          _impl->version));
}

void ServerSocket::SetVersion(uint64_t version) {
  MOCHI_ASSERT(
      _impl->transport.Load() == nullptr,
      "ServerSocket::SetVersion must be called before Start / StartInProc");
  _impl->version = version;
}

void ServerSocket::SetDiscoveryPort(uint16_t discoveryPort) {
  MOCHI_ASSERT(
      _impl->transport.Load() == nullptr,
      "ServerSocket::SetDiscoveryPort must be called before Start / StartInProc or after Stop");
  _impl->discoveryPort = discoveryPort;
}

void ServerSocket::Stop() {
  MOCHI_ASSERT(!InInProcCallback(), "ServerSocket::Stop must not be called from a net callback.");

  // Exchange under the lock, drop outside it: the old transport's destructor (joining threads,
  // transitioning in-proc clients to Lost) never runs while the Guarded lock is held.
  auto transport = _impl->transport.Exchange({});
}

uint16_t ServerSocket::GetPort() const {
  auto transport = _impl->transport.Load();
  return transport ? transport->GetPort() : 0;
}

void ServerSocket::SetLabel(std::string_view label) {
  auto transport = _impl->transport.Load();
  if (transport) {
    transport->SetLabel(label);
  }
}

DynamicArray<ClientId> ServerSocket::GetClients() const {
  auto transport = _impl->transport.Load();
  return transport ? transport->GetClients() : DynamicArray<ClientId>{};
}

bool ServerSocket::SendTo(ClientId client, void const* data, size_t size) {
  auto transport = _impl->transport.Load();
  return transport ? transport->SendTo(client, data, size) : false;
}

void ServerSocket::Broadcast(void const* data, size_t size) {
  auto transport = _impl->transport.Load();
  if (transport) {
    transport->Broadcast(data, size);
  }
}

void ServerSocket::SetReceiveCallback(
    std::function<void(ClientId, void const* data, size_t size)> callback) {
  _impl->receiveHolder->callback.Store(std::move(callback));
}

void ServerSocket::SetClientCallback(std::function<void(ClientId, ClientEvent)> callback) {
  _impl->eventHolder->callback.Store(std::move(callback));
}

// -------------------------------------------------------------------------------------------------
// ClientSocket::Impl
// -------------------------------------------------------------------------------------------------

struct ClientSocket::Impl {
  // Callbacks live in shared_ptr-owned holders so they survive transport replacement (reconnect)
  // and can be set before/after connecting. The StatusHolder also backs the lock-free GetStatus().
  std::shared_ptr<ClientReceiveHolder> receiveHolder{std::make_shared<ClientReceiveHolder>()};
  std::shared_ptr<StatusHolder> statusHolder{std::make_shared<StatusHolder>()};

  // The active transport (TCP or in-proc), or null when not connected. See ServerSocket::Impl for
  // the copy-under-lock / Exchange pattern.
  Guarded<std::shared_ptr<ClientTransport>> transport;

  // User-defined connection version. Set via SetVersion before connecting; captured when a
  // transport is created. `connectionStarted` enforces the call-before-connect ordering.
  uint64_t version{0};
  bool connectionStarted{false};
};

ClientSocket::ClientSocket() : _impl(std::make_unique<Impl>()) {}

ClientSocket::~ClientSocket() {
  Disconnect();
}

void ClientSocket::Connect(std::string_view address, uint16_t port) {
  Disconnect();
  PlatformInit();
  _impl->connectionStarted = true;
  auto transport = std::make_shared<TcpClientTransport>(
      _impl->receiveHolder, _impl->statusHolder, _impl->version);
  transport->Start(std::string(address), port);
  _impl->transport.Store(std::move(transport));
}

void ClientSocket::ConnectInProc(ServerSocket& server) {
  Disconnect();
  _impl->connectionStarted = true;

  // Hold the server transport alive for the duration of the connect (friend access to the server's
  // Impl), then reach the in-proc server through the abstract handle.
  auto serverTransport = server._impl->transport.Load();
  InProcServerTransport* inProc = serverTransport ? serverTransport->AsInProc() : nullptr;
  // A live transport that is not in-proc means a TCP server was passed: that is a misuse of
  // ConnectInProc, not a runtime connection failure.
  MOCHI_ASSERT(
      serverTransport == nullptr || inProc != nullptr,
      "ClientSocket::ConnectInProc requires an in-process server (ServerSocket::StartInProc); a "
      "TCP server (ServerSocket::Start) cannot be connected to in-process.");
  if (inProc == nullptr) {
    _impl->statusHolder->Deliver(SocketStatus::Lost);
    return;
  }

  auto transport = std::make_shared<InProcClientTransport>(
      _impl->receiveHolder, _impl->statusHolder, _impl->version);
  ClientId const id = transport->ConnectTo(*inProc);
  if (id != 0) {
    // Publish the transport before delivering any callback, so a Send issued from the Connected
    // status or client-connected event callback finds it. Deliver the client's Connected status and
    // then the server's Connected event inside one pump, so sends from either are queued and
    // flushed in order (and the client sees Connected before any message).
    _impl->transport.Store(std::move(transport));
    PumpInProc([&] {
      _impl->statusHolder->Deliver(SocketStatus::Connected);
      inProc->FireConnected(id);
    });
  } else {
    // Rejected (server full or version mismatch): transport drops here, never published.
    PumpInProc([&] { _impl->statusHolder->Deliver(SocketStatus::Lost); });
  }
}

void ClientSocket::SetVersion(uint64_t version) {
  MOCHI_ASSERT(
      !_impl->connectionStarted,
      "ClientSocket::SetVersion must be called before Connect / ConnectInProc");
  _impl->version = version;
}

void ClientSocket::Disconnect() {
  MOCHI_ASSERT(
      !InInProcCallback(), "ClientSocket::Disconnect must not be called from a net callback.");

  // Exchange under the lock, drop outside it: the old transport's destructor tears down (TCP joins
  // threads; in-proc leaves the server, firing its disconnect event).
  auto transport = _impl->transport.Exchange({});
  // Drop the transport now (running its teardown, which for TCP delivers SocketStatus::Lost) so the
  // SocketStatus::None below is the last status delivered, rather than racing the destructor's
  // Lost.
  transport.reset();
  _impl->statusHolder->Deliver(SocketStatus::None);

  _impl->connectionStarted = false;
}

SocketStatus ClientSocket::GetStatus() const {
  return _impl->statusHolder->status.load();
}

void ClientSocket::GetAddress(std::string& outAddress, uint16_t& outPort) const {
  // Read the endpoint straight off the active transport, atomically with the transport itself: a
  // null transport (never connected / disconnected) or an in-process transport yields empty/0.
  _impl->transport.Read([&](std::shared_ptr<ClientTransport> const& transport) {
    if (transport) {
      transport->GetAddress(outAddress, outPort);
    } else {
      outAddress.clear();
      outPort = 0;
    }
  });
}

bool ClientSocket::Send(void const* data, size_t size) {
  auto transport = _impl->transport.Load();
  return transport ? transport->Send(data, size) : false;
}

void ClientSocket::SetReceiveCallback(std::function<void(void const* data, size_t size)> callback) {
  _impl->receiveHolder->callback.Store(std::move(callback));
}

void ClientSocket::SetStatusCallback(std::function<void(SocketStatus)> callback) {
  _impl->statusHolder->callback.Store(std::move(callback));
}

// Collect this host's own unicast IPv4 addresses (host order). Best-effort: on any
// failure returns an empty list (the caller still treats 127/8 as local).
static DynamicArray<uint32_t> CollectLocalHostAddresses() {
  DynamicArray<uint32_t> result;

#if MOCHI_PLATFORM_WINDOWS
  // Windows: resolving the local hostname reliably yields every adapter's IPv4 address, so
  // gethostname()+getaddrinfo() suffices and needs no extra platform link dependency.
  char host[256];
  if (gethostname(host, sizeof(host)) != 0) {
    return result;
  }
  host[sizeof(host) - 1] = '\0';

  addrinfo hints{};
  hints.ai_family = AF_INET; // IPv4 only: discovery is IPv4.
  hints.ai_socktype = SOCK_DGRAM;

  addrinfo* infos = nullptr;
  if (getaddrinfo(host, nullptr, &hints, &infos) != 0) {
    return result;
  }
  for (addrinfo const* it = infos; it != nullptr; it = it->ai_next) {
    if (it->ai_family != AF_INET || it->ai_addr == nullptr) {
      continue;
    }
    auto const* sin = reinterpret_cast<sockaddr_in const*>(it->ai_addr);
    result.push_back(ntohl(sin->sin_addr.s_addr));
  }
  freeaddrinfo(infos);
#else
  // POSIX (macOS/Linux/Android): enumerate interface addresses directly with getifaddrs().
  // getaddrinfo() on the local hostname is unreliable here -- macOS commonly resolves it to
  // only 127.0.0.1 or the mDNS ".local" address -- so a real adapter IP (LAN or VPN, e.g.
  // 192.0.0.2) would be missed and the local server would double-list. getifaddrs() reports
  // every assigned address, which is exactly what the local-host check needs.
  ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) {
    return result;
  }
  for (ifaddrs const* it = ifaddr; it != nullptr; it = it->ifa_next) {
    if (it->ifa_addr == nullptr || it->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    auto const* sin = reinterpret_cast<sockaddr_in const*>(it->ifa_addr);
    result.push_back(ntohl(sin->sin_addr.s_addr));
  }
  freeifaddrs(ifaddr);
#endif

  return result;
}

// True if `addr` is an address of this host: the loopback block 127.0.0.0/8 (covers
// 127.0.0.1 and loopback-range virtual adapters), or any address in `localAddrs`.
static bool IsLocalHostAddress(in_addr addr, DynamicArray<uint32_t> const& localAddrs) {
  uint32_t const host = ntohl(addr.s_addr);
  if ((host & 0xFF000000u) == 0x7F000000u) { // 127.0.0.0/8
    return true;
  }
  for (uint32_t const local : localAddrs) {
    if (local == host) {
      return true;
    }
  }
  return false;
}

// -------------------------------------------------------------------------------------------------
// Discovery Receiver (UDP)
// -------------------------------------------------------------------------------------------------

// Owns the UDP discovery socket and the background thread that collects server responses into the
// shared result list. Destruction wakes the thread (via the wake pipe), joins it, then closes its
// fds via RAII. The result list is owned by ServerList::Impl and outlives this receiver (declared
// before it), so the held reference stays valid for the receiver's whole lifetime.
class DiscoveryReceiver {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(DiscoveryReceiver);

 public:
  DiscoveryReceiver(Socket socket, Guarded<DynamicArray<ServerInfo>>& servers)
      : _socket(std::move(socket)), _servers(servers) {
    // Gather this host's own addresses once, on the constructing thread: it keeps the
    // getaddrinfo() call off the receive thread and avoids any data race on _localAddrs.
    // Recomputed per Refresh (a new receiver), so a later VPN/adapter change is picked up.
    _localAddrs = CollectLocalHostAddresses();

    // The receive loop is woken for teardown only by the wake pipe; without it the destructor's
    // join() could hang. Discovery is best-effort, so if the pipe is unavailable simply do not
    // start the thread (yielding no results). This is client-side, so a warning is acceptable.
    if (!_wakePipe.IsValid()) {
      MOCHI_LOG_WARNING("mochi::net: wake pipe unavailable; discovery disabled");
      return;
    }
    SocketHandle const fd = _socket.Get();
    _running.store(true);
    _thread = std::thread([this, fd] {
      marl::Thread::setName("Mochi Net Discovery");
      RunReceiveLoop(fd);
    });
  }

  ~DiscoveryReceiver() {
    _running.store(false);
    _wakePipe.Signal();
    if (_thread.joinable()) {
      _thread.join();
    }
  }

 private:
  void RunReceiveLoop(SocketHandle s) {
    SocketHandle const wakeFd = _wakePipe.ReadEnd();
    while (_running.load()) {
      if (!WaitForSocket(s, wakeFd)) {
        continue;
      }

      char buf[kDiscoveryBufferSize];
      sockaddr_in fromAddr{};
      auto fromLen = static_cast<socklen_t>(sizeof(fromAddr));

      // Read up to the full buffer: a maximal reply is exactly kDiscoveryBufferSize bytes (the
      // responder caps it there), and the payload is length-delimited below (no null terminator
      // needed), so there is no reason to reserve a byte.
      auto const n =
          recvfrom(s, buf, sizeof(buf), 0, reinterpret_cast<sockaddr*>(&fromAddr), &fromLen);
      if (n <= 0 || static_cast<size_t>(n) < sizeof(uint32_t)) {
        continue;
      }

      uint32_t netMagic = 0;
      std::memcpy(&netMagic, buf, sizeof(netMagic));
      if (ntohl(netMagic) != kDiscoveryMagic) {
        continue;
      }

      // Parse the JSON payload following the magic prefix into a ServerInfo. Skip the datagram on
      // any parse failure (best-effort discovery).
      std::string const json(buf + sizeof(netMagic), static_cast<size_t>(n) - sizeof(netMagic));
      ServerInfo info;
      if (!SReflect::FromJsonString(info, json)) {
        continue;
      }

      // The sender leaves address empty; fill it in from the UDP source IP. If the reply
      // originated from this host (loopback range or any local interface address), report it
      // as 127.0.0.1. On Windows the same local server answers both the loopback probes
      // (source 127.0.0.1) and the LAN broadcast (source = a local interface address), which
      // would otherwise list one server twice under two addresses.
      if (IsLocalHostAddress(fromAddr.sin_addr, _localAddrs)) {
        info.address = DynamicString("127.0.0.1");
      } else {
        char addrStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &fromAddr.sin_addr, addrStr, sizeof(addrStr));
        info.address = DynamicString(addrStr);
      }

      // Deduplicate by address+port: a single host may answer both the broadcast and the loopback
      // probe (see ServerList::Refresh), and a probe may be retried.
      _servers.Mutate([&](DynamicArray<ServerInfo>& list) {
        for (auto const& existing : list) {
          if (existing.port == info.port && existing.address == info.address) {
            return;
          }
        }
        list.push_back(std::move(info));
      });
    }
  }

  Socket _socket;
  WakePipe _wakePipe;
  Guarded<DynamicArray<ServerInfo>>& _servers;
  DynamicArray<uint32_t> _localAddrs;
  std::atomic<bool> _running{false};
  std::thread _thread;
};

// -------------------------------------------------------------------------------------------------
// ServerList::Impl
// -------------------------------------------------------------------------------------------------

struct ServerList::Impl {
  explicit Impl(uint16_t discoveryPort_) : discoveryPort(discoveryPort_) {}

  uint16_t discoveryPort{0};
  // Declared before `receiver` so it outlives it: the receiver's thread writes here, and the
  // receiver (joined in its destructor) must be torn down before this list is destroyed.
  Guarded<DynamicArray<ServerInfo>> servers;
  std::unique_ptr<DiscoveryReceiver> receiver;
};

ServerList::ServerList(uint16_t discoveryPort) : _impl(std::make_unique<Impl>(discoveryPort)) {}

ServerList::~ServerList() = default;

void ServerList::Refresh() {
  // ServerSocket::SetDiscoveryPort(0) means "disable discovery".
  if (_impl->discoveryPort == 0) {
    return;
  }

  PlatformInit();

  // Tear down any previous receiver (joins its thread) before starting a new probe.
  _impl->receiver.reset();
  _impl->servers.Mutate([](DynamicArray<ServerInfo>& list) { list.clear(); });

  SocketHandle const raw = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (raw == kInvalidSocket) {
    return;
  }
  Socket sock(raw);
  SetSocketBroadcast(raw);

  uint32_t const requestMagic = htonl(kDiscoveryMagic);

  // Send the discovery probe to several destinations, each best-effort and independent:
  //   * INADDR_BROADCAST   - LAN broadcast, for servers on other hosts.
  //   * kLoopbackBroadcast - loopback directed broadcast (127.255.255.255). Same-host servers share
  //                          the discovery port via SO_REUSEADDR/SO_REUSEPORT, and a broadcast
  //                          datagram is delivered to *every* member of that group, so this reaches
  //                          all local servers at once (a unicast probe would hit only one). Linux
  //                          only: macOS/BSD does not deliver a broadcast on the loopback
  //                          interface.
  //   * multicast group    - same-host discovery over the loopback interface (see
  //                          SetMulticastLoopbackEgress). Multicast IS delivered to every
  //                          SO_REUSEPORT member on all platforms, so this is the path that reaches
  //                          all local servers on macOS, where the loopback broadcast above does
  //                          not.
  //   * INADDR_LOOPBACK    - loopback unicast fallback for platforms that deliver neither loopback
  //                          fan-out mechanism; reaches at least one same-host server.
  // Many hosts (e.g. devservers / containers) have no broadcast-capable route, so the LAN broadcast
  // fails outright there; the loopback probes still reach same-host servers, which bind their
  // discovery socket on INADDR_ANY. Duplicate replies (a server answering more than one probe) are
  // deduplicated by the receiver.
  auto sendProbe = [&](uint32_t address) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(_impl->discoveryPort);
    addr.sin_addr.s_addr = address;
    sendto(
        raw,
        reinterpret_cast<char const*>(&requestMagic),
        sizeof(requestMagic),
        0,
        reinterpret_cast<sockaddr*>(&addr),
        sizeof(addr));
  };
  sendProbe(INADDR_BROADCAST);
  sendProbe(htonl(kLoopbackBroadcast));
  // Configure loopback multicast egress before the multicast probe so it goes over (and is looped
  // back on) the loopback interface. This only affects the multicast send below; the broadcast
  // sends above and the unicast fallback below route normally.
  SetMulticastLoopbackEgress(raw);
  sendProbe(htonl(kDiscoveryMulticastGroup));
  sendProbe(htonl(INADDR_LOOPBACK));

  if (!SetNonBlocking(raw)) {
    return;
  }

  _impl->receiver = std::make_unique<DiscoveryReceiver>(std::move(sock), _impl->servers);
}

void ServerList::GetServers(DynamicArray<ServerInfo>& outList) const {
  outList = _impl->servers.Load();
}

} // namespace mochi::net
