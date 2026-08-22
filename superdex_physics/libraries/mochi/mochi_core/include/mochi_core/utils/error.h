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
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/log.h>

#include <string>

namespace mochi {

/**************************************************************************************************
  Error Reporting

  The Error class is used to return success or failure when calling a function. It is usually the
  last function parameter and is passed by reference. Call IsOK() to check for success. Call
  GetDescription() to find out what went wrong. Example:

      Error error;
      DoStuff(arg1, arg2, error);
      if (error.IsOK()) {
        // celebrate
      }

  By convention, functions that take an Error parameter always start with a MOCHI_ERROR_RETURN macro
  so they return quickly and safely if an error has already been set. Thus, the caller does not need
  to check the error after every call. They can simply pass the same error object to each call and
  check for errors at the end, or let the error bubble up. Example:

      void DoStuff(int a, int b, Error& error) {
        MOCHI_ERROR_RETURN(error); // do this first

        DoPartA(a, error);
        DoPartB(b, error); // does nothing if DoPartA failed
        Thing* thing = CreateThing(a, b, error); // returns nullptr if DoPartA or DoPartB failed
        UseThing(thing, error); // does nothing if any of the above failed
        DestroyThing(thing, error); // does nothing if any of the above failed
      }

  When you need to report a new error, use MOCHI_ERROR_SET. It captures the file & line number along
  with your description (must be a string literal). Example:

      MOCHI_ERROR_SET(error, "Here's why it failed");

  There are times when it does not make sense to handle errors because they don't matter, or because
  you believe they will never happen. There are inline helpers you can use in such cases. Examples:

      DoStuff(ErrorAssert{}); // MOCHI_ASSERT if it fails
      DoStuff(ErrorLog{}); // MOCHI_LOG if it fails
*/
class Error final {
 public:
  Error() = default;
  ~Error() = default;
  Error(Error&&) = default;
  Error& operator=(Error&&) = default;

  // Return true if NO error was set, else return false.
  bool IsOK() const {
    return _description == nullptr;
  }

  // Get the error message text (if any)
  char const* GetDescription() const {
    return IsOK() ? "" : _description;
  }

  // Get the name of the source file that set the error (if any)
  char const* GetFile() const {
    return IsOK() ? "" : _file;
  }

  // Get the line of source code that set the error (if any)
  int GetLine() const {
    return IsOK() ? 0 : _line;
  }

  // Format the error for logging.
  std::string ToString() const {
    // This format for file & line numer lets Visual Studio users double-click to
    // go to the source code.
    return IsOK() ? "OK" : Format("%s(%d): %s", _file, _line, _description);
  }

  // Errors are normally passed by reference, so we made the class non-copyable to
  // prevent mistakes. However, it is possible to make an explicit copy if you ever
  // need to store the Error for later use.
  Error Copy() const {
    return Error{*this};
  }

  // SetFirstError is usually called by the MOCHI_ERROR_SET macro.
  // It stores the first error and ignores subsequent errors.
  MOCHI_NO_INLINE void SetFirstError(char const* description, char const* file, int line) {
    if (IsOK()) {
      _description = description;
      _file = file;
      _line = line;
    }
  }

 private:
  // No implicit copy
  Error(Error const&) = default; // Used by Error::Copy()
  Error& operator=(Error const&) = default;

  char const* _description = nullptr; // nullptr means "OK"
  char const* _file; // NOLINT(cppcoreguidelines-pro-type-member-init) - Lazy init
  int _line; // NOLINT(cppcoreguidelines-pro-type-member-init) - Lazy init
};

/**************************************************************************************************
  Error Macros
*/

// Use MOCHI_ERROR_RETURN at the top of every function that takes an Error* argument.
// Returns immediately if an error has already been set. Note: The static_cast supports wrappers
// like ErrorAssert being used directly.
#define MOCHI_ERROR_RETURN(error, ...)                 \
  if (!static_cast<mochi::Error const&>(error).IsOK()) \
    MOCHI_UNLIKELY {                                   \
      return __VA_ARGS__;                              \
    }

// Use MOCHI_ERROR_SET to indicate that something went wrong (see notes on the Error class above).
// Ignored if an error was already set. Example: MOCHI_ERROR_SET(error, "Here's why it failed");
// Note: The static_cast supports wrappers like ErrorAssert being used directly.
#define MOCHI_ERROR_SET(error, descriptionStringLiteral) \
  static_cast<mochi::Error&>(error).SetFirstError("" descriptionStringLiteral, __FILE__, __LINE__);

// Sets an error and returns if the condition is true
#define MOCHI_ERROR_IF(condition, error, descriptionStringLiteral) \
  if (condition)                                                   \
    MOCHI_UNLIKELY {                                               \
      MOCHI_ERROR_SET(error, descriptionStringLiteral);            \
    }                                                              \
  else {                                                           \
  }

// Sets and error and returns if the condition is false
#define MOCHI_ERROR_IF_NOT(condition, error, descriptionStringLiteral) \
  MOCHI_ERROR_IF(!(condition), error, descriptionStringLiteral)

// Sets an error to indicate that a particular function still needs to be implemented.
// MSVC supports compile-time concatenation of __FUNCTION__ with a string literal, but clang/gcc
// does not.
#if MOCHI_COMPILER_MSVC
#define MOCHI_ERROR_NOT_IMPLEMENTED(error)         \
  static_cast<mochi::Error&>(error).SetFirstError( \
      __FUNCTION__ " not implemented", __FILE__, __LINE__)
#else
#define MOCHI_ERROR_NOT_IMPLEMENTED(error) \
  static_cast<mochi::Error&>(error).SetFirstError("Not implemented", __FILE__, __LINE__)
#endif

/**************************************************************************************************
  ErrorAssert:
      If you are certain that a call will never fail, then you can pass ErrorAssert{} in place of
      the Error& argument. If you were wrong, your mistake will be reported via a MOCHI_ASSERT. By
      default, asserts are fatal unless a debugger is connected (then break point). You can
      customize that behavior (see Debug.h).

  Example:
      DoStuff(arg1, arg2, ErrorAssert{});

      // Equivalent to:
      Error error;
      DoStuff(arg1, arg2, error);
      MOCHI_ASSERT(error.IsOK(), "[MOCHI ERROR]...");

*/
class ErrorAssert final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ErrorAssert);

 public:
  ErrorAssert() = default;
  ~ErrorAssert() {
#if MOCHI_ASSERT_ENABLED
    if (!_error.IsOK())
      MOCHI_UNLIKELY {
        MOCHI_ASSERT_ON_FAILURE(
            _error.GetFile(), _error.GetLine(), "error.IsOK()", "%s", _error.GetDescription());
      }
#endif // MOCHI_ASSERT_ENABLED
  }

  operator Error&() {
    return _error;
  }
  Error _error;
};

/**************************************************************************************************
  ErrorLog:
      If you do not plan to write custom error handling code, then consider passing ErrorLog{} in
      place of the Error& argument. It will automatically report any error using the MOCHI_LOG
      mechanism.

  Example:
      DoStuff(arg1, arg2, ErrorLog{});
*/
class ErrorLog final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ErrorLog);

 public:
  ErrorLog(LogChannel channel = LogChannel::Error) : _channel(channel) {}
  ~ErrorLog() {
    if (!_error.IsOK())
      MOCHI_UNLIKELY {
        MOCHI_LOG_IMPL(
            _channel,
            _error.GetFile(),
            _error.GetLine(),
            "[MOCHI ERROR] %s",
            _error.GetDescription());
      }
  }

  operator Error&() {
    return _error;
  }
  bool IsOK() const {
    return _error.IsOK();
  }
  LogChannel _channel;
  Error _error;
};

} // namespace mochi
