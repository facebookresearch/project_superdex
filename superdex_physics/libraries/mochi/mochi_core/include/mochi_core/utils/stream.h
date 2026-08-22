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

#include <mochi_core/utils/reflection.h>

// The types in this header require reflection because StreamReader and StreamWriter implement an
// interface form the simple_reflection library.
#if MOCHI_USE_REFLECTION

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace mochi {

// ------------------------------------------------------------------------------------------------
// StreamReader Interface
// ------------------------------------------------------------------------------------------------

/**
 * @brief Generic interface for reading binary data from an input stream.
 *
 * @note Compatible with Simple Reflection's binary serialization APIs.
 * @note The virtual call overhead can be avoided if your code uses one of the concrete
 * implementation classes directly.
 *
 * @see StreamWriter
 */
class StreamReader : public SReflect::StreamReader {
 public:
  /**
   * @brief Read the next numBytes and copy them to the destination address.
   *
   * @param[in] dst Address to which data will be written.
   * @param[in] numBytes Number of bytes to be copied.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   */
  virtual void Read(void* dst, size_t numBytes, Error& error) = 0;

  /**
   * @brief Get the current stream position.
   */
  virtual size_t GetPosition() const = 0;

 private:
  // SReflect::StreamReader override:
  bool Read(void* dst, size_t numBytes) final {
    Error error;
    Read(dst, numBytes, error);
    return error.IsOK();
  }
};

// ------------------------------------------------------------------------------------------------
// StreamWriter Interface
// ------------------------------------------------------------------------------------------------

/**
 * @brief Generic interface for writing binary data to an output stream.
 *
 * @note Compatible with Simple Reflection's binary serialization APIs.
 * @note The virtual call overhead can be avoided if your code uses one of the concrete
 * implementation classes directly.
 *
 * @see StreamReader
 */
class StreamWriter : public SReflect::StreamWriter {
 public:
  /**
   * @brief Read numBytes from the source address, write them to the end of the stream, and advance
   * the stream position.
   *
   * @param[in] src Address from which data will be read.
   * @param[in] numBytes Number of bytes to be copied.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   */
  virtual void Write(void const* src, size_t numBytes, Error& error) = 0;

  /**
   * @brief Rewrite numBytes at a specified position within the stream, if possible.
   *
   * @warning This does not extend the stream. The value of (position + numBytes) must be <= the
   * current stream position.
   *
   * @param[in] position Position (offset) of the first byte to write, within the output stream.
   * @param[in] src Address from which data will be read.
   * @param[in] numBytes Number of bytes to be copied.
   * @param[in,out] error Error status. Check @ref Error::IsOK for success.
   */
  virtual void WriteAt(size_t position, void const* src, size_t numBytes, Error& error) = 0;

  /**
   * @brief Get the current stream position.
   */
  [[nodiscard]] virtual size_t GetPosition() const = 0;

 private:
  // SReflect::StreamWriter override:
  bool Write(void const* src, size_t numBytes) final {
    Error error;
    Write(src, numBytes, error);
    return error.IsOK();
  }
};

// ------------------------------------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------------------------------------

/**
 * @brief Read a trivially-copyable object from binary stream.
 *
 * @tparam T Object type to write
 * @tparam StreamReaderT StreamReader or a derived type (no virtual call overhead if it is final)
 * @param[in] obj Object to write
 * @param[in] stream Stream to write to.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
template <class T, class StreamReaderT>
inline void StreamRead(T& obj, StreamReaderT& stream, Error& error) {
  static_assert(
      std::is_trivially_copyable_v<T>,
      "The object must be trivially copyable to read it from a binary stream in this way");
  stream.Read(&obj, sizeof(obj), error);
}

/**
 * @brief Write a trivially-copyable object to a binary stream.
 *
 * @tparam T Object type to write
 * @tparam StreamWriterT StreamWriter or a derived type (no virtual call overhead if it is final).
 * @param[in] obj Object to write
 * @param[in] stream Stream to write to.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
template <class T, class StreamWriterT>
inline void StreamWrite(T const& obj, StreamWriterT& stream, Error& error) {
  static_assert(
      std::is_trivially_copyable_v<T>,
      "The object must be trivially copyable to write it to a binary stream in this way");
  stream.Write(&obj, sizeof(obj), error);
}

/**
 * @brief Write a trivially-copyable object to a specific position within a binary stream.
 *
 * @tparam T Object type to write
 * @tparam StreamWriterT StreamWriter or a derived type (no virtual call overhead if it is final).
 * @param[in] position Stream position to which bytes should be written.
 * @param[in] obj Object to write
 * @param[in] stream Stream to write to.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
template <class T, class StreamWriterT>
inline void StreamWriteAt(size_t position, T const& obj, StreamWriterT& stream, Error& error) {
  static_assert(
      std::is_trivially_copyable_v<T>,
      "The object must be trivially copyable to write it to a binary stream in this way");
  stream.WriteAt(position, &obj, sizeof(obj), error);
}

// ------------------------------------------------------------------------------------------------
// StreamReader/StreamWriter Implementations
// ------------------------------------------------------------------------------------------------

/**
 * @brief Binary StreamReader implementation which reads data from a @ref Span<uint8_t const>.
 *
 * @see StreamReader
 */
class SpanStreamReader final : public StreamReader {
 public:
  SpanStreamReader(Span<uint8_t const> src) : _src(src) {}

  // StreamReader overrides
  void Read(void* dst, size_t numBytes, Error& error) final {
    MOCHI_ERROR_IF(numBytes > GetNumBytesRemaining(), error, "End of stream");
    MOCHI_ERROR_RETURN(error);
    if (numBytes)
      MOCHI_LIKELY {
        MOCHI_ASSERT_VERBOSE(dst != nullptr);
        std::memcpy(dst, &_src[_pos], numBytes);
        _pos += numBytes;
      }
  }
  [[nodiscard]] size_t GetPosition() const final {
    return _pos;
  }

  /**
   * @brief Advance the stream position by @p numBytes without reading the data.
   */
  void Advance(size_t numBytes, Error& error) {
    MOCHI_ERROR_IF(numBytes > GetNumBytesRemaining(), error, "End of stream");
    MOCHI_ERROR_RETURN(error);
    _pos += numBytes;
  }

  [[nodiscard]] size_t GetNumBytesRemaining() const {
    return _src.size() - _pos;
  }

 private:
  Span<uint8_t const> _src;
  size_t _pos = 0;
};

/**
 * @brief Binary StreamWriter implementation which appends data to a DynamicArray<uint8_t>.
 *
 * @see StreamWriter
 */
class DynamicArrayStreamWriter final : public StreamWriter {
 public:
  DynamicArrayStreamWriter(DynamicArray<uint8_t>& dst) : _dst(dst) {}

  // StreamWriter overrides:
  void Write(void const* src, size_t numBytes, Error& error) final {
    MOCHI_ASSERT_VERBOSE(src || !numBytes, "Null pointer");
    MOCHI_ERROR_RETURN(error);
    auto const* srcBegin = static_cast<uint8_t const*>(src);
    _dst.append(srcBegin, srcBegin + numBytes);
  }
  void WriteAt(size_t position, void const* src, size_t numBytes, Error& error) final {
    MOCHI_ASSERT_VERBOSE(src || !numBytes, "Null pointer");
    MOCHI_ERROR_IF(position + numBytes > _dst.size(), error, "Write position out-of-bounds");
    if (error.IsOK() && numBytes)
      MOCHI_LIKELY {
        std::memcpy(&_dst[position], src, numBytes);
      }
  }
  [[nodiscard]] size_t GetPosition() const final {
    return _dst.size();
  }

 private:
  DynamicArray<uint8_t>& _dst;
};

} // namespace mochi

#endif // MOCHI_USE_REFLECTION
