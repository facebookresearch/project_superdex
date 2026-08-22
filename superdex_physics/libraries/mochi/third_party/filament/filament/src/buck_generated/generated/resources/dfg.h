#ifndef DFG_H_
#define DFG_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t DFG_PACKAGE[];
    // This package is compressed with zstd.
    // The entire package is compressed as a single blob.
    // The _SIZE and _OFFSET macros are for the uncompressed data.
    // You must decompress the entire package before using them.
    // Original package size: 98304 bytes.
    // Compressed package size: 77133 bytes.

}

#define DFG_DFG_OFFSET 0
#define DFG_DFG_SIZE 98304
#endif
