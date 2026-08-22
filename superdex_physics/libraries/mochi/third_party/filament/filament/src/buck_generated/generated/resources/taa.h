#ifndef TAA_H_
#define TAA_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t TAA_PACKAGE[];
}

#define TAA_TAA_OFFSET 0
#define TAA_TAA_SIZE 50081
#define TAA_TAA_DATA (TAA_PACKAGE + TAA_TAA_OFFSET)

#endif
